"""The vectors of one index in memory, and nearest-neighbour search over them.

Two implementations with the same interface, chosen per index when it is
created:

- `FlatVectors` (the default): exact search, FAISS's flat index. Every query
  compares against every vector, which is fast up to some hundreds of
  thousands of vectors and is always exactly right.
- `HnswVectors`: approximate search over an HNSW graph. Much faster on large
  indexes (50,000 clustered 384-d vectors: ~0.08 ms a query instead of
  ~3.7 ms, finding the true top 10 ~100% of the time), at the cost of a graph
  that takes seconds to build and ~300 bytes a vector.

Vectors are identified by int64 labels the caller assigns. Callers serialise
all calls with their own lock, which `HnswVectors` also takes (as `lock`)
when a background graph build finishes.
"""

import itertools
import logging
import os
import threading

import faiss
import numpy as np

logger = logging.getLogger(__name__)

EXACT = "exact"
APPROXIMATE = "approximate"
# HNSW parameters: M links per node, and the candidate-list sizes used while
# building and while searching. Larger means better recall, slower, bigger.
DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 64
DEFAULT_EF_SEARCH = 64
# Below this many vectors a graph isn't worth building: exact search is
# already well under a millisecond.
MIN_GRAPH_VECTORS = 2000
# A filter matching at most this many vectors (or this fraction of the
# index) is searched exactly over just those: as cheap as walking the graph,
# and exact, where a graph search restricted to a few nodes loses recall.
EXACT_FILTER_ROWS = 20000
EXACT_FILTER_FRACTION = 0.05
# Deleted vectors stay in the graph, skipped by searches, until they are more
# than this fraction of it (and at least MIN_GRAPH_VECTORS); then the vectors
# are compacted and the graph rebuilt.
MAX_DEAD_FRACTION = 0.25
# On an exact index, a filter matching at most this fraction of it is
# answered by copying out just the matching vectors and comparing against
# those; a broader one by a full scan that skips the rest. Measured on
# 100,000 vectors: copying wins below ~4% at 384 dimensions (1%: 0.26 vs
# 0.40 ms), and by more with fewer dimensions; scanning wins above.
FLAT_GATHER_FRACTION = 0.03
# Writes of at least this many vectors to an index with a graph are linked
# into it with the build threads.
PARALLEL_ADD = 256


def search_config(mode: str | None = None, **params) -> dict:
    """The manifest's "search" setting for an index: {"mode", ...params}."""
    mode = mode or EXACT
    if mode == EXACT:
        if params:
            raise ValueError("exact search takes no parameters")
        return {"mode": EXACT}
    if mode != APPROXIMATE:
        raise ValueError(f"search must be '{EXACT}' or '{APPROXIMATE}', not {mode!r}")
    out = {"mode": APPROXIMATE, "m": DEFAULT_HNSW_M, "ef_construction": DEFAULT_EF_CONSTRUCTION,
           "ef_search": DEFAULT_EF_SEARCH}
    for key, value in params.items():
        if key not in out or key == "mode":
            raise ValueError(f"unknown approximate search parameter {key!r}")
        if value is None:
            continue
        limits = {"m": (4, 128), "ef_construction": (8, 1024), "ef_search": (8, 4096)}[key]
        if not isinstance(value, int) or isinstance(value, bool) or not limits[0] <= value <= limits[1]:
            raise ValueError(f"{key} must be an integer from {limits[0]} to {limits[1]}")
        out[key] = value
    return out


def build_threads_from_env() -> int:
    """S3VAH_GRAPH_BUILD_THREADS: threads one graph build may use (default:
    half the cores, so serving keeps the rest)."""
    raw = os.environ.get("S3VAH_GRAPH_BUILD_THREADS", "").strip()
    if not raw:
        return max(1, (os.cpu_count() or 2) // 2)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"S3VAH_GRAPH_BUILD_THREADS must be a number, not {raw!r}") from None
    if value < 1:
        raise ValueError("S3VAH_GRAPH_BUILD_THREADS must be at least 1")
    return value


def _hits(distances: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[int]]:
    # FAISS pads short result sets with label -1: drop those.
    keep = labels != -1
    return distances[keep].tolist(), labels[keep].tolist()


class FlatVectors:
    """Exact search: FAISS's flat L2 index, keyed by label."""

    mode = EXACT
    extra_bytes_per_vector = 0

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        self._index.add_with_ids(vectors, labels)

    def remove(self, labels: np.ndarray) -> None:
        if len(labels):
            self._index.remove_ids(labels)

    def search(self, q: np.ndarray, k: int, allowed=None, exact: bool = False):
        """(squared L2 distances, labels), closest first, of up to k vectors
        (among the `allowed` labels, a collection, if given)."""
        k = min(k, self.ntotal if allowed is None else len(allowed))
        if k <= 0:
            return [], []
        if allowed is None:
            distances, labels = self._index.search(q, k)
            return _hits(distances[0], labels[0])
        wanted = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
        if len(wanted) <= FLAT_GATHER_FRACTION * self.ntotal:
            # Few matches: compare against just their vectors (looked up by
            # label in C++), not a scan of the whole index.
            distances, rows = faiss.knn(q, self._index.reconstruct_batch(wanted), k)
            keep = rows[0] != -1
            return distances[0][keep].tolist(), wanted[rows[0][keep]].tolist()
        sel = faiss.IDSelectorBatch(wanted)
        distances, labels = self._index.search(q, k, params=faiss.SearchParameters(sel=sel))
        return _hits(distances[0], labels[0])

    def close(self) -> None:
        pass


class HnswVectors:
    """Approximate search: an HNSW graph, built in the background.

    Vectors live in rows of an append-only flat array; `_labels[row]` is a
    row's label and `_alive[row]` whether it's still current. A removal
    only marks its row dead (an HNSW graph can't drop nodes), and searches
    skip dead rows; once they pass MAX_DEAD_FRACTION, live rows are copied
    into a fresh array and the graph rebuilt.

    Until the graph exists, every search is exact, over the flat array. The
    graph is built from a copy of the rows there are when the build starts,
    off the caller's lock; when it's done, rows added meanwhile are added to
    it, and the graph's own storage (the same rows, in the same order)
    replaces the flat array, so the vectors are held once. A build that
    finds the array replaced meanwhile (by compaction) is thrown away.
    """

    mode = APPROXIMATE

    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        lock=None,
        builder=None,
        build_threads: int = 1,
        build: bool = True,
    ):
        """`builder` (an Executor) runs graph builds in the background, taking
        `lock` to install the result; None builds inline, under the
        caller's lock (for tests). `build=False` never builds a graph: every
        search is exact (for short-lived processes such as the CLI)."""
        if builder is not None and lock is None:
            raise ValueError("a background graph builder needs the caller's lock")
        self.dim = dim
        self.m, self.ef_construction, self.ef_search = m, ef_construction, ef_search
        self._lock = lock or threading.Lock()
        self._builder = builder
        self._build_threads = build_threads
        self._flat = faiss.IndexFlatL2(dim)  # rows; the graph's storage once there is one
        self._graph = None
        self._labels = np.empty(0, dtype=np.int64)
        self._alive = np.empty(0, dtype=bool)
        self._rows = 0
        self._row_of: dict[int, int] = {}
        self._dead = 0
        self._generation = 0  # bumped whenever the rows are compacted
        self._building = False
        self._closed = not build  # a closed index never builds or installs a graph
        self.graph_builds = 0

    # Graph links (~2M int32 at level 0, more on upper levels) plus row bookkeeping.
    @property
    def extra_bytes_per_vector(self) -> int:
        return int(self.m * 2 * 4 * 1.2) + 16 + 100

    @property
    def ntotal(self) -> int:
        return len(self._row_of)

    @property
    def has_graph(self) -> bool:
        return self._graph is not None

    def _view(self) -> np.ndarray:
        """The rows, as a (rows, dim) float32 array without copying."""
        if self._rows == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return faiss.rev_swig_ptr(self._flat.get_xb(), self._rows * self.dim).reshape(self._rows, self.dim)

    def add(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        n = len(labels)
        if n == 0:
            return
        start, end = self._rows, self._rows + n
        if end > len(self._labels):
            cap = max(end, 2 * len(self._labels), 1024)
            self._labels = np.resize(self._labels, cap)
            self._alive = np.concatenate([self._alive, np.zeros(cap - len(self._alive), dtype=bool)])
        if self._graph is not None:
            # Appends to its storage, which is self._flat, and links the new
            # nodes: ~0.1 ms a vector at 100k, so a big write gets the build
            # threads (FAISS inserts a batch in parallel).
            self._with_threads(self._build_threads if n >= PARALLEL_ADD else 1, self._graph.add, vectors)
        else:
            self._flat.add(vectors)
        self._labels[start:end] = labels
        self._alive[start:end] = True
        self._rows = end
        for row, label in enumerate(labels.tolist(), start):
            old = self._row_of.get(label)
            if old is not None:  # re-added without a remove: the new row wins
                self._alive[old] = False
                self._dead += 1
            self._row_of[label] = row
        self._maybe_build()

    def remove(self, labels: np.ndarray) -> None:
        for label in labels.tolist():
            row = self._row_of.pop(label, None)
            if row is not None:
                self._alive[row] = False
                self._dead += 1
        if self._dead > max(MIN_GRAPH_VECTORS, MAX_DEAD_FRACTION * self._rows):
            self._compact()

    def _compact(self) -> None:
        """Copy the live rows into a fresh array; the graph is rebuilt."""
        rows = np.flatnonzero(self._alive[: self._rows])
        vectors = np.ascontiguousarray(self._view()[rows])
        labels = self._labels[rows].copy()
        self._flat = faiss.IndexFlatL2(self.dim)
        self._flat.add(vectors)
        self._graph = None
        self._labels = labels
        self._alive = np.ones(len(labels), dtype=bool)
        self._rows = len(labels)
        self._row_of = dict(zip(labels.tolist(), range(len(labels))))
        self._dead = 0
        self._generation += 1
        self._maybe_build()

    def _maybe_build(self) -> None:
        if self._graph is not None or self._building or self._closed or self.ntotal < MIN_GRAPH_VECTORS:
            return
        self._building = True
        generation, rows = self._generation, self._rows
        vectors = np.array(self._view())  # a copy: the build runs without the lock
        if self._builder is None:
            self._install(generation, rows, self._build_graph(vectors))
            return
        try:
            self._builder.submit(self._build_in_background, generation, rows, vectors)
        except RuntimeError:  # the pool is shut down (interpreter exiting): stay exact
            self._building = False

    def _build_graph(self, vectors: np.ndarray):
        graph = faiss.IndexHNSWFlat(self.dim, self.m)
        graph.hnsw.efConstruction = self.ef_construction
        graph.hnsw.efSearch = self.ef_search
        self._with_threads(self._build_threads, graph.add, vectors)
        return graph

    @staticmethod
    def _with_threads(threads: int, fn, *args):
        # OpenMP's thread count is per calling thread: this doesn't change
        # the single-threaded searches the server runs on other threads.
        if threads <= 1:
            return fn(*args)
        faiss.omp_set_num_threads(threads)
        try:
            return fn(*args)
        finally:
            faiss.omp_set_num_threads(1)

    def _build_in_background(self, generation: int, rows: int, vectors: np.ndarray) -> None:
        try:
            graph = self._build_graph(vectors)
        except Exception:
            # Nobody reads the future: log it, and let the next write retry.
            logger.warning("building an HNSW graph failed; searches stay exact until a later write retries it",
                           exc_info=True)
            with self._lock:
                self._building = False
            return
        with self._lock:
            self._install(generation, rows, graph)

    def _install(self, generation: int, rows: int, graph) -> None:
        """Make `graph`, built from rows [0, rows), the graph (caller holds the lock)."""
        self._building = False
        if self._closed or generation != self._generation:
            self._maybe_build()  # compacted meanwhile: build again from the new rows
            return
        if self._rows > rows:
            graph.add(np.ascontiguousarray(self._view()[rows:]))
        self._graph = graph
        self._flat = faiss.downcast_index(graph.storage)
        self.graph_builds += 1

    def search(self, q: np.ndarray, k: int, allowed=None, exact: bool = False):
        """(squared L2 distances, labels), closest first, of up to k vectors
        (among the `allowed` labels, a collection, if given). Approximate
        once the graph is built, unless `exact`; a selective filter is
        always searched exactly."""
        if allowed is not None:
            # Label -> row in C (map over dict.get), not a Python loop: this
            # is most of the cost of a broad filter.
            rows = np.fromiter(map(self._row_of.get, allowed, itertools.repeat(-1)), np.int64, len(allowed))
            rows = rows[rows >= 0]
            candidates = len(rows)
        else:
            rows, candidates = None, self.ntotal
        k = min(k, candidates)
        if k <= 0:
            return [], []
        selective = rows is not None and candidates <= max(EXACT_FILTER_ROWS, EXACT_FILTER_FRACTION * self.ntotal)
        if self._graph is None or exact or selective:
            return self._exact(q, k, rows)
        mask = None
        if rows is not None:
            mask = np.zeros(self._rows, dtype=bool)
            mask[rows] = True
        elif self._dead:
            mask = self._alive[: self._rows]
        params = faiss.SearchParametersHNSW(efSearch=max(self.ef_search, k))
        if mask is not None:
            bits = np.packbits(mask, bitorder="little")  # must outlive the search
            params.sel = sel = faiss.IDSelectorBitmap(self._rows, faiss.swig_ptr(bits))  # noqa: F841
        distances, found = self._graph.search(q, k, params=params)
        return self._labelled(distances[0], found[0])

    def _exact(self, q: np.ndarray, k: int, rows: np.ndarray | None):
        if rows is not None and len(rows) <= EXACT_FILTER_ROWS:
            # Few rows: copy just those and compare against them.
            distances, found = faiss.knn(q, np.ascontiguousarray(self._view()[rows]), k)
            found = found[0]
            keep = found != -1
            return self._labelled(distances[0][keep], rows[found[keep]])
        if rows is None and not self._dead:
            distances, found = self._flat.search(q, k)
        else:
            if rows is None:
                mask = self._alive[: self._rows]
            else:
                mask = np.zeros(self._rows, dtype=bool)
                mask[rows] = True
            bits = np.packbits(mask, bitorder="little")
            sel = faiss.IDSelectorBitmap(self._rows, faiss.swig_ptr(bits))
            distances, found = self._flat.search(q, k, params=faiss.SearchParameters(sel=sel))
        return self._labelled(distances[0], found[0])

    def _labelled(self, distances: np.ndarray, rows: np.ndarray):
        keep = rows != -1
        return distances[keep].tolist(), self._labels[rows[keep]].tolist()

    def close(self) -> None:
        """Stop installing graphs into this (evicted or replaced) index."""
        self._closed = True
