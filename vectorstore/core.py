import contextlib
import gc
import hashlib
import json
import logging
import os
import random
import re
import struct
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import faiss
import numpy as np
from minio.error import S3Error

from minio import Minio
from vectorstore.embedding import DEFAULT_CHUNK_OVERLAP, MODEL_NAME, OnnxEmbedder
from vectorstore.filters import FilterIndex, validate_filter, validate_metadata
from vectorstore.vectors import (
    APPROXIMATE,
    FlatVectors,
    HnswVectors,
    build_threads_from_env,
    search_config,
)

logger = logging.getLogger(__name__)

BUCKET = "vectors"
DEFAULT_INDEX = "default"
MAX_DOC_ID_LENGTH = 512
MAX_TOP_K = 1000
MAX_PAGE_SIZE = 1000
# Index names: 1-63 of [a-z0-9-], not starting or ending with "-".
_INDEX_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# -- storage layout (v4) ------------------------------------------------------
# An index is a small JSON *manifest* plus immutable *segments*:
#
#   indexes/<name>.manifest.json                 the one object written with CAS
#   segments/<name>/<incarnation>/<uuid>.seg     never modified once written
#
# A segment holds records: an upsert (doc_id, text, metadata, vector) or a
# tombstone (doc_id). A write uploads one segment with its records, then
# appends it to the manifest with a conditional PUT, so its cost depends on
# the size of the write, not of the index. Every record has a sequence
# number (a segment written by a writer takes the one the manifest assigns
# it; merged segments store each record's own), and a document's current
# version is its record with the highest seq. That makes merging segments
# (compaction) invisible to readers, and lets a process that already has
# the index in memory catch up by reading only records newer than what it
# has seen.
#
# The manifest:
#   {"format": 4, "incarnation": "<hex>", "dim": 384, "next_seq": 12,
#    "segments": [{"key", "first_seq", "last_seq", "bytes", "full"}, ...],
#    "writers": {"<writer id>": [<last write counter>, <unix time>], ...}}
# "full" marks a merge of the whole index, which drops tombstones. Each
# incarnation (one create/delete lifetime of a name) has its own segment
# prefix, so deleting an index can never sweep away the segments of a newer
# index of the same name. "writers" lets a writer whose append succeeded
# but was reported as a conflict see that it already landed (see `_commit`).
FORMAT = 4
INDEX_PREFIX = "indexes/"
MANIFEST_SUFFIX = ".manifest.json"
SEGMENT_PREFIX = "segments/"
SEGMENT_MAGIC = b"S3VAH-SEG\x01"

# Compaction: after a write, segments are merged size-tiered, like a binary
# counter, so there are O(log n) of them and each record is rewritten
# O(log n) times over its life. Merges run in the background and swap the
# merged segment in with the same conditional PUT as any write. A merge
# result is capped so it stays a single PUT; `compact()` (full compaction)
# splits its output into parts of at most this size instead.
MAX_SEGMENT_BYTES = 1024**3
# A merge waits until it can take at least this many segments. Every merge
# is a manifest write, which makes concurrent writers retry, so batching
# them keeps contention down (~4x fewer than merging pairwise) without
# changing the O(log n) bounds.
MIN_MERGE = 4
# Merge sooner than the size rule would when there are more segments than this.
MAX_SEGMENTS = 64
# Safety margins for deleting segments nothing references (orphans: left by
# a crash between uploading a segment and appending it, or by a failed
# write). A writer re-uploads its segment if it has waited longer than
# SEGMENT_MAX_PENDING_S to append it, and gives up after INDEX_WRITE_TIMEOUT_S,
# so an unreferenced segment older than ORPHAN_GRACE_S can't be one that is
# about to be appended. Assumes clocks within ~45 minutes of each other.
SEGMENT_MAX_PENDING_S = 600
ORPHAN_GRACE_S = 3600
# How long a writer's entry stays in the manifest. Must exceed INDEX_WRITE_TIMEOUT_S.
WRITER_TTL_S = 3600
COMPACTION_ATTEMPTS = 5

# Older layouts, only ever read, to migrate the first time an index is used:
#   v3: one object per index, "indexes/<name>.v3.bin" (FAISS + all docs);
#   v2: root "index.v2.bin" (FAISS + {id: doc_id}) plus per-doc objects;
#   v1: root "faiss.index" + "index.ids.json" plus per-doc objects.
# v1/v2 only ever held the default index. Per-doc objects are "<doc_id>.npy"
# (vector) and "<doc_id>.meta.json" ({"doc_id", "text"}). Old objects are
# left in place after migration (nothing reads them again; delete them once
# you no longer need to roll back) -- except that deleting an index also
# deletes its v3 object, so it can't be migrated back to life.
LEGACY_V3_SUFFIX = ".v3.bin"
LEGACY_V3_MAGIC = b"S3VAH-IDX\x03"
LEGACY_V2_INDEX_OBJ = "index.v2.bin"
LEGACY_V2_MAGIC = b"S3VAH-IDX\x01"

# Reads of several segments (or of legacy per-doc objects) run in parallel.
# Kept below minio-py's urllib3 pool size (10) so workers don't queue for
# connections.
DEFAULT_IO_WORKERS = 8
# Compare-and-swap retries for manifest writes that lose a race to another
# writer. Each retry reloads the manifest; backoff is exponential with full
# jitter, capped, so contending writers spread out instead of colliding in
# lockstep. A writer keeps retrying for up to INDEX_WRITE_TIMEOUT_S: a time
# budget rather than a count, because how many attempts a write needs grows
# with the number of writers (32 processes writing flat out against local
# MinIO needed up to 36), while what a caller can wait for doesn't.
# INDEX_WRITE_ATTEMPTS is only a backstop.
INDEX_WRITE_TIMEOUT_S = 30.0
INDEX_WRITE_ATTEMPTS = 1000
INDEX_WRITE_BACKOFF_BASE_S = 0.005
INDEX_WRITE_BACKOFF_CAP_S = 0.25
# Conditional writes go out as a single PUT (conditions on multipart
# uploads are a different, less widely supported mechanism), and S3 caps a
# single PUT at 5 GiB.
MAX_SINGLE_PUT_BYTES = 5 * 1024**3
# Throwaway objects for the startup conditional-write probe.
PROBE_PREFIX = "_probe/"
# S3 error codes meaning "a conditional write lost to a concurrent change".
_CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict"}


class ObjectNotFoundError(Exception):
    """Raised when a requested object does not exist in the backing store."""


class IndexNotFoundError(LookupError):
    """The named index doesn't exist (create it first)."""


class IndexAlreadyExistsError(Exception):
    """An index with that name already exists."""


class ConditionalWritesUnsupportedError(RuntimeError):
    """The object store doesn't enforce If-Match / If-None-Match on PUT.

    Without that, concurrent writers in different processes would silently
    overwrite each other's index updates, so the store refuses to run.
    """


class IndexWriteConflictError(RuntimeError):
    """An index update kept losing to concurrent writers and gave up.

    Nothing was applied: the index is exactly as other writers left it. The
    operation can be retried.
    """


class IndexCorruptError(RuntimeError):
    """Stored index data is missing or unreadable (a referenced segment is
    gone, or an object doesn't parse). Not the caller's fault: the API
    reports it as a server error."""


class _WriteConflict(Exception):
    """Internal: one conditional PUT lost a race."""


class _SegmentMissing(Exception):
    """Internal: a segment listed in a manifest is gone (merged away meanwhile?)."""


def object_key_id(doc_id: str) -> int:
    """Deterministic 63-bit id for a doc_id, used as its FAISS vector id.

    A hash rather than insertion order keeps the vector tied to its doc_id
    through removals and updates. Two doc_ids hashing to the same id is
    astronomically unlikely, but it is detected (see `add_documents`)
    rather than allowed to silently overwrite a document.
    """
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def require_unicode(value: str, what: str) -> None:
    """Reject strings that aren't valid Unicode (lone surrogates such as
    "\\ud800", which JSON can carry): they can't be encoded as UTF-8, so the
    tokenizer or the hash of a key would fail on them later, as a 500."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{what} must be valid Unicode (it contains a lone surrogate)") from None


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length, as float32. Norms are taken in float64:
    in float32, squaring 1e20 overflows to infinity and squaring 1e-20
    underflows to zero, which would wreck (or falsely reject as all-zero)
    perfectly good vectors."""
    v64 = np.asarray(vectors, dtype=np.float64)
    return (v64 / np.linalg.norm(v64, axis=-1, keepdims=True)).astype(np.float32)


def row_norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=-1)


def validate_doc_id(doc_id: str) -> None:
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    require_unicode(doc_id, "doc_id")
    # Not an object key any more, but it is a URL path segment in the API.
    if "/" in doc_id:
        raise ValueError("doc_id must not contain '/'")
    if len(doc_id) > MAX_DOC_ID_LENGTH:
        raise ValueError(f"doc_id must be at most {MAX_DOC_ID_LENGTH} characters")


def validate_index_name(name: str) -> None:
    if not isinstance(name, str) or not _INDEX_NAME_RE.match(name):
        raise ValueError(
            "index name must be 1-63 characters of lowercase letters, digits and '-', "
            "not starting or ending with '-'"
        )


def manifest_key(name: str) -> str:
    return f"{INDEX_PREFIX}{name}{MANIFEST_SUFFIX}"


def legacy_v3_key(name: str) -> str:
    return f"{INDEX_PREFIX}{name}{LEGACY_V3_SUFFIX}"


def _segment_dir(name: str, incarnation: str | None = None) -> str:
    return f"{SEGMENT_PREFIX}{name}/" + (f"{incarnation}/" if incarnation else "")


# -- segments -----------------------------------------------------------------
# A record is (seq, doc_id, text, metadata, vector, spans); text is None for
# a tombstone (then metadata, vector and spans are None too). seq is None in
# a segment a writer uploads: it is the seq the manifest gives that segment.
# `vector` is a (block, row) reference into a 2-D float32 array, so a
# segment's vectors stay one block and are gathered with one fancy-index per
# block, not stacked row by row (which made loading a 50k-doc index 2x
# slower). `spans` is None for a document with one vector; a long text split
# into chunks has one vector per chunk, rows row..row+len(spans)-1, and
# spans [[start, end], ...] gives the characters of the text each covers.
#
# Blob: SEGMENT_MAGIC | u64 big-endian length of header JSON |
#   header JSON {"dim", "records": [[seq, doc_id, text, metadata(, spans)], ...]} |
#   float32 little-endian vectors of the upsert records, in record order.
# A record without a fifth element has one vector, so segments written
# before chunking existed read unchanged.


def _vector_count(rec: tuple) -> int:
    return len(rec[5]) if rec[5] else 1


def _encode_segment(dim: int, records: list[tuple]) -> bytes:
    header = json.dumps(
        {"dim": dim, "records": [[seq, d, t, m, sp] if sp else [seq, d, t, m] for seq, d, t, m, _, sp in records]},
        separators=(",", ":"),
    ).encode()
    upserts = [r for r in records if r[2] is not None]
    body = (
        _gather([r[4] for r in upserts], dim, [_vector_count(r) for r in upserts]).astype("<f4", copy=False).tobytes()
        if upserts
        else b""
    )
    return b"".join((SEGMENT_MAGIC, struct.pack(">Q", len(header)), header, body))


def _corrupt_on_error(what: str):
    """Decorate a decoder of stored data: any failure to parse it (bad magic,
    truncation, bad JSON, wrong sizes) becomes IndexCorruptError naming the
    object. It is the store's data that is broken, not the caller's
    request, so it must not surface as a ValueError (which the API reports
    as 400 Bad Request)."""

    def wrap(fn):
        def decode(blob, key, *args):
            try:
                return fn(blob, key, *args)
            except IndexCorruptError:
                raise
            except (ValueError, TypeError, KeyError, IndexError, struct.error) as e:
                raise IndexCorruptError(f"{what} {key!r} is unreadable: {e}") from e

        decode.__name__ = fn.__name__
        decode.__doc__ = fn.__doc__
        return decode

    return wrap


@_corrupt_on_error("segment")
def _decode_segment(blob: bytes, key: str, dim: int, default_seq: int) -> list[tuple]:
    if not blob.startswith(SEGMENT_MAGIC):
        raise ValueError(f"{key} is not a recognised segment")
    pos = len(SEGMENT_MAGIC)
    (header_len,) = struct.unpack_from(">Q", blob, pos)
    pos += 8
    header = json.loads(blob[pos: pos + header_len])
    if header["dim"] != dim:
        raise ValueError(f"{key} has dimension {header['dim']}, the index {dim}")
    n_vectors = sum((len(r[4]) if len(r) > 4 else 1) for r in header["records"] if r[2] is not None)
    vectors = np.frombuffer(blob, dtype="<f4", offset=pos + header_len)
    if vectors.size != n_vectors * dim:
        raise ValueError(f"{key} is corrupt: {vectors.size} floats for {n_vectors} vectors of {dim}")
    block = vectors.reshape(n_vectors, dim).astype(np.float32, copy=False)
    out, i = [], 0
    for seq, doc_id, text, metadata, *rest in header["records"]:
        seq = seq if seq is not None else default_seq
        if text is None:
            out.append((seq, doc_id, None, None, None, None))
        else:
            spans = rest[0] if rest else None
            out.append((seq, doc_id, text, metadata, (block, i), spans))
            i += len(spans) if spans else 1
    return out


def _gather(refs: list[tuple[np.ndarray, int]], dim: int, counts: list[int] | None = None) -> np.ndarray:
    """Stack (block, row) vector references into one (len(refs), dim) array.

    With `counts`, reference i stands for counts[i] consecutive rows from
    `row` on (a chunked document's vectors), and the result has sum(counts) rows.
    """
    blocks: dict[int, tuple[int, np.ndarray]] = {}
    which = np.fromiter((blocks.setdefault(id(b), (len(blocks), b))[0] for b, _ in refs), np.int64, len(refs))
    rows = np.fromiter((r for _, r in refs), np.int64, len(refs))
    if counts is not None and any(c != 1 for c in counts):
        c = np.asarray(counts, dtype=np.int64)
        firsts = np.cumsum(c) - c  # where each reference's rows start in the output
        rows = np.repeat(rows - firsts, c) + np.arange(int(c.sum()))
        which = np.repeat(which, c)
    if len(blocks) == 1:
        (_, block), = blocks.values()
        if len(rows) == len(block) and np.array_equal(rows, np.arange(len(block))):
            return np.ascontiguousarray(block, dtype=np.float32)  # the whole block, in order
        return np.ascontiguousarray(block[rows], dtype=np.float32)
    out = np.empty((len(rows), dim), dtype=np.float32)
    for i, block in blocks.values():
        mask = which == i
        out[mask] = block[rows[mask]]
    return out


def _latest(records) -> dict[str, tuple]:
    """doc_id -> its record with the highest seq."""
    latest: dict[str, tuple] = {}
    for rec in records:
        cur = latest.get(rec[1])
        if cur is None or rec[0] > cur[0]:
            latest[rec[1]] = rec
    return latest


def _plan_merge(segments: list[dict]) -> tuple[int, int] | None:
    """Which run of segments [i, j) to merge after a write, if any.

    Size-tiered, like carrying in a binary counter: the run is the newest
    segments, extended while the one before them is no bigger than their
    total, and is merged once it has MIN_MERGE segments. With equal-sized
    writes that keeps O(log n + MIN_MERGE) segments. Runs are capped at
    MAX_SEGMENT_BYTES; past MAX_SEGMENTS segments, more are merged.
    """
    n = len(segments)
    if n < 2:
        return None
    sizes = [s["bytes"] for s in segments]
    i, total = n - 1, sizes[-1]
    while i > 0 and sizes[i - 1] <= total:
        i -= 1
        total += sizes[i]
    forced = n > MAX_SEGMENTS
    if forced:
        i = min(i, MAX_SEGMENTS // 2 - 1)
    while n - i >= 2 and sum(sizes[i:]) > MAX_SEGMENT_BYTES:
        i += 1
    return (i, n) if n - i >= (2 if forced else MIN_MERGE) else None


def chunk_label(doc_id: str, chunk: int) -> int:
    """FAISS id of a document's vector for chunk `chunk`. Chunk 0's is the
    document's own id, so unchunked documents need no extra bookkeeping."""
    return object_key_id(doc_id) if chunk == 0 else object_key_id(f"{doc_id}\x00{chunk}")


class _IndexState:
    """One index in memory: its vectors plus every document's fields.

    `docs` maps a document's id (`object_key_id`) -> {"doc_id", "text",
    "metadata"}, plus "spans" and "labels" for a document split into
    chunks. `by_id` is the reverse lookup, derived from `docs`. `vectors`
    (see vectorstore.vectors) holds one vector per chunk: chunk 0 under the
    document's id, chunk i under `chunk_label(doc_id, i)`; `owner` maps each
    of the latter to (document id, i). `filters` indexes the metadata
    fields filters have used, built on the first filtered search.
    """

    __slots__ = ("_filter_entries", "_vector_bytes", "by_id", "dim", "docs", "filters", "metric", "nbytes",
                 "owner", "raw", "vectors")

    def __init__(self, dim: int, metric: str = "cosine", raw: bool = False, vectors=None):
        # Vectors are compared by squared L2 distance. For "cosine" they are
        # unit length (text embeddings already are; raw vectors are
        # normalised on the way in), so cosine distance is d/2. With `raw`,
        # each doc also keeps the vector it was given, returned as-is.
        self.dim = dim
        self.metric = metric
        self.raw = raw
        self.vectors = vectors or FlatVectors(dim)
        self.docs: dict[int, dict] = {}
        self.by_id: dict[str, int] = {}
        self.owner: dict[int, tuple[int, int]] = {}
        self.filters: FilterIndex | None = None
        # Estimated memory held, kept up to date by `apply` (see `_batch_bytes`).
        self.nbytes = 0
        self._filter_entries = 0  # filter index entries counted in nbytes
        self._vector_bytes = (
            int(4.4 * dim) + 56 + (4 * dim + 112 if raw else 0) + self.vectors.extra_bytes_per_vector
        )

    def _count_filter_bytes(self):
        # ~100 bytes a (document, value) entry: a set slot plus its share of the value's dict entry.
        entries = self.filters.entries if self.filters is not None else 0
        self.nbytes += 100 * (entries - self._filter_entries)
        self._filter_entries = entries

    def apply(self, latest: dict[str, tuple]):
        """Make each doc_id in `latest` ({doc_id: record}) match its record."""
        if not latest:
            return
        keys = {doc_id: object_key_id(doc_id) for doc_id in latest}
        filters = self.filters
        if self.docs:
            gone = []
            for k in keys.values():
                doc = self.docs.pop(k, None)
                if doc is None:
                    continue
                del self.by_id[doc["doc_id"]]
                if filters is not None:
                    filters.remove(k, doc["metadata"])
                self.nbytes -= _batch_bytes(
                    self._vector_bytes,
                    [(None, doc["doc_id"], doc["text"], doc["metadata"], None, doc.get("spans"))],
                    1 + len(doc.get("labels", ())),
                )
                gone.append(k)
                for label in doc.get("labels", ()):
                    del self.owner[label]
                    gone.append(label)
            if gone:
                self.vectors.remove(np.array(gone, dtype="int64"))
        upserts = [rec for rec in latest.values() if rec[2] is not None]
        if not upserts:
            self._count_filter_bytes()
            return
        chunked = any(rec[5] for rec in upserts)
        counts = [len(rec[5]) if rec[5] else 1 for rec in upserts] if chunked else None
        vectors = _gather([rec[4] for rec in upserts], self.dim, counts)
        searchable = vectors
        if self.raw and self.metric == "cosine":
            searchable = unit_rows(vectors)
        docs, by_id, raw = self.docs, self.by_id, self.raw
        # Runs once per document on every load, so the common case (no
        # chunks) is one tight loop, and the size estimate is batch sums.
        labels, row = [], 0
        for rec in upserts:
            doc_id = rec[1]
            key = keys[doc_id]
            doc = {"doc_id": doc_id, "text": rec[2], "metadata": rec[3]}
            if raw:
                doc["vector"] = vectors[row].copy()
            labels.append(key)
            n = len(rec[5]) if chunked and rec[5] else 1
            if n > 1:
                doc["spans"] = rec[5]
                doc["labels"] = [chunk_label(doc_id, i) for i in range(1, n)]
                for i, label in enumerate(doc["labels"], 1):
                    self.owner[label] = (key, i)
                labels.extend(doc["labels"])
            docs[key] = doc
            by_id[doc_id] = key
            row += n
        if filters is not None:
            for rec in upserts:
                filters.add(keys[rec[1]], rec[3])
        self.nbytes += _batch_bytes(self._vector_bytes, upserts, row)
        self._count_filter_bytes()
        self.vectors.add(np.ascontiguousarray(searchable, dtype=np.float32), np.array(labels, dtype="int64"))

    def matching(self, filter: dict) -> set[int]:
        """Ids of the documents whose metadata matches `filter`."""
        if self.filters is None:
            self.filters = FilterIndex(self.docs, metadata=lambda doc: doc["metadata"])
        selected = self.filters.select(filter)
        self._count_filter_bytes()
        return selected

    def search(
        self, q: np.ndarray, top_k: int, filter: dict | None, exact: bool = False
    ) -> list[tuple[float, dict, int]]:
        """[(squared L2 distance, doc, chunk)] of the top_k closest documents
        matching `filter`, closest first. A chunked document counts once, at
        its closest chunk: more vectors are asked for until top_k distinct
        documents are found or every candidate has been seen. Exact unless
        the index searches approximately (and `exact` isn't set)."""
        allowed = None
        if filter:
            keys = self.matching(filter)
            if not keys:
                return []
            if self.owner:
                keys = [*keys, *(label for k in keys for label in self.docs[k].get("labels", ()))]
            allowed = keys  # labels: a document's id is its first chunk's
            candidates = len(allowed)
        else:
            candidates = self.vectors.ntotal
            if candidates == 0:
                return []
        k = min(top_k * 2 if self.owner else top_k, candidates)
        while True:
            distances, labels = self.vectors.search(q, k, allowed, exact)
            hits, seen = [], set()
            for d, label in zip(distances, labels):
                key, chunk = self.owner.get(label, (label, 0))
                if key in seen:
                    continue
                seen.add(key)
                hits.append((d, self.docs[key], chunk))
                if len(hits) == top_k:
                    return hits
            if k >= candidates:
                return hits
            k = min(k * 2, candidates)


class IndexCache:
    """Which loaded indexes a process keeps in memory, under one byte budget.

    Shared by every VectorStore of a process (the text API's and one per S3
    Vectors bucket), so the budget is per process, as memory is. Each entry
    is a (store, index) whose in-memory state was estimated at `nbytes`.
    When the total exceeds `max_bytes`, the least recently used indexes are
    dropped from their store's cache; they're reloaded from the object store
    the next time they're used. The index being used right now is never
    dropped, so one index bigger than the budget still works (alone).
    `max_bytes=None` keeps everything, as before there was a budget.

    All changes to the stores' cache dicts are made here, under this
    object's lock, so what it counts is exactly what the stores hold.
    Evicting pops another store's entry without that store's lock: a thread
    using the index keeps its reference and finishes normally, and an entry
    is only ever a correct snapshot for its manifest ETag, so a missing one
    just means a reload. (Taking the other store's lock here could deadlock
    two stores evicting each other.)
    """

    def __init__(self, max_bytes: int | None = None):
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("the cache budget must not be negative")
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # (id(store), name) -> (store, name, bytes), least recently used first.
        self._entries: OrderedDict[tuple[int, str], tuple[VectorStore, str, int]] = OrderedDict()
        self._total = 0
        self.evictions = 0

    def put(self, store: "VectorStore", name: str, cached: "_Cached") -> None:
        """Make `cached` store's entry for `name`, as the most recently used;
        then evict others until the total fits the budget."""
        key = (id(store), name)
        with self._lock:
            self._discard(key)
            store._cache[name] = cached
            self._entries[key] = (store, name, cached.state.nbytes)
            self._total += cached.state.nbytes
            if self.max_bytes is None:
                return
            while self._total > self.max_bytes and len(self._entries) > 1:
                vkey, (vstore, vname, _) = next(iter(self._entries.items()))
                self._discard(vkey)
                evicted = vstore._cache.pop(vname, None)
                if evicted is not None:
                    evicted.state.vectors.close()  # don't finish building a graph nobody will use
                self.evictions += 1
                logger.debug("dropped index %r from memory to stay within %s bytes", vname, self.max_bytes)

    def drop(self, store: "VectorStore", name: str) -> None:
        with self._lock:
            self._discard((id(store), name))
            store._cache.pop(name, None)

    def touch(self, store: "VectorStore", name: str) -> None:
        """Mark store's index `name` as just used."""
        with self._lock:
            key = (id(store), name)
            if key in self._entries:
                self._entries.move_to_end(key)

    def _discard(self, key) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._total -= old[2]

    def stats(self) -> dict:
        """{"indexes", "bytes", "max_bytes", "evictions"} for this process."""
        with self._lock:
            return {
                "indexes": len(self._entries),
                "bytes": self._total,
                "max_bytes": self.max_bytes,
                "evictions": self.evictions,
            }


def _batch_bytes(vector_bytes: int, records: list[tuple], n_vectors: int) -> int:
    """Rough memory a set of upsert records takes once applied, with
    n_vectors vectors between them: each vector's float32 row (FAISS grows
    its arrays geometrically, so ~10% over) and FAISS id-map entry, and the
    raw copy if kept (all in `vector_bytes`), plus the Python objects for
    text, metadata and bookkeeping. An estimate, to compare indexes against
    the cache budget: within ~5% of measured RSS for 384- and 1024-dimension
    indexes. Computed as sums over the batch, since it runs on every load."""
    getsizeof = sys.getsizeof
    size = n_vectors * vector_bytes + 200 * len(records)
    size += sum(map(getsizeof, (r[2] for r in records))) + sum(map(getsizeof, (r[1] for r in records)))
    size += 110 * sum(map(len, (r[3] for r in records)))
    size += sum(56 + 40 * len(v) for r in records if r[3] for v in r[3].values() if type(v) is list)
    size += 150 * sum(len(r[5]) for r in records if r[5])  # chunk labels, owner entries and spans
    return size


class _Cached:
    """A loaded index: manifest, its ETag, and the state it describes.

    `applied` is the highest seq the state reflects: every record with a
    seq up to it has been applied, none after.
    """

    __slots__ = ("applied", "etag", "manifest", "state")

    def __init__(self, etag: str, manifest: dict, state: _IndexState, applied: int):
        self.etag, self.manifest, self.state, self.applied = etag, manifest, state, applied


def _new_manifest(
    dim: int, incarnation: str, metric: str = "cosine", raw: bool = False, config=None, search=None
) -> dict:
    # "metric": "cosine" or "euclidean". "raw": the index holds caller-given
    # vectors (S3 Vectors API) rather than unit-length text embeddings.
    # "config": free-form settings of the API that created the index.
    # "search": how it's searched (vectorstore.vectors.search_config);
    # manifests without one (older ones) are searched exactly.
    return {
        "format": FORMAT,
        "incarnation": incarnation,
        "dim": dim,
        "metric": metric,
        "raw": raw,
        "created": time.time(),
        "config": config or {},
        "search": search or search_config(),
        "next_seq": 1,
        "segments": [],
        "writers": {},
    }


def _encode_manifest(manifest: dict) -> bytes:
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()


@_corrupt_on_error("manifest")
def _decode_manifest(blob: bytes, key: str) -> dict:
    try:
        manifest = json.loads(blob)
    except ValueError:
        raise ValueError(f"{key} is not a recognised index manifest") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"{key} is not a recognised index manifest")
    return manifest


class VectorStore:
    """Named vector indexes over an object store, searched with FAISS.

    Takes its object-store client and embedder as constructor arguments, so
    it can be built once from env vars for the real app (`get_store`) or
    directly with fakes in tests.

    Indexes. A store holds any number of named indexes; "default" always
    exists (it's created on first use, migrating data from older layouts if
    there is any). Others are created and deleted explicitly. Every method
    takes `index=`, defaulting to "default".

    Concurrency. Any number of processes (uvicorn workers, CLI runs) may
    read and write the same bucket. A write uploads its records as a new,
    immutable segment, then appends the segment to the index's manifest
    with a conditional PUT (`If-Match` on the ETag it read). If another
    writer got there first the PUT is rejected, and the writer reloads,
    re-checks and re-appends the same segment, so no update is lost. The
    manifest assigns each append a sequence number, and a document's newest
    record wins, so concurrent writes to the *same* doc_id are ordered too:
    the last successful append wins, text, metadata and vector together.
    `self._lock` serialises the same within one process and guards the
    cached FAISS objects, which aren't safe to search while being mutated.

    The default index is never deleted, so its one-time migration from the
    old layout can't run twice. A deleted named index stays deleted: a
    writer that loses to the delete reloads, finds no index and fails with
    IndexNotFoundError rather than re-creating it.

    Caching. Each loaded index is kept in memory with its manifest's ETag.
    Every operation does a cheap HEAD on the manifest; when it changed, only
    the new manifest and the segments with records not yet seen are read.

    Compaction. `compaction` is "background" (default: merges run on a
    background thread after writes), "inline" (run before the write
    returns; for tests) or "off".
    """

    def __init__(
        self,
        client: Minio,
        embedder,
        bucket: str = BUCKET,
        io_workers: int = DEFAULT_IO_WORKERS,
        compaction: str = "background",
        prefix: str = "",
        default_index: str | None = DEFAULT_INDEX,
        index_name_re: re.Pattern = _INDEX_NAME_RE,
        raw_vectors: bool = False,
        io_pool: ThreadPoolExecutor | None = None,
        compactor: ThreadPoolExecutor | None = None,
        verify_conditional_writes: bool = True,
        chunk_overlap: int | None = DEFAULT_CHUNK_OVERLAP,
        cache: IndexCache | None = None,
        graph_build: str = "background",
        graph_builder: ThreadPoolExecutor | None = None,
        graph_build_threads: int | None = None,
    ):
        """`prefix`, `default_index`, `index_name_re` and `raw_vectors` let one
        bucket hold several independent namespaces of indexes (the S3
        Vectors API uses one per vector bucket): keys go under `prefix`;
        `default_index` (None: none) always exists; names must match
        `index_name_re`; and with `raw_vectors` an index stores the vectors
        it is given (not embeddings of text) and returns them exactly.

        `io_pool` and `compactor` share thread pools between stores (many
        namespaces must not mean many threads each), and
        `verify_conditional_writes=False` skips the startup probe for a store
        whose client another store has already checked.

        `chunk_overlap` is how many tokens consecutive chunks of a long text
        share; None turns chunking off (long texts are embedded truncated).

        `cache` is the process's memory budget for loaded indexes, shared
        between stores (see IndexCache); by default the store has its own,
        unlimited one.

        Approximate-search indexes build their HNSW graphs on `graph_builder`
        (shared between stores like `io_pool`; default: one thread of the
        store's own) with `graph_build_threads` threads each (default:
        S3VAH_GRAPH_BUILD_THREADS, or half the cores). `graph_build="inline"`
        builds them before the write or load returns instead, for tests, and
        "off" never builds them: approximate indexes are searched exactly
        (for short-lived processes such as the CLI, which would otherwise
        wait at exit for a graph nobody will use).
        """
        if compaction not in ("background", "inline", "off"):
            raise ValueError("compaction must be 'background', 'inline' or 'off'")
        if graph_build not in ("background", "inline", "off"):
            raise ValueError("graph_build must be 'background', 'inline' or 'off'")
        self._prefix = prefix
        self._default = default_index
        self._name_re = index_name_re
        self._raw = raw_vectors
        self._chunk_overlap = chunk_overlap
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._io = io_pool or ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="vectorstore-io")
        self._cache: dict[str, _Cached] = {}
        self._cache_budget = cache or IndexCache()
        self._graph_build = graph_build
        self._graph_builder = graph_builder
        self._graph_build_threads = graph_build_threads or build_threads_from_env()
        self._dim: int | None = None
        # Identifies this store's writes in manifests; see `_commit`.
        self._writer_id = uuid.uuid4().hex[:16]
        self._write_counter = 0
        self._compaction = compaction
        self._compactor = compactor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="vectorstore-compact")
        self._compact_lock = threading.Lock()
        self._compact_idle = threading.Condition(self._compact_lock)
        self._compacting: set[str] = set()
        self._compact_again: set[str] = set()
        if verify_conditional_writes:
            self._ensure_bucket()
            self._verify_conditional_writes()

    # -- object store helpers -------------------------------------------

    def _mkey(self, name: str) -> str:
        return self._prefix + manifest_key(name)

    def _v3key(self, name: str) -> str:
        return self._prefix + legacy_v3_key(name)

    def _segdir(self, name: str, incarnation: str | None = None) -> str:
        return self._prefix + _segment_dir(name, incarnation)

    def _check_name(self, name: str) -> None:
        if self._name_re is _INDEX_NAME_RE:
            validate_index_name(name)
        elif not isinstance(name, str) or not self._name_re.match(name):
            raise ValueError(f"invalid index name {name!r}")

    def _ensure_bucket(self):
        if self._client.bucket_exists(self._bucket):
            return
        try:
            self._client.make_bucket(self._bucket)
        except S3Error as e:
            # Lost the race to another process creating it: that's fine.
            # (BucketAlreadyExists, owned by someone else, is a real error.)
            if e.code != "BucketAlreadyOwnedByYou":
                raise

    def _put_conditional(self, key: str, data: bytes, if_match: str | None) -> str:
        """PUT `data` only if `key` is still at ETag `if_match` (None: only if absent).

        Returns the new ETag, or raises _WriteConflict if the condition
        failed. Uses minio-py's single-request `_put_object` because its
        public `put_object` treats unknown headers as user metadata: an
        If-Match passed there is sent as `x-amz-meta-if-match` and silently
        ignored. `_verify_conditional_writes` checks at startup that the
        conditions are really enforced, so a library or server change can't
        quietly turn this back into an unconditional write.
        """
        if len(data) > MAX_SINGLE_PUT_BYTES:
            raise ValueError(f"object is {len(data)} bytes; a single PUT is capped at {MAX_SINGLE_PUT_BYTES}")
        headers = {"Content-Type": "application/octet-stream"}
        if if_match is None:
            headers["If-None-Match"] = "*"
        else:
            headers["If-Match"] = f'"{if_match}"'
        try:
            result = self._client._put_object(self._bucket, key, data, headers)
        except S3Error as e:
            # NoSuchKey: we expected a version to replace but the key is gone.
            if e.code in _CONFLICT_CODES or (if_match is not None and e.code == "NoSuchKey"):
                raise _WriteConflict(key) from e
            raise
        return _normalize_etag(result.etag)

    def _verify_conditional_writes(self):
        key = f"{PROBE_PREFIX}{uuid.uuid4().hex}"
        try:
            self._put_conditional(key, b"1", if_match=None)
            for label, if_match in (("If-None-Match: *", None), ("a stale If-Match", "0" * 32)):
                try:
                    self._put_conditional(key, b"2", if_match=if_match)
                except _WriteConflict:
                    continue
                raise ConditionalWritesUnsupportedError(
                    f"the object store accepted a PUT with {label} that should have been "
                    "rejected; it doesn't enforce conditional writes, which concurrent "
                    "index updates rely on (MinIO supports them in releases since late 2024)"
                )
        finally:
            self._delete_object(key)

    def _get_with_etag(self, key: str) -> tuple[bytes, str]:
        try:
            resp = self._client.get_object(self._bucket, key)
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectNotFoundError(key) from e
            raise
        try:
            return resp.read(), _require_etag(resp.headers.get("ETag"), key)
        finally:
            resp.close()
            resp.release_conn()

    def _get_bytes(self, key: str) -> bytes:
        return self._get_with_etag(key)[0]

    def _get_if_exists(self, key: str) -> bytes | None:
        try:
            return self._get_bytes(key)
        except ObjectNotFoundError:
            return None

    def _etag(self, key: str) -> str | None:
        try:
            return _require_etag(self._client.stat_object(self._bucket, key).etag, key)
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject"):
                return None
            raise

    def _delete_object(self, key: str):
        try:
            self._client.remove_object(self._bucket, key)
        except S3Error as e:
            if e.code != "NoSuchKey":
                raise

    def _delete_quietly(self, keys):
        """Best-effort cleanup: a failure only leaves an orphan for a later sweep."""
        for key in keys:
            try:
                self._delete_object(key)
            except Exception:
                logger.warning("could not delete %s; a later compaction will", key, exc_info=True)

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    def _chunk_spans(self, text: str) -> list[tuple[int, int]]:
        """Spans of `text` to embed one vector each (see OnnxEmbedder.chunk_spans)."""
        split = getattr(self._embedder, "chunk_spans", None)
        if self._chunk_overlap is None or split is None:
            return [(0, len(text))]
        return split(text, self._chunk_overlap) or [(0, len(text))]

    def _embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed("dimension probe").shape[0])
        return self._dim

    def _new_state(self, manifest: dict) -> "_IndexState":
        dim = manifest["dim"]
        search = manifest.get("search") or search_config()
        vectors = None
        if search["mode"] == APPROXIMATE:
            builder = None
            if self._graph_build == "background":
                if self._graph_builder is None:
                    self._graph_builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vectorstore-graph")
                builder = self._graph_builder
            vectors = HnswVectors(
                dim,
                m=search["m"],
                ef_construction=search["ef_construction"],
                ef_search=search["ef_search"],
                lock=self._lock,
                builder=builder,
                build_threads=self._graph_build_threads,
                build=self._graph_build != "off",
            )
        return _IndexState(dim, manifest.get("metric", "cosine"), manifest.get("raw", False), vectors)

    # -- the in-memory cache ----------------------------------------------------
    # Every change to `self._cache` goes through these, so the process-wide
    # IndexCache knows what is held and can evict to stay in its budget.

    def _cache_put(self, name: str, cached: _Cached) -> None:
        self._cache_budget.put(self, name, cached)

    def _cache_drop(self, name: str) -> None:
        self._cache_budget.drop(self, name)

    def cache_stats(self) -> dict:
        """Memory held by loaded indexes in this process: see IndexCache.stats."""
        return self._cache_budget.stats()

    # -- reading an index -----------------------------------------------------

    def _read_manifest(self, name: str) -> tuple[dict, str] | None:
        try:
            blob, etag = self._get_with_etag(self._mkey(name))
        except ObjectNotFoundError:
            return None
        return _decode_manifest(blob, self._mkey(name)), etag

    def _fetch(self, manifest: dict, entries: list[dict]) -> list[tuple]:
        """All records of `entries`, read in parallel. Raises _SegmentMissing."""

        def read(entry):
            try:
                blob = self._get_bytes(entry["key"])
            except ObjectNotFoundError:
                raise _SegmentMissing(entry["key"]) from None
            return _decode_segment(blob, entry["key"], manifest["dim"], entry["first_seq"])

        try:
            results = list(self._io.map(read, entries)) if len(entries) > 1 else [read(e) for e in entries]
        except RuntimeError:
            # The pool is shut down: the interpreter is exiting while a
            # background merge runs (say, at the end of a CLI command).
            # Finish it serially rather than fail it.
            results = [read(e) for e in entries]
        return [rec for recs in results for rec in recs]

    def _load(self, name: str, stale: bool = False) -> _Cached | None:
        """The index as currently stored, or None if it doesn't exist (yet).

        Must be called with `self._lock` held. Serves the cache when the
        manifest's ETag is unchanged; otherwise reads the manifest and only
        the records the cached state hasn't seen (or everything, for a new
        incarnation or after a full merge the cache can't follow). `stale`
        skips the HEAD when the caller knows the manifest changed.
        """
        key = self._mkey(name)
        missing_before = None
        while True:
            cached = self._cache.get(name)
            if not stale:
                etag = self._etag(key)
                if etag is None:
                    self._cache_drop(name)
                    return None
                if cached is not None and cached.etag == etag:
                    self._cache_budget.touch(self, name)
                    return cached
            stale = False
            got = self._read_manifest(name)
            if got is None:
                self._cache_drop(name)
                return None
            manifest, etag = got
            try:
                fresh = self._catch_up(name, cached, manifest, etag)
            except _SegmentMissing as e:
                # Merged away after we read the manifest: a newer manifest
                # no longer lists it. The same manifest listing it twice
                # means the segment is really gone.
                if missing_before == (etag, e.args[0]):
                    raise IndexCorruptError(f"index {name!r} lists segment {e.args[0]!r}, which doesn't exist") from None
                missing_before = (etag, e.args[0])
                continue
            self._cache_put(name, fresh)
            return fresh

    def _catch_up(self, name: str, cached: _Cached | None, manifest: dict, etag: str) -> _Cached:
        applied = manifest["next_seq"] - 1
        entries = manifest["segments"]
        if (
            cached is not None
            and cached.manifest["incarnation"] == manifest["incarnation"]
            and cached.applied <= applied
            # A full merge dropped tombstones the cache may not have seen.
            and not any(e["full"] and e["first_seq"] <= cached.applied < e["last_seq"] for e in entries)
        ):
            newer = self._fetch(manifest, [e for e in entries if e["last_seq"] > cached.applied])
            latest = _latest(r for r in newer if r[0] > cached.applied)
            # Fetching can't fail past this point; drop the cache entry while
            # mutating so a failure can't leave a half-applied state cached.
            self._cache_drop(name)
            state = cached.state
        else:
            with _gc_paused():
                latest = {d: r for d, r in _latest(self._fetch(manifest, entries)).items() if r[2] is not None}
                state = self._new_state(manifest)
                state.apply(latest)
            return _Cached(etag, manifest, state, applied)
        state.apply(latest)
        return _Cached(etag, manifest, state, applied)

    # -- creating an index (and migrating older layouts) -----------------

    def _ensure(self, name: str, stale: bool = False) -> _Cached:
        """The index, creating the default one (migrated, or empty) if needed.

        Must hold `self._lock`. Only ever *creates* (If-None-Match), so if two
        processes migrate at once exactly one result is kept.
        """
        for _ in range(INDEX_WRITE_ATTEMPTS):
            cached = self._load(name, stale)
            if cached is not None:
                return cached
            stale = False
            initial = self._initial_records(name)
            if initial is None:
                raise IndexNotFoundError(name)
            dim, records = initial
            try:
                self._create(name, dim, records)
            except _WriteConflict:
                continue  # someone else created it first: use theirs
        raise IndexWriteConflictError("index kept changing while being created; retry the operation")

    def _create(self, name: str, dim: int, records: list[tuple], metric: str = "cosine", config=None, search=None):
        """Create the manifest, with `records` (seq None) as its first segment.

        Raises _WriteConflict if the index already exists.
        """
        manifest = _new_manifest(dim, uuid.uuid4().hex[:16], metric, self._raw, config, search)
        seg_key = None
        if records:
            blob = _encode_segment(dim, records)
            seg_key = f"{self._segdir(name, manifest['incarnation'])}{uuid.uuid4().hex}.seg"
            self._put_conditional(seg_key, blob, if_match=None)
            manifest["segments"].append(
                {"key": seg_key, "first_seq": 1, "last_seq": 1, "bytes": len(blob), "full": True}
            )
            manifest["next_seq"] = 2
        try:
            self._put_conditional(self._mkey(name), _encode_manifest(manifest), if_match=None)
        except _WriteConflict:
            if seg_key:  # never referenced: the manifest already existed
                self._delete_quietly([seg_key])
            raise

    def _initial_records(self, name: str) -> tuple[int, list[tuple]] | None:
        """(dim, records) to create index `name` from; None if it doesn't exist."""
        blob = self._get_if_exists(self._v3key(name))
        if blob is not None:
            return _decode_v3(blob, self._v3key(name))
        if name != self._default or self._prefix:
            return None  # legacy layouts only ever held the root default index
        legacy = self._legacy_docs()
        if not legacy:
            return self._embedding_dim(), []
        dim = next(iter(legacy.values()))[0].shape[0]
        return dim, [(None, d, text, {}, (vec.reshape(1, -1), 0), None) for d, (vec, text) in legacy.items()]

    def _legacy_docs(self) -> dict[str, tuple[np.ndarray, str]]:
        """{doc_id: (vector, text)} from a v1/v2 bucket; {} if there is none.

        A v2 index decides which docs are live (its map is what search used);
        without one, every doc with both a vector and a text object is. Docs
        missing either object are skipped: on the old layout they were
        unreachable (list skipped them) or had no vector to search by.
        """
        names = {obj.object_name for obj in self._client.list_objects(self._bucket)}
        if LEGACY_V2_INDEX_OBJ in names:
            blob = self._get_bytes(LEGACY_V2_INDEX_OBJ)
            if not blob.startswith(LEGACY_V2_MAGIC):
                raise IndexCorruptError(f"{LEGACY_V2_INDEX_OBJ} is not a recognised v2 index blob")
            (ids_len,) = struct.unpack_from(">Q", blob, len(LEGACY_V2_MAGIC))
            start = len(LEGACY_V2_MAGIC) + 8
            live = list(json.loads(blob[start: start + ids_len]).values())
        else:
            live = [n.removesuffix(".npy") for n in names if n.endswith(".npy")]
        live = [d for d in live if f"{d}.npy" in names and f"{d}.meta.json" in names]

        def read(doc_id):
            vec = np.frombuffer(self._get_bytes(f"{doc_id}.npy"), dtype=np.float32)
            text = json.loads(self._get_bytes(f"{doc_id}.meta.json")).get("text", "")
            return doc_id, (vec, text)

        return dict(self._io.map(read, live))

    # -- writing --------------------------------------------------------------

    def _commit(self, name: str, records: list[tuple], needed) -> bool:
        """Append `records` (seq None) to index `name` as one segment, with CAS.

        `needed(state)` is called on the current state before each attempt:
        it may raise to abort (nothing is written), or return False if the
        write has become a no-op. Returns True if the records are in the
        index when this returns, False if they turned out not to be needed.

        Retrying is safe because the segment is the same on every attempt
        and each attempt starts from a freshly loaded manifest. One case
        needs care: an append that *landed* but was reported as a conflict
        (the response was lost and urllib3's retry of the PUT hit our own
        new ETag). Re-appending would re-apply our records over any newer
        write to the same doc_ids, and deleting "our unused" segment would
        break the index. So every append also records this store's write
        counter in the manifest's "writers" map, which compaction keeps,
        and each attempt first checks whether that counter already landed.

        The segment is deleted only when it is known to be unreferenced: a
        manifest read after our last append attempt didn't contain it. If an
        append's outcome is unknown (an error other than a conflict), the
        segment stays; if nothing references it, a later sweep removes it.
        """
        self._write_counter += 1
        counter = self._write_counter
        mkey = self._mkey(name)
        seg_key, seg_blob, uploaded_at = None, None, 0.0
        unresolved = False  # an append whose outcome we haven't seen yet
        conflicted = False
        started = time.monotonic()
        try:
            for attempt in range(INDEX_WRITE_ATTEMPTS + 1):
                try:
                    cached = self._ensure(name, stale=conflicted)
                except IndexNotFoundError:
                    unresolved = False  # no manifest, so nothing references the segment
                    raise
                manifest = cached.manifest
                if manifest["writers"].get(self._writer_id, [0])[0] >= counter:
                    unresolved, seg_key = False, None  # it landed; the segment is in use
                    return True
                unresolved = False
                if attempt == INDEX_WRITE_ATTEMPTS:
                    break  # that was the final check, after the last attempt
                if not needed(cached.state):
                    return False
                if time.monotonic() - started > INDEX_WRITE_TIMEOUT_S:
                    break
                seg_dir = self._segdir(name, manifest["incarnation"])
                stale = seg_key is not None and (
                    not seg_key.startswith(seg_dir) or time.monotonic() - uploaded_at > SEGMENT_MAX_PENDING_S
                )
                if stale:
                    self._delete_quietly([seg_key])  # known unreferenced: just checked
                    seg_key = None
                if seg_key is None:
                    if seg_blob is None:
                        seg_blob = _encode_segment(manifest["dim"], records)
                    seg_key = f"{seg_dir}{uuid.uuid4().hex}.seg"
                    self._put_conditional(seg_key, seg_blob, if_match=None)
                    uploaded_at = time.monotonic()
                seq = manifest["next_seq"]
                now = time.time()
                writers = {w: v for w, v in manifest["writers"].items() if v[1] >= now - WRITER_TTL_S}
                writers[self._writer_id] = [counter, now]
                new = {
                    **manifest,
                    "next_seq": seq + 1,
                    "segments": [
                        *manifest["segments"],
                        {"key": seg_key, "first_seq": seq, "last_seq": seq, "bytes": len(seg_blob), "full": False},
                    ],
                    "writers": writers,
                }
                unresolved = True
                try:
                    new_etag = self._put_conditional(mkey, _encode_manifest(new), if_match=cached.etag)
                except _WriteConflict:
                    conflicted = True
                    time.sleep(
                        random.uniform(0, min(INDEX_WRITE_BACKOFF_CAP_S, INDEX_WRITE_BACKOFF_BASE_S * 2 ** min(attempt, 30)))
                    )
                    continue
                unresolved, seg_key = False, None
                # The new manifest is exactly the old one plus our segment, so
                # the new state is the cached one plus our records. Drop the
                # entry while mutating in case applying fails half-way.
                self._cache_drop(name)
                cached.state.apply(_latest((seq, *r[1:]) for r in records))
                self._cache_put(name, _Cached(new_etag, new, cached.state, seq))
                return True
            raise IndexWriteConflictError(
                f"index update kept losing to concurrent writers for {time.monotonic() - started:.0f} s "
                f"({attempt} attempts); retry the operation"
            )
        finally:
            if seg_key is not None and not unresolved:
                self._delete_quietly([seg_key])

    def _after_write(self, name: str):
        """Merge segments if the size rule says so (see `_plan_merge`)."""
        if self._compaction == "off":
            return
        with self._lock:
            cached = self._cache.get(name)
            if cached is None or _plan_merge(cached.manifest["segments"]) is None:
                return
        if self._compaction == "inline":
            self._compact_safely(name, full=False)
            return
        with self._compact_lock:
            if name in self._compacting:
                self._compact_again.add(name)  # run once more when the current merge ends
                return
            self._compacting.add(name)
        self._compactor.submit(self._compact_loop, name)

    def _compact_loop(self, name: str):
        while True:
            self._compact_safely(name, full=False)
            with self._compact_lock:
                if name in self._compact_again:
                    self._compact_again.discard(name)
                    continue
                self._compacting.discard(name)
                self._compact_idle.notify_all()
                return

    def _compact_safely(self, name: str, full: bool):
        try:
            self._compact(name, full)
        except Exception:
            logger.warning("compaction of index %r failed; a later write will retry it", name, exc_info=True)

    def wait_for_compaction(self, timeout: float | None = None) -> bool:
        """Block until no background merge is running. False on timeout."""
        with self._compact_lock:
            return self._compact_idle.wait_for(lambda: not self._compacting, timeout)

    def _compact(self, name: str, full: bool) -> bool:
        """Merge a run of segments into one (or, for `full`, all into parts).

        Doesn't hold `self._lock`: it only reads immutable segments and
        changes the manifest by CAS like any writer. The merged segments
        replace the run in the manifest only if that run is still there,
        unchanged; if another merge got to it first, ours is dropped.
        Returns True if a merge was committed.
        """
        got = self._read_manifest(name)
        if got is None:
            return False
        manifest, etag = got
        entries = manifest["segments"]
        run = (0, len(entries)) if full else _plan_merge(entries)
        if run is None or (full and len(entries) < 2 and all(e["full"] for e in entries)):
            if full:
                self._sweep(name, manifest)
            return False
        i, j = run
        merged = entries[i:j]
        try:
            latest = _latest(self._fetch(manifest, merged))
        except _SegmentMissing:
            return False  # merged away by someone else meanwhile
        is_full = i == 0
        if is_full:  # nothing older remains for a tombstone to hide
            latest = {d: r for d, r in latest.items() if r[2] is not None}
        first_seq, last_seq = min(e["first_seq"] for e in merged), max(e["last_seq"] for e in merged)
        parts = _split_parts(sorted(latest.values()), manifest["dim"])
        incarnation = manifest["incarnation"]
        seg_dir = self._segdir(name, incarnation)
        new_entries, started = [], time.monotonic()
        for blob in parts:
            key = f"{seg_dir}{uuid.uuid4().hex}.seg"
            self._put_conditional(key, blob, if_match=None)
            new_entries.append(
                {"key": key, "first_seq": first_seq, "last_seq": last_seq, "bytes": len(blob), "full": is_full}
            )
        new_keys = {e["key"] for e in new_entries}
        old_keys = [e["key"] for e in merged]
        unresolved = False  # a manifest PUT whose outcome we haven't seen yet
        try:
            for _ in range(COMPACTION_ATTEMPTS):
                if time.monotonic() - started > SEGMENT_MAX_PENDING_S:
                    break
                entries = manifest["segments"]
                pos = _find_run([e["key"] for e in entries], old_keys)
                if manifest["incarnation"] != incarnation or pos is None or (is_full and pos != 0):
                    break  # the run changed under us: another merge got to it first
                new = {**manifest, "segments": [*entries[:pos], *new_entries, *entries[pos + len(old_keys):]]}
                unresolved = True
                try:
                    self._put_conditional(self._mkey(name), _encode_manifest(new), if_match=etag)
                except _WriteConflict:
                    got = self._read_manifest(name)
                    if got is None:
                        unresolved = False  # index deleted: nothing references our parts
                        break
                    manifest, etag = got
                    unresolved = False
                    if not new_keys & {e["key"] for e in manifest["segments"]}:
                        continue
                    # It landed after all (the response was lost). If our parts
                    # aren't listed any more they were merged again, which
                    # also means the run is gone; either way it's committed.
                new_keys = set()
                self._delete_quietly(old_keys)  # readers that still list them reload
                if is_full:
                    self._sweep(name, new)
                return True
            return False
        finally:
            if new_keys and not unresolved:
                self._delete_quietly(new_keys)

    def _sweep(self, name: str, manifest: dict):
        """Delete this index's segments that `manifest` doesn't reference and
        that are older than ORPHAN_GRACE_S (see the constants)."""
        referenced = {e["key"] for e in manifest["segments"]}
        cutoff = datetime.now(UTC).timestamp() - ORPHAN_GRACE_S
        orphans = [
            obj.object_name
            for obj in self._client.list_objects(self._bucket, prefix=self._segdir(name), recursive=True)
            if obj.object_name not in referenced
            and obj.last_modified is not None
            and obj.last_modified.timestamp() < cutoff
        ]
        self._delete_quietly(orphans)

    # -- indexes --------------------------------------------------------------

    def list_indexes(self) -> list[str]:
        names = {self._default} if self._default else set()
        prefix = self._prefix + INDEX_PREFIX
        for obj in self._client.list_objects(self._bucket, prefix=prefix):
            for suffix in (MANIFEST_SUFFIX, LEGACY_V3_SUFFIX):
                if obj.object_name.endswith(suffix):
                    names.add(obj.object_name[len(prefix): -len(suffix)])
        return sorted(names)

    def create_index(
        self, name: str, dim: int | None = None, metric: str = "cosine", config=None, search: dict | None = None
    ) -> None:
        """Create an empty index. `dim` defaults to the embedder's.

        `metric` ("cosine" or "euclidean") and `config` (any JSON, returned by
        `index_info`) matter to raw-vector indexes; text indexes are cosine.
        `search` is {"mode": "exact"} (the default) or {"mode": "approximate",
        and optionally "m", "ef_construction", "ef_search"}: see
        vectorstore.vectors. It's fixed for the index's lifetime.
        """
        self._check_name(name)
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        search = dict(search or {})
        search = search_config(search.pop("mode", None), **search)
        if name == self._default or self._etag(self._v3key(name)) is not None:
            raise IndexAlreadyExistsError(name)
        with self._lock:
            try:
                self._create(name, dim or self._embedding_dim(), [], metric, config, search)
            except _WriteConflict:
                raise IndexAlreadyExistsError(name) from None

    def index_info(self, name: str) -> dict:
        """{"dim", "metric", "created" (unix time), "config"}; IndexNotFoundError if absent."""
        self._check_name(name)
        if name == self._default:
            with self._lock:
                self._ensure(name)  # create it on first use, like every other read
        got = self._read_manifest(name)  # just the manifest: no need to load the index
        if got is None:
            raise IndexNotFoundError(name)
        m = got[0]
        return {
            "dim": m["dim"],
            "metric": m.get("metric", "cosine"),
            "created": m.get("created", 0.0),
            "config": m.get("config", {}),
            "search": m.get("search") or search_config(),
        }

    def delete_index(self, name: str) -> None:
        """Delete a named index and its segments.

        The legacy v3 object goes first, so a concurrent reader can't
        migrate the index back from it once the manifest is gone.
        """
        self._check_name(name)
        if name == self._default:
            raise ValueError("the default index can't be deleted")
        with self._lock:
            had_v3 = self._etag(self._v3key(name)) is not None
            if had_v3:
                self._delete_object(self._v3key(name))
            got = self._read_manifest(name)
            if got is None and not had_v3:
                raise IndexNotFoundError(name)
            self._cache_drop(name)
            if got is None:
                return
            self._delete_object(self._mkey(name))
            # Only this incarnation's segments: a newer index of the same
            # name, created since, has its own prefix.
            prefix = self._segdir(name, got[0]["incarnation"])
            self._delete_quietly(
                [o.object_name for o in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)]
            )

    def compact(self, index: str = DEFAULT_INDEX) -> dict:
        """Merge all of an index's segments, dropping overwritten and deleted
        records, and delete orphaned segments. Returns `index_stats` after."""
        self._check_name(index)
        with self._lock:
            self._ensure(index)
        self._compact(index, full=True)
        return self.index_stats(index)

    def index_stats(self, index: str = DEFAULT_INDEX) -> dict:
        """{"documents", "vectors", "segments", "bytes", "search", "graph_ready"}
        (vectors: one per chunk; bytes: all segments, live or not; search:
        "exact" or "approximate"; graph_ready: whether an approximate index's
        graph is built, so searches use it, or still exact)."""
        self._check_name(index)
        with self._lock:
            cached = self._ensure(index)
            entries = cached.manifest["segments"]
            return {
                "documents": len(cached.state.docs),
                "vectors": cached.state.vectors.ntotal,
                "segments": len(entries),
                "bytes": sum(e["bytes"] for e in entries),
                "search": cached.state.vectors.mode,
                "graph_ready": getattr(cached.state.vectors, "has_graph", False),
            }

    # -- raw vectors (the S3 Vectors API) -------------------------------------
    # Keys are opaque strings (validated by the caller); metadata is stored
    # as given (the caller validates it against its own limits).

    def put_vectors(self, index: str, items) -> int:
        """Add or overwrite [(key, vector, metadata)] as one write; the last
        occurrence of a repeated key wins. Raises ValueError for a vector of
        the wrong dimension, with NaN/infinity, or all zero on a cosine index."""
        self._check_name(index)
        batch: dict[str, tuple[np.ndarray, dict]] = {}
        for key, vector, metadata in items:
            if not isinstance(key, str) or not key:
                raise ValueError("vector keys must be non-empty strings")
            require_unicode(key, "a vector key")
            batch[key] = (np.asarray(vector, dtype=np.float32).reshape(-1), metadata or {})
        if not batch:
            return 0
        keys = list(batch)
        dims = {v.shape[0] for v, _ in batch.values()}
        vecs = np.vstack([batch[k][0] for k in keys]) if len(dims) == 1 else None
        if vecs is not None and not np.isfinite(vecs).all():
            raise ValueError("vector data must be finite (no NaN or infinity)")
        records = [(None, k, "", batch[k][1], (vecs, i), None) for i, k in enumerate(keys)] if vecs is not None else []
        ids = [object_key_id(k) for k in keys]

        def check(state: _IndexState) -> bool:
            if vecs is None or vecs.shape[1] != state.dim:
                raise ValueError(f"vectors must have the index's dimension, {state.dim}; got {sorted(dims)}")
            if state.metric == "cosine" and not row_norms(vecs).all():
                raise ValueError("an all-zero vector has no direction; it can't be stored in a cosine index")
            for key, kid in zip(keys, ids):
                clash = _collision(state, key, kid)
                if clash is not None:
                    raise ValueError(f"key {key!r} collides with {clash!r}; use another key")
            return True

        with self._lock:
            self._commit(index, records, check)
        self._after_write(index)
        return len(keys)

    def get_vectors(self, index: str, keys) -> list[dict]:
        """[{"key", "vector", "metadata"}] for the keys that exist, in request order."""
        self._check_name(index)
        with self._lock:
            state = self._ensure(index).state
            out = []
            for key in keys:
                kid = state.by_id.get(key)
                if kid is not None:
                    doc = state.docs[kid]
                    out.append({"key": key, "vector": doc.get("vector"), "metadata": _copy_metadata(doc["metadata"])})
            return out

    def list_vectors(self, index: str, after: str | None, limit: int, segment: tuple[int, int] = (1, 0)):
        """Up to `limit` vectors with keys > `after`, in key order, restricted to
        those where hash(key) % segment[0] == segment[1] (for parallel
        listing). Returns (vectors, last_key_or_None_if_done)."""
        self._check_name(index)
        count, part = segment
        with self._lock:
            state = self._ensure(index).state
            keys = sorted(k for k in state.by_id if (after is None or k > after) and (count == 1 or state.by_id[k] % count == part))
            page = keys[:limit]
            out = [
                {"key": k, "vector": doc.get("vector"), "metadata": _copy_metadata(doc["metadata"])}
                for k in page
                for doc in (state.docs[state.by_id[k]],)
            ]
        return out, (page[-1] if len(keys) > limit else None)

    def delete_vectors(self, index: str, keys) -> None:
        """Delete these keys; ones that don't exist are ignored."""
        self._check_name(index)
        keys = list(dict.fromkeys(keys))
        if not keys:
            return
        with self._lock:
            self._commit(index, [(None, k, None, None, None, None) for k in keys], lambda state: any(k in state.by_id for k in keys))
        self._after_write(index)

    def query_vectors(
        self, index: str, vector, top_k: int, filter: dict | None = None, exact: bool = False
    ) -> tuple[str, list[dict]]:
        """(metric, [{"key", "distance", "metadata"}]) for the top_k closest
        vectors, closest first: exact, unless the index searches
        approximately and `exact` isn't set. Distance is 1 - cosine
        similarity for "cosine", plain Euclidean distance for "euclidean"."""
        self._check_name(index)
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(q).all():
            raise ValueError("query vector must be finite (no NaN or infinity)")
        if filter:
            validate_filter(filter)
        with self._lock:
            state = self._ensure(index).state
            if q.shape[1] != state.dim:
                raise ValueError(f"query vector must have the index's dimension, {state.dim}; got {q.shape[1]}")
            if state.metric == "cosine":
                if not row_norms(q).all():
                    raise ValueError("an all-zero query vector has no direction for cosine distance")
                q = unit_rows(q)
            hits = state.search(q, top_k, filter, exact)
            out = []
            for d, doc, _ in hits:
                dist = d / 2.0 if state.metric == "cosine" else float(np.sqrt(max(d, 0.0)))
                out.append({"key": doc["doc_id"], "distance": dist, "metadata": _copy_metadata(doc["metadata"])})
            return state.metric, out

    # -- documents ------------------------------------------------------------

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None, index: str = DEFAULT_INDEX):
        """Add a new document, or replace the text, metadata and vector of an existing one."""
        self.add_documents([(doc_id, text, metadata)], index=index)

    def add_documents(self, docs, index: str = DEFAULT_INDEX) -> int:
        """Add or replace many documents with one embed call and one index write.

        `docs` is an iterable of (doc_id, text) or (doc_id, text, metadata).
        Everything is validated before anything is written, so one bad
        document rejects the whole batch. If a doc_id appears more than once,
        the last occurrence wins. Returns the number of distinct documents
        written.
        """
        self._check_name(index)
        batch: dict[str, tuple[str, dict]] = {}
        for doc in docs:
            doc_id, text, metadata = (*doc, None) if len(doc) == 2 else doc
            validate_doc_id(doc_id)
            if not isinstance(text, str):
                raise ValueError(f"text for {doc_id!r} must be a string")  # noqa: TRY004 -- API maps ValueError to 400
            require_unicode(text, f"text for {doc_id!r}")
            batch[doc_id] = (text, validate_metadata(metadata))
        if not batch:
            return 0
        doc_ids = list(batch)
        spans = {d: self._chunk_spans(batch[d][0]) for d in doc_ids}
        pieces = [batch[d][0] if len(spans[d]) == 1 else batch[d][0][a:b] for d in doc_ids for a, b in spans[d]]
        vecs = self._embed_many(pieces)
        records, row = [], 0
        for d in doc_ids:
            chunked = [list(sp) for sp in spans[d]] if len(spans[d]) > 1 else None
            records.append((None, d, batch[d][0], batch[d][1], (vecs, row), chunked))
            row += len(spans[d])
        labels = [(d, chunk_label(d, i)) for d in doc_ids for i in range(len(spans[d]))]

        def check(state: _IndexState) -> bool:
            if vecs.shape[1] != state.dim:
                raise ValueError(f"embedding dimension {vecs.shape[1]} doesn't match index {index!r} ({state.dim})")
            for doc_id, label in labels:
                clash = _collision(state, doc_id, label)
                if clash is not None:
                    raise ValueError(f"doc_id {doc_id!r} collides with {clash!r}; use another doc_id")
            return True

        with self._lock:
            self._commit(index, records, check)
        self._after_write(index)
        return len(doc_ids)

    def get_document(self, doc_id: str, index: str = DEFAULT_INDEX):
        """{"doc_id", "text", "metadata"}, or None if there's no such document."""
        self._check_name(index)
        with self._lock:
            state = self._ensure(index).state
            key = state.by_id.get(doc_id)
            return None if key is None else _public(state.docs[key])

    def list_documents(self, skip: int = 0, limit: int = 100, index: str = DEFAULT_INDEX):
        if skip < 0:
            raise ValueError("skip must not be negative")
        if not 0 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 0 and {MAX_PAGE_SIZE}")
        self._check_name(index)
        with self._lock:
            state = self._ensure(index).state
            page = sorted(state.by_id)[skip: skip + limit]
            return [
                {"doc_id": d, "text_preview": doc["text"][:64], "metadata": _copy_metadata(doc["metadata"])}
                for d in page
                for doc in (state.docs[state.by_id[d]],)
            ]

    def delete_document(self, doc_id: str, index: str = DEFAULT_INDEX) -> bool:
        """Delete a document; False if it wasn't there.

        True once *any* attempt found the document: from then on its removal
        is guaranteed. (If another process deleted it concurrently, both
        report True, as idempotent deletes do.)
        """
        self._check_name(index)
        found = False

        def check(state: _IndexState) -> bool:
            nonlocal found
            present = doc_id in state.by_id
            found = found or present
            return present

        with self._lock:
            self._commit(index, [(None, doc_id, None, None, None, None)], check)
        self._after_write(index)
        return found

    def search(
        self, query: str, top_k: int = 5, index: str = DEFAULT_INDEX, filter: dict | None = None, exact: bool = False
    ) -> list[str]:
        """doc_ids of the top_k closest matching documents, closest first."""
        return [r["doc_id"] for r in self.search_with_scores(query, top_k, index=index, filter=filter, exact=exact)]

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        index: str = DEFAULT_INDEX,
        filter: dict | None = None,
        passages: bool = False,
        exact: bool = False,
    ) -> list[dict]:
        """Top_k matches as [{"doc_id", "score", "metadata"}], closest first.

        `filter` (see vectorstore.filters) restricts the search to documents
        whose metadata matches. On an exact index (the default) these are
        the true top_k among the matching documents; on an approximate one,
        almost always the same, unless `exact` asks for a full comparison
        (see vectorstore.vectors). `score` is cosine
        similarity (-1..1, higher is closer): vectors are unit-normalised,
        so it is 1 - d/2 from FAISS's squared L2 distance d. A long document
        is split into chunks, and scores as its closest chunk; with
        `passages`, each match also has "passage" (that chunk's text, or the
        whole text of an unchunked document) and "span" ([start, end), its
        characters in the text).
        """
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        self._check_name(index)
        if filter:
            validate_filter(filter)
        qv = np.expand_dims(self._embed(query), axis=0)
        with self._lock:
            state = self._ensure(index).state
            out = []
            for d, doc, chunk in state.search(qv, top_k, filter, exact):
                hit = {"doc_id": doc["doc_id"], "score": float(1.0 - d / 2.0), "metadata": _copy_metadata(doc["metadata"])}
                if passages:
                    start, end = doc["spans"][chunk] if "spans" in doc else (0, len(doc["text"]))
                    hit["passage"] = doc["text"][start:end]
                    hit["span"] = [start, end]
                out.append(hit)
            return out

    def count_documents(self, index: str = DEFAULT_INDEX) -> int:
        self._check_name(index)
        with self._lock:
            return len(self._ensure(index).state.docs)

    def embedding_model_name(self) -> str:
        return MODEL_NAME


_gc_pause_lock = threading.Lock()
_gc_pauses = 0


@contextlib.contextmanager
def _gc_paused():
    """Pause the cyclic GC while building a large index in memory.

    A rebuild allocates a few objects per document (records, dicts) and no
    reference cycles; without this the collector repeatedly walks them all,
    which made loading a 50k-doc index ~40% slower. Counted, so concurrent
    rebuilds don't re-enable it early, and only re-enabled if it was on.
    """
    global _gc_pauses
    with _gc_pause_lock:
        if _gc_pauses == 0:
            _gc_paused.was_enabled = gc.isenabled()
            gc.disable()
        _gc_pauses += 1
    try:
        yield
    finally:
        with _gc_pause_lock:
            _gc_pauses -= 1
            if _gc_pauses == 0 and _gc_paused.was_enabled:
                gc.enable()


def _find_run(keys: list[str], run: list[str]) -> int | None:
    """Position of `run` as a contiguous slice of `keys`, or None."""
    if not run or run[0] not in keys:
        return None
    pos = keys.index(run[0])
    return pos if keys[pos: pos + len(run)] == run else None


def _split_parts(records: list[tuple], dim: int) -> list[bytes]:
    """Encode records as segments of at most ~MAX_SEGMENT_BYTES each."""
    parts, current, size = [], [], 0
    for rec in records:
        rec_size = len(rec[1]) + len(rec[2] or "") + len(json.dumps(rec[3])) + (4 * dim + 16) * _vector_count(rec) + 16
        if current and size + rec_size > MAX_SEGMENT_BYTES:
            parts.append(_encode_segment(dim, current))
            current, size = [], 0
        current.append(rec)
        size += rec_size
    if current or not parts:
        parts.append(_encode_segment(dim, current))
    return parts


@_corrupt_on_error("v3 index")
def _decode_v3(blob: bytes, key: str) -> tuple[int, list[tuple]]:
    """(dim, records) from a v3 single-object index."""
    if not blob.startswith(LEGACY_V3_MAGIC):
        raise ValueError(f"{key} is not a recognised index blob")
    pos = len(LEGACY_V3_MAGIC)
    (header_len,) = struct.unpack_from(">Q", blob, pos)
    pos += 8
    header = json.loads(blob[pos: pos + header_len])
    index = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8, offset=pos + header_len))
    if index.ntotal != len(header["docs"]):
        raise ValueError(f"{key} is corrupt: {index.ntotal} vectors but {len(header['docs'])} documents")
    dim = header["dim"]
    if not header["docs"]:
        return dim, []
    vectors = index.index.reconstruct_n(0, index.ntotal)
    row = {int(k): i for i, k in enumerate(faiss.vector_to_array(index.id_map))}
    return dim, [(None, d, t, m, (vectors, row[int(k)]), None) for k, d, t, m in header["docs"]]


def _collision(state: "_IndexState", doc_id: str, label: int) -> str | None:
    """The other document already using FAISS id `label`, if any (see
    `object_key_id`): writing `doc_id` would silently overwrite it."""
    doc = state.docs.get(label)
    if doc is None and label in state.owner:
        doc = state.docs[state.owner[label][0]]
    return None if doc is None or doc["doc_id"] == doc_id else doc["doc_id"]


def _copy_metadata(metadata: dict) -> dict:
    # Values are scalars or lists of scalars, so copying the lists makes
    # this a full copy: callers can't reach into the cached state.
    return {k: list(v) if isinstance(v, list) else v for k, v in metadata.items()}


def _public(doc: dict) -> dict:
    return {"doc_id": doc["doc_id"], "text": doc["text"], "metadata": _copy_metadata(doc["metadata"])}


def _normalize_etag(etag: str | None) -> str:
    return (etag or "").strip('"')


def _require_etag(etag: str | None, key: str) -> str:
    # Index writes are conditional on the ETag a read returned. Without one
    # every write would be sent as `If-Match: ""` and fail forever, so fail
    # loudly here instead.
    normalized = _normalize_etag(etag)
    if not normalized:
        raise RuntimeError(f"object store returned no ETag for {key!r}")
    return normalized


def _chunk_overlap_from_env() -> int | None:
    """S3VAH_CHUNK_OVERLAP: tokens shared by consecutive chunks, or "off"."""
    raw = os.environ.get("S3VAH_CHUNK_OVERLAP", "").strip().lower()
    if not raw:
        return DEFAULT_CHUNK_OVERLAP
    if raw in ("off", "false", "none"):
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"S3VAH_CHUNK_OVERLAP must be a number of tokens or 'off', not {raw!r}") from None
    if value < 0:
        raise ValueError("S3VAH_CHUNK_OVERLAP must not be negative")
    return value


def _cache_budget_from_env() -> int | None:
    """S3VAH_CACHE_MB: memory for loaded indexes per process, in MiB; unset: no limit."""
    raw = os.environ.get("S3VAH_CACHE_MB", "").strip()
    if not raw:
        return None
    try:
        mb = float(raw)
    except ValueError:
        raise ValueError(f"S3VAH_CACHE_MB must be a number of MiB, not {raw!r}") from None
    if mb < 0:
        raise ValueError("S3VAH_CACHE_MB must not be negative")
    return int(mb * 1024 * 1024)


def client_from_env() -> Minio:
    """A MinIO client for MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY and MINIO_SECURE."""
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


_store: VectorStore | None = None
_store_init_lock = threading.Lock()


def get_store() -> VectorStore:
    """Lazily build the process-wide VectorStore from env vars.

    Deferred to first use (rather than constructed at import time) so
    importing this module never requires a live MinIO connection or a
    model download -- both the app and tests can import it freely and only
    pay that cost when a store is actually needed.
    """
    global _store
    if _store is None:
        with _store_init_lock:
            if _store is None:
                client = client_from_env()
                # FAISS's OpenMP pool and ONNX Runtime's threads otherwise
                # contend for cores: a 0.05 ms search over 1,000 docs took
                # ~18 ms. A single-query flat search runs on one thread anyway.
                faiss.omp_set_num_threads(1)
                _store = VectorStore(
                    client,
                    OnnxEmbedder.from_hub(),
                    chunk_overlap=_chunk_overlap_from_env(),
                    cache=IndexCache(_cache_budget_from_env()),
                    graph_build=os.environ.get("S3VAH_GRAPH_BUILD", "").strip() or "background",
                )
    return _store
