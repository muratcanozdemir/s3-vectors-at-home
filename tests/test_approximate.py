"""Approximate (HNSW) search: vectorstore.vectors.HnswVectors and the store around it."""

import random
import threading

import numpy as np
import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore import core
from vectorstore import vectors as vs
from vectorstore.core import VectorStore
from vectorstore.s3vectors import S3VectorsService

DIM = 16


@pytest.fixture(autouse=True)
def small_graphs(monkeypatch):
    # Build graphs (and compact) at test-sized indexes, not thousands of vectors.
    monkeypatch.setattr(vs, "MIN_GRAPH_VECTORS", 50)
    monkeypatch.setattr(vs, "EXACT_FILTER_ROWS", 20)
    monkeypatch.setattr(vs, "EXACT_FILTER_FRACTION", 0.0)


def clustered(n, dim=DIM, seed=0, centers=20):
    """Unit vectors around a few centres, like real embeddings (uniform
    random vectors in many dimensions are the worst case for any ANN index)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(centers, dim))
    x = c[rng.integers(0, centers, n)] + 0.3 * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def brute(vectors: dict[int, np.ndarray], q, k, allowed=None):
    items = [(float(((v - q) ** 2).sum()), label) for label, v in vectors.items() if allowed is None or label in allowed]
    return sorted(items)[:k]


def recall(got_labels, want):
    return len(set(got_labels) & {label for _, label in want}) / max(1, len(want))


def hnsw(**kwargs):
    return vs.HnswVectors(DIM, **kwargs)


def test_small_indexes_have_no_graph_and_search_exactly():
    v = hnsw()
    x = clustered(30)
    v.add(x, np.arange(30, dtype=np.int64))
    assert not v.has_graph
    d, labels = v.search(x[3:4], 5)
    assert labels[0] == 3 and d[0] == pytest.approx(0, abs=1e-6)
    want = brute(dict(enumerate(x)), x[3], 5)
    assert labels == [label for _, label in want]


def test_a_graph_is_built_once_there_are_enough_vectors_and_finds_the_neighbours():
    v = hnsw()
    x = clustered(400)
    v.add(x[:200], np.arange(200, dtype=np.int64))
    assert v.has_graph and v.graph_builds == 1
    v.add(x[200:], np.arange(200, 400, dtype=np.int64))  # added straight to the graph
    assert v.graph_builds == 1
    table = dict(enumerate(x))
    queries = clustered(50, seed=1)
    r = np.mean([recall(v.search(q[None], 10)[1], brute(table, q, 10)) for q in queries])
    assert r >= 0.95
    # exact=True is a full comparison, whatever the graph says.
    for q in queries[:10]:
        _, labels = v.search(q[None], 10, exact=True)
        assert labels == [label for _, label in brute(table, q, 10)]


def test_removed_vectors_are_never_returned_and_compaction_keeps_the_rest():
    v = hnsw()
    x = clustered(400)
    v.add(x, np.arange(400, dtype=np.int64))
    table = dict(enumerate(x))
    rng = random.Random(0)
    gone = rng.sample(range(400), 90)  # under the compaction threshold
    v.remove(np.array(gone, dtype=np.int64))
    for g in gone:
        del table[g]
    assert v.ntotal == 310 and v._dead == 90
    for q in clustered(20, seed=2):
        _, labels = v.search(q[None], 10)
        assert not set(labels) & set(gone)
    more = [k for k in table if rng.random() < 0.5][:60]
    v.remove(np.array(more, dtype=np.int64))  # now past MAX_DEAD_FRACTION: compacted
    for g in more:
        del table[g]
    assert v._dead == 0 and v._rows == v.ntotal == len(table)
    for q in clustered(20, seed=3):
        assert v.search(q[None], 10, exact=True)[1] == [label for _, label in brute(table, q, 10)]
        assert recall(v.search(q[None], 10)[1], brute(table, q, 10)) >= 0.8


def test_re_adding_a_label_replaces_its_vector():
    v = hnsw()
    x = clustered(100)
    v.add(x, np.arange(100, dtype=np.int64))
    v.remove(np.array([7], dtype=np.int64))
    v.add(x[50:51], np.array([7], dtype=np.int64))  # 7 now has 50's vector
    _, labels = v.search(x[50:51], 2, exact=True)
    assert sorted(labels) == [7, 50]
    assert v.ntotal == 100


@pytest.mark.parametrize("size", [5, 300])
def test_filtered_search_selective_is_exact_and_broad_uses_the_graph(size):
    v = hnsw()
    x = clustered(600)
    v.add(x, np.arange(600, dtype=np.int64))
    table = dict(enumerate(x))
    allowed = np.array(random.Random(size).sample(range(600), size), dtype=np.int64)
    for q in clustered(15, seed=4):
        want = brute(table, q, 10, set(allowed.tolist()))
        _, labels = v.search(q[None], 10, allowed)
        assert set(labels) <= set(allowed.tolist())
        if size <= vs.EXACT_FILTER_ROWS:
            assert labels == [label for _, label in want]  # small filters: exact
        else:
            assert recall(labels, want) >= 0.9


class ManualExecutor:
    """Runs submitted jobs only when told to, to test what happens meanwhile."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


def test_background_build_searches_exactly_until_done_then_includes_later_rows():
    lock = threading.Lock()
    pool = ManualExecutor()
    v = hnsw(lock=lock, builder=pool)
    x = clustered(300)
    with lock:
        v.add(x[:100], np.arange(100, dtype=np.int64))
    assert len(pool.jobs) == 1 and not v.has_graph
    with lock:
        v.add(x[100:], np.arange(100, 300, dtype=np.int64))  # arrive while the graph builds
        assert v.search(x[250:251], 1)[1] == [250]  # exact meanwhile
    pool.run()
    assert v.has_graph and len(pool.jobs) == 0
    with lock:
        assert v.search(x[250:251], 1)[1] == [250]  # the later rows made it into the graph
        assert v.search(x[250:251], 1, exact=True)[1] == [250]
        assert v._graph.ntotal == 300


def test_a_build_overtaken_by_compaction_is_thrown_away_and_redone():
    lock = threading.Lock()
    pool = ManualExecutor()
    v = hnsw(lock=lock, builder=pool)
    x = clustered(400)
    with lock:
        v.add(x, np.arange(400, dtype=np.int64))
        v.remove(np.arange(0, 150, dtype=np.int64))  # compacts: the rows are renumbered
    assert len(pool.jobs) == 1  # the first build is still queued; compaction doesn't queue another
    pool.run()  # stale: discarded, and a new build queued for the compacted rows
    assert not v.has_graph and len(pool.jobs) == 1
    pool.run()
    assert v.has_graph
    with lock:
        assert v.search(x[200:201], 1)[1] == [200]
        assert all(label >= 150 for label in v.search(x[0:1], 20)[1])


def test_a_closed_index_never_installs_its_graph():
    lock = threading.Lock()
    pool = ManualExecutor()
    v = hnsw(lock=lock, builder=pool)
    with lock:
        v.add(clustered(100), np.arange(100, dtype=np.int64))
    v.close()
    pool.run()
    assert not v.has_graph


def test_graph_build_off_searches_exactly_and_never_builds():
    store = _raw_store(graph_build="off")
    store.create_index("ann", DIM, search={"mode": "approximate"})
    x = clustered(200)
    store.put_vectors("ann", [(f"k{i}", x[i], {}) for i in range(200)])
    assert store.index_stats("ann")["graph_ready"] is False
    assert store.query_vectors("ann", x[5], 1)[1][0]["key"] == "k5"


def test_a_failed_background_build_is_logged_and_retried(caplog, monkeypatch):
    lock = threading.Lock()
    pool = ManualExecutor()
    v = hnsw(lock=lock, builder=pool)
    real = v._build_graph
    monkeypatch.setattr(v, "_build_graph", lambda vectors: (_ for _ in ()).throw(MemoryError("boom")))
    with lock:
        v.add(clustered(100), np.arange(100, dtype=np.int64))
    pool.run()
    assert not v.has_graph and "building an HNSW graph failed" in caplog.text
    monkeypatch.setattr(v, "_build_graph", real)
    with lock:
        v.add(clustered(1, seed=9), np.array([100], dtype=np.int64))  # the next write retries
    pool.run()
    assert v.has_graph


def test_a_background_builder_needs_a_lock():
    with pytest.raises(ValueError, match="lock"):
        vs.HnswVectors(DIM, builder=ManualExecutor())


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("fuzzy",), "search must be"),
        (("exact",), None),
        (("approximate",), None),
    ],
)
def test_search_config(args, error):
    if error:
        with pytest.raises(ValueError, match=error):
            vs.search_config(*args)
    else:
        assert vs.search_config(*args)["mode"] == args[0]


@pytest.mark.parametrize(("params", "error"), [({"m": 2}, "m must be"), ({"ef_search": True}, "ef_search must be"),
                                              ({"bogus": 1}, "unknown")])
def test_bad_approximate_parameters_are_refused(params, error):
    with pytest.raises(ValueError, match=error):
        vs.search_config("approximate", **params)
    with pytest.raises(ValueError, match="no parameters"):
        vs.search_config("exact", m=16)


# -- through the store -----------------------------------------------------------


def _raw_store(client=None, graph_build="inline", **kwargs):
    return VectorStore(client or FakeMinioClient(), None, compaction="inline", default_index=None, raw_vectors=True,
                       graph_build=graph_build, **kwargs)


def test_an_approximate_index_is_remembered_and_rebuilt_on_load():
    client = FakeMinioClient()
    store = _raw_store(client)
    store.create_index("ann", DIM, search={"mode": "approximate", "ef_search": 100})
    x = clustered(300)
    store.put_vectors("ann", [(f"k{i}", x[i], {"i": i}) for i in range(300)])
    assert store.index_info("ann")["search"] == {"mode": "approximate", "m": 32, "ef_construction": 64, "ef_search": 100}
    stats = store.index_stats("ann")
    assert (stats["search"], stats["graph_ready"]) == ("approximate", True)
    fresh = _raw_store(client)
    assert fresh.index_stats("ann")["graph_ready"]
    assert fresh._cache["ann"].state.vectors.ef_search == 100
    _, hits = fresh.query_vectors("ann", x[42], 1)
    assert hits[0]["key"] == "k42"


def test_indexes_are_exact_by_default_and_older_manifests_too():
    store = _raw_store()
    store.create_index("plain", DIM)
    assert store.index_info("plain")["search"] == {"mode": "exact"}
    assert store.index_stats("plain")["search"] == "exact"
    manifest = core._new_manifest(DIM, "inc")
    del manifest["search"]  # as written before approximate search existed
    assert isinstance(store._new_state(manifest).vectors, vs.FlatVectors)


@pytest.mark.parametrize("seed", range(6))
def test_store_level_random_ops_exact_queries_match_brute_force(seed):
    """Writes, deletes, re-writes, filters and chunk-free raw vectors: with
    exact=True an approximate index must answer exactly; without, closely."""
    rng = random.Random(seed)
    store = _raw_store()
    store.create_index("ann", DIM, "euclidean", search={"mode": "approximate"})
    x = clustered(500, seed=seed)
    table: dict[str, tuple[np.ndarray, dict]] = {}
    for _ in range(12):
        batch = {f"k{rng.randint(0, 299)}": rng.randint(0, 499) for _ in range(rng.randint(10, 60))}
        store.put_vectors("ann", [(k, x[i], {"tag": i % 3}) for k, i in batch.items()])
        table.update({k: (x[i], {"tag": i % 3}) for k, i in batch.items()})
        gone = rng.sample(sorted(table), min(len(table), rng.randint(0, 25)))
        store.delete_vectors("ann", gone)
        for k in gone:
            del table[k]
        q = clustered(1, seed=rng.randint(0, 10**6))[0]
        flt = {"tag": rng.randint(0, 2)} if rng.random() < 0.5 else None
        want = sorted(
            (float(np.sqrt(((v - q) ** 2).sum())), k) for k, (v, md) in table.items() if flt is None or md["tag"] == flt["tag"]
        )[:10]
        _, exact_hits = store.query_vectors("ann", q, 10, flt, exact=True)
        # Keys share vectors here (random rows of x), so compare distances: tied keys may come in either order.
        assert [h["distance"] for h in exact_hits] == pytest.approx([d for d, _ in want], abs=1e-5)
        assert all(table[h["key"]][1]["tag"] == flt["tag"] for h in exact_hits) if flt else True
        _, approx_hits = store.query_vectors("ann", q, 10, flt)
        assert len({h["key"] for h in approx_hits} & {k for _, k in want}) >= 0.8 * len(want)


def test_text_indexes_can_be_approximate_with_chunked_documents():
    store = VectorStore(FakeMinioClient(), FakeEmbedder(window=4), compaction="inline", graph_build="inline")
    store.create_index("t", search={"mode": "approximate"})
    store.add_documents([(f"d{i}", " ".join(f"w{i}x{j}" for j in range(12)), {"n": i}) for i in range(30)], index="t")
    assert store.index_stats("t")["graph_ready"]  # 30 docs x 4+ chunks > 50 vectors
    hit = store.search_with_scores("w7x4 w7x5 w7x6 w7x7", top_k=1, index="t", passages=True)[0]
    assert (hit["doc_id"], hit["score"]) == ("d7", pytest.approx(1.0))
    assert len(store.search("w3x0", top_k=10, index="t")) == 10  # distinct documents, not chunks


def test_s3vectors_service_creates_approximate_indexes_when_configured(monkeypatch):
    monkeypatch.setenv("S3VAH_S3VECTORS_SEARCH", "approximate")
    svc = S3VectorsService(FakeMinioClient(), compaction="inline", graph_build="inline")
    svc.create_vector_bucket({"vectorBucketName": "bkt"})
    svc.create_index({"vectorBucketName": "bkt", "indexName": "idx", "dataType": "float32", "dimension": DIM,
                      "distanceMetric": "cosine"})
    x = clustered(120)
    svc.put_vectors({"vectorBucketName": "bkt", "indexName": "idx",
                     "vectors": [{"key": f"k{i}", "data": {"float32": x[i].tolist()}} for i in range(120)]})
    state = svc._store("bkt")._cache["idx"].state
    assert state.vectors.mode == "approximate" and state.vectors.has_graph
    got = svc.query_vectors({"vectorBucketName": "bkt", "indexName": "idx", "topK": 1,
                             "queryVector": {"float32": x[9].tolist()}})
    assert got["vectors"][0]["key"] == "k9"


def test_a_bad_s3vectors_search_setting_is_refused(monkeypatch):
    monkeypatch.setenv("S3VAH_S3VECTORS_SEARCH", "fast")
    with pytest.raises(ValueError, match="search must be"):
        S3VectorsService(FakeMinioClient())


@pytest.mark.parametrize("size", [3, 30, 400])  # gathered (<= 3% of 1,000), then scanned
def test_flat_filtered_search_is_exact_whichever_way_it_runs(size):
    v = vs.FlatVectors(DIM)
    x = clustered(1000)
    labels = np.arange(10**12, 10**12 + 1000, dtype=np.int64)  # labels aren't row numbers
    v.add(x, labels)
    table = dict(zip(labels.tolist(), x))
    allowed = set(random.Random(size).sample(labels.tolist(), size))
    for q in clustered(10, seed=5):
        d, got = v.search(q[None], 10, allowed)
        want = brute(table, q, 10, allowed)
        assert d == pytest.approx([dist for dist, _ in want], abs=1e-5)
        assert set(got) <= allowed and len(got) == min(10, size)
