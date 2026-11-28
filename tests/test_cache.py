"""The per-process memory budget for loaded indexes (IndexCache)."""

import random
import threading

import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore import core
from vectorstore.core import IndexCache, VectorStore
from vectorstore.s3vectors import S3VectorsService


def _store(client=None, cache=None, **kwargs):
    return VectorStore(client or FakeMinioClient(), FakeEmbedder(), compaction="inline", cache=cache, **kwargs)


def _fill(store, name, n):
    store.create_index(name)
    store.add_documents([(f"{name}-{i}", f"{name} text {i}") for i in range(n)], index=name)


def _held(store):
    return sorted(store._cache)


def _index_bytes(n):
    """What an index of n _fill documents is estimated at."""
    probe = _store()
    _fill(probe, "probe", n)
    return probe._cache["probe"].state.nbytes


def test_estimate_follows_writes_and_returns_to_zero():
    store = _store()
    store.create_index("a")
    store.add_documents([(f"d{i}", "x " * i, {"k": i, "tags": ["p", "q"]}) for i in range(50)], index="a")
    state = store._cache["a"].state
    assert state.nbytes > 50 * 4 * FakeEmbedder.dim
    store.add_document("d3", "replaced with a much longer text " * 20, index="a")
    for i in range(50):
        store.delete_document(f"d{i}", index="a")
    assert store._cache["a"].state.nbytes == 0


def test_chunked_documents_cost_more():
    vb = core._IndexState(8)._vector_bytes
    one = core._batch_bytes(vb, [(None, "d", "t", {}, None, None)], 1)
    five = core._batch_bytes(vb, [(None, "d", "t", {}, None, [[0, 1]] * 5)], 5)
    assert five > one + 4 * 4 * 8


def test_the_estimate_is_the_same_document_by_document_or_in_one_batch():
    """apply adds a batch's estimate at once but subtracts document by
    document; the two must agree or the total drifts."""
    store = _store(client=None)
    store._embedder.window = 3
    store.create_index("a")
    docs = [(f"d{i}", "word " * (i % 9), {"k": i, "tags": ["x"] * (i % 4)}) for i in range(40)]
    store.add_documents(docs, index="a")
    store.add_documents([(d, "other text") for d, _, _ in docs[::2]], index="a")
    for d, _, _ in docs:
        store.delete_document(d, index="a")
    assert store._cache["a"].state.nbytes == 0


def test_unlimited_by_default_keeps_everything():
    store = _store()
    for name in ("a", "b", "c"):
        _fill(store, name, 20)
    assert _held(store) == ["a", "b", "c"]  # "default" is only loaded once used
    assert store.cache_stats()["max_bytes"] is None
    assert store.cache_stats()["evictions"] == 0


def test_least_recently_used_index_is_dropped_first():
    per_index = _index_bytes(20)
    cache = IndexCache(max_bytes=int(per_index * 2.5))
    store = _store(cache=cache)
    for name in ("a", "b", "c"):
        _fill(store, name, 20)
    assert "a" not in store._cache
    assert {"b", "c"} <= set(store._cache)
    store.search("b text 1", index="b")  # b is now the most recently used
    _fill(store, "d", 20)
    assert set(store._cache) == {"b", "d"}
    assert cache.stats()["bytes"] <= cache.max_bytes
    assert cache.stats()["evictions"] == 2  # a, then c


def test_an_evicted_index_reloads_with_the_same_answers():
    cache = IndexCache(max_bytes=0)  # keeps only the index in use
    store = _store(cache=cache)
    _fill(store, "a", 30)
    before = store.search_with_scores("a text 7", top_k=3, index="a")
    _fill(store, "b", 30)
    assert "a" not in store._cache
    assert store.search_with_scores("a text 7", top_k=3, index="a") == before
    assert store.count_documents(index="a") == 30
    assert _held(store) == ["a"]


def test_one_index_larger_than_the_budget_still_works():
    cache = IndexCache(max_bytes=1)
    store = _store(cache=cache)
    _fill(store, "huge", 100)
    assert store.search("huge text 5", top_k=1, index="huge") == ["huge-5"]
    assert _held(store) == ["huge"]
    assert cache.stats()["indexes"] == 1


def test_the_accounting_matches_what_stores_hold():
    cache = IndexCache(max_bytes=_index_bytes(10) * 3)
    stores = [_store(cache=cache) for _ in range(3)]
    rng = random.Random(1)
    for step in range(200):
        store = rng.choice(stores)
        name = rng.choice(["a", "b", "c", "default"])
        if name != "default" and name not in store.list_indexes():
            store.create_index(name)
        if rng.random() < 0.6:
            store.add_document(f"d{rng.randint(0, 15)}", f"text {step}", index=name)
        else:
            store.search("text", top_k=3, index=name)
        held = {(id(s), n): s._cache[n].state.nbytes for s in stores for n in s._cache}
        assert set(held) == set(cache._entries)
        assert cache.stats()["bytes"] == sum(held.values())


def test_deleting_an_index_frees_its_share():
    cache = IndexCache()
    store = _store(cache=cache)
    _fill(store, "a", 10)
    held = cache.stats()["bytes"]
    store.delete_index("a")
    assert cache.stats()["bytes"] < held
    assert (id(store), "a") not in cache._entries


def test_stores_sharing_a_budget_evict_each_other():
    client = FakeMinioClient()
    cache = IndexCache(max_bytes=int(_index_bytes(20) * 1.5))
    one = _store(client, cache, prefix="one/", default_index=None)
    two = _store(client, cache, prefix="two/", default_index=None)
    _fill(one, "a", 20)
    _fill(two, "a", 20)
    assert "a" not in one._cache and "a" in two._cache
    assert one.search("a text 3", top_k=1, index="a") == ["a-3"]
    assert "a" not in two._cache


def test_s3vectors_buckets_share_the_service_budget():
    cache = IndexCache(max_bytes=0)
    svc = S3VectorsService(FakeMinioClient(), compaction="inline", cache=cache)
    for bucket in ("b-one", "b-two"):
        svc.create_vector_bucket({"vectorBucketName": bucket})
        svc.create_index({"vectorBucketName": bucket, "indexName": "idx", "dataType": "float32",
                          "dimension": 2, "distanceMetric": "cosine"})
        svc.put_vectors({"vectorBucketName": bucket, "indexName": "idx",
                         "vectors": [{"key": "k", "data": {"float32": [1.0, 0.0]}}]})
    assert cache.stats()["indexes"] == 1
    got = svc.query_vectors({"vectorBucketName": "b-one", "indexName": "idx", "topK": 1,
                             "queryVector": {"float32": [1.0, 0.0]}})
    assert [v["key"] for v in got["vectors"]] == ["k"]


def test_threads_on_stores_that_evict_each_other_dont_deadlock():
    client = FakeMinioClient()
    cache = IndexCache(max_bytes=0)
    stores = [_store(client, cache, prefix=f"p{i}/", default_index=None) for i in range(4)]
    for s in stores:
        _fill(s, "a", 10)
        _fill(s, "b", 10)
    errors = []

    def work(store, seed):
        rng = random.Random(seed)
        try:
            for _ in range(60):
                name = rng.choice(["a", "b"])
                assert store.search(f"{name} text 4", top_k=1, index=name) == [f"{name}-4"]
        except Exception as e:  # noqa: BLE001 -- reported by the main thread
            errors.append(e)

    threads = [threading.Thread(target=work, args=(s, i)) for i, s in enumerate(stores * 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert errors == []


@pytest.mark.parametrize(("value", "expected"), [("", None), ("512", 512 * 1024**2), ("0.5", 512 * 1024), ("0", 0)])
def test_cache_budget_from_env(monkeypatch, value, expected):
    monkeypatch.setenv("S3VAH_CACHE_MB", value)
    assert core._cache_budget_from_env() == expected


@pytest.mark.parametrize("value", ["lots", "-1"])
def test_a_bad_cache_budget_is_refused(monkeypatch, value):
    monkeypatch.setenv("S3VAH_CACHE_MB", value)
    with pytest.raises(ValueError, match="S3VAH_CACHE_MB"):
        core._cache_budget_from_env()
