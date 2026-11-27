"""Fast, deterministic tests against VectorStore with fake collaborators.

No MinIO, no model download, no Docker -- these run in-process and assert
exact expected values (hence "golden"), which is what the previous
integration-only test suite could not do: it only asserted shapes
(`"x" in matches`, `count >= 2`), which is exactly why the -1/duplicate
search bug and the stale-index-on-update bug went uncaught (see
features/core-hardening/exploration.md).
"""

import threading

import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore.core import INDEX_OBJ, MAX_PAGE_SIZE, MAX_TOP_K, VectorStore


@pytest.fixture
def store():
    return VectorStore(FakeMinioClient(), FakeEmbedder())


def test_add_and_get_document(store):
    store.add_document("doc-1", "hello world")
    assert store.get_document("doc-1") == {"doc_id": "doc-1", "text": "hello world"}


def test_get_missing_document_returns_none(store):
    assert store.get_document("nope") is None


def test_search_finds_its_own_text_first(store):
    store.add_document("doc-a", "alpha text")
    store.add_document("doc-b", "beta text")
    # A document's own text embeds to the exact same vector under the fake
    # embedder, so it is always the (single) exact match for its own query.
    assert store.search("alpha text", top_k=1) == ["doc-a"]


def test_search_top_k_larger_than_corpus_returns_exact_set_no_duplicates(store):
    store.add_document("doc-a", "alpha text")
    store.add_document("doc-b", "beta text")
    results = store.search("alpha text", top_k=5)
    assert len(results) == 2
    assert set(results) == {"doc-a", "doc-b"}
    assert len(set(results)) == len(results)  # no id repeated to pad out top_k


def test_search_empty_store_returns_empty_list(store):
    assert store.search("anything", top_k=5) == []


def test_updating_existing_doc_id_does_not_duplicate_index_entry(store):
    store.add_document("doc-1", "first version")
    store.add_document("doc-1", "second version")
    results = store.search("second version", top_k=10)
    assert results == ["doc-1"]  # not ["doc-1", "doc-1"]
    assert store.get_document("doc-1")["text"] == "second version"
    assert store.count_documents() == 1


def test_delete_document_removes_it_from_search(store):
    store.add_document("doc-1", "keep me")
    store.add_document("doc-2", "delete me")
    assert store.delete_document("doc-2") is True
    assert store.get_document("doc-2") is None
    assert "doc-2" not in store.search("delete me", top_k=10)
    assert store.search("keep me", top_k=10) == ["doc-1"]


def test_delete_missing_document_returns_false(store):
    assert store.delete_document("nope") is False


def test_delete_last_document_leaves_store_searchable(store):
    store.add_document("doc-1", "only doc")
    store.delete_document("doc-1")
    assert store.count_documents() == 0
    assert store.search("only doc", top_k=5) == []


def test_list_documents_pagination_is_sorted_and_exact(store):
    for doc_id in ["c", "a", "b"]:
        store.add_document(doc_id, f"text for {doc_id}")
    all_docs = [d["doc_id"] for d in store.list_documents(skip=0, limit=100)]
    assert all_docs == ["a", "b", "c"]
    page = [d["doc_id"] for d in store.list_documents(skip=1, limit=1)]
    assert page == ["b"]


def test_list_documents_text_preview_is_truncated(store):
    store.add_document("doc-1", "x" * 100)
    preview = store.list_documents()[0]["text_preview"]
    assert preview == "x" * 64


def test_count_documents(store):
    assert store.count_documents() == 0
    store.add_document("doc-1", "a")
    store.add_document("doc-2", "b")
    assert store.count_documents() == 2


def test_embedding_model_name_is_exact(store):
    assert store.embedding_model_name() == "all-MiniLM-L6-v2"


@pytest.mark.parametrize(
    "bad_doc_id",
    ["", "has/slash", "index.v2.bin", "faiss.index", "index.ids.json", "x" * 513],
)
def test_add_document_rejects_invalid_doc_id(store, bad_doc_id):
    with pytest.raises(ValueError):
        store.add_document(bad_doc_id, "some text")


def test_search_rejects_non_positive_top_k(store):
    store.add_document("doc-1", "text")
    with pytest.raises(ValueError):
        store.search("text", top_k=0)


def test_concurrent_adds_all_land(store):
    # Regression test for the read-modify-write race on the shared index:
    # every add must be reflected, none silently lost to a lost update.
    def add(i):
        store.add_document(f"doc-{i}", f"text {i}")

    threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count_documents() == 20
    results = store.search("text 0", top_k=20)
    assert len(results) == 20
    assert len(set(results)) == 20


# -- batch writes ----------------------------------------------------------


def test_add_documents_batch_is_searchable_and_counted(store):
    assert store.add_documents([("a", "alpha"), ("b", "beta"), ("c", "gamma")]) == 3
    assert store.count_documents() == 3
    assert store.search("beta", top_k=1) == ["b"]
    assert store.get_document("c") == {"doc_id": "c", "text": "gamma"}


def test_add_documents_writes_index_once():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    index_puts = []
    real_put = client._put_object

    def counting_put(bucket, key, data, headers):
        index_puts.append(key)
        return real_put(bucket, key, data, headers)

    client._put_object = counting_put
    store.add_documents([(f"d{i}", f"text {i}") for i in range(10)])
    assert index_puts.count(INDEX_OBJ) == 1


def test_add_documents_rejects_whole_batch_on_invalid_id(store):
    with pytest.raises(ValueError):
        store.add_documents([("ok-1", "fine"), ("bad/id", "nope"), ("ok-2", "fine")])
    # Validation happens before any write: nothing from the batch landed.
    assert store.count_documents() == 0
    assert store.search("fine", top_k=5) == []


def test_add_documents_duplicate_ids_last_wins_single_entry(store):
    store.add_documents([("x", "first"), ("x", "second")])
    assert store.get_document("x")["text"] == "second"
    assert store.search("second", top_k=10) == ["x"]


def test_add_documents_replaces_existing_entries(store):
    store.add_document("x", "old")
    store.add_documents([("x", "new"), ("y", "other")])
    results = store.search("new", top_k=10)
    assert sorted(results) == ["x", "y"]
    assert results[0] == "x"


def test_add_documents_empty_batch_is_noop(store):
    assert store.add_documents([]) == 0
    assert store.search("anything", top_k=5) == []


# -- index cache -----------------------------------------------------------


def test_repeated_searches_do_not_redownload_index():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("doc-1", "cached text")
    before = client.get_counts.get(INDEX_OBJ, 0)
    for _ in range(5):
        assert store.search("cached text", top_k=1) == ["doc-1"]
    assert client.get_counts.get(INDEX_OBJ, 0) == before


def test_cache_picks_up_writes_from_another_process():
    # Two VectorStores over the same bucket stand in for two server
    # processes: each has its own cache, coordinated only via the store.
    client = FakeMinioClient()
    writer = VectorStore(client, FakeEmbedder())
    reader = VectorStore(client, FakeEmbedder())
    writer.add_document("doc-1", "first")
    assert reader.search("first", top_k=10) == ["doc-1"]
    writer.add_document("doc-2", "second")
    assert sorted(reader.search("second", top_k=10)) == ["doc-1", "doc-2"]
    writer.delete_document("doc-1")
    assert reader.search("first", top_k=10) == ["doc-2"]
    writer.delete_document("doc-2")
    assert reader.search("second", top_k=10) == []


def test_index_rebuilt_when_index_objects_go_missing(store):
    store.add_documents([("a", "alpha"), ("b", "beta")])
    store._client.remove_object(store._bucket, INDEX_OBJ)
    assert sorted(store.search("alpha", top_k=10)) == ["a", "b"]


# -- bounds ----------------------------------------------------------------


def test_search_rejects_top_k_above_max(store):
    with pytest.raises(ValueError):
        store.search("text", top_k=MAX_TOP_K + 1)


@pytest.mark.parametrize("skip,limit", [(-1, 10), (0, -1), (0, MAX_PAGE_SIZE + 1)])
def test_list_documents_rejects_out_of_range_paging(store, skip, limit):
    with pytest.raises(ValueError):
        store.list_documents(skip=skip, limit=limit)


# -- index format ----------------------------------------------------------


def test_index_is_a_single_object(store):
    store.add_documents([("a", "alpha"), ("b", "beta")])
    names = set(store._client.objects[store._bucket])
    assert INDEX_OBJ in names
    assert not names & {"faiss.index", "index.ids.json"}


def test_index_blob_round_trips(store):
    store.add_documents([("a", "alpha"), ("b", "beta")])
    blob = store._client.objects[store._bucket][INDEX_OBJ]
    index, id_map = VectorStore._decode_index(blob)
    assert index.ntotal == 2
    assert sorted(id_map.values()) == ["a", "b"]


def test_legacy_two_object_layout_is_migrated(store):
    # A bucket written by the pre-v2 code: per-doc objects plus the old
    # faiss.index / index.ids.json pair, and no index.v2.bin.
    store.add_documents([("a", "alpha"), ("b", "beta")])
    objects = store._client.objects[store._bucket]
    del objects[INDEX_OBJ]
    objects["faiss.index"] = b"legacy"
    objects["index.ids.json"] = b"{}"
    fresh = VectorStore(store._client, FakeEmbedder())
    assert fresh.search("alpha", top_k=1) == ["a"]
    assert INDEX_OBJ in objects
    assert "faiss.index" not in objects and "index.ids.json" not in objects


def test_unrecognised_index_blob_is_an_error(store):
    store.add_document("a", "alpha")
    store._client.objects[store._bucket][INDEX_OBJ] = b"garbage"
    fresh = VectorStore(store._client, FakeEmbedder())
    with pytest.raises(ValueError):
        fresh.search("alpha", top_k=1)


# -- scores ----------------------------------------------------------------


def test_search_with_scores_exact_match_scores_one(store):
    store.add_documents([("a", "alpha"), ("b", "beta")])
    results = store.search_with_scores("alpha", top_k=2)
    assert results[0]["doc_id"] == "a"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_search_with_scores_is_cosine_similarity_sorted_desc(store):
    import numpy as np

    texts = {"a": "alpha", "b": "beta", "c": "gamma"}
    store.add_documents(texts.items())
    emb = FakeEmbedder()
    q = emb.encode(["query"])[0]
    expected = {d: float(np.dot(q, emb.encode([t])[0])) for d, t in texts.items()}
    results = store.search_with_scores("query", top_k=3)
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    for r in results:
        assert r["score"] == pytest.approx(expected[r["doc_id"]], abs=1e-5)


def test_search_matches_search_with_scores_order(store):
    store.add_documents([(f"d{i}", f"text {i}") for i in range(6)])
    assert store.search("text 3", top_k=4) == [r["doc_id"] for r in store.search_with_scores("text 3", top_k=4)]


# -- performance-shaped behaviour -------------------------------------------


def test_bulk_spanning_several_embed_chunks_keeps_ids_aligned(store, monkeypatch):
    from vectorstore import core

    monkeypatch.setattr(core, "EMBED_CHUNK_SIZE", 4)
    docs = [(f"d{i}", f"text {i}") for i in range(11)]  # 3 chunks, last one partial
    assert store.add_documents(docs) == 11
    for i in (0, 3, 4, 7, 10):
        assert store.search(f"text {i}", top_k=1) == [f"d{i}"]


def test_list_and_count_read_the_cached_index_not_the_bucket():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_documents([("b", "beta"), ("a", "alpha"), ("c", "gamma")])
    listed = []
    real_list = client.list_objects
    client.list_objects = lambda bucket: listed.append(bucket) or real_list(bucket)
    assert store.count_documents() == 3
    assert [d["doc_id"] for d in store.list_documents(skip=1, limit=5)] == ["b", "c"]
    assert listed == []


def test_upload_failure_in_bulk_leaves_index_untouched():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("keep", "kept text")
    real_put = client.put_object

    def flaky_put(bucket, key, data, length):
        if key == "boom.meta.json":
            raise RuntimeError("upload failed")
        return real_put(bucket, key, data, length)

    client.put_object = flaky_put
    with pytest.raises(RuntimeError):
        store.add_documents([("ok", "fine"), ("boom", "bad")])
    assert store.search("fine", top_k=10) == ["keep"]
    assert store.count_documents() == 1


# -- concurrent writers (compare-and-swap index updates) --------------------
# Two VectorStores over one FakeMinioClient stand in for two processes:
# separate locks and caches, coordinated only through the object store.


def _intercept_index_puts(client, before_put):
    """Run before_put(n) just before the n-th (0-based) conditional index PUT."""
    real = client._put_object
    calls = {"n": 0}

    def wrapped(bucket, key, data, headers):
        if key == INDEX_OBJ:
            n = calls["n"]
            calls["n"] += 1
            before_put(n)
        return real(bucket, key, data, headers)

    client._put_object = wrapped
    return calls


def test_lost_race_is_retried_on_top_of_the_winner():
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    a.add_document("base", "base text")
    b.search("base text", top_k=1)  # b now caches the same index version

    # Right before a's next index PUT, b commits its own update. a's PUT
    # carries the now-stale ETag, so it must be rejected and redone.
    def b_sneaks_in(n):
        if n == 0:
            client._put_object = real_put
            b.add_document("from-b", "b text")
            client._put_object = wrapped

    real_put = client._put_object
    _intercept_index_puts(client, b_sneaks_in)
    wrapped = client._put_object
    a.add_document("from-a", "a text")

    fresh = VectorStore(client, FakeEmbedder())
    assert sorted(d["doc_id"] for d in fresh.list_documents()) == ["base", "from-a", "from-b"]
    assert sorted(fresh.search("a text", top_k=10)) == ["base", "from-a", "from-b"]


def test_delete_racing_an_add_keeps_both_effects():
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    a.add_documents([("keep", "k"), ("gone", "g")])

    def b_adds(n):
        if n == 0:
            client._put_object = real_put
            b.add_document("new", "n")
            client._put_object = wrapped

    real_put = client._put_object
    _intercept_index_puts(client, b_adds)
    wrapped = client._put_object
    assert a.delete_document("gone")

    fresh = VectorStore(client, FakeEmbedder())
    assert sorted(d["doc_id"] for d in fresh.list_documents()) == ["keep", "new"]


def test_write_applied_but_reported_as_conflict_is_idempotent():
    # The PUT lands, but the client sees a 412 -- e.g. the response was
    # lost and urllib3's automatic retry of the PUT hit our own new ETag.
    # Re-applying the edit on top of its own effect must be harmless.
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("x", "x text")
    real = client._put_object
    state = {"armed": True}

    def apply_then_conflict(bucket, key, data, headers):
        result = real(bucket, key, data, headers)
        if key == INDEX_OBJ and state["armed"]:
            state["armed"] = False
            from tests.fakes import _s3_error

            raise _s3_error("PreconditionFailed", bucket, key)
        return result

    client._put_object = apply_then_conflict
    store.add_document("y", "y text")
    assert sorted(store.search("y text", top_k=10)) == ["x", "y"]  # no duplicate y

    state["armed"] = True
    assert store.delete_document("y")
    fresh = VectorStore(client, FakeEmbedder())
    assert fresh.search("x text", top_k=10) == ["x"]


def test_gives_up_after_bounded_retries_without_corrupting(monkeypatch):
    from tests.fakes import _s3_error
    from vectorstore import core

    monkeypatch.setattr(core, "INDEX_WRITE_ATTEMPTS", 3)
    monkeypatch.setattr(core, "INDEX_WRITE_BACKOFF_BASE_S", 0)
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("x", "x text")
    real = client._put_object
    attempts = []

    def always_conflict(bucket, key, data, headers):
        if key == INDEX_OBJ:
            attempts.append(1)
            raise _s3_error("PreconditionFailed", bucket, key)
        return real(bucket, key, data, headers)

    client._put_object = always_conflict
    with pytest.raises(core.IndexWriteConflictError):
        store.add_document("y", "y text")
    assert len(attempts) == 3
    client._put_object = real
    assert store.search("y text", top_k=10) == ["x"]  # index untouched by the failed write


def test_non_conflict_put_error_propagates_and_drops_cache():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("x", "x text")
    real = client._put_object

    def broken(bucket, key, data, headers):
        if key == INDEX_OBJ:
            raise ConnectionError("network down")
        return real(bucket, key, data, headers)

    client._put_object = broken
    with pytest.raises(ConnectionError):
        store.add_document("y", "y text")
    client._put_object = real
    # The in-memory index was mutated before the failed PUT; it must not
    # be served as if it were the stored version.
    assert store.search("y text", top_k=10) == ["x"]


def test_deleting_last_doc_keeps_an_empty_index_object(store):
    store.add_document("only", "only text")
    store.delete_document("only")
    assert INDEX_OBJ in store._client.objects[store._bucket]
    assert store.search("only text", top_k=5) == []
    assert store.count_documents() == 0
    store.add_document("next", "next text")
    assert store.search("next text", top_k=5) == ["next"]


def test_bootstrap_from_stale_listing_cannot_replace_a_real_index():
    # b doesn't see the index at load time (as if it raced a creator) and
    # builds one from a vector listing that still has a deleted doc. Its
    # write is create-only, so it loses and b uses the real index instead.
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    a.add_documents([("keep", "k"), ("gone", "g")])
    stale_vec = client.objects[a._bucket]["gone.npy"]
    a.delete_document("gone")
    client.objects[a._bucket]["gone.npy"] = stale_vec  # stale listing

    b = VectorStore(client, FakeEmbedder())
    real_load = b._load_index
    hidden = {"once": True}

    def load_hiding_index_once():
        if hidden["once"]:
            hidden["once"] = False
            return None, {}, None
        return real_load()

    b._load_index = load_hiding_index_once
    assert b.search("k", top_k=10) == ["keep"]


def test_concurrent_first_writes_to_an_empty_bucket_both_land():
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())

    def b_creates_first(n):
        if n == 0:
            client._put_object = real_put
            b.add_document("from-b", "b text")
            client._put_object = wrapped

    real_put = client._put_object
    _intercept_index_puts(client, b_creates_first)
    wrapped = client._put_object
    a.add_document("from-a", "a text")  # its If-None-Match: * create must lose, then retry
    fresh = VectorStore(client, FakeEmbedder())
    assert sorted(d["doc_id"] for d in fresh.list_documents()) == ["from-a", "from-b"]


# -- startup checks ------------------------------------------------------------


def test_refuses_backend_that_ignores_conditional_writes():
    from vectorstore.core import ConditionalWritesUnsupportedError

    client = FakeMinioClient(enforce_conditions=False)
    with pytest.raises(ConditionalWritesUnsupportedError):
        VectorStore(client, FakeEmbedder())
    assert client.objects["vectors"] == {}  # probe cleaned up after itself


def test_probe_leaves_no_objects_behind(store):
    assert store._client.objects[store._bucket] == {}


def test_losing_bucket_creation_race_is_fine():
    from tests.fakes import _s3_error

    client = FakeMinioClient()

    def already_created(bucket):
        client.buckets.add(bucket)
        client.objects.setdefault(bucket, {})
        raise _s3_error("BucketAlreadyOwnedByYou", bucket, "")

    client.make_bucket = already_created
    VectorStore(client, FakeEmbedder()).add_document("x", "x")


def test_bucket_owned_by_someone_else_is_an_error():
    from minio.error import S3Error

    from tests.fakes import _s3_error

    client = FakeMinioClient()

    def taken(bucket):
        raise _s3_error("BucketAlreadyExists", bucket, "")

    client.make_bucket = taken
    with pytest.raises(S3Error):
        VectorStore(client, FakeEmbedder())


def test_missing_etag_on_read_is_a_hard_error():
    client = FakeMinioClient()
    store = VectorStore(client, FakeEmbedder())
    store.add_document("x", "x text")
    real_stat = client.stat_object
    client.stat_object = lambda b, k: real_stat(b, k)._replace(etag="")
    fresh = VectorStore(client, FakeEmbedder())
    with pytest.raises(RuntimeError, match="no ETag"):
        fresh.search("x text", top_k=1)
