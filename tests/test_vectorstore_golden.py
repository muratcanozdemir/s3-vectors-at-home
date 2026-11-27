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
from vectorstore.core import (
    DEFAULT_INDEX,
    MAX_PAGE_SIZE,
    MAX_TOP_K,
    VectorStore,
    index_object_key,
)

INDEX_OBJ = index_object_key(DEFAULT_INDEX)


@pytest.fixture
def store():
    return VectorStore(FakeMinioClient(), FakeEmbedder())


def test_add_and_get_document(store):
    store.add_document("doc-1", "hello world")
    assert store.get_document("doc-1") == {"doc_id": "doc-1", "text": "hello world", "metadata": {}}


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
    ["", "has/slash", "x" * 513],
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
    assert store.get_document("c") == {"doc_id": "c", "text": "gamma", "metadata": {}}


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


# -- bounds ----------------------------------------------------------------


def test_search_rejects_top_k_above_max(store):
    with pytest.raises(ValueError):
        store.search("text", top_k=MAX_TOP_K + 1)


@pytest.mark.parametrize("skip,limit", [(-1, 10), (0, -1), (0, MAX_PAGE_SIZE + 1)])
def test_list_documents_rejects_out_of_range_paging(store, skip, limit):
    with pytest.raises(ValueError):
        store.list_documents(skip=skip, limit=limit)


# -- index format ----------------------------------------------------------


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


def test_bulk_keeps_vectors_aligned_with_their_ids(store):
    docs = [(f"d{i}", f"text {i}") for i in range(11)]
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


# -- metadata & filtered search ----------------------------------------------


def _corpus(store, index=DEFAULT_INDEX):
    docs = [
        ("en-old", "apples and pears", {"lang": "en", "year": 2001, "tags": ["fruit"]}),
        ("en-new", "apples and oranges", {"lang": "en", "year": 2024, "tags": ["fruit", "citrus"]}),
        ("de-new", "Äpfel und Birnen", {"lang": "de", "year": 2023}),
        ("plain", "no metadata here", None),
    ]
    store.add_documents(docs, index=index)
    return docs


def test_metadata_round_trips_through_get_list_and_search(store):
    _corpus(store)
    assert store.get_document("en-new")["metadata"] == {"lang": "en", "year": 2024, "tags": ["fruit", "citrus"]}
    assert store.get_document("plain")["metadata"] == {}
    listed = {d["doc_id"]: d["metadata"] for d in store.list_documents()}
    assert listed["de-new"] == {"lang": "de", "year": 2023}
    top = store.search_with_scores("apples and oranges", top_k=1)[0]
    assert top["doc_id"] == "en-new" and top["metadata"]["year"] == 2024


@pytest.mark.parametrize(
    "flt,expected",
    [
        ({"lang": "en"}, {"en-old", "en-new"}),
        ({"year": {"$gte": 2023}}, {"en-new", "de-new"}),
        ({"tags": "citrus"}, {"en-new"}),
        ({"lang": {"$ne": "en"}}, {"de-new", "plain"}),
        ({"$or": [{"lang": "de"}, {"year": {"$lt": 2010}}]}, {"de-new", "en-old"}),
        ({"year": {"$exists": False}}, {"plain"}),
        ({"lang": "fr"}, set()),
    ],
)
def test_filtered_search_returns_only_matching_docs(store, flt, expected):
    _corpus(store)
    assert set(store.search("apples", top_k=10, filter=flt)) == expected


def test_filtered_search_is_the_exact_top_k_among_matches(store):
    import numpy as np

    emb = FakeEmbedder()
    docs = [(f"d{i}", f"text {i}", {"parity": i % 2}) for i in range(40)]
    store.add_documents(docs)
    q = emb.encode(["query"])[0]
    odd = [(d, t) for d, t, m in docs if m["parity"] == 1]
    expected = sorted(odd, key=lambda dt: -float(np.dot(q, emb.encode([dt[1]])[0])))[:5]
    assert store.search("query", top_k=5, filter={"parity": 1}) == [d for d, _ in expected]


def test_upsert_replaces_metadata(store):
    store.add_document("x", "text", {"v": 1})
    store.add_document("x", "text", {"v": 2})
    assert store.search("text", top_k=5, filter={"v": 1}) == []
    assert store.search("text", top_k=5, filter={"v": 2}) == ["x"]


def test_invalid_filter_is_rejected(store):
    store.add_document("x", "text")
    with pytest.raises(ValueError):
        store.search("text", top_k=5, filter={"v": {"$regex": "."}})


def test_bad_metadata_rejects_whole_batch(store):
    with pytest.raises(ValueError):
        store.add_documents([("ok", "fine", {"a": 1}), ("bad", "nope", {"a": {"nested": True}})])
    assert store.count_documents() == 0


# -- named indexes -------------------------------------------------------------


def test_default_index_always_listed_and_others_created_explicitly(store):
    from vectorstore.core import IndexAlreadyExistsError

    assert store.list_indexes() == ["default"]
    store.create_index("papers")
    store.create_index("notes-2024")
    assert store.list_indexes() == ["default", "notes-2024", "papers"]
    with pytest.raises(IndexAlreadyExistsError):
        store.create_index("papers")
    with pytest.raises(IndexAlreadyExistsError):
        store.create_index("default")


@pytest.mark.parametrize("bad", ["", "Upper", "under_score", "-lead", "trail-", "a" * 64, "has/slash", "dot.ted"])
def test_invalid_index_names_are_rejected(store, bad):
    with pytest.raises(ValueError):
        store.create_index(bad)


def test_indexes_are_isolated(store):
    store.create_index("other")
    store.add_document("same-id", "in default")
    store.add_document("same-id", "in other", {"where": "other"}, index="other")
    assert store.get_document("same-id")["text"] == "in default"
    assert store.get_document("same-id", index="other")["text"] == "in other"
    assert store.count_documents() == 1 and store.count_documents(index="other") == 1
    assert store.search("in other", top_k=5, filter={"where": "other"}) == []
    assert store.search("in other", top_k=5, index="other", filter={"where": "other"}) == ["same-id"]
    store.delete_document("same-id", index="other")
    assert store.get_document("same-id")["text"] == "in default"


@pytest.mark.parametrize(
    "op",
    [
        lambda s: s.add_document("x", "t", index="nope"),
        lambda s: s.get_document("x", index="nope"),
        lambda s: s.list_documents(index="nope"),
        lambda s: s.count_documents(index="nope"),
        lambda s: s.delete_document("x", index="nope"),
        lambda s: s.search("t", index="nope"),
        lambda s: s.delete_index("nope"),
    ],
)
def test_missing_index_raises_not_found_and_is_not_created(store, op):
    from vectorstore.core import IndexNotFoundError

    with pytest.raises(IndexNotFoundError):
        op(store)
    assert store.list_indexes() == ["default"]


def test_delete_index_removes_it_and_its_documents(store):
    from vectorstore.core import IndexNotFoundError

    store.create_index("tmp")
    store.add_document("x", "t", index="tmp")
    store.delete_index("tmp")
    assert store.list_indexes() == ["default"]
    with pytest.raises(IndexNotFoundError):
        store.get_document("x", index="tmp")
    store.create_index("tmp")  # re-created empty
    assert store.count_documents(index="tmp") == 0


def test_default_index_cannot_be_deleted(store):
    with pytest.raises(ValueError):
        store.delete_index("default")


def test_writer_losing_to_index_delete_does_not_resurrect_it():
    from vectorstore.core import IndexNotFoundError

    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    a.create_index("tmp")
    a.add_document("x", "t", index="tmp")
    key = index_object_key("tmp")
    real_put = client._put_object

    def b_deletes_index_first(bucket, k, data, headers):
        if k == key:
            client._put_object = real_put
            b.delete_index("tmp")
        return real_put(bucket, k, data, headers)

    client._put_object = b_deletes_index_first
    with pytest.raises(IndexNotFoundError):
        a.add_document("y", "t2", index="tmp")
    assert key not in client.objects[a._bucket]


# -- same-doc_id races are now consistent --------------------------------------


def _race(client, key, other_action):
    """Run other_action() just before the next conditional PUT to `key`."""
    real_put = client._put_object

    def wrapped(bucket, k, data, headers):
        if k == key:
            client._put_object = real_put
            other_action()
        return real_put(bucket, k, data, headers)

    client._put_object = wrapped


def test_concurrent_writes_to_same_doc_id_keep_text_metadata_and_vector_together():
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    a.add_document("x", "original", {"v": 0})
    _race(client, INDEX_OBJ, lambda: b.add_document("x", "from b", {"v": "b"}))
    a.add_document("x", "from a", {"v": "a"})  # loses, retries on top of b's write

    fresh = VectorStore(client, FakeEmbedder())
    doc = fresh.get_document("x")
    assert doc == {"doc_id": "x", "text": "from a", "metadata": {"v": "a"}}
    top = fresh.search_with_scores("from a", top_k=1)[0]  # the vector is a's too
    assert top["doc_id"] == "x" and top["score"] == pytest.approx(1.0, abs=1e-5)
    assert fresh.count_documents() == 1


def test_add_racing_delete_of_same_doc_id_ends_consistent():
    client = FakeMinioClient()
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    a.add_document("x", "v1")
    _race(client, INDEX_OBJ, lambda: b.delete_document("x"))
    a.add_document("x", "v2")
    fresh = VectorStore(client, FakeEmbedder())
    assert fresh.get_document("x")["text"] == "v2"
    assert fresh.search("v2", top_k=5) == ["x"]


# -- guards ------------------------------------------------------------------------


def test_doc_id_hash_collision_is_detected_not_overwritten(store, monkeypatch):
    from vectorstore import core

    monkeypatch.setattr(core, "object_key_id", lambda doc_id: 42)
    store.add_document("first", "first text")
    with pytest.raises(ValueError, match="collides"):
        store.add_document("second", "second text")
    assert store.get_document("first")["text"] == "first text"
    assert store.count_documents() == 1


def test_embedding_dimension_mismatch_is_rejected():
    client = FakeMinioClient()
    VectorStore(client, FakeEmbedder()).add_document("x", "t")

    class SmallerEmbedder(FakeEmbedder):
        dim = 4

    with pytest.raises(ValueError, match="dimension"):
        VectorStore(client, SmallerEmbedder()).add_document("y", "t")


def test_index_blob_round_trips(store):
    from vectorstore.core import _IndexState

    _corpus(store)
    state = _IndexState.decode(store._client.objects[store._bucket][INDEX_OBJ], INDEX_OBJ)
    assert state.faiss.ntotal == 4 and sorted(state.by_id) == ["de-new", "en-new", "en-old", "plain"]
    assert state.docs[state.by_id["en-old"]]["metadata"]["year"] == 2001


def test_index_is_a_single_object_with_no_per_doc_objects(store):
    _corpus(store)
    assert set(store._client.objects[store._bucket]) == {INDEX_OBJ}


# -- migration from older layouts ---------------------------------------------------


def _legacy_bucket(docs, v2_live=None, vector_only=(), text_only=()):
    """A bucket as the pre-v3 code left it: per-doc .npy/.meta.json, plus an
    index.v2.bin listing `v2_live` doc_ids if given (else no index: v1/none)."""
    import json
    import struct

    import faiss
    import numpy as np

    from vectorstore import core

    client = FakeMinioClient()
    client.make_bucket("vectors")
    objs = client.objects["vectors"]
    emb = FakeEmbedder()
    for doc_id, text in docs:
        objs[f"{doc_id}.npy"] = emb.encode([text])[0].tobytes()
        objs[f"{doc_id}.meta.json"] = json.dumps({"doc_id": doc_id, "text": text}).encode()
    for doc_id in vector_only:
        objs[f"{doc_id}.npy"] = emb.encode(["orphan"])[0].tobytes()
    for doc_id in text_only:
        objs[f"{doc_id}.meta.json"] = json.dumps({"doc_id": doc_id, "text": "orphan"}).encode()
    if v2_live is not None:
        idx = faiss.IndexIDMap2(faiss.IndexFlatL2(emb.dim))
        keys = [core.object_key_id(d) for d in v2_live]
        vecs = np.vstack([np.frombuffer(objs[f"{d}.npy"], dtype=np.float32) for d in v2_live])
        idx.add_with_ids(vecs, np.array(keys, dtype="int64"))
        ids_json = json.dumps({str(k): d for k, d in zip(keys, v2_live)}).encode()
        objs["index.v2.bin"] = b"".join(
            (b"S3VAH-IDX\x01", struct.pack(">Q", len(ids_json)), ids_json, faiss.serialize_index(idx).tobytes())
        )
    return client


def test_migrates_v2_layout_into_default_index():
    client = _legacy_bucket([("a", "alpha"), ("b", "beta"), ("stale", "not in v2 index")], v2_live=["a", "b"])
    legacy = dict(client.objects["vectors"])
    store = VectorStore(client, FakeEmbedder())
    assert sorted(d["doc_id"] for d in store.list_documents()) == ["a", "b"]  # v2 index decides what's live
    assert store.get_document("a") == {"doc_id": "a", "text": "alpha", "metadata": {}}
    assert store.search("beta", top_k=1) == ["b"]
    objs = client.objects["vectors"]
    assert INDEX_OBJ in objs
    assert all(objs[k] == v for k, v in legacy.items())  # legacy objects left untouched


def test_migrates_v1_or_index_less_layout_skipping_half_written_docs():
    client = _legacy_bucket([("a", "alpha"), ("b", "beta")], vector_only=["vec-only"], text_only=["text-only"])
    store = VectorStore(client, FakeEmbedder())
    assert sorted(d["doc_id"] for d in store.list_documents()) == ["a", "b"]
    assert store.search("alpha", top_k=1) == ["a"]


def test_migration_runs_once():
    client = _legacy_bucket([("a", "alpha")])
    VectorStore(client, FakeEmbedder()).count_documents()
    client.objects["vectors"]["late.npy"] = FakeEmbedder().encode(["late"])[0].tobytes()
    client.objects["vectors"]["late.meta.json"] = b'{"doc_id": "late", "text": "late"}'
    assert VectorStore(client, FakeEmbedder()).count_documents() == 1


def test_concurrent_migrations_keep_exactly_one_result():
    client = _legacy_bucket([("a", "alpha"), ("b", "beta")])
    a = VectorStore(client, FakeEmbedder())
    b = VectorStore(client, FakeEmbedder())
    _race(client, INDEX_OBJ, lambda: b.add_document("c", "gamma"))  # b migrates + writes first
    assert a.count_documents() == 3  # a's create loses; it reads b's migrated index
    assert sorted(d["doc_id"] for d in a.list_documents()) == ["a", "b", "c"]


def test_empty_bucket_gets_an_empty_default_index(store):
    assert store.count_documents() == 0
    assert INDEX_OBJ in store._client.objects[store._bucket]


def test_returned_metadata_is_a_copy_not_the_cached_state(store):
    # Callers editing what they got back must not change what later reads
    # (served from the in-memory cache) return.
    store.add_document("x", "text", {"tags": ["a"], "v": 1})
    store.search_with_scores("text", top_k=1)[0]["metadata"]["v"] = 99
    store.get_document("x")["metadata"]["v"] = 98
    store.list_documents()[0]["metadata"]["v"] = 97
    assert store.get_document("x")["metadata"] == {"tags": ["a"], "v": 1}
    store.get_document("x")["metadata"]["tags"].append("b")
    assert store.get_document("x")["metadata"]["tags"] == ["a"]
