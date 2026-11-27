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
    puts = []
    real_put = client.put_object

    def counting_put(bucket, key, data, length):
        puts.append(key)
        return real_put(bucket, key, data, length)

    client.put_object = counting_put
    store.add_documents([(f"d{i}", f"text {i}") for i in range(10)])
    assert puts.count(INDEX_OBJ) == 1


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
