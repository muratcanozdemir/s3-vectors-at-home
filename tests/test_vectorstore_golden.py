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
from vectorstore.core import VectorStore


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
    ["", "has/slash", "faiss.index", "index.ids.json", "x" * 513],
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
