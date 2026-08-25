"""End-to-end sanity checks against a real MinIO + real embedding model.

Exact-value regression coverage (index bookkeeping, pagination, validation,
concurrency) lives in test_vectorstore_golden.py and runs without any of
this. These tests exist to confirm the real collaborators (MinIO, FAISS
serialization, the actual sentence-transformers model) still wire together
correctly end to end.
"""

import uuid

from vectorstore.core import get_store


def unique_id():
    return f"test-{uuid.uuid4()}"


def test_add_and_get_document():
    store = get_store()
    doc_id = unique_id()
    text = "Document for core API testing"
    store.add_document(doc_id, text)
    doc = store.get_document(doc_id)
    assert doc == {"doc_id": doc_id, "text": text}
    store.delete_document(doc_id)


def test_search_vectors_finds_doc():
    store = get_store()
    doc_id = unique_id()
    text = "unicorn and dragon"
    store.add_document(doc_id, text)
    matches = store.search("dragon", top_k=3)
    assert doc_id in matches
    store.delete_document(doc_id)


def test_list_documents_and_count_documents():
    store = get_store()
    doc_id1, doc_id2 = unique_id(), unique_id()
    store.add_document(doc_id1, "first test doc")
    store.add_document(doc_id2, "second test doc")
    docs = store.list_documents()
    ids = [d["doc_id"] for d in docs]
    assert doc_id1 in ids and doc_id2 in ids
    store.delete_document(doc_id1)
    store.delete_document(doc_id2)


def test_delete_document_removes_all():
    store = get_store()
    doc_id = unique_id()
    text = "document to delete"
    store.add_document(doc_id, text)
    assert store.get_document(doc_id) is not None
    assert store.delete_document(doc_id) is True
    assert store.get_document(doc_id) is None


def test_embedding_model_name():
    assert get_store().embedding_model_name() == "all-MiniLM-L6-v2"
