import uuid
from vectorstore import core

def unique_id():
    return f"test-{uuid.uuid4()}"

def test_add_and_get_document():
    doc_id = unique_id()
    text = "Document for core API testing"
    core.add_document(doc_id, text)
    doc = core.get_document(doc_id)
    assert doc is not None
    assert doc["doc_id"] == doc_id
    assert text in doc["text"]

def test_search_vectors_finds_doc():
    doc_id = unique_id()
    text = "unicorn and dragon"
    core.add_document(doc_id, text)
    matches = core.search_vectors("dragon", top_k=3)
    assert doc_id in matches
    core.delete_document(doc_id)

def test_list_documents_and_count_documents():
    doc_id1, doc_id2 = unique_id(), unique_id()
    core.add_document(doc_id1, "first test doc")
    core.add_document(doc_id2, "second test doc")
    docs = core.list_documents()
    ids = [d["doc_id"] for d in docs]
    assert doc_id1 in ids and doc_id2 in ids
    assert core.count_documents() >= 2
    # Cleanup
    core.delete_document(doc_id1)
    core.delete_document(doc_id2)

def test_delete_document_removes_all():
    doc_id = unique_id()
    text = "document to delete"
    core.add_document(doc_id, text)
    assert core.get_document(doc_id) is not None
    assert core.delete_document(doc_id) is True
    assert core.get_document(doc_id) is None

def test_embedding_model_name():
    name = core.embedding_model_name()
    assert isinstance(name, str)
    assert len(name) > 0
