from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_status_works():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "embedding_model" in data

def test_upload_and_search_and_delete():
    doc_id = "unittest-doc-1"
    text = "The quick brown fox jumps over the lazy dog."
    # Upload
    r = client.post("/upload", json={"doc_id": doc_id, "text": text})
    assert r.status_code == 200
    # Search
    search_r = client.post("/search", json={"query": "fox", "top_k": 1})
    assert search_r.status_code == 200
    matches = search_r.json()["matches"]
    assert doc_id in matches
    # Get doc
    get_r = client.get(f"/document/{doc_id}")
    assert get_r.status_code == 200
    assert get_r.json()["doc_id"] == doc_id
    # List docs
    list_r = client.get("/documents")
    assert list_r.status_code == 200
    doc_ids = [d["doc_id"] for d in list_r.json()["documents"]]
    assert doc_id in doc_ids
    # Delete
    del_r = client.delete(f"/document/{doc_id}")
    assert del_r.status_code == 200
    # Get deleted
    get_r2 = client.get(f"/document/{doc_id}")
    assert get_r2.status_code == 404

def test_bulk_upload():
    docs = [
        {"doc_id": "bulk1", "text": "First doc"},
        {"doc_id": "bulk2", "text": "Second doc"}
    ]
    r = client.post("/upload/bulk", json={"docs": docs})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    # Both should be searchable
    for doc in docs:
        s = client.post("/search", json={"query": doc["text"], "top_k": 1})
        assert doc["doc_id"] in s.json()["matches"]
        # Clean up
        client.delete(f"/document/{doc['doc_id']}")

def test_count_documents_consistent():
    # Should match list length
    resp = client.get("/status")
    count = resp.json()["document_count"]
    resp2 = client.get("/documents")
    assert count == len(resp2.json()["documents"])
