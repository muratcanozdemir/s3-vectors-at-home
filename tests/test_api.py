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
    results = search_r.json()["results"]
    assert [r["doc_id"] for r in results] == matches
    assert all(-1.0 <= r["score"] <= 1.0 for r in results)
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


def test_index_write_conflict_maps_to_503(monkeypatch):
    from vectorstore import core

    class Conflicted:
        def add_document(self, *a, **k):
            raise core.IndexWriteConflictError("busy")

    monkeypatch.setattr("api.main.get_store", lambda: Conflicted())
    r = client.post("/upload", json={"doc_id": "x", "text": "y"})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"


def test_metadata_filter_and_named_indexes_over_http():
    import uuid

    name = f"api-{uuid.uuid4().hex[:8]}"
    r = client.post("/indexes", json={"name": name})
    assert r.status_code == 201
    assert client.post("/indexes", json={"name": name}).status_code == 409
    assert name in client.get("/indexes").json()["indexes"]
    try:
        base = f"/indexes/{name}"
        r = client.post(f"{base}/upload/bulk", json={"docs": [
            {"doc_id": "en", "text": "a guide to sourdough bread", "metadata": {"lang": "en", "year": 2024}},
            {"doc_id": "de", "text": "ein Leitfaden für Sauerteigbrot", "metadata": {"lang": "de", "year": 2020}},
        ]})
        assert r.status_code == 200 and r.json()["count"] == 2
        r = client.post(f"{base}/search", json={"query": "sourdough", "top_k": 5, "filter": {"lang": "de"}})
        assert r.json()["matches"] == ["de"]
        assert r.json()["results"][0]["metadata"] == {"lang": "de", "year": 2020}
        assert client.get(f"{base}/document/en").json()["metadata"]["year"] == 2024
        assert client.get(f"{base}/status").json()["document_count"] == 2
        assert client.get("/document/en").status_code in (200, 404)  # default index is separate
        assert client.post(f"{base}/search", json={"query": "x", "filter": {"lang": {"$regex": "."}}}).status_code == 400
        assert client.post(f"{base}/upload", json={"doc_id": "bad", "text": "t", "metadata": {"a": {"b": 1}}}).status_code == 400
        assert client.delete(f"{base}/document/en").status_code == 200
        assert client.delete(f"{base}/document/en").status_code == 404
    finally:
        assert client.delete(f"/indexes/{name}").status_code == 200
    assert client.get(f"/indexes/{name}/documents").status_code == 404
    assert client.post(f"/indexes/{name}/upload", json={"doc_id": "x", "text": "t"}).status_code == 404
    assert client.delete("/indexes/default").status_code == 400
    assert client.post("/indexes", json={"name": "Bad_Name"}).status_code == 400
