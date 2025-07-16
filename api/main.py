from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vectorstore.core import (
    add_document, search_vectors, get_document, list_documents, delete_document, count_documents, embedding_model_name
)

app = FastAPI()

@app.on_event("startup")
def ensure_index():
    from vectorstore.core import _update_index
    _update_index()

class DocUpload(BaseModel):
    doc_id: str
    text: str

class BulkDocUpload(BaseModel):
    docs: list[DocUpload]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/upload")
def upload_doc(payload: DocUpload):
    add_document(payload.doc_id, payload.text)
    return {"status": "ok"}

@app.post("/upload/bulk")
def upload_bulk(payload: BulkDocUpload):
    for doc in payload.docs:
        add_document(doc.doc_id, doc.text)
    return {"status": "ok", "count": len(payload.docs)}

@app.get("/document/{doc_id}")
def get_doc(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.get("/documents")
def list_docs(skip: int = 0, limit: int = 100):
    return {"documents": list_documents(skip=skip, limit=limit)}

@app.delete("/document/{doc_id}")
def delete_doc(doc_id: str):
    if not delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}

@app.post("/search")
def search(req: SearchRequest):
    results = search_vectors(req.query, req.top_k)
    return {"matches": results}

@app.get("/status")
def status():
    return {
        "status": "ok",
        "document_count": count_documents(),
        "embedding_model": embedding_model_name()
    }
