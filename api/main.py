from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vectorstore.core import get_store

app = FastAPI()


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
    try:
        get_store().add_document(payload.doc_id, payload.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.post("/upload/bulk")
def upload_bulk(payload: BulkDocUpload):
    store = get_store()
    try:
        for doc in payload.docs:
            store.add_document(doc.doc_id, doc.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "count": len(payload.docs)}


@app.get("/document/{doc_id}")
def get_doc(doc_id: str):
    doc = get_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/documents")
def list_docs(skip: int = 0, limit: int = 100):
    return {"documents": get_store().list_documents(skip=skip, limit=limit)}


@app.delete("/document/{doc_id}")
def delete_doc(doc_id: str):
    if not get_store().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@app.post("/search")
def search(req: SearchRequest):
    try:
        results = get_store().search(req.query, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"matches": results}


@app.get("/status")
def status():
    store = get_store()
    return {
        "status": "ok",
        "document_count": store.count_documents(),
        "embedding_model": store.embedding_model_name(),
    }
