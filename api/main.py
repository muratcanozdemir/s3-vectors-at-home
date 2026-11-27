from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vectorstore.core import (
    MAX_PAGE_SIZE,
    MAX_TOP_K,
    IndexWriteConflictError,
    get_store,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the store (MinIO client + embedding model) and load the index
    # at startup, so the first request doesn't pay seconds of model loading,
    # and a misconfigured MinIO fails the server at boot, not mid-request.
    get_store().count_documents()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(IndexWriteConflictError)
def index_write_conflict(_request: Request, exc: IndexWriteConflictError):
    # Contention, not a bad request: nothing was half-applied and the same
    # request can simply be sent again.
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "1"})


class DocUpload(BaseModel):
    doc_id: str
    text: str


class BulkDocUpload(BaseModel):
    docs: list[DocUpload]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)


@app.post("/upload")
def upload_doc(payload: DocUpload):
    try:
        get_store().add_document(payload.doc_id, payload.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.post("/upload/bulk")
def upload_bulk(payload: BulkDocUpload):
    try:
        count = get_store().add_documents((doc.doc_id, doc.text) for doc in payload.docs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "count": count}


@app.get("/document/{doc_id}")
def get_doc(doc_id: str):
    doc = get_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/documents")
def list_docs(skip: int = Query(0, ge=0), limit: int = Query(100, ge=0, le=MAX_PAGE_SIZE)):
    return {"documents": get_store().list_documents(skip=skip, limit=limit)}


@app.delete("/document/{doc_id}")
def delete_doc(doc_id: str):
    if not get_store().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@app.post("/search")
def search(req: SearchRequest):
    try:
        results = get_store().search_with_scores(req.query, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # `matches` (bare doc_ids) is kept as-is for existing clients; `results`
    # adds the cosine-similarity score per match.
    return {"matches": [r["doc_id"] for r in results], "results": results}


@app.get("/status")
def status():
    store = get_store()
    return {
        "status": "ok",
        "document_count": store.count_documents(),
        "embedding_model": store.embedding_model_name(),
    }
