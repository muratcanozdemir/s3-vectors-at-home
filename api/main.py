from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import s3vectors as _s3vectors
from vectorstore.core import (
    DEFAULT_INDEX,
    MAX_PAGE_SIZE,
    MAX_TOP_K,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    IndexWriteConflictError,
    get_store,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the store (MinIO client + embedding model) and load the default
    # index at startup, so the first request doesn't pay seconds of model
    # loading, and a misconfigured MinIO fails the server at boot.
    get_store().count_documents()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(IndexWriteConflictError)
def index_write_conflict(_request: Request, exc: IndexWriteConflictError):
    # Contention, not a bad request: nothing was half-applied and the same
    # request can simply be sent again.
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "1"})


@app.exception_handler(IndexNotFoundError)
def index_not_found(_request: Request, exc: IndexNotFoundError):
    return JSONResponse(status_code=404, content={"detail": f"Index {exc.args[0]!r} not found"})


@app.exception_handler(IndexAlreadyExistsError)
def index_exists(_request: Request, exc: IndexAlreadyExistsError):
    return JSONResponse(status_code=409, content={"detail": f"Index {exc.args[0]!r} already exists"})


@app.exception_handler(ValueError)
def bad_request(_request: Request, exc: ValueError):
    # The store raises ValueError for invalid input (doc_id, metadata,
    # filter, index name, bounds).
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class DocUpload(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] | None = None


class BulkDocUpload(BaseModel):
    docs: list[DocUpload]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)
    filter: dict[str, Any] | None = Field(
        None, description="Metadata filter, e.g. {\"lang\": \"en\", \"year\": {\"$gte\": 2020}}"
    )


class IndexCreate(BaseModel):
    name: str


# -- documents: the same routes at "/" (default index) and "/indexes/{index}"


def _default_index() -> str:
    return DEFAULT_INDEX


def _path_index(index: str) -> str:
    return index


def _document_routes(router: APIRouter, index_dep):
    """Register the document routes on `router`, taking the index from `index_dep`.

    The original unprefixed routes use the default index; the same routes
    under /indexes/{index} use the one named in the path.
    """
    from_route = Depends(index_dep)

    @router.post("/upload")
    def upload_doc(payload: DocUpload, index: str = from_route):
        get_store().add_document(payload.doc_id, payload.text, payload.metadata, index=index)
        return {"status": "ok"}

    @router.post("/upload/bulk")
    def upload_bulk(payload: BulkDocUpload, index: str = from_route):
        count = get_store().add_documents(((d.doc_id, d.text, d.metadata) for d in payload.docs), index=index)
        return {"status": "ok", "count": count}

    @router.get("/document/{doc_id}")
    def get_doc(doc_id: str, index: str = from_route):
        doc = get_store().get_document(doc_id, index=index)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc

    @router.get("/documents")
    def list_docs(skip: int = Query(0, ge=0), limit: int = Query(100, ge=0, le=MAX_PAGE_SIZE), index: str = from_route):
        return {"documents": get_store().list_documents(skip=skip, limit=limit, index=index)}

    @router.delete("/document/{doc_id}")
    def delete_doc(doc_id: str, index: str = from_route):
        if not get_store().delete_document(doc_id, index=index):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted"}

    @router.post("/search")
    def search(req: SearchRequest, index: str = from_route):
        results = get_store().search_with_scores(req.query, req.top_k, index=index, filter=req.filter)
        # `matches` (bare doc_ids) is kept as-is for existing clients;
        # `results` adds score and metadata per match.
        return {"matches": [r["doc_id"] for r in results], "results": results}

    @router.get("/status")
    def status(index: str = from_route):
        store = get_store()
        stats = store.index_stats(index=index)
        return {
            "status": "ok",
            "index": index,
            "document_count": stats["documents"],
            "segments": stats["segments"],
            "storage_bytes": stats["bytes"],
            "embedding_model": store.embedding_model_name(),
        }

    @router.post("/compact")
    def compact(index: str = from_route):
        # Merges happen automatically after writes; this also reclaims space
        # held by overwritten and deleted documents, and orphaned segments.
        stats = get_store().compact(index=index)
        return {
            "status": "ok",
            "index": index,
            "document_count": stats["documents"],
            "segments": stats["segments"],
            "storage_bytes": stats["bytes"],
        }


_default_routes = APIRouter()
_document_routes(_default_routes, _default_index)
_named_routes = APIRouter(prefix="/indexes/{index}")
_document_routes(_named_routes, _path_index)


# -- indexes


@app.get("/indexes")
def list_indexes():
    return {"indexes": get_store().list_indexes()}


@app.post("/indexes", status_code=201)
def create_index(payload: IndexCreate):
    get_store().create_index(payload.name)
    return {"status": "created", "index": payload.name}


@app.delete("/indexes/{index}")
def delete_index(index: str):
    get_store().delete_index(index)
    return {"status": "deleted", "index": index}


app.include_router(_default_routes)
app.include_router(_named_routes)

# The Amazon S3 Vectors API (POST /CreateIndex, /PutVectors, /QueryVectors, ...),
# for the AWS SDKs. S3VAH_S3VECTORS_API=false turns it off.
if _s3vectors.enabled():
    app.include_router(_s3vectors.router)
