"""HTTP routes for the Amazon S3 Vectors API (see vectorstore/s3vectors.py).

The wire protocol is botocore's for `s3vectors`: rest-json, every operation
a `POST /<OperationName>` with a JSON body; errors carry their code in the
`x-amzn-ErrorType` header and `{"message"}` in the body. Point an SDK at
the server with `endpoint_url`:

    boto3.client("s3vectors", endpoint_url="http://localhost:8000", region_name="us-east-1")

Request signatures are not checked (any credentials work), the same as the
rest of this API: run it on a network you trust.
"""

import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from vectorstore.s3vectors import S3VectorsError, S3VectorsService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["S3 Vectors API"])

_service: S3VectorsService | None = None


def configure(service: S3VectorsService | None) -> None:
    """Use `service` (tests pass one over a fake client); None: build from env."""
    global _service
    _service = service


def get_service() -> S3VectorsService:
    global _service
    if _service is None:
        from vectorstore.core import get_store

        store = get_store()  # same MinIO client and bucket as the rest of the API
        _service = S3VectorsService(store._client, store._bucket)
    return _service


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"message": message}, headers={"x-amzn-ErrorType": code})


OPERATIONS = {
    "CreateVectorBucket": "create_vector_bucket",
    "GetVectorBucket": "get_vector_bucket",
    "ListVectorBuckets": "list_vector_buckets",
    "DeleteVectorBucket": "delete_vector_bucket",
    "CreateIndex": "create_index",
    "GetIndex": "get_index",
    "ListIndexes": "list_indexes",
    "DeleteIndex": "delete_index",
    "PutVectors": "put_vectors",
    "GetVectors": "get_vectors",
    "ListVectors": "list_vectors",
    "DeleteVectors": "delete_vectors",
    "QueryVectors": "query_vectors",
}

# In the service model but not implemented: answered with a clear error
# rather than a 404 the SDK would report as an unknown failure.
UNSUPPORTED = [
    "PutVectorBucketPolicy",
    "GetVectorBucketPolicy",
    "DeleteVectorBucketPolicy",
    "PutVectorBucketDefaultIndexMode",
    "UpdateIndexMode",
    "UpdateMetadataConfiguration",
]


def _route(method_name: str):
    async def handler(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return _error("ValidationException", "request body is not valid JSON", 400)
        if not isinstance(body, dict):
            return _error("ValidationException", "request body must be a JSON object", 400)
        try:
            result = await run_in_threadpool(getattr(get_service(), method_name), body)
        except S3VectorsError as e:
            return _error(e.code, e.message, e.status)
        except Exception:
            logger.exception("S3 Vectors %s failed", method_name)
            return _error("InternalServerException", "internal error", 500)
        return JSONResponse(result)

    return handler


for _op, _method in OPERATIONS.items():
    router.add_api_route(f"/{_op}", _route(_method), methods=["POST"], name=_op, include_in_schema=True)


def _unsupported(op: str):
    async def handler():
        return _error("ValidationException", f"{op} is not supported by this server", 400)

    return handler


for _op in UNSUPPORTED:
    router.add_api_route(f"/{_op}", _unsupported(_op), methods=["POST"], include_in_schema=False)
for _method in ("GET", "POST", "DELETE"):
    router.add_api_route(
        "/tags/{resource_arn:path}", _unsupported("Tagging"), methods=[_method], include_in_schema=False
    )


def enabled() -> bool:
    return os.environ.get("S3VAH_S3VECTORS_API", "true").lower() != "false"
