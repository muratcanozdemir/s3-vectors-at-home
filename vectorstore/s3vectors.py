"""The Amazon S3 Vectors API, on top of VectorStore.

Implements the operations that make up S3 Vectors' data model -- vector
buckets, indexes, and vectors (Put/Get/List/Delete/Query) -- with the
request and response shapes of botocore's `s3vectors` service model, so
the AWS SDKs work against it unchanged (see api/s3vectors.py for the HTTP
side, and README for what's not supported).

Storage, all in the one MinIO bucket:
    s3vectors/buckets/<vector bucket>.json     registry entry
    s3vectors/<vector bucket>/indexes/...      a VectorStore namespace per bucket
    s3vectors/<vector bucket>/segments/...

Where AWS's documentation leaves behaviour open, this picks one and says
so: cosine distance is 1 - cosine similarity and euclidean distance is the
plain (not squared) distance, results sorted closest first; GetVectors
omits missing keys and DeleteVectors ignores them; a filter that refers to
a non-filterable metadata key is a ValidationException rather than
silently matching nothing; deleting a vector bucket that still has indexes
is a ConflictException.
"""

import base64
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vectorstore import core
from vectorstore.filters import (
    _is_scalar,
    _require_unicode,
    count_constraints,
    filter_fields,
    validate_filter,
)

ROOT = "s3vectors/"
# "_" can't occur in a bucket name, so no bucket's namespace (ROOT + name)
# can overlap the registry.
REGISTRY_PREFIX = ROOT + "_buckets/"
# Bucket and index names: 3-63 of lowercase letters, digits, '-' and '.',
# starting and ending with a letter or digit (the ARN pattern in the model).
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_ARN_RE = re.compile(
    r"^arn:aws[-a-z0-9]*:s3vectors:[a-z0-9-]+:[0-9]{12}:bucket/([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])"
    r"(?:/index/([a-z0-9][a-z0-9.-]{1,61}[a-z0-9]))?$"
)

# Limits from the service model and AWS's S3 Vectors limits page.
MAX_DIMENSION = 4096
MAX_PUT_VECTORS = 500
MAX_GET_VECTORS = 100
MAX_DELETE_VECTORS = 500
MAX_LIST_VECTORS = 1000
DEFAULT_LIST_VECTORS = 500
MAX_LIST_SEGMENTS = 16
MAX_LIST_RESULTS = 500  # ListVectorBuckets / ListIndexes
MAX_TOP_K = 10_000
QUERY_PAGE_SIZE = 100
MAX_KEY_LENGTH = 1024
MAX_METADATA_KEYS = 50
MAX_METADATA_KEY_LENGTH = 63
MAX_METADATA_BYTES = 100 * 1024
MAX_FILTERABLE_METADATA_BYTES = 40 * 1024
MAX_FILTERABLE_STRING_BYTES = 2 * 1024
MAX_NON_FILTERABLE_KEYS = 10
MAX_FILTER_CONSTRAINTS = 100
DISTANCE_METRICS = ("euclidean", "cosine")


class S3VectorsError(Exception):
    """An error as the API reports it: an error code, a message, an HTTP status."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _invalid(message: str) -> S3VectorsError:
    return S3VectorsError("ValidationException", message, 400)


def _not_found(message: str) -> S3VectorsError:
    return S3VectorsError("NotFoundException", message, 404)


def _conflict(message: str) -> S3VectorsError:
    return S3VectorsError("ConflictException", message, 409)


# -- request field helpers --------------------------------------------------------


def _str(req: dict, field: str, lo: int, hi: int, required: bool = False) -> str | None:
    v = req.get(field)
    if v is None:
        if required:
            raise _invalid(f"{field} is required")
        return None
    if not isinstance(v, str) or not lo <= len(v) <= hi:
        raise _invalid(f"{field} must be a string of {lo} to {hi} characters")
    return v


def _int(req: dict, field: str, lo: int, hi: int, default: int | None = None) -> int | None:
    v = req.get(field)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _invalid(f"{field} must be an integer from {lo} to {hi}")
    return v


def _bool(req: dict, field: str) -> bool:
    v = req.get(field, False)
    if not isinstance(v, bool):
        raise _invalid(f"{field} must be a boolean")
    return v


def _name(value: str, what: str) -> str:
    if not NAME_RE.match(value):
        raise _invalid(
            f"{what} must be 3-63 characters of lowercase letters, digits, '-' and '.', "
            "starting and ending with a letter or digit"
        )
    return value


def _vector(data, field: str) -> list:
    if not isinstance(data, dict) or not isinstance(data.get("float32"), list) or not data["float32"]:
        raise _invalid(f"{field} must be {{'float32': [numbers]}}")
    values = data["float32"]
    if len(values) > MAX_DIMENSION:
        raise _invalid(f"{field} has more than {MAX_DIMENSION} values")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
        raise _invalid(f"{field}.float32 must contain only numbers")
    if not all(math.isfinite(x) for x in values):
        raise _invalid(f"{field} must be finite (no NaN or infinity)")
    return values


def _encode_token(payload: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _decode_token(token: str | None) -> dict | None:
    if token is None:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()))
    except ValueError:
        raise _invalid("nextToken is not valid") from None
    if not isinstance(payload, dict):
        raise _invalid("nextToken is not valid")
    return payload


def validate_metadata(metadata, non_filterable: set[str]) -> dict:
    """Check vector metadata against S3 Vectors' rules; returns a clean copy.

    Values: strings, finite numbers, booleans, or lists of those. Up to 50
    keys of at most 63 characters, 100 KB in all; the filterable part (keys
    not declared non-filterable) at most 40 KB, each string in it at most
    2 KB.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise _invalid("metadata must be a JSON object")
    if len(metadata) > MAX_METADATA_KEYS:
        raise _invalid(f"metadata may have at most {MAX_METADATA_KEYS} keys")
    clean, filterable = {}, {}
    for key, value in metadata.items():
        if not key or len(key) > MAX_METADATA_KEY_LENGTH:
            raise _invalid(f"metadata keys must be 1 to {MAX_METADATA_KEY_LENGTH} characters")
        if key.startswith("$"):
            raise _invalid(f"metadata key {key!r} must not start with '$'")
        try:
            _require_unicode(key, value)
        except ValueError as e:
            raise _invalid(str(e)) from None
        if isinstance(value, list):
            if not all(_is_scalar(v) for v in value):
                raise _invalid(f"metadata list {key!r} may only contain strings, finite numbers and booleans")
            value = list(value)
        elif not _is_scalar(value):
            raise _invalid(f"metadata value for {key!r} must be a string, finite number, boolean, or a list of those")
        clean[key] = value
        if key not in non_filterable:
            filterable[key] = value
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, str) and len(v.encode()) > MAX_FILTERABLE_STRING_BYTES:
                    raise _invalid(f"filterable metadata string in {key!r} exceeds {MAX_FILTERABLE_STRING_BYTES} bytes")
    if len(json.dumps(clean).encode()) > MAX_METADATA_BYTES:
        raise _invalid(f"metadata exceeds {MAX_METADATA_BYTES} bytes")
    if len(json.dumps(filterable).encode()) > MAX_FILTERABLE_METADATA_BYTES:
        raise _invalid(f"filterable metadata exceeds {MAX_FILTERABLE_METADATA_BYTES} bytes")
    return clean


def validate_query_filter(flt, non_filterable: set[str]) -> dict:
    if not isinstance(flt, dict) or not flt:
        raise _invalid("filter must be a non-empty JSON object")
    try:
        validate_filter(flt)
    except ValueError as e:
        raise _invalid(f"invalid filter: {e}") from None
    _require_non_empty_lists(flt)
    if count_constraints(flt) > MAX_FILTER_CONSTRAINTS:
        raise _invalid(f"filter has more than {MAX_FILTER_CONSTRAINTS} constraints")
    bad = filter_fields(flt) & non_filterable
    if bad:
        raise _invalid(f"filter refers to non-filterable metadata keys: {sorted(bad)}")
    return flt


def _require_non_empty_lists(flt: dict):
    for key, cond in flt.items():
        if key in ("$and", "$or"):
            for sub in cond:
                _require_non_empty_lists(sub)
        elif isinstance(cond, dict):
            for op, operand in cond.items():
                if op in ("$in", "$nin") and not operand:
                    raise _invalid(f"{op} on {key!r} takes a non-empty array")


class S3VectorsService:
    """The S3 Vectors operations. Each method takes the request body (a dict,
    as the SDK sends it) and returns the response body; errors are raised as
    S3VectorsError."""

    def __init__(
        self,
        client,
        bucket: str = core.BUCKET,
        region: str | None = None,
        account_id: str | None = None,
        compaction: str = "background",
    ):
        self.region = region or os.environ.get("S3VAH_REGION", "us-east-1")
        self.account_id = account_id or os.environ.get("S3VAH_ACCOUNT_ID", "000000000000")
        self._client = client
        self._bucket = bucket
        self._compaction = compaction
        # One set of threads for every vector bucket's store, however many.
        self._io = ThreadPoolExecutor(max_workers=core.DEFAULT_IO_WORKERS, thread_name_prefix="s3vectors-io")
        self._compactor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3vectors-compact")
        self._stores: dict[str, core.VectorStore] = {}
        self._stores_lock = threading.Lock()
        # Holds no indexes; used for the registry objects, and runs the one
        # startup check that the object store enforces conditional writes.
        self._registry = self._new_store("_registry", verify=True)

    def _new_store(self, vector_bucket: str, verify: bool = False) -> core.VectorStore:
        return core.VectorStore(
            self._client,
            None,
            bucket=self._bucket,
            compaction=self._compaction,
            prefix=f"{ROOT}{vector_bucket}/",
            default_index=None,
            index_name_re=NAME_RE,
            raw_vectors=True,
            io_pool=self._io,
            compactor=self._compactor,
            verify_conditional_writes=verify,
        )

    def _store(self, vector_bucket: str) -> core.VectorStore:
        with self._stores_lock:
            store = self._stores.get(vector_bucket)
            if store is None:
                store = self._stores[vector_bucket] = self._new_store(vector_bucket)
            return store

    # -- ARNs and request targets ----------------------------------------------

    def bucket_arn(self, name: str) -> str:
        return f"arn:aws:s3vectors:{self.region}:{self.account_id}:bucket/{name}"

    def index_arn(self, bucket: str, index: str) -> str:
        return f"{self.bucket_arn(bucket)}/index/{index}"

    def _target_bucket(self, req: dict) -> str:
        name = _str(req, "vectorBucketName", 3, 63)
        arn = req.get("vectorBucketArn")
        if arn is not None:
            m = _ARN_RE.match(arn) if isinstance(arn, str) else None
            if not m or m.group(2):
                raise _invalid("vectorBucketArn is not a vector bucket ARN")
            if name is not None and name != m.group(1):
                raise _invalid("vectorBucketName and vectorBucketArn name different buckets")
            name = m.group(1)
        if name is None:
            raise _invalid("specify vectorBucketName or vectorBucketArn")
        return _name(name, "vectorBucketName")

    def _target_index(self, req: dict) -> tuple[str, str]:
        arn = req.get("indexArn")
        if arn is not None:
            m = _ARN_RE.match(arn) if isinstance(arn, str) else None
            if not m or not m.group(2):
                raise _invalid("indexArn is not a vector index ARN")
            if req.get("vectorBucketName") not in (None, m.group(1)) or req.get("indexName") not in (None, m.group(2)):
                raise _invalid("indexArn and the names given refer to different indexes")
            return m.group(1), m.group(2)
        bucket = _str(req, "vectorBucketName", 3, 63)
        index = _str(req, "indexName", 3, 63)
        if bucket is None or index is None:
            raise _invalid("specify indexArn, or vectorBucketName and indexName")
        return _name(bucket, "vectorBucketName"), _name(index, "indexName")

    # -- the registry of vector buckets -----------------------------------------

    def _registry_key(self, name: str) -> str:
        return f"{REGISTRY_PREFIX}{name}.json"

    def _read_bucket(self, name: str) -> tuple[dict, str] | None:
        """(entry, etag) of an existing, not-being-deleted vector bucket, else None."""
        try:
            blob, etag = self._registry._get_with_etag(self._registry_key(name))
        except core.ObjectNotFoundError:
            return None
        entry = json.loads(blob)
        return (entry, etag) if entry.get("state") == "active" else None

    def _require_bucket(self, name: str) -> dict:
        got = self._read_bucket(name)
        if got is None:
            raise _not_found(f"vector bucket {name!r} not found")
        return got[0]

    def _index_store(self, req: dict) -> tuple[str, str, core.VectorStore]:
        bucket, index = self._target_index(req)
        self._require_bucket(bucket)
        return bucket, index, self._store(bucket)

    # -- vector buckets ---------------------------------------------------------

    def create_vector_bucket(self, req: dict) -> dict:
        name = _name(_str(req, "vectorBucketName", 3, 63, required=True), "vectorBucketName")
        encryption = self._encryption(req.get("encryptionConfiguration"))
        entry = {"name": name, "state": "active", "created": time.time(), "encryption": encryption}
        try:
            self._registry._put_conditional(self._registry_key(name), json.dumps(entry).encode(), if_match=None)
        except core._WriteConflict:
            raise _conflict(f"vector bucket {name!r} already exists") from None
        return {"vectorBucketArn": self.bucket_arn(name)}

    @staticmethod
    def _encryption(cfg) -> dict:
        if cfg is None:
            return {"sseType": "AES256"}
        if not isinstance(cfg, dict) or cfg.get("sseType", "AES256") != "AES256":
            raise _invalid("only sseType AES256 is supported (aws:kms is not)")
        return {"sseType": "AES256"}

    def get_vector_bucket(self, req: dict) -> dict:
        name = self._target_bucket(req)
        entry = self._require_bucket(name)
        return {
            "vectorBucket": {
                "vectorBucketName": name,
                "vectorBucketArn": self.bucket_arn(name),
                "creationTime": entry["created"],
                "encryptionConfiguration": entry.get("encryption", {"sseType": "AES256"}),
            }
        }

    def list_vector_buckets(self, req: dict) -> dict:
        limit = _int(req, "maxResults", 1, MAX_LIST_RESULTS, MAX_LIST_RESULTS)
        prefix = _str(req, "prefix", 1, 63) or ""
        after = (_decode_token(_str(req, "nextToken", 1, 512)) or {}).get("after")
        names = sorted(
            o.object_name[len(REGISTRY_PREFIX): -len(".json")]
            for o in self._client.list_objects(self._bucket, prefix=REGISTRY_PREFIX)
            if o.object_name.endswith(".json")
        )
        names = [n for n in names if n.startswith(prefix) and (after is None or n > after)]
        page, more = names[:limit], len(names) > limit
        entries = list(self._io.map(self._read_bucket, page))
        out = {
            "vectorBuckets": [
                {"vectorBucketName": n, "vectorBucketArn": self.bucket_arn(n), "creationTime": got[0]["created"]}
                for n, got in zip(page, entries)
                if got is not None
            ]
        }
        if more:
            out["nextToken"] = _encode_token({"after": page[-1]})
        return out

    def delete_vector_bucket(self, req: dict) -> dict:
        """Delete an empty vector bucket.

        Race-free against CreateIndex: the entry is first marked "deleting"
        (CAS), then checked for indexes. CreateIndex re-checks the entry
        after creating its index and undoes it if the bucket is being
        deleted, so an index can't slip into a bucket as it goes.
        """
        name = self._target_bucket(req)
        key = self._registry_key(name)
        for _ in range(core.INDEX_WRITE_ATTEMPTS):
            got = self._read_bucket(name)
            if got is None:
                raise _not_found(f"vector bucket {name!r} not found")
            entry, etag = got
            try:
                etag = self._registry._put_conditional(key, json.dumps({**entry, "state": "deleting"}).encode(), etag)
            except core._WriteConflict:
                continue
            break
        else:
            raise S3VectorsError("ServiceUnavailableException", "vector bucket kept changing; retry", 503)
        if self._store(name).list_indexes():
            try:
                self._registry._put_conditional(key, json.dumps(entry).encode(), etag)
            except core._WriteConflict:
                pass  # can't happen: only this call writes a "deleting" entry
            raise _conflict(f"vector bucket {name!r} is not empty: delete its indexes first")
        self._registry._delete_object(key)
        return {}

    # -- indexes ----------------------------------------------------------------

    def create_index(self, req: dict) -> dict:
        bucket = self._target_bucket(req)
        index = _name(_str(req, "indexName", 3, 63, required=True), "indexName")
        if req.get("dataType") != "float32":
            raise _invalid("dataType must be float32")
        dim = _int(req, "dimension", 1, MAX_DIMENSION)
        if dim is None:
            raise _invalid("dimension is required")
        metric = req.get("distanceMetric")
        if metric not in DISTANCE_METRICS:
            raise _invalid(f"distanceMetric must be one of {list(DISTANCE_METRICS)}")
        mcfg = req.get("metadataConfiguration") or {}
        if not isinstance(mcfg, dict):
            raise _invalid("metadataConfiguration must be an object")
        if mcfg.get("textSearchConfigurations"):
            raise _invalid("full-text search (textSearchConfigurations) is not supported")
        non_filterable = mcfg.get("nonFilterableMetadataKeys") or []
        if (
            not isinstance(non_filterable, list)
            or len(non_filterable) > MAX_NON_FILTERABLE_KEYS
            or not all(isinstance(k, str) and 1 <= len(k) <= MAX_METADATA_KEY_LENGTH for k in non_filterable)
        ):
            raise _invalid(f"nonFilterableMetadataKeys takes up to {MAX_NON_FILTERABLE_KEYS} keys of 1-63 characters")
        config = {
            "nonFilterableMetadataKeys": sorted(set(non_filterable)),
            "encryption": self._encryption(req.get("encryptionConfiguration")),
        }
        self._require_bucket(bucket)
        store = self._store(bucket)
        try:
            store.create_index(index, dim, metric, config)
        except core.IndexAlreadyExistsError:
            raise _conflict(f"index {index!r} already exists in {bucket!r}") from None
        if self._read_bucket(bucket) is None:  # the bucket is being (or was) deleted: undo
            try:
                store.delete_index(index)
            except core.IndexNotFoundError:
                pass
            raise _not_found(f"vector bucket {bucket!r} not found")
        return {"indexArn": self.index_arn(bucket, index)}

    def _index_info(self, store: core.VectorStore, bucket: str, index: str) -> dict:
        try:
            return store.index_info(index)
        except core.IndexNotFoundError:
            raise _not_found(f"index {index!r} not found in {bucket!r}") from None

    def _index_summary(self, bucket: str, index: str, info: dict) -> dict:
        return {
            "vectorBucketName": bucket,
            "indexName": index,
            "indexArn": self.index_arn(bucket, index),
            "creationTime": info["created"],
        }

    def get_index(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        info = self._index_info(store, bucket, index)
        out = {
            **self._index_summary(bucket, index, info),
            "dataType": "float32",
            "dimension": info["dim"],
            "distanceMetric": info["metric"],
            "encryptionConfiguration": info["config"].get("encryption", {"sseType": "AES256"}),
        }
        if info["config"].get("nonFilterableMetadataKeys"):
            out["metadataConfiguration"] = {"nonFilterableMetadataKeys": info["config"]["nonFilterableMetadataKeys"]}
        return {"index": out}

    def list_indexes(self, req: dict) -> dict:
        bucket = self._target_bucket(req)
        self._require_bucket(bucket)
        limit = _int(req, "maxResults", 1, MAX_LIST_RESULTS, MAX_LIST_RESULTS)
        prefix = _str(req, "prefix", 1, 63) or ""
        after = (_decode_token(_str(req, "nextToken", 1, 512)) or {}).get("after")
        store = self._store(bucket)
        names = [n for n in store.list_indexes() if n.startswith(prefix) and (after is None or n > after)]
        page, more = names[:limit], len(names) > limit

        def info(name):
            try:
                return store.index_info(name)
            except core.IndexNotFoundError:
                return None  # deleted meanwhile

        out = {
            "indexes": [
                self._index_summary(bucket, n, i) for n, i in zip(page, self._io.map(info, page)) if i is not None
            ]
        }
        if more:
            out["nextToken"] = _encode_token({"after": page[-1]})
        return out

    def delete_index(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        try:
            store.delete_index(index)
        except core.IndexNotFoundError:
            raise _not_found(f"index {index!r} not found in {bucket!r}") from None
        return {}

    # -- vectors -----------------------------------------------------------------

    def _non_filterable(self, store, bucket, index) -> set[str]:
        return set(self._index_info(store, bucket, index)["config"].get("nonFilterableMetadataKeys", []))

    def put_vectors(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        vectors = req.get("vectors")
        if not isinstance(vectors, list) or not 1 <= len(vectors) <= MAX_PUT_VECTORS:
            raise _invalid(f"vectors must be a list of 1 to {MAX_PUT_VECTORS} vectors")
        non_filterable = self._non_filterable(store, bucket, index)
        items = []
        for i, v in enumerate(vectors):
            if not isinstance(v, dict):
                raise _invalid(f"vectors[{i}] must be an object")
            key = _str(v, "key", 1, MAX_KEY_LENGTH, required=True)
            data = _vector(v.get("data"), f"vectors[{i}].data")
            items.append((key, np.asarray(data, dtype=np.float32), validate_metadata(v.get("metadata"), non_filterable)))
        self._call(store.put_vectors, bucket, index, index, items)
        return {}

    def get_vectors(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        keys = self._keys(req, MAX_GET_VECTORS)
        return_data, return_meta = _bool(req, "returnData"), _bool(req, "returnMetadata")  # checked even if nothing matches
        found = self._call(store.get_vectors, bucket, index, index, keys)
        return {"vectors": [self._out(v, return_data, return_meta) for v in found]}

    def list_vectors(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        limit = _int(req, "maxResults", 1, MAX_LIST_VECTORS, DEFAULT_LIST_VECTORS)
        count = _int(req, "segmentCount", 1, MAX_LIST_SEGMENTS)
        part = _int(req, "segmentIndex", 0, MAX_LIST_SEGMENTS - 1)
        if (count is None) != (part is None):
            raise _invalid("segmentCount and segmentIndex must be given together")
        segment = (count or 1, part or 0)
        if segment[1] >= segment[0]:
            raise _invalid("segmentIndex must be less than segmentCount")
        token = _decode_token(_str(req, "nextToken", 1, 2048))
        if token is not None and token.get("seg") != list(segment):
            raise _invalid("nextToken belongs to a different segment")
        after = token.get("after") if token else None
        return_data, return_meta = _bool(req, "returnData"), _bool(req, "returnMetadata")
        page, last = self._call(store.list_vectors, bucket, index, index, after, limit, segment)
        out = {"vectors": [self._out(v, return_data, return_meta) for v in page]}
        if last is not None:
            out["nextToken"] = _encode_token({"after": last, "seg": list(segment)})
        return out

    def delete_vectors(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        self._call(store.delete_vectors, bucket, index, index, self._keys(req, MAX_DELETE_VECTORS))
        return {}

    def query_vectors(self, req: dict) -> dict:
        bucket, index, store = self._index_store(req)
        if req.get("queryTexts"):
            raise _invalid("full-text and hybrid search (queryTexts) are not supported; use queryVector")
        if req.get("queryVector") is None:
            raise _invalid("queryVector is required")
        query = _vector(req["queryVector"], "queryVector")
        return_meta, return_distance = _bool(req, "returnMetadata"), _bool(req, "returnDistance")
        _bool(req, "returnScores")
        top_k = _int(req, "topK", 1, MAX_TOP_K)
        if top_k is None:
            raise _invalid("topK is required")
        if req.get("queryMode") not in (None, "CLASSIC", "ENHANCED"):
            raise _invalid("queryMode must be CLASSIC or ENHANCED")
        flt = req.get("filter")
        if flt is not None:
            flt = validate_query_filter(flt, self._non_filterable(store, bucket, index))
        # Pages of QUERY_PAGE_SIZE: a token carries the offset and a digest of
        # the query, so it can't be replayed against a different one.
        digest = hashlib.sha256(
            json.dumps([bucket, index, query, top_k, flt], sort_keys=True).encode()
        ).hexdigest()[:16]
        token = _decode_token(_str(req, "nextToken", 1, 4096))
        offset = 0
        if token is not None:
            if token.get("q") != digest or not isinstance(token.get("offset"), int):
                raise _invalid("nextToken belongs to a different query")
            offset = token["offset"]
        metric, hits = self._call(store.query_vectors, bucket, index, index, query, top_k, flt)
        page = hits[offset: offset + QUERY_PAGE_SIZE]
        vectors = []
        for h in page:
            v = {"key": h["key"]}
            if return_distance:
                v["distance"] = h["distance"]
            if return_meta:
                v["metadata"] = h["metadata"]
            vectors.append(v)
        out = {"vectors": vectors, "distanceMetric": metric}
        if offset + QUERY_PAGE_SIZE < len(hits):
            out["nextToken"] = _encode_token({"q": digest, "offset": offset + QUERY_PAGE_SIZE})
        return out

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _keys(req: dict, most: int) -> list[str]:
        keys = req.get("keys")
        if not isinstance(keys, list) or not 1 <= len(keys) <= most:
            raise _invalid(f"keys must be a list of 1 to {most} keys")
        if not all(isinstance(k, str) and 1 <= len(k) <= MAX_KEY_LENGTH for k in keys):
            raise _invalid(f"each key must be a string of 1 to {MAX_KEY_LENGTH} characters")
        return keys

    @staticmethod
    def _out(v: dict, return_data: bool, return_meta: bool) -> dict:
        out = {"key": v["key"]}
        if return_data:
            out["data"] = {"float32": v["vector"].tolist()}
        if return_meta:
            out["metadata"] = v["metadata"]
        return out

    @staticmethod
    def _call(fn, bucket, index, *args):
        """Run a store operation, translating its errors to the API's."""
        try:
            return fn(*args)
        except core.IndexNotFoundError:
            raise _not_found(f"index {index!r} not found in {bucket!r}") from None
        except core.IndexWriteConflictError as e:
            raise S3VectorsError("ServiceUnavailableException", str(e), 503) from None
        except core.IndexCorruptError as e:
            raise S3VectorsError("InternalServerException", str(e), 500) from None
        except ValueError as e:
            raise _invalid(str(e)) from None
