"""In-memory test doubles for VectorStore's collaborators.

Neither of these talks to the network or loads a real model, which is what
makes the golden tests in test_vectorstore_golden.py fast and deterministic:
they exercise VectorStore's own bookkeeping (index maintenance, pagination,
validation) in isolation from MinIO and the real embedding model.
"""

import hashlib
import threading
from collections import namedtuple

import numpy as np
from minio.error import S3Error

_Object = namedtuple("Object", ["object_name"])
_Stat = namedtuple("Stat", ["object_name", "etag"])
_WriteResult = namedtuple("WriteResult", ["object_name", "etag"])


def _etag(data: bytes) -> str:
    # Matches what MinIO/S3 use for single-part uploads: the hex MD5.
    return hashlib.md5(data).hexdigest()


class FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.closed = False
        self.headers = {"ETag": f'"{_etag(data)}"'}

    def read(self):
        return self._data

    def close(self):
        self.closed = True

    def release_conn(self):
        pass


def _s3_error(code: str, bucket: str, key: str) -> S3Error:
    return S3Error(
        response=None,
        code=code,
        message=code,
        resource=f"/{bucket}/{key}",
        request_id="fake-request-id",
        host_id="fake-host-id",
    )


def _not_found(bucket: str, key: str) -> S3Error:
    return _s3_error("NoSuchKey", bucket, key)


class FakeMinioClient:
    """Implements the small subset of the Minio client surface VectorStore uses.

    `_put_object` enforces If-Match / If-None-Match atomically, with the
    error codes MinIO returns (checked against a real server: 412
    PreconditionFailed, and NoSuchKey for If-Match on a missing key).
    `enforce_conditions=False` imitates a server that ignores them.
    """

    def __init__(self, enforce_conditions: bool = True):
        self.buckets = set()
        self.objects: dict[str, dict[str, bytes]] = {}
        self.get_counts: dict[str, int] = {}
        self.enforce_conditions = enforce_conditions
        self._write_lock = threading.Lock()

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)
        self.objects.setdefault(bucket, {})

    def put_object(self, bucket, key, data, length):
        payload = data.read()
        with self._write_lock:
            self.objects[bucket][key] = payload
        return _WriteResult(key, _etag(payload))

    def _put_object(self, bucket, key, data, headers):
        payload = bytes(data)
        with self._write_lock:
            if self.enforce_conditions:
                current = self.objects[bucket].get(key)
                if_match = headers.get("If-Match")
                if headers.get("If-None-Match") == "*" and current is not None:
                    raise _s3_error("PreconditionFailed", bucket, key)
                if if_match is not None:
                    if current is None:
                        raise _not_found(bucket, key)
                    if if_match.strip('"') != _etag(current):
                        raise _s3_error("PreconditionFailed", bucket, key)
            self.objects[bucket][key] = payload
        return _WriteResult(key, _etag(payload))

    def get_object(self, bucket, key):
        self.get_counts[key] = self.get_counts.get(key, 0) + 1
        try:
            return FakeResponse(self.objects[bucket][key])
        except KeyError:
            raise _not_found(bucket, key)

    def stat_object(self, bucket, key):
        try:
            return _Stat(key, _etag(self.objects[bucket][key]))
        except KeyError:
            raise _not_found(bucket, key)

    def remove_object(self, bucket, key):
        try:
            with self._write_lock:
                del self.objects[bucket][key]
        except KeyError:
            raise _not_found(bucket, key)

    def list_objects(self, bucket, prefix=None):
        return [
            _Object(name)
            for name in sorted(self.objects.get(bucket, {}))
            if prefix is None or name.startswith(prefix)
        ]


class FakeEmbedder:
    """Deterministic stand-in for the embedder (vectorstore.embedding.OnnxEmbedder).

    Maps each distinct text to a fixed, reproducible unit vector (seeded
    from a hash of the text) rather than a real semantic embedding. This is
    enough to golden-test VectorStore's plumbing exactly (search with a
    document's own text as the query is always its closest/only-exact
    match), without needing a real model or non-determinism in assertions.
    """

    dim = 8

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self.dim
