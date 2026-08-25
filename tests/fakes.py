"""In-memory test doubles for VectorStore's collaborators.

Neither of these talks to the network or loads a real model, which is what
makes the golden tests in test_vectorstore_golden.py fast and deterministic:
they exercise VectorStore's own bookkeeping (index maintenance, pagination,
validation) in isolation from MinIO and the real embedding model.
"""

import hashlib
from collections import namedtuple

import numpy as np
from minio.error import S3Error

_Object = namedtuple("Object", ["object_name"])


class FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.closed = False

    def read(self):
        return self._data

    def close(self):
        self.closed = True

    def release_conn(self):
        pass


def _not_found(bucket: str, key: str) -> S3Error:
    return S3Error(
        response=None,
        code="NoSuchKey",
        message="not found",
        resource=f"/{bucket}/{key}",
        request_id="fake-request-id",
        host_id="fake-host-id",
    )


class FakeMinioClient:
    """Implements the small subset of the Minio client surface VectorStore uses."""

    def __init__(self):
        self.buckets = set()
        self.objects: dict[str, dict[str, bytes]] = {}

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)
        self.objects.setdefault(bucket, {})

    def put_object(self, bucket, key, data, length):
        self.objects[bucket][key] = data.read()

    def get_object(self, bucket, key):
        try:
            return FakeResponse(self.objects[bucket][key])
        except KeyError:
            raise _not_found(bucket, key)

    def remove_object(self, bucket, key):
        try:
            del self.objects[bucket][key]
        except KeyError:
            raise _not_found(bucket, key)

    def list_objects(self, bucket):
        return [_Object(name) for name in sorted(self.objects.get(bucket, {}))]


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.

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
