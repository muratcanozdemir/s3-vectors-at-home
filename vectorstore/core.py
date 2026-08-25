import hashlib
import io
import json
import os
import threading

import faiss
import numpy as np
from minio.error import S3Error

from minio import Minio

MODEL_NAME = "all-MiniLM-L6-v2"
BUCKET = "vectors"
INDEX_OBJ = "faiss.index"
IDS_OBJ = "index.ids.json"
RESERVED_OBJECT_NAMES = {INDEX_OBJ, IDS_OBJ}
MAX_DOC_ID_LENGTH = 512


class ObjectNotFoundError(Exception):
    """Raised when a requested object does not exist in the backing store."""


def object_key_id(doc_id: str) -> int:
    """Deterministic id for a doc_id, used as its FAISS vector id.

    Derived from a hash rather than insertion order so the FAISS index and
    the doc_id it maps to stay tied together even after removals/updates --
    the previous design tracked ids as a plain list kept in lockstep with
    FAISS insertion order, which silently drifted whenever an add and an
    index update weren't perfectly paired (see exploration.md bug #4).
    """
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def validate_doc_id(doc_id: str) -> None:
    if not doc_id:
        raise ValueError("doc_id must not be empty")
    if "/" in doc_id:
        raise ValueError("doc_id must not contain '/'")
    if doc_id in RESERVED_OBJECT_NAMES:
        raise ValueError(f"doc_id {doc_id!r} is reserved")
    if len(doc_id) > MAX_DOC_ID_LENGTH:
        raise ValueError(f"doc_id must be at most {MAX_DOC_ID_LENGTH} characters")


class VectorStore:
    """Text + vector storage backed by an object store, indexed with FAISS.

    Takes its object-store client and embedder as constructor arguments
    instead of reaching for module-level globals, so it can be built once
    (lazily, from env vars -- see `get_store`) for the real app, or built
    directly with fakes in tests with no network/model download involved.

    The FAISS index is a single `IndexIDMap2`, with ids derived from doc_id
    (`object_key_id`). That lets add/update/delete mutate the index in place
    via `add_with_ids`/`remove_ids` -- no full rebuild needed except to
    bootstrap a missing index from the individually-stored vectors.
    """

    def __init__(self, client: Minio, embedder, bucket: str = BUCKET):
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._ensure_bucket()

    # -- object store helpers -------------------------------------------

    def _ensure_bucket(self):
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def _get_bytes(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(self._bucket, key)
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectNotFoundError(key) from e
            raise
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def _put_bytes(self, key: str, data: bytes):
        self._client.put_object(self._bucket, key, io.BytesIO(data), length=len(data))

    def _delete_object(self, key: str):
        try:
            self._client.remove_object(self._bucket, key)
        except S3Error as e:
            if e.code != "NoSuchKey":
                raise
        except ObjectNotFoundError:
            pass

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embedder.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]

    # -- index persistence ------------------------------------------------
    # faiss.write_index/read_index require a path/FILE*/IOWriter, not a
    # plain file-like object -- serialize_index/deserialize_index are the
    # supported way to round-trip an index through an in-memory buffer.

    def _load_index(self):
        """Returns (IndexIDMap2 or None, {int_id: doc_id})."""
        try:
            idx_bytes = self._get_bytes(INDEX_OBJ)
            ids_bytes = self._get_bytes(IDS_OBJ)
        except ObjectNotFoundError:
            return None, {}
        index = faiss.deserialize_index(np.frombuffer(idx_bytes, dtype=np.uint8))
        id_map = {int(k): v for k, v in json.loads(ids_bytes).items()}
        return index, id_map

    def _save_index(self, index, id_map: dict):
        self._put_bytes(INDEX_OBJ, faiss.serialize_index(index).tobytes())
        self._put_bytes(IDS_OBJ, json.dumps({str(k): v for k, v in id_map.items()}).encode())

    def _rebuild_index_from_scratch(self):
        """Bootstrap the index from the individually-stored vectors.

        Only used when no index exists yet (first write, or a previous
        index/ids object went missing) -- ordinary add/update/delete mutate
        the existing index directly instead of paying for this.
        """
        vectors, doc_ids = [], []
        for obj in self._client.list_objects(self._bucket):
            if obj.object_name.endswith(".npy"):
                doc_id = obj.object_name.removesuffix(".npy")
                vec = np.frombuffer(self._get_bytes(obj.object_name), dtype=np.float32)
                vectors.append(vec)
                doc_ids.append(doc_id)
        if not vectors:
            return None, {}
        dim = vectors[0].shape[0]
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        keys = np.array([object_key_id(d) for d in doc_ids], dtype="int64")
        index.add_with_ids(np.vstack(vectors), keys)
        id_map = dict(zip((int(k) for k in keys), doc_ids))
        self._save_index(index, id_map)
        return index, id_map

    def _ensure_index(self):
        index, id_map = self._load_index()
        if index is None:
            index, id_map = self._rebuild_index_from_scratch()
        return index, id_map

    # -- public API ---------------------------------------------------------

    def add_document(self, doc_id: str, text: str):
        """Add a new document, or replace the vector+text of an existing one."""
        validate_doc_id(doc_id)
        vec = self._embed(text)
        self._put_bytes(f"{doc_id}.npy", vec.tobytes())
        self._put_bytes(f"{doc_id}.meta.json", json.dumps({"doc_id": doc_id, "text": text}).encode())
        with self._lock:
            index, id_map = self._ensure_index()
            key = object_key_id(doc_id)
            if index is None:
                index = faiss.IndexIDMap2(faiss.IndexFlatL2(vec.shape[0]))
            elif key in id_map:
                index.remove_ids(np.array([key], dtype="int64"))
            index.add_with_ids(np.expand_dims(vec, axis=0), np.array([key], dtype="int64"))
            id_map[key] = doc_id
            self._save_index(index, id_map)

    def get_document(self, doc_id: str):
        try:
            return json.loads(self._get_bytes(f"{doc_id}.meta.json"))
        except ObjectNotFoundError:
            return None

    def list_documents(self, skip: int = 0, limit: int = 100):
        doc_ids = sorted(
            obj.object_name.removesuffix(".meta.json")
            for obj in self._client.list_objects(self._bucket)
            if obj.object_name.endswith(".meta.json")
        )
        docs = []
        for doc_id in doc_ids[skip: skip + limit]:
            meta = self.get_document(doc_id)
            if meta:
                docs.append({"doc_id": doc_id, "text_preview": meta.get("text", "")[:64]})
        return docs

    def delete_document(self, doc_id: str) -> bool:
        if self.get_document(doc_id) is None:
            return False
        self._delete_object(f"{doc_id}.npy")
        self._delete_object(f"{doc_id}.meta.json")
        with self._lock:
            index, id_map = self._load_index()
            key = object_key_id(doc_id)
            if index is not None and key in id_map:
                index.remove_ids(np.array([key], dtype="int64"))
                del id_map[key]
                if index.ntotal == 0:
                    self._delete_object(INDEX_OBJ)
                    self._delete_object(IDS_OBJ)
                else:
                    self._save_index(index, id_map)
        return True

    def search(self, query: str, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            index, id_map = self._ensure_index()
        if index is None:
            return []
        qv = self._embed(query)
        _, result_ids = index.search(np.expand_dims(qv, axis=0), top_k)
        # FAISS pads short result sets with sentinel id -1; must be excluded
        # explicitly or it aliases to the last id_map entry looked up.
        return [id_map[int(i)] for i in result_ids[0] if i != -1]

    def count_documents(self) -> int:
        return sum(1 for obj in self._client.list_objects(self._bucket) if obj.object_name.endswith(".meta.json"))

    def embedding_model_name(self) -> str:
        return MODEL_NAME


_store: VectorStore | None = None
_store_init_lock = threading.Lock()


def get_store() -> VectorStore:
    """Lazily build the process-wide VectorStore from env vars.

    Deferred to first use (rather than constructed at import time) so
    importing this module never requires a live MinIO connection or a
    model download -- both the app and tests can import it freely and only
    pay that cost when a store is actually needed.
    """
    global _store
    if _store is None:
        with _store_init_lock:
            if _store is None:
                client = Minio(
                    os.environ["MINIO_ENDPOINT"],
                    access_key=os.environ["MINIO_ACCESS_KEY"],
                    secret_key=os.environ["MINIO_SECRET_KEY"],
                    secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
                )
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer(MODEL_NAME, device="cpu")
                _store = VectorStore(client, embedder)
    return _store
