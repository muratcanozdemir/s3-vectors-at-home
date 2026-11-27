import hashlib
import io
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from minio.error import S3Error

from minio import Minio

MODEL_NAME = "all-MiniLM-L6-v2"
BUCKET = "vectors"
# The FAISS index and its {int id: doc_id} map live in ONE object, so a
# single PUT replaces both atomically. (They used to be two objects,
# written one after the other: a crash or a concurrent writer in between
# could leave an index whose ids the map didn't match.)
INDEX_OBJ = "index.v2.bin"
INDEX_MAGIC = b"S3VAH-IDX\x01"
# Pre-v2 layout. Never read: a bucket that only has these has no INDEX_OBJ,
# so the index is rebuilt from the per-document vectors and these are
# removed. Kept reserved so no doc_id can collide with them.
LEGACY_INDEX_OBJS = ("faiss.index", "index.ids.json")
RESERVED_OBJECT_NAMES = {INDEX_OBJ, *LEGACY_INDEX_OBJS}
MAX_DOC_ID_LENGTH = 512
MAX_TOP_K = 1000
MAX_PAGE_SIZE = 1000
# Object-store calls are network round trips, so independent ones (the
# per-document objects of a bulk write, a page of metadata reads, the
# vectors of a rebuild) run in parallel. Kept below minio-py's urllib3
# pool size (10) so workers don't queue for connections.
DEFAULT_IO_WORKERS = 8
# Bulk writes embed this many texts at a time, uploading each chunk while
# the next one is embedded (CPU-bound model work overlaps network I/O).
EMBED_CHUNK_SIZE = 256


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

    The loaded index is cached in memory, keyed by the ETag of the index
    object. Each read does a cheap
    HEAD (`stat_object`) instead of downloading the whole index; if another
    process has rewritten the index since, the ETag differs and it is
    reloaded. All access to the cached FAISS object happens under
    `self._lock`, since FAISS indexes aren't safe to search while another
    thread mutates them.
    """

    def __init__(self, client: Minio, embedder, bucket: str = BUCKET, io_workers: int = DEFAULT_IO_WORKERS):
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="vectorstore-io")
        # (ids-object etag, index, id_map) or None
        self._cache: tuple[str, object, dict] | None = None
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

    def _put_bytes(self, key: str, data: bytes) -> str:
        """Write an object and return its (unquoted) ETag."""
        result = self._client.put_object(self._bucket, key, io.BytesIO(data), length=len(data))
        return _normalize_etag(result.etag)

    def _etag(self, key: str) -> str | None:
        try:
            return _normalize_etag(self._client.stat_object(self._bucket, key).etag)
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject"):
                return None
            raise

    def _delete_object(self, key: str):
        try:
            self._client.remove_object(self._bucket, key)
        except S3Error as e:
            if e.code != "NoSuchKey":
                raise
        except ObjectNotFoundError:
            pass

    def _parallel(self, fn, items) -> list:
        """fn over items on the I/O pool, results in order; re-raises the first error."""
        return list(self._io.map(fn, items))

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    # -- index persistence ------------------------------------------------
    # Blob layout: INDEX_MAGIC | u64 big-endian length of ids JSON | ids JSON
    # | faiss.serialize_index bytes. faiss.write_index/read_index need a
    # path/FILE*/IOWriter, not a file-like object -- serialize_index /
    # deserialize_index are the supported in-memory round trip.

    @staticmethod
    def _encode_index(index, id_map: dict) -> bytes:
        ids_json = json.dumps({str(k): v for k, v in id_map.items()}).encode()
        return b"".join(
            (INDEX_MAGIC, struct.pack(">Q", len(ids_json)), ids_json, faiss.serialize_index(index).tobytes())
        )

    @staticmethod
    def _decode_index(blob: bytes):
        if not blob.startswith(INDEX_MAGIC):
            raise ValueError(f"{INDEX_OBJ} is not a recognised index blob")
        pos = len(INDEX_MAGIC)
        (ids_len,) = struct.unpack_from(">Q", blob, pos)
        pos += 8
        id_map = {int(k): v for k, v in json.loads(blob[pos: pos + ids_len]).items()}
        index = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8, offset=pos + ids_len))
        return index, id_map

    def _load_index(self):
        """Returns (IndexIDMap2 or None, {int_id: doc_id}).

        Must be called with `self._lock` held. Serves the cached index when
        the index object's ETag is unchanged. The ETag is read *before* the
        download, so a concurrent writer in another process can only cause
        newer data to be cached under an older ETag -- which just triggers
        one extra reload next time, never a stale read.
        """
        etag = self._etag(INDEX_OBJ)
        if etag is None:
            self._cache = None
            return None, {}
        if self._cache is not None and self._cache[0] == etag:
            return self._cache[1], self._cache[2]
        try:
            blob = self._get_bytes(INDEX_OBJ)
        except ObjectNotFoundError:
            self._cache = None
            return None, {}
        index, id_map = self._decode_index(blob)
        self._cache = (etag, index, id_map)
        return index, id_map

    def _save_index(self, index, id_map: dict):
        # Drop the cache first: callers mutate the cached index in place, so
        # if a write below fails the cache must not keep serving the
        # mutated-but-unpersisted index as if it matched the stored ETag.
        self._cache = None
        etag = self._put_bytes(INDEX_OBJ, self._encode_index(index, id_map))
        self._cache = (etag, index, id_map)

    def _drop_index(self):
        self._cache = None
        self._delete_object(INDEX_OBJ)

    def _rebuild_index_from_scratch(self, persist: bool = True):
        """Bootstrap the index from the individually-stored vectors.

        Only used when no index exists yet (first write, a bucket still on
        the pre-v2 two-object layout, or the index object went missing) --
        ordinary add/update/delete mutate the existing index directly
        instead of paying for this.
        """
        for legacy in LEGACY_INDEX_OBJS:
            self._delete_object(legacy)
        doc_ids = [
            obj.object_name.removesuffix(".npy")
            for obj in self._client.list_objects(self._bucket)
            if obj.object_name.endswith(".npy")
        ]
        if not doc_ids:
            return None, {}
        vectors = self._parallel(
            lambda d: np.frombuffer(self._get_bytes(f"{d}.npy"), dtype=np.float32), doc_ids
        )
        dim = vectors[0].shape[0]
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        keys = np.array([object_key_id(d) for d in doc_ids], dtype="int64")
        index.add_with_ids(np.vstack(vectors), keys)
        id_map = dict(zip((int(k) for k in keys), doc_ids))
        if persist:
            self._save_index(index, id_map)
        return index, id_map

    def _ensure_index(self, persist: bool = True):
        """Load the index, bootstrapping it from stored vectors if missing.

        Writers pass persist=False because they save the index themselves
        right after mutating it; persisting the bootstrap too would upload
        the whole index twice.
        """
        index, id_map = self._load_index()
        if index is None:
            index, id_map = self._rebuild_index_from_scratch(persist=persist)
        return index, id_map

    # -- public API ---------------------------------------------------------

    def add_document(self, doc_id: str, text: str):
        """Add a new document, or replace the vector+text of an existing one."""
        self.add_documents([(doc_id, text)])

    def add_documents(self, docs) -> int:
        """Add or replace many documents with one embed call and one index write.

        `docs` is an iterable of (doc_id, text) pairs. Every doc_id is
        validated before anything is written, so an invalid id rejects the
        whole batch instead of leaving it half-applied. If a doc_id appears
        more than once, the last occurrence wins. Returns the number of
        distinct documents written.
        """
        batch: dict[str, str] = {}
        for doc_id, text in docs:
            validate_doc_id(doc_id)
            batch[doc_id] = text
        if not batch:
            return 0
        doc_ids = list(batch)
        # Embed chunk by chunk, handing each chunk's uploads to the I/O pool
        # before embedding the next, so uploads overlap embedding.
        chunks, uploads = [], []
        for start in range(0, len(doc_ids), EMBED_CHUNK_SIZE):
            chunk_ids = doc_ids[start: start + EMBED_CHUNK_SIZE]
            chunk_vecs = self._embed_many([batch[d] for d in chunk_ids])
            chunks.append(chunk_vecs)
            for doc_id, vec in zip(chunk_ids, chunk_vecs):
                meta = json.dumps({"doc_id": doc_id, "text": batch[doc_id]}).encode()
                uploads.append(self._io.submit(self._put_bytes, f"{doc_id}.npy", vec.tobytes()))
                uploads.append(self._io.submit(self._put_bytes, f"{doc_id}.meta.json", meta))
        for f in uploads:
            f.result()  # wait for all; re-raise the first failure before touching the index
        vecs = np.vstack(chunks)
        keys = np.array([object_key_id(d) for d in doc_ids], dtype="int64")
        with self._lock:
            index, id_map = self._ensure_index(persist=False)
            if index is None:
                index = faiss.IndexIDMap2(faiss.IndexFlatL2(vecs.shape[1]))
            else:
                existing = np.array([k for k in keys if int(k) in id_map], dtype="int64")
                if existing.size:
                    index.remove_ids(existing)
            index.add_with_ids(vecs, keys)
            id_map.update(zip((int(k) for k in keys), doc_ids))
            self._save_index(index, id_map)
        return len(doc_ids)

    def get_document(self, doc_id: str):
        try:
            return json.loads(self._get_bytes(f"{doc_id}.meta.json"))
        except ObjectNotFoundError:
            return None

    def list_documents(self, skip: int = 0, limit: int = 100):
        if skip < 0:
            raise ValueError("skip must not be negative")
        if not 0 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 0 and {MAX_PAGE_SIZE}")
        # The (cached) index's id map is the list of searchable documents,
        # so paging reads it instead of listing every object in the bucket.
        page = sorted(self._indexed_doc_ids())[skip: skip + limit]
        metas = self._parallel(self.get_document, page)
        return [
            {"doc_id": doc_id, "text_preview": meta.get("text", "")[:64]}
            for doc_id, meta in zip(page, metas)
            if meta
        ]

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
                    self._drop_index()
                else:
                    self._save_index(index, id_map)
        return True

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """doc_ids of the top_k closest documents, closest first."""
        return [r["doc_id"] for r in self.search_with_scores(query, top_k)]

    def search_with_scores(self, query: str, top_k: int = 5) -> list[dict]:
        """Top_k matches as [{"doc_id", "score"}], closest first.

        `score` is cosine similarity (-1..1, higher is closer). Vectors are
        unit-normalised, so it follows from FAISS's squared L2 distance d
        as 1 - d/2 without changing the index type.
        """
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        qv = self._embed(query)
        with self._lock:
            index, id_map = self._ensure_index()
            if index is None:
                return []
            distances, result_ids = index.search(np.expand_dims(qv, axis=0), top_k)
        # FAISS pads short result sets with sentinel id -1; must be excluded
        # explicitly or it aliases to the last id_map entry looked up.
        return [
            {"doc_id": id_map[int(i)], "score": float(1.0 - d / 2.0)}
            for d, i in zip(distances[0], result_ids[0])
            if i != -1
        ]

    def count_documents(self) -> int:
        return len(self._indexed_doc_ids())

    def _indexed_doc_ids(self) -> list[str]:
        with self._lock:
            _, id_map = self._ensure_index()
            return list(id_map.values())

    def embedding_model_name(self) -> str:
        return MODEL_NAME


def _normalize_etag(etag: str | None) -> str:
    return (etag or "").strip('"')


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
