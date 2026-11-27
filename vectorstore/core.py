import hashlib
import io
import json
import os
import random
import struct
import threading
import time
import uuid
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
# Compare-and-swap retries for index writes that lose a race to another
# writer. Each retry reloads the index; backoff is exponential with full
# jitter, capped, so contending writers spread out instead of colliding in
# lockstep. Worst case before giving up is a few seconds of waiting.
INDEX_WRITE_ATTEMPTS = 25
INDEX_WRITE_BACKOFF_BASE_S = 0.005
INDEX_WRITE_BACKOFF_CAP_S = 0.25
# Conditional writes go out as a single PUT (conditions on multipart
# uploads are a different, less widely supported mechanism), and S3 caps a
# single PUT at 5 GiB. At 384 float32 dims that is ~3.4M documents.
MAX_SINGLE_PUT_BYTES = 5 * 1024**3
# Throwaway objects for the startup conditional-write probe. doc_ids can't
# contain "/", so nothing under this prefix can collide with a document.
PROBE_PREFIX = "_probe/"
# S3 error codes meaning "a conditional write lost to a concurrent change".
_CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict"}


class ObjectNotFoundError(Exception):
    """Raised when a requested object does not exist in the backing store."""


class ConditionalWritesUnsupportedError(RuntimeError):
    """The object store doesn't enforce If-Match / If-None-Match on PUT.

    Without that, concurrent writers in different processes would silently
    overwrite each other's index updates, so the store refuses to run.
    """


class IndexWriteConflictError(RuntimeError):
    """An index update kept losing to concurrent writers and gave up.

    Nothing was half-applied: the index is exactly as some other writer
    left it. The operation can be retried.
    """


class _WriteConflict(Exception):
    """Internal: one conditional index PUT lost a race."""


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

    Concurrency. Any number of processes (uvicorn workers, CLI runs) may
    read and write the same bucket. Index updates are compare-and-swap: a
    writer loads the index *and the ETag it was served with*, applies its
    change, and PUTs with `If-Match: <that etag>` (or `If-None-Match: *` if
    there was no index yet). If another writer got there first the PUT is
    rejected, and the writer reloads and re-applies its change on top. Every
    change is an explicit, idempotent edit ("upsert these ids", "remove this
    id") rather than "write back what I read", so re-applying it -- even on
    a state that already contains it, as when a PUT succeeded but its
    response was lost and the client's retry got a 412 -- is harmless.
    The index object is never deleted once created (an empty corpus is an
    empty index), so a slow writer can never re-create it from stale data.
    `self._lock` serialises the same within one process and guards the
    cached FAISS object, which isn't safe to search while being mutated.

    Not covered: two callers writing the *same* doc_id at the same time
    (e.g. one adding it while another deletes it). The index stays
    internally consistent, but its entry for that doc may disagree with the
    doc's `.npy`/`.meta.json` objects until the doc is written again.

    Caching. The loaded index is kept in memory with its ETag. Each read
    does a cheap HEAD and only downloads the index if the ETag changed.
    """

    def __init__(self, client: Minio, embedder, bucket: str = BUCKET, io_workers: int = DEFAULT_IO_WORKERS):
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="vectorstore-io")
        # (etag, index, id_map) of the last index this process read or wrote.
        self._cache: tuple[str, object, dict] | None = None
        self._ensure_bucket()
        self._verify_conditional_writes()

    # -- object store helpers -------------------------------------------

    def _ensure_bucket(self):
        if self._client.bucket_exists(self._bucket):
            return
        try:
            self._client.make_bucket(self._bucket)
        except S3Error as e:
            # Lost the race to another process creating it: that's fine.
            # (BucketAlreadyExists, owned by someone else, is a real error.)
            if e.code != "BucketAlreadyOwnedByYou":
                raise

    def _put_conditional(self, key: str, data: bytes, if_match: str | None) -> str:
        """PUT `data` only if `key` is still at ETag `if_match` (None: only if absent).

        Returns the new ETag, or raises _WriteConflict if the condition
        failed. Uses minio-py's single-request `_put_object` because its
        public `put_object` treats unknown headers as user metadata: an
        If-Match passed there is sent as `x-amz-meta-if-match` and silently
        ignored. `_verify_conditional_writes` checks at startup that the
        conditions are really enforced, so a library or server change can't
        quietly turn this back into an unconditional write.
        """
        if len(data) > MAX_SINGLE_PUT_BYTES:
            raise ValueError(f"index is {len(data)} bytes; a single PUT is capped at {MAX_SINGLE_PUT_BYTES}")
        headers = {"Content-Type": "application/octet-stream"}
        if if_match is None:
            headers["If-None-Match"] = "*"
        else:
            headers["If-Match"] = f'"{if_match}"'
        try:
            result = self._client._put_object(self._bucket, key, data, headers)
        except S3Error as e:
            # NoSuchKey: we expected a version to replace but the key is gone.
            if e.code in _CONFLICT_CODES or (if_match is not None and e.code == "NoSuchKey"):
                raise _WriteConflict(key) from e
            raise
        return _normalize_etag(result.etag)

    def _verify_conditional_writes(self):
        key = f"{PROBE_PREFIX}{uuid.uuid4().hex}"
        try:
            self._put_conditional(key, b"1", if_match=None)
            for label, if_match in (("If-None-Match: *", None), ("a stale If-Match", "0" * 32)):
                try:
                    self._put_conditional(key, b"2", if_match=if_match)
                except _WriteConflict:
                    continue
                raise ConditionalWritesUnsupportedError(
                    f"the object store accepted a PUT with {label} that should have been "
                    "rejected; it doesn't enforce conditional writes, which concurrent "
                    "index updates rely on (MinIO supports them in releases since late 2024)"
                )
        finally:
            self._delete_object(key)

    def _get_with_etag(self, key: str) -> tuple[bytes, str]:
        try:
            resp = self._client.get_object(self._bucket, key)
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectNotFoundError(key) from e
            raise
        try:
            return resp.read(), _require_etag(resp.headers.get("ETag"), key)
        finally:
            resp.close()
            resp.release_conn()

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
            return _require_etag(self._client.stat_object(self._bucket, key).etag, key)
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
        """(index, id_map, etag) as currently stored; (None, {}, None) if there is none.

        Must be called with `self._lock` held. Serves the cache when the
        HEAD shows an unchanged ETag. Otherwise downloads, and keys the cache
        on the ETag of that same GET response, so the content and the ETag a
        later If-Match is checked against always belong together.
        """
        etag = self._etag(INDEX_OBJ)
        if etag is None:
            self._cache = None
            return None, {}, None
        if self._cache is not None and self._cache[0] == etag:
            return self._cache[1], self._cache[2], etag
        try:
            blob, etag = self._get_with_etag(INDEX_OBJ)
        except ObjectNotFoundError:
            self._cache = None
            return None, {}, None
        index, id_map = self._decode_index(blob)
        self._cache = (etag, index, id_map)
        return index, id_map, etag

    def _build_index_from_vectors(self):
        """Index the individually-stored vectors; (None, {}) if there are none.

        Only needed when there is no index object yet: first write, a bucket
        still on the pre-v2 two-object layout, or the index object was
        removed by hand. The result is only ever *created* (If-None-Match),
        so a build from a stale listing can't replace a real index.
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
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(vectors[0].shape[0]))
        keys = np.array([object_key_id(d) for d in doc_ids], dtype="int64")
        index.add_with_ids(np.vstack(vectors), keys)
        return index, dict(zip((int(k) for k in keys), doc_ids))

    def _update_index(self, edit):
        """Apply `edit` to the stored index with compare-and-swap, retrying on conflict.

        `edit(index_or_None, id_map)` mutates in place and returns
        `(index_or_None, changed)`. It must be idempotent -- it may run
        again on a fresh state after a lost race, including a state that
        already contains its own effect.
        """
        with self._lock:
            for attempt in range(INDEX_WRITE_ATTEMPTS):
                index, id_map, etag = self._load_index()
                bootstrapped = False
                if index is None:
                    index, id_map = self._build_index_from_vectors()
                    bootstrapped = index is not None
                # From here the cached index may be mutated in place, so drop
                # it unless the write succeeds; a later read must not take
                # unpersisted state for the stored version.
                self._cache = None
                index, changed = edit(index, id_map)
                if index is None or not (changed or bootstrapped):
                    if etag is not None and index is not None and not changed:
                        self._cache = (etag, index, id_map)
                    return
                try:
                    new_etag = self._put_conditional(INDEX_OBJ, self._encode_index(index, id_map), if_match=etag)
                except _WriteConflict:
                    time.sleep(random.uniform(0, min(INDEX_WRITE_BACKOFF_CAP_S, INDEX_WRITE_BACKOFF_BASE_S * 2**attempt)))
                    continue
                self._cache = (new_etag, index, id_map) if new_etag else None
                return
        raise IndexWriteConflictError(
            f"index update lost to concurrent writers {INDEX_WRITE_ATTEMPTS} times in a row; retry the operation"
        )

    def _read_index(self):
        """Index and id map for reading; (None, {}) if there are no documents.

        Must be called with `self._lock` held. If there's no index object
        but there are stored vectors, builds it and tries to create it so
        other readers needn't rebuild too; losing that race just means
        someone else's index is the one to use.
        """
        for _ in range(INDEX_WRITE_ATTEMPTS):
            index, id_map, _etag = self._load_index()
            if index is not None:
                return index, id_map
            index, id_map = self._build_index_from_vectors()
            if index is None:
                return None, {}
            try:
                new_etag = self._put_conditional(INDEX_OBJ, self._encode_index(index, id_map), if_match=None)
            except _WriteConflict:
                continue
            self._cache = (new_etag, index, id_map) if new_etag else None
            return index, id_map
        raise IndexWriteConflictError("index kept changing while being bootstrapped; retry the operation")

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

        def upsert(index, id_map):
            if index is None:
                index = faiss.IndexIDMap2(faiss.IndexFlatL2(vecs.shape[1]))
            else:
                existing = np.array([k for k in keys if int(k) in id_map], dtype="int64")
                if existing.size:
                    index.remove_ids(existing)
            index.add_with_ids(vecs, keys)
            id_map.update(zip((int(k) for k in keys), doc_ids))
            return index, True

        self._update_index(upsert)
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
        # Objects first, then the index: if a rebuild-from-vectors races
        # with this delete, the vector is already gone from its listing.
        self._delete_object(f"{doc_id}.npy")
        self._delete_object(f"{doc_id}.meta.json")
        key = object_key_id(doc_id)

        def remove(index, id_map):
            if index is None or key not in id_map:
                return index, False
            # An emptied index is written as-is, never deleted: see the
            # class docstring for why the index object must not disappear.
            index.remove_ids(np.array([key], dtype="int64"))
            del id_map[key]
            return index, True

        self._update_index(remove)
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
            index, id_map = self._read_index()
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
            _, id_map = self._read_index()
            return list(id_map.values())

    def embedding_model_name(self) -> str:
        return MODEL_NAME


def _normalize_etag(etag: str | None) -> str:
    return (etag or "").strip('"')


def _require_etag(etag: str | None, key: str) -> str:
    # Index writes are conditional on the ETag a read returned. Without one
    # every write would be sent as `If-Match: ""` and fail forever, so fail
    # loudly here instead.
    normalized = _normalize_etag(etag)
    if not normalized:
        raise RuntimeError(f"object store returned no ETag for {key!r}")
    return normalized


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
