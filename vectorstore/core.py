import hashlib
import json
import os
import random
import re
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from minio.error import S3Error

from minio import Minio
from vectorstore.embedding import MODEL_NAME, OnnxEmbedder
from vectorstore.filters import matches, validate_filter, validate_metadata

BUCKET = "vectors"
DEFAULT_INDEX = "default"
MAX_DOC_ID_LENGTH = 512
MAX_TOP_K = 1000
MAX_PAGE_SIZE = 1000
# Index names: 1-63 of [a-z0-9-], not starting or ending with "-".
_INDEX_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# Each named index is ONE object holding everything about its documents:
# doc_id, text, metadata and vector. A single conditional PUT therefore
# updates a document's text, metadata and vector together, so they can never
# disagree, even when several processes write the same doc_id at once.
INDEX_PREFIX = "indexes/"
INDEX_SUFFIX = ".v3.bin"
INDEX_MAGIC = b"S3VAH-IDX\x03"

# Older layouts, only ever read, to migrate into the default index the first
# time it is used:
#   v2: root "index.v2.bin" (FAISS + {id: doc_id}) plus per-doc objects;
#   v1: root "faiss.index" + "index.ids.json" plus per-doc objects.
# Per-doc objects are "<doc_id>.npy" (vector) and "<doc_id>.meta.json"
# ({"doc_id", "text"}). They are left in place after migration (nothing
# reads them again; delete them once you no longer need to roll back).
LEGACY_V2_INDEX_OBJ = "index.v2.bin"
LEGACY_V2_MAGIC = b"S3VAH-IDX\x01"

# Migration reads legacy per-doc objects in parallel. Kept below minio-py's
# urllib3 pool size (10) so workers don't queue for connections.
DEFAULT_IO_WORKERS = 8
# Compare-and-swap retries for index writes that lose a race to another
# writer. Each retry reloads the index; backoff is exponential with full
# jitter, capped, so contending writers spread out instead of colliding in
# lockstep. Worst case before giving up is a few seconds of waiting.
INDEX_WRITE_ATTEMPTS = 25
INDEX_WRITE_BACKOFF_BASE_S = 0.005
INDEX_WRITE_BACKOFF_CAP_S = 0.25
# Conditional writes go out as a single PUT (conditions on multipart
# uploads are a different, less widely supported mechanism), and S3 caps a
# single PUT at 5 GiB.
MAX_SINGLE_PUT_BYTES = 5 * 1024**3
# Throwaway objects for the startup conditional-write probe.
PROBE_PREFIX = "_probe/"
# S3 error codes meaning "a conditional write lost to a concurrent change".
_CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict"}


class ObjectNotFoundError(Exception):
    """Raised when a requested object does not exist in the backing store."""


class IndexNotFoundError(LookupError):
    """The named index doesn't exist (create it first)."""


class IndexAlreadyExistsError(Exception):
    """An index with that name already exists."""


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
    """Deterministic 63-bit id for a doc_id, used as its FAISS vector id.

    A hash rather than insertion order keeps the vector tied to its doc_id
    through removals and updates. Two doc_ids hashing to the same id is
    astronomically unlikely, but it is detected (see `add_documents`)
    rather than allowed to silently overwrite a document.
    """
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def validate_doc_id(doc_id: str) -> None:
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    # Not an object key any more, but it is a URL path segment in the API.
    if "/" in doc_id:
        raise ValueError("doc_id must not contain '/'")
    if len(doc_id) > MAX_DOC_ID_LENGTH:
        raise ValueError(f"doc_id must be at most {MAX_DOC_ID_LENGTH} characters")


def validate_index_name(name: str) -> None:
    if not isinstance(name, str) or not _INDEX_NAME_RE.match(name):
        raise ValueError(
            "index name must be 1-63 characters of lowercase letters, digits and '-', "
            "not starting or ending with '-'"
        )


def index_object_key(name: str) -> str:
    return f"{INDEX_PREFIX}{name}{INDEX_SUFFIX}"


class _IndexState:
    """One index in memory: the FAISS index plus every document's fields.

    `docs` maps FAISS id -> {"doc_id", "text", "metadata"}. `by_id` is the
    reverse lookup, derived from `docs`.
    """

    __slots__ = ("by_id", "dim", "docs", "faiss")

    def __init__(self, dim: int, faiss_index=None, docs=None):
        self.dim = dim
        self.faiss = faiss_index if faiss_index is not None else faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.docs: dict[int, dict] = docs or {}
        self.by_id = {d["doc_id"]: k for k, d in self.docs.items()}

    # Blob layout: INDEX_MAGIC | u64 big-endian length of header JSON |
    # header JSON {"dim", "docs": [[id, doc_id, text, metadata], ...]} |
    # faiss.serialize_index bytes. (faiss.write_index needs a path/FILE*;
    # serialize_index/deserialize_index are the in-memory round trip.)

    def encode(self) -> bytes:
        header = json.dumps(
            {"dim": self.dim, "docs": [[k, d["doc_id"], d["text"], d["metadata"]] for k, d in self.docs.items()]}
        ).encode()
        return b"".join((INDEX_MAGIC, struct.pack(">Q", len(header)), header, faiss.serialize_index(self.faiss).tobytes()))

    @classmethod
    def decode(cls, blob: bytes, key: str) -> "_IndexState":
        if not blob.startswith(INDEX_MAGIC):
            raise ValueError(f"{key} is not a recognised index blob")
        pos = len(INDEX_MAGIC)
        (header_len,) = struct.unpack_from(">Q", blob, pos)
        pos += 8
        header = json.loads(blob[pos: pos + header_len])
        index = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8, offset=pos + header_len))
        docs = {int(k): {"doc_id": d, "text": t, "metadata": m} for k, d, t, m in header["docs"]}
        if index.ntotal != len(docs):
            raise ValueError(f"{key} is corrupt: {index.ntotal} vectors but {len(docs)} documents")
        return cls(header["dim"], index, docs)


class VectorStore:
    """Named vector indexes over an object store, searched with FAISS.

    Takes its object-store client and embedder as constructor arguments, so
    it can be built once from env vars for the real app (`get_store`) or
    directly with fakes in tests.

    Indexes. A store holds any number of named indexes; "default" always
    exists (it's created on first use, migrating data from older layouts if
    there is any). Others are created and deleted explicitly. Every method
    takes `index=`, defaulting to "default".

    Concurrency. Any number of processes (uvicorn workers, CLI runs) may
    read and write the same bucket. Index updates are compare-and-swap: a
    writer loads the index *and the ETag it was served with*, applies its
    change, and PUTs with `If-Match: <that etag>` (or `If-None-Match: *` to
    create). If another writer got there first the PUT is rejected, and the
    writer reloads and re-applies its change on top. Every change is an
    explicit, idempotent edit ("upsert these docs", "remove this doc"), so
    re-applying it -- even on a state that already contains it, as when a
    PUT succeeded but its response was lost and the client's retry got a
    412 -- is harmless. Because a document's text, metadata and vector live
    in the same object, concurrent writes to the *same* doc_id are ordered
    too: the last successful write wins, whole. `self._lock` serialises the
    same within one process and guards the cached FAISS objects, which
    aren't safe to search while being mutated.

    The default index is never deleted, so its one-time migration from the
    old layout can't run twice. A deleted named index stays deleted: a
    writer that loses to the delete reloads, finds no index and fails with
    IndexNotFoundError rather than re-creating it.

    Caching. Each loaded index is kept in memory with its ETag. Every
    operation does a cheap HEAD and only downloads an index whose ETag
    changed.
    """

    def __init__(self, client: Minio, embedder, bucket: str = BUCKET, io_workers: int = DEFAULT_IO_WORKERS):
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="vectorstore-io")
        # index name -> (etag, _IndexState) of the last version this process read or wrote.
        self._cache: dict[str, tuple[str, _IndexState]] = {}
        self._dim: int | None = None
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
        return self._get_with_etag(key)[0]

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

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    def _embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed("dimension probe").shape[0])
        return self._dim

    # -- index state ------------------------------------------------------

    def _load(self, name: str) -> tuple[_IndexState | None, str | None]:
        """(state, etag) as currently stored; (None, None) if the index doesn't exist.

        Must be called with `self._lock` held. Serves the cache when the
        HEAD shows an unchanged ETag. Otherwise downloads, and keys the cache
        on the ETag of that same GET response, so the content and the ETag a
        later If-Match is checked against always belong together.
        """
        key = index_object_key(name)
        etag = self._etag(key)
        if etag is None:
            self._cache.pop(name, None)
            return None, None
        cached = self._cache.get(name)
        if cached is not None and cached[0] == etag:
            return cached[1], etag
        try:
            blob, etag = self._get_with_etag(key)
        except ObjectNotFoundError:
            self._cache.pop(name, None)
            return None, None
        state = _IndexState.decode(blob, key)
        self._cache[name] = (etag, state)
        return state, etag

    def _initial_default_state(self) -> _IndexState:
        """State for a default index that doesn't exist yet: migrated, or empty.

        Only ever *created* (If-None-Match), so if two processes migrate at
        once exactly one result is kept and the other process uses it.
        """
        legacy = self._legacy_docs()
        if not legacy:
            return _IndexState(self._embedding_dim())
        doc_ids = list(legacy)
        vectors = np.vstack([legacy[d][0] for d in doc_ids])
        state = _IndexState(vectors.shape[1])
        keys = np.array([object_key_id(d) for d in doc_ids], dtype="int64")
        state.faiss.add_with_ids(vectors, keys)
        for k, d in zip(keys, doc_ids):
            state.docs[int(k)] = {"doc_id": d, "text": legacy[d][1], "metadata": {}}
            state.by_id[d] = int(k)
        return state

    def _legacy_docs(self) -> dict[str, tuple[np.ndarray, str]]:
        """{doc_id: (vector, text)} from a pre-v3 bucket; {} if there is none.

        A v2 index decides which docs are live (its map is what search used);
        without one, every doc with both a vector and a text object is. Docs
        missing either object are skipped: on the old layout they were
        unreachable (list skipped them) or had no vector to search by.
        """
        names = {obj.object_name for obj in self._client.list_objects(self._bucket)}
        if LEGACY_V2_INDEX_OBJ in names:
            blob = self._get_bytes(LEGACY_V2_INDEX_OBJ)
            if not blob.startswith(LEGACY_V2_MAGIC):
                raise ValueError(f"{LEGACY_V2_INDEX_OBJ} is not a recognised v2 index blob")
            (ids_len,) = struct.unpack_from(">Q", blob, len(LEGACY_V2_MAGIC))
            start = len(LEGACY_V2_MAGIC) + 8
            live = list(json.loads(blob[start: start + ids_len]).values())
        else:
            live = [n.removesuffix(".npy") for n in names if n.endswith(".npy")]
        live = [d for d in live if f"{d}.npy" in names and f"{d}.meta.json" in names]

        def read(doc_id):
            vec = np.frombuffer(self._get_bytes(f"{doc_id}.npy"), dtype=np.float32)
            text = json.loads(self._get_bytes(f"{doc_id}.meta.json")).get("text", "")
            return doc_id, (vec, text)

        return dict(self._io.map(read, live))

    def _read(self, name: str) -> _IndexState:
        """Current state of index `name` for reading. Must hold `self._lock`."""
        for _ in range(INDEX_WRITE_ATTEMPTS):
            state, _ = self._load(name)
            if state is not None:
                return state
            if name != DEFAULT_INDEX:
                raise IndexNotFoundError(name)
            state = self._initial_default_state()
            try:
                etag = self._put_conditional(index_object_key(name), state.encode(), if_match=None)
            except _WriteConflict:
                continue  # someone else created it first: use theirs
            self._cache[name] = (etag, state)
            return state
        raise IndexWriteConflictError("index kept changing while being created; retry the operation")

    def _update(self, name: str, edit):
        """Apply `edit` to index `name` with compare-and-swap, retrying on conflict.

        `edit(state)` mutates `state` in place and returns `(changed, result)`;
        `result` is what this method returns. It must be idempotent -- it may
        run again on a fresh state after a lost race, including a state that
        already contains its own effect -- and must raise (to abort, writing
        nothing) before it mutates anything.
        """
        key = index_object_key(name)
        with self._lock:
            for attempt in range(INDEX_WRITE_ATTEMPTS):
                state, etag = self._load(name)
                created = False
                if state is None:
                    if name != DEFAULT_INDEX:
                        raise IndexNotFoundError(name)
                    state, created = self._initial_default_state(), True
                # From here the cached state may be mutated in place, so drop
                # it unless the write succeeds; a later read must not take
                # unpersisted state for the stored version.
                self._cache.pop(name, None)
                changed, result = edit(state)
                if not (changed or created):
                    assert etag is not None  # not created, so it was loaded with one
                    self._cache[name] = (etag, state)  # untouched: still that version
                    return result
                try:
                    new_etag = self._put_conditional(key, state.encode(), if_match=etag)
                except _WriteConflict:
                    time.sleep(random.uniform(0, min(INDEX_WRITE_BACKOFF_CAP_S, INDEX_WRITE_BACKOFF_BASE_S * 2**attempt)))
                    continue
                if new_etag:
                    self._cache[name] = (new_etag, state)
                return result
        raise IndexWriteConflictError(
            f"index update lost to concurrent writers {INDEX_WRITE_ATTEMPTS} times in a row; retry the operation"
        )

    # -- indexes --------------------------------------------------------------

    def list_indexes(self) -> list[str]:
        names = {DEFAULT_INDEX}
        for obj in self._client.list_objects(self._bucket, prefix=INDEX_PREFIX):
            if obj.object_name.endswith(INDEX_SUFFIX):
                names.add(obj.object_name[len(INDEX_PREFIX): -len(INDEX_SUFFIX)])
        return sorted(names)

    def create_index(self, name: str) -> None:
        validate_index_name(name)
        if name == DEFAULT_INDEX:
            raise IndexAlreadyExistsError(name)
        state = _IndexState(self._embedding_dim())
        with self._lock:
            try:
                etag = self._put_conditional(index_object_key(name), state.encode(), if_match=None)
            except _WriteConflict:
                raise IndexAlreadyExistsError(name) from None
            self._cache[name] = (etag, state)

    def delete_index(self, name: str) -> None:
        validate_index_name(name)
        if name == DEFAULT_INDEX:
            raise ValueError("the default index can't be deleted")
        key = index_object_key(name)
        with self._lock:
            if self._etag(key) is None:
                raise IndexNotFoundError(name)
            self._delete_object(key)
            self._cache.pop(name, None)

    # -- documents ------------------------------------------------------------

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None, index: str = DEFAULT_INDEX):
        """Add a new document, or replace the text, metadata and vector of an existing one."""
        self.add_documents([(doc_id, text, metadata)], index=index)

    def add_documents(self, docs, index: str = DEFAULT_INDEX) -> int:
        """Add or replace many documents with one embed call and one index write.

        `docs` is an iterable of (doc_id, text) or (doc_id, text, metadata).
        Everything is validated before anything is written, so one bad
        document rejects the whole batch. If a doc_id appears more than once,
        the last occurrence wins. Returns the number of distinct documents
        written.
        """
        validate_index_name(index)
        batch: dict[str, tuple[str, dict]] = {}
        for doc in docs:
            doc_id, text, metadata = (*doc, None) if len(doc) == 2 else doc
            validate_doc_id(doc_id)
            if not isinstance(text, str):
                raise ValueError(f"text for {doc_id!r} must be a string")  # noqa: TRY004 -- API maps ValueError to 400
            batch[doc_id] = (text, validate_metadata(metadata))
        if not batch:
            return 0
        doc_ids = list(batch)
        vecs = self._embed_many([batch[d][0] for d in doc_ids])
        keys = [object_key_id(d) for d in doc_ids]

        def upsert(state: _IndexState):
            # Validate against the current state first: raising after a
            # mutation would leave the (uncached) state half-edited.
            if vecs.shape[1] != state.dim:
                raise ValueError(f"embedding dimension {vecs.shape[1]} doesn't match index {index!r} ({state.dim})")
            for doc_id, key in zip(doc_ids, keys):
                owner = state.docs.get(key)
                if owner is not None and owner["doc_id"] != doc_id:
                    raise ValueError(f"doc_id {doc_id!r} collides with {owner['doc_id']!r}; use another doc_id")
            existing = np.array([k for k in keys if k in state.docs], dtype="int64")
            if existing.size:
                state.faiss.remove_ids(existing)
            state.faiss.add_with_ids(vecs, np.array(keys, dtype="int64"))
            for doc_id, key in zip(doc_ids, keys):
                text, metadata = batch[doc_id]
                state.docs[key] = {"doc_id": doc_id, "text": text, "metadata": metadata}
                state.by_id[doc_id] = key
            return True, len(doc_ids)

        return self._update(index, upsert)

    def get_document(self, doc_id: str, index: str = DEFAULT_INDEX):
        """{"doc_id", "text", "metadata"}, or None if there's no such document."""
        validate_index_name(index)
        with self._lock:
            state = self._read(index)
            key = state.by_id.get(doc_id)
            return None if key is None else _public(state.docs[key])

    def list_documents(self, skip: int = 0, limit: int = 100, index: str = DEFAULT_INDEX):
        if skip < 0:
            raise ValueError("skip must not be negative")
        if not 0 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 0 and {MAX_PAGE_SIZE}")
        validate_index_name(index)
        with self._lock:
            state = self._read(index)
            page = sorted(state.by_id)[skip: skip + limit]
            return [
                {"doc_id": d, "text_preview": doc["text"][:64], "metadata": _copy_metadata(doc["metadata"])}
                for d in page
                for doc in (state.docs[state.by_id[d]],)
            ]

    def delete_document(self, doc_id: str, index: str = DEFAULT_INDEX) -> bool:
        """Delete a document; False if it wasn't there.

        True once *any* attempt found the document: from then on its removal
        is guaranteed. That matters when a PUT lands but is reported as a
        conflict -- the retry no longer finds the doc, and answering "not
        found" would tell the caller its own successful delete failed. (If
        another process deleted it concurrently, both report True, as
        idempotent deletes do.)
        """
        validate_index_name(index)
        found = False

        def remove(state: _IndexState):
            nonlocal found
            key = state.by_id.get(doc_id)
            if key is None:
                return False, found
            found = True
            state.faiss.remove_ids(np.array([key], dtype="int64"))
            del state.docs[key]
            del state.by_id[doc_id]
            return True, True

        return self._update(index, remove)

    def search(self, query: str, top_k: int = 5, index: str = DEFAULT_INDEX, filter: dict | None = None) -> list[str]:
        """doc_ids of the top_k closest matching documents, closest first."""
        return [r["doc_id"] for r in self.search_with_scores(query, top_k, index=index, filter=filter)]

    def search_with_scores(
        self, query: str, top_k: int = 5, index: str = DEFAULT_INDEX, filter: dict | None = None
    ) -> list[dict]:
        """Top_k matches as [{"doc_id", "score", "metadata"}], closest first.

        `filter` (see vectorstore.filters) restricts the search to documents
        whose metadata matches; the search is still exact, i.e. these are the
        true top_k among the matching documents. `score` is cosine
        similarity (-1..1, higher is closer): vectors are unit-normalised,
        so it is 1 - d/2 from FAISS's squared L2 distance d.
        """
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        validate_index_name(index)
        if filter:
            validate_filter(filter)
        qv = np.expand_dims(self._embed(query), axis=0)
        with self._lock:
            state = self._read(index)
            if filter:
                allowed = np.array([k for k, d in state.docs.items() if matches(filter, d["metadata"])], dtype="int64")
                if not allowed.size:
                    return []
                params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
                distances, result_ids = state.faiss.search(qv, top_k, params=params)
            else:
                if state.faiss.ntotal == 0:
                    return []
                distances, result_ids = state.faiss.search(qv, top_k)
            # FAISS pads short result sets with id -1: skip those.
            return [
                {
                    "doc_id": state.docs[int(i)]["doc_id"],
                    "score": float(1.0 - d / 2.0),
                    "metadata": _copy_metadata(state.docs[int(i)]["metadata"]),
                }
                for d, i in zip(distances[0], result_ids[0])
                if i != -1
            ]

    def count_documents(self, index: str = DEFAULT_INDEX) -> int:
        validate_index_name(index)
        with self._lock:
            return len(self._read(index).docs)

    def embedding_model_name(self) -> str:
        return MODEL_NAME


def _copy_metadata(metadata: dict) -> dict:
    # Values are scalars or lists of scalars, so copying the lists makes
    # this a full copy: callers can't reach into the cached state.
    return {k: list(v) if isinstance(v, list) else v for k, v in metadata.items()}


def _public(doc: dict) -> dict:
    return {"doc_id": doc["doc_id"], "text": doc["text"], "metadata": _copy_metadata(doc["metadata"])}


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
                # FAISS's OpenMP pool and ONNX Runtime's threads otherwise
                # contend for cores: a 0.05 ms search over 1,000 docs took
                # ~18 ms. A single-query flat search runs on one thread anyway.
                faiss.omp_set_num_threads(1)
                _store = VectorStore(client, OnnxEmbedder.from_hub())
    return _store
