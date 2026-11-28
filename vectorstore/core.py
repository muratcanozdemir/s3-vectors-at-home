import contextlib
import gc
import hashlib
import json
import logging
import os
import random
import re
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import faiss
import numpy as np
from minio.error import S3Error

from minio import Minio
from vectorstore.embedding import MODEL_NAME, OnnxEmbedder
from vectorstore.filters import matches, validate_filter, validate_metadata

logger = logging.getLogger(__name__)

BUCKET = "vectors"
DEFAULT_INDEX = "default"
MAX_DOC_ID_LENGTH = 512
MAX_TOP_K = 1000
MAX_PAGE_SIZE = 1000
# Index names: 1-63 of [a-z0-9-], not starting or ending with "-".
_INDEX_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# -- storage layout (v4) ------------------------------------------------------
# An index is a small JSON *manifest* plus immutable *segments*:
#
#   indexes/<name>.manifest.json                 the one object written with CAS
#   segments/<name>/<incarnation>/<uuid>.seg     never modified once written
#
# A segment holds records: an upsert (doc_id, text, metadata, vector) or a
# tombstone (doc_id). A write uploads one segment with its records, then
# appends it to the manifest with a conditional PUT, so its cost depends on
# the size of the write, not of the index. Every record has a sequence
# number (a segment written by a writer takes the one the manifest assigns
# it; merged segments store each record's own), and a document's current
# version is its record with the highest seq. That makes merging segments
# (compaction) invisible to readers, and lets a process that already has
# the index in memory catch up by reading only records newer than what it
# has seen.
#
# The manifest:
#   {"format": 4, "incarnation": "<hex>", "dim": 384, "next_seq": 12,
#    "segments": [{"key", "first_seq", "last_seq", "bytes", "full"}, ...],
#    "writers": {"<writer id>": [<last write counter>, <unix time>], ...}}
# "full" marks a merge of the whole index, which drops tombstones. Each
# incarnation (one create/delete lifetime of a name) has its own segment
# prefix, so deleting an index can never sweep away the segments of a newer
# index of the same name. "writers" lets a writer whose append succeeded
# but was reported as a conflict see that it already landed (see `_commit`).
FORMAT = 4
INDEX_PREFIX = "indexes/"
MANIFEST_SUFFIX = ".manifest.json"
SEGMENT_PREFIX = "segments/"
SEGMENT_MAGIC = b"S3VAH-SEG\x01"

# Compaction: after a write, segments are merged size-tiered, like a binary
# counter, so there are O(log n) of them and each record is rewritten
# O(log n) times over its life. Merges run in the background and swap the
# merged segment in with the same conditional PUT as any write. A merge
# result is capped so it stays a single PUT; `compact()` (full compaction)
# splits its output into parts of at most this size instead.
MAX_SEGMENT_BYTES = 1024**3
# A merge waits until it can take at least this many segments. Every merge
# is a manifest write, which makes concurrent writers retry, so batching
# them keeps contention down (~4x fewer than merging pairwise) without
# changing the O(log n) bounds.
MIN_MERGE = 4
# Merge sooner than the size rule would when there are more segments than this.
MAX_SEGMENTS = 64
# Safety margins for deleting segments nothing references (orphans: left by
# a crash between uploading a segment and appending it, or by a failed
# write). A writer re-uploads its segment if it has waited longer than
# SEGMENT_MAX_PENDING_S to append it, and gives up after INDEX_WRITE_TIMEOUT_S,
# so an unreferenced segment older than ORPHAN_GRACE_S can't be one that is
# about to be appended. Assumes clocks within ~45 minutes of each other.
SEGMENT_MAX_PENDING_S = 600
ORPHAN_GRACE_S = 3600
# How long a writer's entry stays in the manifest. Must exceed INDEX_WRITE_TIMEOUT_S.
WRITER_TTL_S = 3600
COMPACTION_ATTEMPTS = 5

# Older layouts, only ever read, to migrate the first time an index is used:
#   v3: one object per index, "indexes/<name>.v3.bin" (FAISS + all docs);
#   v2: root "index.v2.bin" (FAISS + {id: doc_id}) plus per-doc objects;
#   v1: root "faiss.index" + "index.ids.json" plus per-doc objects.
# v1/v2 only ever held the default index. Per-doc objects are "<doc_id>.npy"
# (vector) and "<doc_id>.meta.json" ({"doc_id", "text"}). Old objects are
# left in place after migration (nothing reads them again; delete them once
# you no longer need to roll back) -- except that deleting an index also
# deletes its v3 object, so it can't be migrated back to life.
LEGACY_V3_SUFFIX = ".v3.bin"
LEGACY_V3_MAGIC = b"S3VAH-IDX\x03"
LEGACY_V2_INDEX_OBJ = "index.v2.bin"
LEGACY_V2_MAGIC = b"S3VAH-IDX\x01"

# Reads of several segments (or of legacy per-doc objects) run in parallel.
# Kept below minio-py's urllib3 pool size (10) so workers don't queue for
# connections.
DEFAULT_IO_WORKERS = 8
# Compare-and-swap retries for manifest writes that lose a race to another
# writer. Each retry reloads the manifest; backoff is exponential with full
# jitter, capped, so contending writers spread out instead of colliding in
# lockstep. A writer keeps retrying for up to INDEX_WRITE_TIMEOUT_S: a time
# budget rather than a count, because how many attempts a write needs grows
# with the number of writers (32 processes writing flat out against local
# MinIO needed up to 36), while what a caller can wait for doesn't.
# INDEX_WRITE_ATTEMPTS is only a backstop.
INDEX_WRITE_TIMEOUT_S = 30.0
INDEX_WRITE_ATTEMPTS = 1000
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

    Nothing was applied: the index is exactly as other writers left it. The
    operation can be retried.
    """


class IndexCorruptError(RuntimeError):
    """The manifest references a segment that doesn't exist."""


class _WriteConflict(Exception):
    """Internal: one conditional PUT lost a race."""


class _SegmentMissing(Exception):
    """Internal: a segment listed in a manifest is gone (merged away meanwhile?)."""


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


def manifest_key(name: str) -> str:
    return f"{INDEX_PREFIX}{name}{MANIFEST_SUFFIX}"


def legacy_v3_key(name: str) -> str:
    return f"{INDEX_PREFIX}{name}{LEGACY_V3_SUFFIX}"


def _segment_dir(name: str, incarnation: str | None = None) -> str:
    return f"{SEGMENT_PREFIX}{name}/" + (f"{incarnation}/" if incarnation else "")


# -- segments -----------------------------------------------------------------
# A record is (seq, doc_id, text, metadata, vector); text is None for a
# tombstone (then metadata and vector are None too). seq is None in a segment
# a writer uploads: it is the seq the manifest gives that segment. `vector`
# is a (block, row) reference into a 2-D float32 array, so a segment's
# vectors stay one block and are gathered with one fancy-index per block,
# not stacked row by row (which made loading a 50k-doc index 2x slower).
#
# Blob: SEGMENT_MAGIC | u64 big-endian length of header JSON |
#   header JSON {"dim", "records": [[seq, doc_id, text, metadata], ...]} |
#   float32 little-endian vectors of the upsert records, in record order.


def _encode_segment(dim: int, records: list[tuple]) -> bytes:
    header = json.dumps(
        {"dim": dim, "records": [[seq, d, t, m] for seq, d, t, m, _ in records]}, separators=(",", ":")
    ).encode()
    refs = [v for _, _, t, _, v in records if t is not None]
    body = _gather(refs, dim).astype("<f4", copy=False).tobytes() if refs else b""
    return b"".join((SEGMENT_MAGIC, struct.pack(">Q", len(header)), header, body))


def _decode_segment(blob: bytes, key: str, dim: int, default_seq: int) -> list[tuple]:
    if not blob.startswith(SEGMENT_MAGIC):
        raise ValueError(f"{key} is not a recognised segment")
    pos = len(SEGMENT_MAGIC)
    (header_len,) = struct.unpack_from(">Q", blob, pos)
    pos += 8
    header = json.loads(blob[pos: pos + header_len])
    if header["dim"] != dim:
        raise ValueError(f"{key} has dimension {header['dim']}, the index {dim}")
    n_vectors = sum(1 for r in header["records"] if r[2] is not None)
    vectors = np.frombuffer(blob, dtype="<f4", offset=pos + header_len)
    if vectors.size != n_vectors * dim:
        raise ValueError(f"{key} is corrupt: {vectors.size} floats for {n_vectors} vectors of {dim}")
    block = vectors.reshape(n_vectors, dim).astype(np.float32, copy=False)
    out, i = [], 0
    for seq, doc_id, text, metadata in header["records"]:
        if text is None:
            out.append((seq if seq is not None else default_seq, doc_id, None, None, None))
        else:
            out.append((seq if seq is not None else default_seq, doc_id, text, metadata, (block, i)))
            i += 1
    return out


def _gather(refs: list[tuple[np.ndarray, int]], dim: int) -> np.ndarray:
    """Stack (block, row) vector references into one (len(refs), dim) array."""
    blocks: dict[int, tuple[int, np.ndarray]] = {}
    which = np.fromiter((blocks.setdefault(id(b), (len(blocks), b))[0] for b, _ in refs), np.int64, len(refs))
    rows = np.fromiter((r for _, r in refs), np.int64, len(refs))
    if len(blocks) == 1:
        (_, block), = blocks.values()
        if len(rows) == len(block) and np.array_equal(rows, np.arange(len(block))):
            return np.ascontiguousarray(block, dtype=np.float32)  # the whole block, in order
        return np.ascontiguousarray(block[rows], dtype=np.float32)
    out = np.empty((len(refs), dim), dtype=np.float32)
    for i, block in blocks.values():
        mask = which == i
        out[mask] = block[rows[mask]]
    return out


def _latest(records) -> dict[str, tuple]:
    """doc_id -> its record with the highest seq."""
    latest: dict[str, tuple] = {}
    for rec in records:
        cur = latest.get(rec[1])
        if cur is None or rec[0] > cur[0]:
            latest[rec[1]] = rec
    return latest


def _plan_merge(segments: list[dict]) -> tuple[int, int] | None:
    """Which run of segments [i, j) to merge after a write, if any.

    Size-tiered, like carrying in a binary counter: the run is the newest
    segments, extended while the one before them is no bigger than their
    total, and is merged once it has MIN_MERGE segments. With equal-sized
    writes that keeps O(log n + MIN_MERGE) segments. Runs are capped at
    MAX_SEGMENT_BYTES; past MAX_SEGMENTS segments, more are merged.
    """
    n = len(segments)
    if n < 2:
        return None
    sizes = [s["bytes"] for s in segments]
    i, total = n - 1, sizes[-1]
    while i > 0 and sizes[i - 1] <= total:
        i -= 1
        total += sizes[i]
    forced = n > MAX_SEGMENTS
    if forced:
        i = min(i, MAX_SEGMENTS // 2 - 1)
    while n - i >= 2 and sum(sizes[i:]) > MAX_SEGMENT_BYTES:
        i += 1
    return (i, n) if n - i >= (2 if forced else MIN_MERGE) else None


class _IndexState:
    """One index in memory: the FAISS index plus every document's fields.

    `docs` maps FAISS id -> {"doc_id", "text", "metadata"}. `by_id` is the
    reverse lookup, derived from `docs`.
    """

    __slots__ = ("by_id", "dim", "docs", "faiss")

    def __init__(self, dim: int):
        self.dim = dim
        self.faiss = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.docs: dict[int, dict] = {}
        self.by_id: dict[str, int] = {}

    def apply(self, latest: dict[str, tuple]):
        """Make each doc_id in `latest` ({doc_id: record}) match its record."""
        if not latest:
            return
        keys = {doc_id: object_key_id(doc_id) for doc_id in latest}
        if self.docs:
            gone = np.array([k for k in keys.values() if k in self.docs], dtype="int64")
            if gone.size:
                self.faiss.remove_ids(gone)
            for k in gone.tolist():
                del self.by_id[self.docs.pop(k)["doc_id"]]
        upserts = [rec for rec in latest.values() if rec[2] is not None]
        if upserts:
            self.faiss.add_with_ids(
                _gather([rec[4] for rec in upserts], self.dim),
                np.array([keys[rec[1]] for rec in upserts], dtype="int64"),
            )
            for _, doc_id, text, metadata, _ in upserts:
                self.docs[keys[doc_id]] = {"doc_id": doc_id, "text": text, "metadata": metadata}
                self.by_id[doc_id] = keys[doc_id]


class _Cached:
    """A loaded index: manifest, its ETag, and the state it describes.

    `applied` is the highest seq the state reflects: every record with a
    seq up to it has been applied, none after.
    """

    __slots__ = ("applied", "etag", "manifest", "state")

    def __init__(self, etag: str, manifest: dict, state: _IndexState, applied: int):
        self.etag, self.manifest, self.state, self.applied = etag, manifest, state, applied


def _new_manifest(dim: int, incarnation: str) -> dict:
    return {"format": FORMAT, "incarnation": incarnation, "dim": dim, "next_seq": 1, "segments": [], "writers": {}}


def _encode_manifest(manifest: dict) -> bytes:
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()


def _decode_manifest(blob: bytes, key: str) -> dict:
    try:
        manifest = json.loads(blob)
    except ValueError:
        raise ValueError(f"{key} is not a recognised index manifest") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"{key} is not a recognised index manifest")
    return manifest


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
    read and write the same bucket. A write uploads its records as a new,
    immutable segment, then appends the segment to the index's manifest
    with a conditional PUT (`If-Match` on the ETag it read). If another
    writer got there first the PUT is rejected, and the writer reloads,
    re-checks and re-appends the same segment, so no update is lost. The
    manifest assigns each append a sequence number, and a document's newest
    record wins, so concurrent writes to the *same* doc_id are ordered too:
    the last successful append wins, text, metadata and vector together.
    `self._lock` serialises the same within one process and guards the
    cached FAISS objects, which aren't safe to search while being mutated.

    The default index is never deleted, so its one-time migration from the
    old layout can't run twice. A deleted named index stays deleted: a
    writer that loses to the delete reloads, finds no index and fails with
    IndexNotFoundError rather than re-creating it.

    Caching. Each loaded index is kept in memory with its manifest's ETag.
    Every operation does a cheap HEAD on the manifest; when it changed, only
    the new manifest and the segments with records not yet seen are read.

    Compaction. `compaction` is "background" (default: merges run on a
    background thread after writes), "inline" (run before the write
    returns; for tests) or "off".
    """

    def __init__(
        self,
        client: Minio,
        embedder,
        bucket: str = BUCKET,
        io_workers: int = DEFAULT_IO_WORKERS,
        compaction: str = "background",
    ):
        if compaction not in ("background", "inline", "off"):
            raise ValueError("compaction must be 'background', 'inline' or 'off'")
        self._client = client
        self._embedder = embedder
        self._bucket = bucket
        self._lock = threading.Lock()
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="vectorstore-io")
        self._cache: dict[str, _Cached] = {}
        self._dim: int | None = None
        # Identifies this store's writes in manifests; see `_commit`.
        self._writer_id = uuid.uuid4().hex[:16]
        self._write_counter = 0
        self._compaction = compaction
        self._compactor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vectorstore-compact")
        self._compact_lock = threading.Lock()
        self._compact_idle = threading.Condition(self._compact_lock)
        self._compacting: set[str] = set()
        self._compact_again: set[str] = set()
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
            raise ValueError(f"object is {len(data)} bytes; a single PUT is capped at {MAX_SINGLE_PUT_BYTES}")
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

    def _get_if_exists(self, key: str) -> bytes | None:
        try:
            return self._get_bytes(key)
        except ObjectNotFoundError:
            return None

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

    def _delete_quietly(self, keys):
        """Best-effort cleanup: a failure only leaves an orphan for a later sweep."""
        for key in keys:
            try:
                self._delete_object(key)
            except Exception:
                logger.warning("could not delete %s; a later compaction will", key, exc_info=True)

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    def _embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed("dimension probe").shape[0])
        return self._dim

    # -- reading an index -----------------------------------------------------

    def _read_manifest(self, name: str) -> tuple[dict, str] | None:
        try:
            blob, etag = self._get_with_etag(manifest_key(name))
        except ObjectNotFoundError:
            return None
        return _decode_manifest(blob, manifest_key(name)), etag

    def _fetch(self, manifest: dict, entries: list[dict]) -> list[tuple]:
        """All records of `entries`, read in parallel. Raises _SegmentMissing."""

        def read(entry):
            try:
                blob = self._get_bytes(entry["key"])
            except ObjectNotFoundError:
                raise _SegmentMissing(entry["key"]) from None
            return _decode_segment(blob, entry["key"], manifest["dim"], entry["first_seq"])

        try:
            results = list(self._io.map(read, entries)) if len(entries) > 1 else [read(e) for e in entries]
        except RuntimeError:
            # The pool is shut down: the interpreter is exiting while a
            # background merge runs (say, at the end of a CLI command).
            # Finish it serially rather than fail it.
            results = [read(e) for e in entries]
        return [rec for recs in results for rec in recs]

    def _load(self, name: str, stale: bool = False) -> _Cached | None:
        """The index as currently stored, or None if it doesn't exist (yet).

        Must be called with `self._lock` held. Serves the cache when the
        manifest's ETag is unchanged; otherwise reads the manifest and only
        the records the cached state hasn't seen (or everything, for a new
        incarnation or after a full merge the cache can't follow). `stale`
        skips the HEAD when the caller knows the manifest changed.
        """
        key = manifest_key(name)
        missing_before = None
        while True:
            cached = self._cache.get(name)
            if not stale:
                etag = self._etag(key)
                if etag is None:
                    self._cache.pop(name, None)
                    return None
                if cached is not None and cached.etag == etag:
                    return cached
            stale = False
            got = self._read_manifest(name)
            if got is None:
                self._cache.pop(name, None)
                return None
            manifest, etag = got
            try:
                fresh = self._catch_up(name, cached, manifest, etag)
            except _SegmentMissing as e:
                # Merged away after we read the manifest: a newer manifest
                # no longer lists it. The same manifest listing it twice
                # means the segment is really gone.
                if missing_before == (etag, e.args[0]):
                    raise IndexCorruptError(f"index {name!r} lists segment {e.args[0]!r}, which doesn't exist") from None
                missing_before = (etag, e.args[0])
                continue
            self._cache[name] = fresh
            return fresh

    def _catch_up(self, name: str, cached: _Cached | None, manifest: dict, etag: str) -> _Cached:
        applied = manifest["next_seq"] - 1
        entries = manifest["segments"]
        if (
            cached is not None
            and cached.manifest["incarnation"] == manifest["incarnation"]
            and cached.applied <= applied
            # A full merge dropped tombstones the cache may not have seen.
            and not any(e["full"] and e["first_seq"] <= cached.applied < e["last_seq"] for e in entries)
        ):
            newer = self._fetch(manifest, [e for e in entries if e["last_seq"] > cached.applied])
            latest = _latest(r for r in newer if r[0] > cached.applied)
            # Fetching can't fail past this point; drop the cache entry while
            # mutating so a failure can't leave a half-applied state cached.
            self._cache.pop(name, None)
            state = cached.state
        else:
            with _gc_paused():
                latest = {d: r for d, r in _latest(self._fetch(manifest, entries)).items() if r[2] is not None}
                state = _IndexState(manifest["dim"])
                state.apply(latest)
            return _Cached(etag, manifest, state, applied)
        state.apply(latest)
        return _Cached(etag, manifest, state, applied)

    # -- creating an index (and migrating older layouts) -----------------

    def _ensure(self, name: str, stale: bool = False) -> _Cached:
        """The index, creating the default one (migrated, or empty) if needed.

        Must hold `self._lock`. Only ever *creates* (If-None-Match), so if two
        processes migrate at once exactly one result is kept.
        """
        for _ in range(INDEX_WRITE_ATTEMPTS):
            cached = self._load(name, stale)
            if cached is not None:
                return cached
            stale = False
            initial = self._initial_records(name)
            if initial is None:
                raise IndexNotFoundError(name)
            dim, records = initial
            try:
                self._create(name, dim, records)
            except _WriteConflict:
                continue  # someone else created it first: use theirs
        raise IndexWriteConflictError("index kept changing while being created; retry the operation")

    def _create(self, name: str, dim: int, records: list[tuple]):
        """Create the manifest, with `records` (seq None) as its first segment.

        Raises _WriteConflict if the index already exists.
        """
        manifest = _new_manifest(dim, uuid.uuid4().hex[:16])
        seg_key = None
        if records:
            blob = _encode_segment(dim, records)
            seg_key = f"{_segment_dir(name, manifest['incarnation'])}{uuid.uuid4().hex}.seg"
            self._put_conditional(seg_key, blob, if_match=None)
            manifest["segments"].append(
                {"key": seg_key, "first_seq": 1, "last_seq": 1, "bytes": len(blob), "full": True}
            )
            manifest["next_seq"] = 2
        try:
            self._put_conditional(manifest_key(name), _encode_manifest(manifest), if_match=None)
        except _WriteConflict:
            if seg_key:  # never referenced: the manifest already existed
                self._delete_quietly([seg_key])
            raise

    def _initial_records(self, name: str) -> tuple[int, list[tuple]] | None:
        """(dim, records) to create index `name` from; None if it doesn't exist."""
        blob = self._get_if_exists(legacy_v3_key(name))
        if blob is not None:
            return _decode_v3(blob, legacy_v3_key(name))
        if name != DEFAULT_INDEX:
            return None
        legacy = self._legacy_docs()
        if not legacy:
            return self._embedding_dim(), []
        dim = next(iter(legacy.values()))[0].shape[0]
        return dim, [(None, d, text, {}, (vec.reshape(1, -1), 0)) for d, (vec, text) in legacy.items()]

    def _legacy_docs(self) -> dict[str, tuple[np.ndarray, str]]:
        """{doc_id: (vector, text)} from a v1/v2 bucket; {} if there is none.

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

    # -- writing --------------------------------------------------------------

    def _commit(self, name: str, records: list[tuple], needed) -> bool:
        """Append `records` (seq None) to index `name` as one segment, with CAS.

        `needed(state)` is called on the current state before each attempt:
        it may raise to abort (nothing is written), or return False if the
        write has become a no-op. Returns True if the records are in the
        index when this returns, False if they turned out not to be needed.

        Retrying is safe because the segment is the same on every attempt
        and each attempt starts from a freshly loaded manifest. One case
        needs care: an append that *landed* but was reported as a conflict
        (the response was lost and urllib3's retry of the PUT hit our own
        new ETag). Re-appending would re-apply our records over any newer
        write to the same doc_ids, and deleting "our unused" segment would
        break the index. So every append also records this store's write
        counter in the manifest's "writers" map, which compaction keeps,
        and each attempt first checks whether that counter already landed.

        The segment is deleted only when it is known to be unreferenced: a
        manifest read after our last append attempt didn't contain it. If an
        append's outcome is unknown (an error other than a conflict), the
        segment stays; if nothing references it, a later sweep removes it.
        """
        self._write_counter += 1
        counter = self._write_counter
        mkey = manifest_key(name)
        seg_key, seg_blob, uploaded_at = None, None, 0.0
        unresolved = False  # an append whose outcome we haven't seen yet
        conflicted = False
        started = time.monotonic()
        try:
            for attempt in range(INDEX_WRITE_ATTEMPTS + 1):
                try:
                    cached = self._ensure(name, stale=conflicted)
                except IndexNotFoundError:
                    unresolved = False  # no manifest, so nothing references the segment
                    raise
                manifest = cached.manifest
                if manifest["writers"].get(self._writer_id, [0])[0] >= counter:
                    unresolved, seg_key = False, None  # it landed; the segment is in use
                    return True
                unresolved = False
                if attempt == INDEX_WRITE_ATTEMPTS:
                    break  # that was the final check, after the last attempt
                if not needed(cached.state):
                    return False
                if time.monotonic() - started > INDEX_WRITE_TIMEOUT_S:
                    break
                seg_dir = _segment_dir(name, manifest["incarnation"])
                stale = seg_key is not None and (
                    not seg_key.startswith(seg_dir) or time.monotonic() - uploaded_at > SEGMENT_MAX_PENDING_S
                )
                if stale:
                    self._delete_quietly([seg_key])  # known unreferenced: just checked
                    seg_key = None
                if seg_key is None:
                    if seg_blob is None:
                        seg_blob = _encode_segment(manifest["dim"], records)
                    seg_key = f"{seg_dir}{uuid.uuid4().hex}.seg"
                    self._put_conditional(seg_key, seg_blob, if_match=None)
                    uploaded_at = time.monotonic()
                seq = manifest["next_seq"]
                now = time.time()
                writers = {w: v for w, v in manifest["writers"].items() if v[1] >= now - WRITER_TTL_S}
                writers[self._writer_id] = [counter, now]
                new = {
                    **manifest,
                    "next_seq": seq + 1,
                    "segments": [
                        *manifest["segments"],
                        {"key": seg_key, "first_seq": seq, "last_seq": seq, "bytes": len(seg_blob), "full": False},
                    ],
                    "writers": writers,
                }
                unresolved = True
                try:
                    new_etag = self._put_conditional(mkey, _encode_manifest(new), if_match=cached.etag)
                except _WriteConflict:
                    conflicted = True
                    time.sleep(
                        random.uniform(0, min(INDEX_WRITE_BACKOFF_CAP_S, INDEX_WRITE_BACKOFF_BASE_S * 2 ** min(attempt, 30)))
                    )
                    continue
                unresolved, seg_key = False, None
                # The new manifest is exactly the old one plus our segment, so
                # the new state is the cached one plus our records. Drop the
                # entry while mutating in case applying fails half-way.
                self._cache.pop(name, None)
                cached.state.apply(_latest((seq, *r[1:]) for r in records))
                self._cache[name] = _Cached(new_etag, new, cached.state, seq)
                return True
            raise IndexWriteConflictError(
                f"index update kept losing to concurrent writers for {time.monotonic() - started:.0f} s "
                f"({attempt} attempts); retry the operation"
            )
        finally:
            if seg_key is not None and not unresolved:
                self._delete_quietly([seg_key])

    def _after_write(self, name: str):
        """Merge segments if the size rule says so (see `_plan_merge`)."""
        if self._compaction == "off":
            return
        with self._lock:
            cached = self._cache.get(name)
            if cached is None or _plan_merge(cached.manifest["segments"]) is None:
                return
        if self._compaction == "inline":
            self._compact_safely(name, full=False)
            return
        with self._compact_lock:
            if name in self._compacting:
                self._compact_again.add(name)  # run once more when the current merge ends
                return
            self._compacting.add(name)
        self._compactor.submit(self._compact_loop, name)

    def _compact_loop(self, name: str):
        while True:
            self._compact_safely(name, full=False)
            with self._compact_lock:
                if name in self._compact_again:
                    self._compact_again.discard(name)
                    continue
                self._compacting.discard(name)
                self._compact_idle.notify_all()
                return

    def _compact_safely(self, name: str, full: bool):
        try:
            self._compact(name, full)
        except Exception:
            logger.warning("compaction of index %r failed; a later write will retry it", name, exc_info=True)

    def wait_for_compaction(self, timeout: float | None = None) -> bool:
        """Block until no background merge is running. False on timeout."""
        with self._compact_lock:
            return self._compact_idle.wait_for(lambda: not self._compacting, timeout)

    def _compact(self, name: str, full: bool) -> bool:
        """Merge a run of segments into one (or, for `full`, all into parts).

        Doesn't hold `self._lock`: it only reads immutable segments and
        changes the manifest by CAS like any writer. The merged segments
        replace the run in the manifest only if that run is still there,
        unchanged; if another merge got to it first, ours is dropped.
        Returns True if a merge was committed.
        """
        got = self._read_manifest(name)
        if got is None:
            return False
        manifest, etag = got
        entries = manifest["segments"]
        run = (0, len(entries)) if full else _plan_merge(entries)
        if run is None or (full and len(entries) < 2 and all(e["full"] for e in entries)):
            if full:
                self._sweep(name, manifest)
            return False
        i, j = run
        merged = entries[i:j]
        try:
            latest = _latest(self._fetch(manifest, merged))
        except _SegmentMissing:
            return False  # merged away by someone else meanwhile
        is_full = i == 0
        if is_full:  # nothing older remains for a tombstone to hide
            latest = {d: r for d, r in latest.items() if r[2] is not None}
        first_seq, last_seq = min(e["first_seq"] for e in merged), max(e["last_seq"] for e in merged)
        parts = _split_parts(sorted(latest.values()), manifest["dim"])
        incarnation = manifest["incarnation"]
        seg_dir = _segment_dir(name, incarnation)
        new_entries, started = [], time.monotonic()
        for blob in parts:
            key = f"{seg_dir}{uuid.uuid4().hex}.seg"
            self._put_conditional(key, blob, if_match=None)
            new_entries.append(
                {"key": key, "first_seq": first_seq, "last_seq": last_seq, "bytes": len(blob), "full": is_full}
            )
        new_keys = {e["key"] for e in new_entries}
        old_keys = [e["key"] for e in merged]
        unresolved = False  # a manifest PUT whose outcome we haven't seen yet
        try:
            for _ in range(COMPACTION_ATTEMPTS):
                if time.monotonic() - started > SEGMENT_MAX_PENDING_S:
                    break
                entries = manifest["segments"]
                pos = _find_run([e["key"] for e in entries], old_keys)
                if manifest["incarnation"] != incarnation or pos is None or (is_full and pos != 0):
                    break  # the run changed under us: another merge got to it first
                new = {**manifest, "segments": [*entries[:pos], *new_entries, *entries[pos + len(old_keys):]]}
                unresolved = True
                try:
                    self._put_conditional(manifest_key(name), _encode_manifest(new), if_match=etag)
                except _WriteConflict:
                    got = self._read_manifest(name)
                    if got is None:
                        unresolved = False  # index deleted: nothing references our parts
                        break
                    manifest, etag = got
                    unresolved = False
                    if not new_keys & {e["key"] for e in manifest["segments"]}:
                        continue
                    # It landed after all (the response was lost). If our parts
                    # aren't listed any more they were merged again, which
                    # also means the run is gone; either way it's committed.
                new_keys = set()
                self._delete_quietly(old_keys)  # readers that still list them reload
                if is_full:
                    self._sweep(name, new)
                return True
            return False
        finally:
            if new_keys and not unresolved:
                self._delete_quietly(new_keys)

    def _sweep(self, name: str, manifest: dict):
        """Delete this index's segments that `manifest` doesn't reference and
        that are older than ORPHAN_GRACE_S (see the constants)."""
        referenced = {e["key"] for e in manifest["segments"]}
        cutoff = datetime.now(UTC).timestamp() - ORPHAN_GRACE_S
        orphans = [
            obj.object_name
            for obj in self._client.list_objects(self._bucket, prefix=_segment_dir(name), recursive=True)
            if obj.object_name not in referenced
            and obj.last_modified is not None
            and obj.last_modified.timestamp() < cutoff
        ]
        self._delete_quietly(orphans)

    # -- indexes --------------------------------------------------------------

    def list_indexes(self) -> list[str]:
        names = {DEFAULT_INDEX}
        for obj in self._client.list_objects(self._bucket, prefix=INDEX_PREFIX):
            for suffix in (MANIFEST_SUFFIX, LEGACY_V3_SUFFIX):
                if obj.object_name.endswith(suffix):
                    names.add(obj.object_name[len(INDEX_PREFIX): -len(suffix)])
        return sorted(names)

    def create_index(self, name: str) -> None:
        validate_index_name(name)
        if name == DEFAULT_INDEX or self._etag(legacy_v3_key(name)) is not None:
            raise IndexAlreadyExistsError(name)
        with self._lock:
            try:
                self._create(name, self._embedding_dim(), [])
            except _WriteConflict:
                raise IndexAlreadyExistsError(name) from None

    def delete_index(self, name: str) -> None:
        """Delete a named index and its segments.

        The legacy v3 object goes first, so a concurrent reader can't
        migrate the index back from it once the manifest is gone.
        """
        validate_index_name(name)
        if name == DEFAULT_INDEX:
            raise ValueError("the default index can't be deleted")
        with self._lock:
            had_v3 = self._etag(legacy_v3_key(name)) is not None
            if had_v3:
                self._delete_object(legacy_v3_key(name))
            got = self._read_manifest(name)
            if got is None and not had_v3:
                raise IndexNotFoundError(name)
            self._cache.pop(name, None)
            if got is None:
                return
            self._delete_object(manifest_key(name))
            # Only this incarnation's segments: a newer index of the same
            # name, created since, has its own prefix.
            prefix = _segment_dir(name, got[0]["incarnation"])
            self._delete_quietly(
                [o.object_name for o in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)]
            )

    def compact(self, index: str = DEFAULT_INDEX) -> dict:
        """Merge all of an index's segments, dropping overwritten and deleted
        records, and delete orphaned segments. Returns `index_stats` after."""
        validate_index_name(index)
        with self._lock:
            self._ensure(index)
        self._compact(index, full=True)
        return self.index_stats(index)

    def index_stats(self, index: str = DEFAULT_INDEX) -> dict:
        """{"documents", "segments", "bytes"} (bytes: all segments, live or not)."""
        validate_index_name(index)
        with self._lock:
            cached = self._ensure(index)
            entries = cached.manifest["segments"]
            return {
                "documents": len(cached.state.docs),
                "segments": len(entries),
                "bytes": sum(e["bytes"] for e in entries),
            }

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
        records = [(None, d, batch[d][0], batch[d][1], (vecs, i)) for i, d in enumerate(doc_ids)]

        def check(state: _IndexState) -> bool:
            if vecs.shape[1] != state.dim:
                raise ValueError(f"embedding dimension {vecs.shape[1]} doesn't match index {index!r} ({state.dim})")
            for doc_id, key in zip(doc_ids, keys):
                owner = state.docs.get(key)
                if owner is not None and owner["doc_id"] != doc_id:
                    raise ValueError(f"doc_id {doc_id!r} collides with {owner['doc_id']!r}; use another doc_id")
            return True

        with self._lock:
            self._commit(index, records, check)
        self._after_write(index)
        return len(doc_ids)

    def get_document(self, doc_id: str, index: str = DEFAULT_INDEX):
        """{"doc_id", "text", "metadata"}, or None if there's no such document."""
        validate_index_name(index)
        with self._lock:
            state = self._ensure(index).state
            key = state.by_id.get(doc_id)
            return None if key is None else _public(state.docs[key])

    def list_documents(self, skip: int = 0, limit: int = 100, index: str = DEFAULT_INDEX):
        if skip < 0:
            raise ValueError("skip must not be negative")
        if not 0 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 0 and {MAX_PAGE_SIZE}")
        validate_index_name(index)
        with self._lock:
            state = self._ensure(index).state
            page = sorted(state.by_id)[skip: skip + limit]
            return [
                {"doc_id": d, "text_preview": doc["text"][:64], "metadata": _copy_metadata(doc["metadata"])}
                for d in page
                for doc in (state.docs[state.by_id[d]],)
            ]

    def delete_document(self, doc_id: str, index: str = DEFAULT_INDEX) -> bool:
        """Delete a document; False if it wasn't there.

        True once *any* attempt found the document: from then on its removal
        is guaranteed. (If another process deleted it concurrently, both
        report True, as idempotent deletes do.)
        """
        validate_index_name(index)
        found = False

        def check(state: _IndexState) -> bool:
            nonlocal found
            present = doc_id in state.by_id
            found = found or present
            return present

        with self._lock:
            self._commit(index, [(None, doc_id, None, None, None)], check)
        self._after_write(index)
        return found

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
            state = self._ensure(index).state
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
            return len(self._ensure(index).state.docs)

    def embedding_model_name(self) -> str:
        return MODEL_NAME


_gc_pause_lock = threading.Lock()
_gc_pauses = 0


@contextlib.contextmanager
def _gc_paused():
    """Pause the cyclic GC while building a large index in memory.

    A rebuild allocates a few objects per document (records, dicts) and no
    reference cycles; without this the collector repeatedly walks them all,
    which made loading a 50k-doc index ~40% slower. Counted, so concurrent
    rebuilds don't re-enable it early, and only re-enabled if it was on.
    """
    global _gc_pauses
    with _gc_pause_lock:
        if _gc_pauses == 0:
            _gc_paused.was_enabled = gc.isenabled()
            gc.disable()
        _gc_pauses += 1
    try:
        yield
    finally:
        with _gc_pause_lock:
            _gc_pauses -= 1
            if _gc_pauses == 0 and _gc_paused.was_enabled:
                gc.enable()


def _find_run(keys: list[str], run: list[str]) -> int | None:
    """Position of `run` as a contiguous slice of `keys`, or None."""
    if not run or run[0] not in keys:
        return None
    pos = keys.index(run[0])
    return pos if keys[pos: pos + len(run)] == run else None


def _split_parts(records: list[tuple], dim: int) -> list[bytes]:
    """Encode records as segments of at most ~MAX_SEGMENT_BYTES each."""
    parts, current, size = [], [], 0
    for rec in records:
        rec_size = len(rec[1]) + len(rec[2] or "") + len(json.dumps(rec[3])) + 4 * dim + 32
        if current and size + rec_size > MAX_SEGMENT_BYTES:
            parts.append(_encode_segment(dim, current))
            current, size = [], 0
        current.append(rec)
        size += rec_size
    if current or not parts:
        parts.append(_encode_segment(dim, current))
    return parts


def _decode_v3(blob: bytes, key: str) -> tuple[int, list[tuple]]:
    """(dim, records) from a v3 single-object index."""
    if not blob.startswith(LEGACY_V3_MAGIC):
        raise ValueError(f"{key} is not a recognised index blob")
    pos = len(LEGACY_V3_MAGIC)
    (header_len,) = struct.unpack_from(">Q", blob, pos)
    pos += 8
    header = json.loads(blob[pos: pos + header_len])
    index = faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8, offset=pos + header_len))
    if index.ntotal != len(header["docs"]):
        raise ValueError(f"{key} is corrupt: {index.ntotal} vectors but {len(header['docs'])} documents")
    dim = header["dim"]
    if not header["docs"]:
        return dim, []
    vectors = index.index.reconstruct_n(0, index.ntotal)
    row = {int(k): i for i, k in enumerate(faiss.vector_to_array(index.id_map))}
    return dim, [(None, d, t, m, (vectors, row[int(k)])) for k, d, t, m in header["docs"]]


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
