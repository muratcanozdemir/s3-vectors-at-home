"""Consistent backups of a bucket's indexes to a local directory, and restores.

A backup is a directory holding each object it copied at its object key
(`indexes/papers.manifest.json`, `segments/papers/<incarnation>/<id>.seg`,
...) plus `backup.json`, a catalog written last: a directory without one
is an unfinished backup. The catalog lists every file with its size and
MD5, so a restore can check the copy before it writes anything.

Consistency without pausing writers. Segments are immutable, and the only
object ever rewritten is an index's manifest. So each index is backed up by
reading its manifest, copying every segment that manifest lists, then
saving that same manifest: the copy is the index exactly as it was when the
manifest was read, whatever happens meanwhile. If compaction deletes a
listed segment before it's copied, the manifest is read again and the
copy continues from the new one (segments already copied are kept).

Restores never overwrite unless asked. Every file is checked first; then,
for each index, segments are uploaded (If-None-Match) and the manifest last,
also If-None-Match, so a running server never sees an index whose segments
aren't there yet. With `replace`, an existing index's manifest is swapped
for the backup's by compare-and-swap; servers reload it (it's a new
incarnation), and compaction later deletes the old segments.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vectorstore import core
from vectorstore.s3vectors import REGISTRY_PREFIX, ROOT

CATALOG = "backup.json"
CATALOG_FORMAT = 1
COPY_ATTEMPTS = 20
IO_WORKERS = 8


class BackupError(RuntimeError):
    """A backup or restore that can't go ahead; nothing (more) was written."""


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # a file is complete or absent, never half-written


class _Bucket:
    """The object-store operations backup and restore need, on one bucket."""

    def __init__(self, client, bucket: str):
        # A VectorStore with no indexes of its own: for its conditional PUT
        # and its startup check that the store enforces conditions.
        self.store = core.VectorStore(client, None, bucket=bucket, default_index=None, compaction="off")
        self.client, self.bucket = client, bucket

    def keys(self, prefix: str, recursive: bool = False) -> list[str]:
        return [o.object_name for o in self.client.list_objects(self.bucket, prefix=prefix, recursive=recursive)
                if not getattr(o, "is_dir", False) and not o.object_name.endswith("/")]

    def get(self, key: str) -> tuple[bytes, str] | None:
        try:
            return self.store._get_with_etag(key)
        except core.ObjectNotFoundError:
            return None

    def etag(self, key: str) -> str | None:
        return self.store._etag(key)

    def put_new(self, key: str, data: bytes) -> bool:
        """PUT only if absent. False if the key exists already."""
        try:
            self.store._put_conditional(key, data, if_match=None)
        except core._WriteConflict:
            return False
        return True

    def put_replacing(self, key: str, data: bytes, etag: str) -> bool:
        try:
            self.store._put_conditional(key, data, if_match=etag)
        except core._WriteConflict:
            return False
        return True


def _index_name(manifest_key: str) -> str:
    """Display name: "papers" (text API) or "media/movies" (S3 Vectors bucket/index)."""
    name = manifest_key.removesuffix(core.MANIFEST_SUFFIX)
    if name.startswith(ROOT):
        bucket, _, index = name[len(ROOT):].partition("/" + core.INDEX_PREFIX)
        return f"{bucket}/{index}"
    return name.removeprefix(core.INDEX_PREFIX)


def _manifest_keys(b: _Bucket, vector_buckets: list[str]) -> list[str]:
    prefixes = [core.INDEX_PREFIX] + [f"{ROOT}{vb}/{core.INDEX_PREFIX}" for vb in vector_buckets]
    return sorted(k for p in prefixes for k in b.keys(p) if k.endswith(core.MANIFEST_SUFFIX))


def _legacy_objects(b: _Bucket) -> list[str]:
    """Objects of layouts older than manifests that haven't been migrated
    yet, so the backup can't include their data. (Once an index has a
    manifest, its old objects are just leftovers: nothing reads them.)"""
    index_keys = set(b.keys(core.INDEX_PREFIX))
    legacy = [
        k for k in sorted(index_keys)
        if k.endswith(core.LEGACY_V3_SUFFIX)
        and core.manifest_key(k[len(core.INDEX_PREFIX): -len(core.LEGACY_V3_SUFFIX)]) not in index_keys
    ]
    if core.manifest_key(core.DEFAULT_INDEX) not in index_keys and core.legacy_v3_key(core.DEFAULT_INDEX) not in legacy:
        legacy += [k for k in b.keys("") if k in (core.LEGACY_V2_INDEX_OBJ, "faiss.index", "index.ids.json")]
    return sorted(legacy)


def backup(client, bucket: str, dest, log=print) -> dict:
    """Back up every index in `bucket` (text API and S3 Vectors) to the
    directory `dest`, which must not exist or be empty. Returns the catalog."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise BackupError(f"{dest} isn't empty: back up into a new directory")
    dest.mkdir(parents=True, exist_ok=True)
    b = _Bucket(client, bucket)
    files: dict[str, dict] = {}

    def save(key: str, data: bytes) -> None:
        _write_file(dest / key, data)
        files[key] = {"bytes": len(data), "md5": _md5(data)}

    # S3 Vectors buckets: their registry entries (skipping ones being deleted).
    vector_buckets = []
    for key in sorted(b.keys(REGISTRY_PREFIX)):
        got = b.get(key)
        if got is None:
            continue
        try:
            state = json.loads(got[0]).get("state")
        except ValueError:
            raise BackupError(f"the S3 Vectors registry entry {key!r} is unreadable") from None
        if state == "deleting":
            continue
        save(key, got[0])
        vector_buckets.append(key[len(REGISTRY_PREFIX): -len(".json")])

    indexes = []
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for mkey in _manifest_keys(b, vector_buckets):
            entry = _backup_index(b, mkey, dest, files, pool)
            if entry is not None:
                indexes.append(entry)
                log(f"{entry['name']}: {len(entry['segments'])} segment(s), {entry['bytes']:,} bytes")

    legacy = _legacy_objects(b)
    if legacy:
        log(
            f"warning: not backed up, indexes in an older layout not yet migrated: {', '.join(legacy)}. "
            "Use them once (any request does), then back up again."
        )
    catalog = {
        "format": CATALOG_FORMAT,
        "created": time.time(),
        "bucket": bucket,
        "vector_buckets": vector_buckets,
        "indexes": indexes,
        "files": files,
        "skipped_legacy": legacy,
    }
    _write_file(dest / CATALOG, json.dumps(catalog, indent=1, sort_keys=True).encode())
    return catalog


def _backup_index(b: _Bucket, mkey: str, dest: Path, files: dict, pool) -> dict | None:
    """Copy one index as of one manifest version; None if it was deleted meanwhile."""
    copied: set[str] = set()
    for _ in range(COPY_ATTEMPTS):
        got = b.get(mkey)
        if got is None:
            return None  # deleted since it was listed
        blob = got[0]
        manifest = core._decode_manifest(blob, mkey)
        needed = [e["key"] for e in manifest["segments"] if e["key"] not in copied]

        def copy(key: str) -> bool:
            seg = b.get(key)
            if seg is None:
                return False
            _write_file(dest / key, seg[0])
            files[key] = {"bytes": len(seg[0]), "md5": _md5(seg[0])}
            return True

        results = list(pool.map(copy, needed))
        copied.update(k for k, ok in zip(needed, results) if ok)
        if not all(results):
            continue  # merged away meanwhile: a newer manifest lists what replaced it
        _write_file(dest / mkey, blob)
        files[mkey] = {"bytes": len(blob), "md5": _md5(blob)}
        segments = [e["key"] for e in manifest["segments"]]
        return {
            "name": _index_name(mkey),
            "manifest": mkey,
            "segments": segments,
            "bytes": sum(files[k]["bytes"] for k in segments),
        }
    raise BackupError(f"index {_index_name(mkey)!r} kept changing under the backup; try again")


def _read_catalog(src: Path) -> dict:
    try:
        catalog = json.loads((src / CATALOG).read_bytes())
    except FileNotFoundError:
        raise BackupError(f"{src} has no {CATALOG}: not a backup, or an unfinished one") from None
    except ValueError:
        raise BackupError(f"{src / CATALOG} is unreadable") from None
    if catalog.get("format") != CATALOG_FORMAT:
        raise BackupError(f"{src / CATALOG} is format {catalog.get('format')!r}; this version reads {CATALOG_FORMAT}")
    return catalog


def restore(client, bucket: str, src, only=None, replace: bool = False, dry_run: bool = False, log=print) -> list[str]:
    """Restore the indexes in backup `src` into `bucket`; `only`: just these
    names (as `backup` logs them). Returns the names restored. Raises
    BackupError, writing nothing, if a file is damaged or missing, a name
    isn't in the backup, or an index exists already (without `replace`)."""
    src = Path(src)
    catalog = _read_catalog(src)
    by_name = {e["name"]: e for e in catalog["indexes"]}
    wanted = list(by_name) if not only else list(dict.fromkeys(only))
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise BackupError(f"not in this backup: {', '.join(unknown)} (it has: {', '.join(by_name) or 'nothing'})")
    entries = [by_name[n] for n in wanted]

    # 1. Every file is there and intact, before anything is written.
    # (Checked by streaming; each is read again only to upload it.)
    if only:
        vector_buckets = sorted({e["name"].split("/")[0] for e in entries if "/" in e["name"]})
    else:
        vector_buckets = list(catalog.get("vector_buckets", []))  # empty ones too
    keys = [k for e in entries for k in (*e["segments"], e["manifest"])]
    keys += [f"{REGISTRY_PREFIX}{vb}.json" for vb in vector_buckets]
    for key in keys:
        _verify(src, key, catalog["files"].get(key))

    def data(key: str) -> bytes:
        blob = (src / key).read_bytes()
        if _md5(blob) != catalog["files"][key]["md5"]:
            raise BackupError(f"{src / key} changed during the restore")
        return blob

    # 2. No clashes with what the bucket holds now.
    b = _Bucket(client, bucket)
    existing = {e["manifest"]: b.etag(e["manifest"]) for e in entries}
    clashes = [e["name"] for e in entries if existing[e["manifest"]] is not None]
    if clashes and not replace:
        raise BackupError(f"these indexes exist already: {', '.join(clashes)} (pass --replace to overwrite them)")
    for vb in vector_buckets:
        got = b.get(f"{REGISTRY_PREFIX}{vb}.json")
        if got is not None and json.loads(got[0]).get("state") == "deleting":
            raise BackupError(f"vector bucket {vb!r} is being deleted; restore once it's gone")
    if dry_run:
        for e in entries:
            log(f"would restore {e['name']}" + (" (replacing it)" if e["name"] in clashes else ""))
        return [e["name"] for e in entries]

    # 3. Write: vector buckets, then per index its segments, then its manifest.
    for vb in vector_buckets:
        key = f"{REGISTRY_PREFIX}{vb}.json"
        if b.put_new(key, data(key)):
            log(f"created vector bucket {vb}")
    restored = []
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for e in entries:
            def upload(key: str) -> None:
                blob = data(key)
                if not b.put_new(key, blob) and b.etag(key) != _md5(blob):
                    raise BackupError(f"{key} exists in the bucket with different content")

            list(pool.map(upload, e["segments"]))
            mkey, etag = e["manifest"], existing[e["manifest"]]
            blob = data(mkey)
            ok = b.put_new(mkey, blob) if etag is None else b.put_replacing(mkey, blob, etag)
            if not ok:
                raise BackupError(
                    f"index {e['name']!r} changed while restoring (another process wrote to it); "
                    f"restored so far: {', '.join(restored) or 'none'}"
                )
            restored.append(e["name"])
            log(f"restored {e['name']}" + (" (replaced)" if etag is not None else ""))
    return restored


def _verify(src: Path, key: str, info: dict | None) -> None:
    if info is None:
        raise BackupError(f"{CATALOG} doesn't list {key}")
    h, size = hashlib.md5(), 0
    try:
        with open(src / key, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
                size += len(chunk)
    except FileNotFoundError:
        raise BackupError(f"{src / key} is missing from the backup") from None
    if size != info["bytes"] or h.hexdigest() != info["md5"]:
        raise BackupError(f"{src / key} is damaged (its size or checksum doesn't match {CATALOG})")
