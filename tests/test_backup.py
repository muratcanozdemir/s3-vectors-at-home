"""Backups to a directory and restores (vectorstore.backup), on the fake object store."""

import json

import numpy as np
import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore import backup as bk
from vectorstore import core
from vectorstore.core import BUCKET, VectorStore
from vectorstore.s3vectors import S3VectorsService

QUIET = {"log": lambda *_: None}


def _text_store(client):
    return VectorStore(client, FakeEmbedder(window=4), compaction="inline", graph_build="inline")


def _fill(client):
    """A bucket with text indexes (one chunked, one approximate) and S3 Vectors data."""
    store = _text_store(client)
    store.add_documents([(f"d{i}", f"default doc {i}", {"n": i}) for i in range(10)])
    store.create_index("papers")
    store.add_documents([("long", " ".join(f"w{j}" for j in range(20)), {"k": 1}), ("short", "tiny", {})], index="papers")
    store.delete_document("short", index="papers")
    store.create_index("ann", search={"mode": "approximate"})
    store.add_documents([(f"a{i}", f"ann doc {i}", {}) for i in range(5)], index="ann")
    svc = S3VectorsService(client, compaction="inline", graph_build="inline")
    for vb in ("media", "empty"):
        svc.create_vector_bucket({"vectorBucketName": vb})
    svc.create_index({"vectorBucketName": "media", "indexName": "movies", "dataType": "float32", "dimension": 3,
                      "distanceMetric": "cosine"})
    svc.put_vectors({"vectorBucketName": "media", "indexName": "movies",
                     "vectors": [{"key": "alien", "data": {"float32": [0.2, 0.8, 0.0]}, "metadata": {"year": 1979}}]})
    return store, svc


def _snapshot(client):
    """What the bucket's indexes answer, to compare a restored bucket with."""
    store = _text_store(client)
    svc = S3VectorsService(client, compaction="off")
    return {
        "indexes": store.list_indexes(),
        "docs": {i: [store.get_document(d["doc_id"], index=i) for d in store.list_documents(index=i)]
                 for i in store.list_indexes()},
        "search": store.search_with_scores("w5 w6 w7 w8", top_k=3, index="papers", passages=True),
        "info": {i: store.index_info(i)["search"] for i in store.list_indexes()},
        "vector_buckets": sorted(b["vectorBucketName"] for b in svc.list_vector_buckets({})["vectorBuckets"]),
        "movies": svc.get_vectors({"vectorBucketName": "media", "indexName": "movies", "keys": ["alien"],
                                   "returnData": True, "returnMetadata": True}),
    }


def test_round_trip_restores_every_index_exactly(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    catalog = bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    assert sorted(e["name"] for e in catalog["indexes"]) == ["ann", "default", "media/movies", "papers"]
    assert catalog["vector_buckets"] == ["empty", "media"]
    target = FakeMinioClient()
    restored = bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    assert sorted(restored) == ["ann", "default", "media/movies", "papers"]
    assert _snapshot(target) == _snapshot(source)


def test_the_catalog_is_written_last_and_lists_every_file(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    catalog = json.loads((tmp_path / "b" / bk.CATALOG).read_text())
    on_disk = {str(p.relative_to(tmp_path / "b")) for p in (tmp_path / "b").rglob("*") if p.is_file()}
    assert on_disk == set(catalog["files"]) | {bk.CATALOG}
    assert not any(name.endswith(".tmp") for name in on_disk)


def test_a_segment_merged_away_mid_backup_is_followed_to_the_new_manifest(tmp_path, monkeypatch):
    source = FakeMinioClient()
    store = VectorStore(source, FakeEmbedder(), compaction="off")
    for i in range(6):  # six separate segments
        store.add_document(f"d{i}", f"doc {i}")
    assert store.index_stats()["segments"] == 6
    want = sorted(d["doc_id"] for d in store.list_documents())
    real_get = bk._Bucket.get
    fetched, compacted = [], []

    def get(self, key):
        if key.endswith(".seg") and not compacted:
            compacted.append(key)
            store.compact()  # merges every segment and deletes the originals
        got = real_get(self, key)
        fetched.append((key, got is not None))
        return got

    monkeypatch.setattr(bk._Bucket, "get", get)
    catalog = bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    assert (compacted[0], False) in fetched  # the first segment really was gone when fetched...
    manifest_reads = [k for k, _ in fetched if k.endswith(core.MANIFEST_SUFFIX)]
    assert len(manifest_reads) == 2  # ...so the manifest was read again
    [entry] = [e for e in catalog["indexes"] if e["name"] == "default"]
    assert len(entry["segments"]) == 1  # the compacted manifest's one segment
    target = FakeMinioClient()
    bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    assert sorted(d["doc_id"] for d in _text_store(target).list_documents()) == want


def test_writes_during_a_backup_dont_tear_an_index(tmp_path, monkeypatch):
    """The copy is the index as of one manifest: all of a concurrent write, or none of it."""
    source = FakeMinioClient()
    store = _text_store(source)
    store.add_documents([(f"d{i}", f"doc {i}") for i in range(5)])
    real_get = bk._Bucket.get
    written = []

    def get(self, key):
        got = real_get(self, key)
        if key.endswith(".seg") and not written:
            written.append(1)
            store.add_documents([("late-1", "late"), ("late-2", "late")])
            store.delete_document("d0")
        return got

    monkeypatch.setattr(bk._Bucket, "get", get)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    assert sorted(d["doc_id"] for d in _text_store(target).list_documents()) == [f"d{i}" for i in range(5)]


def test_restore_refuses_existing_indexes_and_writes_nothing(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    _text_store(target).add_document("mine", "keep me")  # a server already created "default"
    before = {k: v for k, v in target.objects[BUCKET].items() if not k.startswith(core.PROBE_PREFIX)}
    with pytest.raises(bk.BackupError, match="exist already: default"):
        bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    after = {k: v for k, v in target.objects[BUCKET].items() if not k.startswith(core.PROBE_PREFIX)}
    assert after == before


def test_replace_swaps_an_index_that_a_running_store_then_reloads(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    running = _text_store(target)
    running.add_document("mine", "replaced soon")
    assert running.get_document("mine") is not None
    bk.restore(target, BUCKET, tmp_path / "b", only=["default"], replace=True, **QUIET)
    assert running.get_document("mine") is None
    assert running.count_documents() == 10
    running.add_document("after", "writes still work")
    assert running.count_documents() == 11


def test_only_restores_the_named_indexes_and_their_vector_buckets(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    assert bk.restore(target, BUCKET, tmp_path / "b", only=["media/movies"], **QUIET) == ["media/movies"]
    svc = S3VectorsService(target, compaction="off")
    assert [b["vectorBucketName"] for b in svc.list_vector_buckets({})["vectorBuckets"]] == ["media"]
    assert _text_store(target).list_indexes() == ["default"]  # only the always-present default
    with pytest.raises(bk.BackupError, match="not in this backup: nope"):
        bk.restore(target, BUCKET, tmp_path / "b", only=["nope"], **QUIET)


def test_dry_run_checks_and_writes_nothing(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    lines = []
    bk.restore(target, BUCKET, tmp_path / "b", dry_run=True, log=lines.append)
    assert "would restore papers" in lines
    assert not [k for k in target.objects.get(BUCKET, {}) if not k.startswith(core.PROBE_PREFIX)]


@pytest.mark.parametrize("damage", ["truncate", "flip", "delete"])
def test_a_damaged_backup_is_refused_before_anything_is_written(tmp_path, damage):
    source = FakeMinioClient()
    _fill(source)
    catalog = bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    seg = next(k for k in catalog["files"] if k.endswith(".seg"))
    path = tmp_path / "b" / seg
    if damage == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "flip":
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(bytes(blob))
    else:
        path.unlink()
    target = FakeMinioClient()
    with pytest.raises(bk.BackupError, match="damaged|missing"):
        bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    assert not [k for k in target.objects.get(BUCKET, {}) if not k.startswith(core.PROBE_PREFIX)]


def test_unfinished_or_foreign_directories_are_refused(tmp_path):
    with pytest.raises(bk.BackupError, match="no backup.json"):
        bk.restore(FakeMinioClient(), BUCKET, tmp_path, **QUIET)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "file").write_text("hi")
    with pytest.raises(bk.BackupError, match="isn't empty"):
        bk.backup(FakeMinioClient(), BUCKET, tmp_path / "x", **QUIET)
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / bk.CATALOG).write_text(json.dumps({"format": 99}))
    with pytest.raises(bk.BackupError, match="format 99"):
        bk.restore(FakeMinioClient(), BUCKET, tmp_path / "y", **QUIET)


def test_vector_buckets_being_deleted_are_left_out(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    key = f"{bk.REGISTRY_PREFIX}empty.json"
    entry = json.loads(source.objects[BUCKET][key])
    source.objects[BUCKET][key] = json.dumps({**entry, "state": "deleting"}).encode()
    catalog = bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    assert catalog["vector_buckets"] == ["media"]


def test_unmigrated_older_layouts_are_reported_not_copied(tmp_path):
    source = FakeMinioClient()
    _text_store(source).add_document("x", "y")  # default: migrated, has a manifest
    objects = source.objects[BUCKET]
    objects[core.legacy_v3_key("old")] = b"S3VAH-IDX\x03..."  # never used since the upgrade
    objects[core.legacy_v3_key("default")] = b"leftover"  # default's, already migrated
    objects[core.LEGACY_V2_INDEX_OBJ] = b"leftover"  # likewise
    lines = []
    catalog = bk.backup(source, BUCKET, tmp_path / "b", log=lines.append)
    assert catalog["skipped_legacy"] == ["indexes/old.v3.bin"]
    assert any("not yet migrated: indexes/old.v3.bin" in line for line in lines)


def test_an_unmigrated_default_index_is_reported():
    client = FakeMinioClient()
    b = bk._Bucket(client, BUCKET)
    client.objects[BUCKET]["faiss.index"] = b"v1"
    client.objects[BUCKET]["index.ids.json"] = b"[]"
    assert bk._legacy_objects(b) == ["faiss.index", "index.ids.json"]


def test_a_restored_s3vectors_index_answers_queries(tmp_path):
    source = FakeMinioClient()
    _fill(source)
    bk.backup(source, BUCKET, tmp_path / "b", **QUIET)
    target = FakeMinioClient()
    bk.restore(target, BUCKET, tmp_path / "b", **QUIET)
    svc = S3VectorsService(target, compaction="off")
    got = svc.query_vectors({"vectorBucketName": "media", "indexName": "movies", "topK": 1,
                             "queryVector": {"float32": list(np.array([0.2, 0.8, 0.0]))}, "returnMetadata": True})
    assert got["vectors"][0]["key"] == "alien" and got["vectors"][0]["metadata"] == {"year": 1979}
