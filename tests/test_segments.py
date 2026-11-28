"""The segmented index layout: write cost, compaction, catch-up, and the
failure cases that make it safe across processes.

Several VectorStores over one FakeMinioClient stand in for several
processes: separate locks and caches, coordinated only through the object
store. Hooks on the fake client inject the races at exact points.
"""

import json
import math
import random
import struct
from datetime import UTC, datetime, timedelta

import faiss
import numpy as np
import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient, _s3_error
from vectorstore import core
from vectorstore.core import (
    DEFAULT_INDEX,
    IndexAlreadyExistsError,
    IndexCorruptError,
    IndexNotFoundError,
    IndexWriteConflictError,
    VectorStore,
    manifest_key,
)

MANIFEST = manifest_key(DEFAULT_INDEX)


def _store(client, compaction="inline"):
    return VectorStore(client, FakeEmbedder(), compaction=compaction)


def _manifest(client, name=DEFAULT_INDEX):
    return json.loads(client.objects["vectors"][manifest_key(name)])


def _segment_keys(client, name=DEFAULT_INDEX):
    return {k for k in client.objects["vectors"] if k.startswith(f"segments/{name}/")}


def _referenced(client, name=DEFAULT_INDEX):
    return {e["key"] for e in _manifest(client, name)["segments"]}


def _assert_no_orphans(client, name=DEFAULT_INDEX):
    assert _segment_keys(client, name) == _referenced(client, name)


def _docs(store, index=DEFAULT_INDEX):
    return {d["doc_id"]: store.get_document(d["doc_id"], index=index)["text"] for d in store.list_documents(limit=1000, index=index)}


def _hook_puts(client, before=None, after=None):
    """Call before(key, n)/after(key, n) around each PUT; n counts PUTs to that key."""
    real = client._put_object
    counts = {}

    def wrapped(bucket, key, data, headers):
        n = counts.get(key, 0)
        counts[key] = n + 1
        if before:
            before(key, n)
        result = real(bucket, key, data, headers)
        if after:
            after(key, n)
        return result

    client._put_object = wrapped
    return real


# -- write cost and segment count ----------------------------------------------


def test_write_cost_does_not_grow_with_index_size():
    client = FakeMinioClient()
    store = _store(client, compaction="off")
    store.add_documents([(f"d{i}", f"text {i}") for i in range(2000)])
    sizes = {}
    _hook_puts(client, before=lambda key, n: None)
    real = client._put_object

    def measure(bucket, key, data, headers):
        sizes[key] = len(data)
        return real(bucket, key, data, headers)

    client._put_object = measure
    store.add_document("one-more", "small")
    seg = [v for k, v in sizes.items() if k.endswith(".seg")]
    assert len(seg) == 1 and seg[0] < 500  # one small segment, not the 2,000-doc index
    assert sizes[MANIFEST] < 2000


def test_segment_count_stays_logarithmic_and_write_amplification_bounded():
    client = FakeMinioClient()
    store = _store(client)
    written = []
    real = client._put_object

    def count(bucket, key, data, headers):
        if key.endswith(".seg"):
            written.append(len(data))
        return real(bucket, key, data, headers)

    client._put_object = count
    n = 512
    for i in range(n):
        store.add_document(f"d{i}", f"text {i}")
        assert store.index_stats()["segments"] <= math.log2(i + 1) + core.MIN_MERGE
    one = min(written)
    # Binary-counter merging rewrites each record about log2(n) times.
    assert sum(written) / (n * one) < math.log2(n) + 2
    assert store.count_documents() == n
    _assert_no_orphans(client)


def test_segments_beyond_the_size_cap_are_not_merged(monkeypatch):
    monkeypatch.setattr(core, "MAX_SEGMENT_BYTES", 2000)
    client = FakeMinioClient()
    store = _store(client)
    for i in range(40):
        store.add_document(f"d{i}", f"text {i}")
    sizes = [e["bytes"] for e in _manifest(client)["segments"]]
    assert all(s <= 2000 for s in sizes)
    assert store.count_documents() == 40


def test_too_many_segments_forces_a_merge(monkeypatch):
    monkeypatch.setattr(core, "MAX_SEGMENTS", 4)
    client = FakeMinioClient()
    store = _store(client)
    # Ever-growing writes defeat the size rule (each is bigger than all before).
    for i in range(10):
        store.add_documents([(f"b{i}-{j}", f"t {i} {j}") for j in range(2**i)])
        assert store.index_stats()["segments"] <= 4
    assert store.count_documents() == 2**10 - 1


# -- catching up with other processes -----------------------------------------------


def test_reader_downloads_only_the_new_segment():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_documents([(f"d{i}", f"text {i}") for i in range(100)])
    assert b.count_documents() == 100
    a.add_document("new", "new text")
    before = dict(client.get_counts)
    assert b.search("new text", top_k=1) == ["new"]
    fetched = {k: v - before.get(k, 0) for k, v in client.get_counts.items() if v != before.get(k, 0)}
    assert set(fetched) == {MANIFEST, _manifest(client)["segments"][-1]["key"]}


def test_reader_follows_a_merge_without_redownloading_old_segments():
    client = FakeMinioClient()
    a, b = _store(client), _store(client, "off")
    for i in range(8):
        a.add_document(f"d{i}", f"text {i}")
    assert b.count_documents() == 8
    a.add_document("d8", "text 8")  # triggers a merge of the tail
    before = dict(client.get_counts)
    assert b.count_documents() == 9
    fetched = {k for k, v in client.get_counts.items() if v != before.get(k, 0)}
    old = set(before) - {MANIFEST}
    assert not (fetched & old)


def test_reader_behind_a_full_merge_rebuilds_and_sees_deletes():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_documents([("keep", "k"), ("gone", "g")])
    assert b.count_documents() == 2
    a.delete_document("gone")
    a.add_document("new", "n")
    a.compact()  # full merge: the tombstone for "gone" is dropped
    assert all(e["full"] for e in _manifest(client)["segments"])
    assert sorted(d["doc_id"] for d in b.list_documents()) == ["keep", "new"]


def test_reader_retries_when_a_segment_is_merged_away_mid_read():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    for i in range(4):
        a.add_document(f"d{i}", f"text {i}")
    real_get = client.get_object
    fired = []

    def get(bucket, key):
        if key.endswith(".seg") and not fired:
            fired.append(key)
            a.compact()  # deletes every segment b is about to read
        return real_get(bucket, key)

    client.get_object = get
    assert b.count_documents() == 4
    assert fired


def test_missing_segment_in_an_unchanged_manifest_is_reported_as_corruption():
    client = FakeMinioClient()
    a = _store(client)
    a.add_document("x", "x")
    for key in _segment_keys(client):
        del client.objects["vectors"][key]
    with pytest.raises(IndexCorruptError):
        _store(client).count_documents()


# -- a write that landed but was reported as a conflict ------------------------------


def _land_then_conflict(client, key_match, between=None):
    """Make the first PUT matching `key_match` land, run `between`, then report 412."""
    real = client._put_object
    state = {"armed": True}

    def put(bucket, key, data, headers):
        result = real(bucket, key, data, headers)
        if state["armed"] and key_match(key):
            state["armed"] = False
            client._put_object = real
            if between:
                between()
            client._put_object = put
            raise _s3_error("PreconditionFailed", bucket, key)
        return result

    client._put_object = put


def test_landed_append_is_not_reapplied_over_a_newer_write():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_document("x", "v0")
    # a's append of v1 lands; before a sees the (false) conflict, b writes v2.
    _land_then_conflict(client, lambda k: k == MANIFEST, between=lambda: b.add_document("x", "v2"))
    a.add_document("x", "v1")
    assert _store(client).get_document("x")["text"] == "v2"
    assert a.get_document("x")["text"] == "v2"
    _assert_no_orphans(client)


def test_landed_append_is_detected_even_after_it_was_merged_away():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_documents([("x", "v0"), ("y", "y")])

    def b_overwrites_and_compacts():
        b.add_document("x", "v2")
        b.compact()  # a's segment is merged and deleted

    _land_then_conflict(client, lambda k: k == MANIFEST, between=b_overwrites_and_compacts)
    a.add_document("x", "v1")
    assert _docs(_store(client)) == {"x": "v2", "y": "y"}
    _assert_no_orphans(client)


def test_landed_delete_reports_found():
    client = FakeMinioClient()
    a = _store(client, "off")
    a.add_document("x", "x")
    _land_then_conflict(client, lambda k: k == MANIFEST)
    assert a.delete_document("x") is True
    assert _store(client).count_documents() == 0


# -- failure paths leave no garbage and no half-applied state --------------------------


def test_exhausted_retries_leave_no_orphan_segment(monkeypatch):
    monkeypatch.setattr(core, "INDEX_WRITE_ATTEMPTS", 3)
    monkeypatch.setattr(core, "INDEX_WRITE_BACKOFF_BASE_S", 0)
    client = FakeMinioClient()
    store = _store(client, "off")
    store.add_document("x", "x")
    real = client._put_object

    def conflict(bucket, key, data, headers):
        if key == MANIFEST:
            raise _s3_error("PreconditionFailed", bucket, key)
        return real(bucket, key, data, headers)

    client._put_object = conflict
    with pytest.raises(IndexWriteConflictError):
        store.add_document("y", "y")
    client._put_object = real
    _assert_no_orphans(client)
    assert _docs(_store(client)) == {"x": "x"}


def test_unknown_outcome_keeps_the_segment():
    # A network error on the manifest PUT: it may have landed, so the
    # segment must not be deleted (a later sweep removes it if unused).
    client = FakeMinioClient()
    store = _store(client, "off")
    store.add_document("x", "x")
    real = client._put_object

    def land_then_drop(bucket, key, data, headers):
        result = real(bucket, key, data, headers)
        if key == MANIFEST:
            raise ConnectionError("connection reset")
        return result

    client._put_object = land_then_drop
    with pytest.raises(ConnectionError):
        store.add_document("y", "y")
    client._put_object = real
    assert _docs(_store(client)) == {"x": "x", "y": "y"}  # it had landed: still readable
    assert store.get_document("y")["text"] == "y"


def test_validation_failure_and_noop_delete_upload_nothing():
    client = FakeMinioClient()
    store = _store(client, "off")
    store.add_document("x", "x")
    puts = []
    real = client._put_object
    client._put_object = lambda b, k, d, h: puts.append(k) or real(b, k, d, h)
    assert store.delete_document("missing") is False
    with pytest.raises(ValueError):
        store.add_documents([("ok", "fine"), ("bad/id", "nope")])
    assert puts == []


def test_writer_losing_to_index_delete_cleans_up_its_segment():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.create_index("tmp")
    a.add_document("x", "x", index="tmp")

    fired = []

    # a has loaded the manifest; b deletes the index (manifest and all
    # segments) before a uploads its segment, so only a can clean it up.
    def b_deletes(key, n):
        if key.startswith("segments/tmp/") and not fired:
            fired.append(1)
            b.delete_index("tmp")

    real = _hook_puts(client, before=b_deletes)
    with pytest.raises(IndexNotFoundError):
        a.add_document("y", "y", index="tmp")
    client._put_object = real
    assert fired
    assert _segment_keys(client, "tmp") == set()


def test_stale_pending_segment_is_reuploaded(monkeypatch):
    monkeypatch.setattr(core, "SEGMENT_MAX_PENDING_S", -1)  # always "too old"
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_document("base", "b")

    uploads = []

    def b_sneaks_in(key, n):
        if key.endswith(".seg"):
            uploads.append(key)
        if key == MANIFEST and n == 0:
            b.add_document("from-b", "fb")

    _hook_puts(client, before=b_sneaks_in)
    a.add_document("from-a", "fa")
    assert len(uploads) == 3  # a, b, then a again (its first upload was "too old")
    assert sorted(_docs(_store(client))) == ["base", "from-a", "from-b"]
    _assert_no_orphans(client)


def test_writers_map_is_pruned(monkeypatch):
    client = FakeMinioClient()
    for _ in range(5):
        _store(client, "off").add_document("x", "x")
    assert len(_manifest(client)["writers"]) == 5
    monkeypatch.setattr(core, "WRITER_TTL_S", -1)
    _store(client, "off").add_document("x", "x")
    assert len(_manifest(client)["writers"]) == 1


# -- compaction races ---------------------------------------------------------


def test_compaction_losing_to_a_write_keeps_the_write():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    for i in range(3):
        a.add_document(f"d{i}", f"t{i}")

    def b_writes(key, n):
        if key == MANIFEST and n == 0:  # a's compaction CAS
            b.add_document("late", "late")

    real = _hook_puts(client, before=b_writes)
    a.compact()
    client._put_object = real
    assert sorted(_docs(_store(client))) == ["d0", "d1", "d2", "late"]
    # The merge still went in, ahead of b's segment.
    entries = _manifest(client)["segments"]
    assert len(entries) == 2 and entries[0]["full"] and not entries[1]["full"]
    _assert_no_orphans(client)


def test_two_compactions_of_the_same_run_commit_once():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    for i in range(4):
        a.add_document(f"d{i}", f"t{i}")

    def b_compacts_first(key, n):
        if key == MANIFEST and n == 0:
            client._put_object = real
            b.compact()
            client._put_object = hooked

    real = _hook_puts(client, before=b_compacts_first)
    hooked = client._put_object
    a.compact()
    client._put_object = real
    assert len(_manifest(client)["segments"]) == 1
    assert sorted(_docs(_store(client))) == ["d0", "d1", "d2", "d3"]
    _assert_no_orphans(client)


def test_compaction_that_landed_but_reported_conflict_keeps_its_output():
    client = FakeMinioClient()
    a = _store(client, "off")
    for i in range(3):
        a.add_document(f"d{i}", f"t{i}")
    _land_then_conflict(client, lambda k: k == MANIFEST)
    a.compact()
    assert sorted(_docs(_store(client))) == ["d0", "d1", "d2"]
    assert len(_manifest(client)["segments"]) == 1
    _assert_no_orphans(client)


def test_full_compaction_splits_into_parts_and_reads_back(monkeypatch):
    monkeypatch.setattr(core, "MAX_SEGMENT_BYTES", 600)
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.add_documents([(f"d{i}", f"text {i}") for i in range(30)])
    assert b.count_documents() == 30
    a.delete_document("d3")
    a.add_document("d4", "changed")
    a.compact()
    entries = _manifest(client)["segments"]
    assert len(entries) > 1 and all(e["full"] for e in entries)
    for reader in (b, _store(client)):
        docs = _docs(reader)
        assert len(docs) == 29 and "d3" not in docs and docs["d4"] == "changed"
        assert reader.search("changed", top_k=1) == ["d4"]


def test_sweep_deletes_old_orphans_but_not_young_ones():
    client = FakeMinioClient()
    store = _store(client, "off")
    store.add_document("x", "x")
    inc = _manifest(client)["incarnation"]
    old, young = f"segments/default/{inc}/old.seg", f"segments/default/{inc}/young.seg"
    gone_incarnation = "segments/default/deadbeef/stale.seg"
    for key in (old, young, gone_incarnation):
        client.objects["vectors"][key] = b"junk"
    client.mtimes["vectors"][old] = datetime.now(UTC) - timedelta(hours=2)
    client.mtimes["vectors"][gone_incarnation] = datetime.now(UTC) - timedelta(hours=2)
    store.compact()
    keys = _segment_keys(client)
    assert old not in keys and gone_incarnation not in keys and young in keys
    assert store.get_document("x")["text"] == "x"


def test_background_compaction():
    client = FakeMinioClient()
    store = _store(client, compaction="background")
    for i in range(64):
        store.add_document(f"d{i}", f"t{i}")
    assert store.wait_for_compaction(timeout=10)
    assert store.index_stats()["segments"] <= 8
    assert store.count_documents() == 64
    _assert_no_orphans(client)


# -- index lifecycle ----------------------------------------------------------


def test_deleting_an_index_never_touches_a_newer_incarnation():
    client = FakeMinioClient()
    a, b = _store(client, "off"), _store(client, "off")
    a.create_index("tmp")
    a.add_document("old", "old", index="tmp")
    real_remove = client.remove_object

    def remove(bucket, key):
        real_remove(bucket, key)
        if key == manifest_key("tmp"):  # between a's manifest delete and its sweep
            client.remove_object = real_remove
            b.create_index("tmp")
            b.add_document("new", "new", index="tmp")

    client.remove_object = remove
    a.delete_index("tmp")
    assert _docs(_store(client), index="tmp") == {"new": "new"}
    _assert_no_orphans(client, "tmp")


def test_recreated_index_starts_empty():
    client = FakeMinioClient()
    store = _store(client)
    store.create_index("tmp")
    store.add_document("x", "x", index="tmp")
    store.delete_index("tmp")
    store.create_index("tmp")
    assert store.count_documents(index="tmp") == 0
    assert _store(client).count_documents(index="tmp") == 0


# -- migration from the v3 single-object layout ----------------------------------------


def _v3_blob(docs, dim=FakeEmbedder.dim):
    """A v3 index object as the previous release wrote it."""
    emb = FakeEmbedder()
    idx = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    rows = []
    if docs:
        keys = [core.object_key_id(d) for d, _, _ in docs]
        idx.add_with_ids(emb.encode([t for _, t, _ in docs]), np.array(keys, dtype="int64"))
        rows = [[k, d, t, m] for k, (d, t, m) in zip(keys, docs)]
    header = json.dumps({"dim": dim, "docs": rows}).encode()
    return b"".join((b"S3VAH-IDX\x03", struct.pack(">Q", len(header)), header, faiss.serialize_index(idx).tobytes()))


def test_migrates_v3_indexes():
    client = FakeMinioClient()
    client.make_bucket("vectors")
    client.objects["vectors"]["indexes/default.v3.bin"] = _v3_blob([("a", "alpha", {"y": 1})])
    client.objects["vectors"]["indexes/papers.v3.bin"] = _v3_blob([("p", "paper", {}), ("q", "quill", {"t": ["x"]})])
    client.objects["vectors"]["indexes/empty.v3.bin"] = _v3_blob([])
    store = _store(client)
    assert store.list_indexes() == ["default", "empty", "papers"]
    assert store.get_document("a") == {"doc_id": "a", "text": "alpha", "metadata": {"y": 1}}
    assert store.search("quill", top_k=1, index="papers") == ["q"]
    assert store.get_document("q", index="papers")["metadata"] == {"t": ["x"]}
    assert store.count_documents(index="empty") == 0
    with pytest.raises(IndexAlreadyExistsError):
        store.create_index("papers")
    # Migrated once; the v3 object is left for rollback but no longer read.
    client.objects["vectors"]["indexes/papers.v3.bin"] = _v3_blob([("other", "other", {})])
    assert _store(client).count_documents(index="papers") == 2


def test_deleted_v3_index_is_not_migrated_back():
    client = FakeMinioClient()
    client.make_bucket("vectors")
    client.objects["vectors"]["indexes/papers.v3.bin"] = _v3_blob([("p", "paper", {})])
    store = _store(client)
    assert store.count_documents(index="papers") == 1
    store.delete_index("papers")
    assert "papers" not in store.list_indexes()
    with pytest.raises(IndexNotFoundError):
        _store(client).count_documents(index="papers")


def test_unmigrated_v3_index_can_be_deleted():
    client = FakeMinioClient()
    client.make_bucket("vectors")
    client.objects["vectors"]["indexes/papers.v3.bin"] = _v3_blob([("p", "paper", {})])
    _store(client).delete_index("papers")
    assert _store(client).list_indexes() == ["default"]


# -- randomized: several processes, injected races, compared with a model ----------------


@pytest.mark.parametrize("seed", range(12))
def test_random_operations_match_a_model(seed, monkeypatch):
    monkeypatch.setattr(core, "INDEX_WRITE_BACKOFF_BASE_S", 0)
    rng = random.Random(seed)
    client = FakeMinioClient()
    stores = [_store(client, rng.choice(["inline", "off"])) for _ in range(3)]
    model: dict[str, str] = {}
    real = client._put_object

    # Randomly make manifest PUTs fail with a conflict, either before or
    # after landing (lost response); both must be invisible to the caller.
    def flaky(bucket, key, data, headers):
        if key == MANIFEST and rng.random() < 0.15:
            if rng.random() < 0.5:
                raise _s3_error("PreconditionFailed", bucket, key)
            real(bucket, key, data, headers)
            raise _s3_error("PreconditionFailed", bucket, key)
        return real(bucket, key, data, headers)

    client._put_object = flaky
    ids = [f"d{i}" for i in range(15)]
    for step in range(150):
        s = rng.choice(stores)
        op = rng.random()
        if op < 0.5:
            batch = {d: f"{d} v{step}" for d in rng.sample(ids, rng.randint(1, 4))}
            s.add_documents(list(batch.items()))
            model.update(batch)
        elif op < 0.8:
            d = rng.choice(ids)
            assert s.delete_document(d) == (d in model)
            model.pop(d, None)
        elif op < 0.9:
            s.compact()
        else:
            assert _docs(s) == model
    client._put_object = real
    for s in [*stores, _store(client)]:
        assert _docs(s) == model
        for d, text in model.items():
            assert s.search(text, top_k=1) == [d]
    stores[0].compact()
    _assert_no_orphans(client)
