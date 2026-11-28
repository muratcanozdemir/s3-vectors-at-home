"""Long texts are split into chunks, one vector each (fake embedder, no MinIO).

FakeEmbedder(window=n) treats whitespace words as tokens and splits texts
longer than n words, so these tests control exactly where chunks fall.
"""

import json
import random
import struct

import numpy as np
import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore import core
from vectorstore.core import VectorStore

WINDOW = 4


def _store(client=None, window=WINDOW, **kwargs):
    return VectorStore(client or FakeMinioClient(), FakeEmbedder(window=window), compaction="inline", **kwargs)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.fixture
def store():
    return _store()


def test_short_text_is_one_vector(store):
    store.add_document("short", "one two three")
    assert store.index_stats()["vectors"] == 1


def test_long_text_gets_one_vector_per_chunk(store):
    text = words(10)
    spans = FakeEmbedder(window=WINDOW).chunk_spans(text, core.DEFAULT_CHUNK_OVERLAP)
    store.add_document("long", text)
    stats = store.index_stats()
    assert stats["documents"] == 1
    assert stats["vectors"] == len(spans) > 1
    assert store.get_document("long") == {"doc_id": "long", "text": text, "metadata": {}}


@pytest.mark.parametrize("chunk", [0, 1, 2])
def test_search_finds_text_past_the_window_and_returns_the_passage(store, chunk):
    text = words(12)
    spans = FakeEmbedder(window=WINDOW).chunk_spans(text, core.DEFAULT_CHUNK_OVERLAP)
    store.add_document("long", text)
    store.add_document("other", "something else entirely")
    start, end = spans[chunk]
    # The fake embeds equal texts to equal vectors, so a chunk's own text
    # is an exact match for that chunk's vector.
    hit = store.search_with_scores(text[start:end], top_k=1, passages=True)[0]
    assert hit["doc_id"] == "long"
    assert hit["score"] == pytest.approx(1.0)
    assert hit["passage"] == text[start:end]
    assert hit["span"] == [start, end]


def test_passage_of_an_unchunked_doc_is_its_whole_text(store):
    store.add_document("short", "hello there")
    hit = store.search_with_scores("hello there", top_k=1, passages=True)[0]
    assert hit["passage"] == "hello there"
    assert hit["span"] == [0, len("hello there")]


def test_without_passages_results_keep_their_shape(store):
    store.add_document("long", words(12))
    assert set(store.search_with_scores("w0 w1 w2 w3", top_k=1)[0]) == {"doc_id", "score", "metadata"}


def test_top_k_counts_documents_not_chunks(store):
    store.add_documents([("a", words(20, "a")), ("b", words(20, "b")), ("c", "c0 c1")])
    hits = store.search_with_scores("a0 a1 a2 a3", top_k=3)
    assert sorted(h["doc_id"] for h in hits) == ["a", "b", "c"]
    assert hits[0]["doc_id"] == "a"


def test_replacing_a_chunked_doc_drops_its_old_chunks(store):
    store.add_document("doc", words(20))
    store.add_document("doc", "now short")
    assert store.index_stats()["vectors"] == 1
    hit = store.search_with_scores("w8 w9 w10 w11", top_k=5, passages=True)
    assert [h["passage"] for h in hit] == ["now short"]


def test_deleting_a_chunked_doc_removes_every_vector(store):
    store.add_document("doc", words(20))
    store.add_document("keep", "keep me")
    assert store.delete_document("doc")
    assert store.index_stats()["vectors"] == 1
    assert store.search("w4 w5 w6 w7", top_k=5) == ["keep"]


def test_filters_apply_to_chunked_docs(store):
    store.add_documents([
        ("en", words(12, "x"), {"lang": "en"}),
        ("fr", words(12, "x"), {"lang": "fr"}),
    ])
    assert store.search("x4 x5 x6 x7", top_k=5, filter={"lang": "fr"}) == ["fr"]


def test_chunks_survive_a_reload_and_compaction():
    client = FakeMinioClient()
    first = _store(client)
    text = words(15)
    first.add_documents([("long", text, {"k": 1}), ("short", "tiny")])
    first.add_document("other", words(9, "o"))
    expected = first.search_with_scores("w5 w6 w7 w8", top_k=3, passages=True)
    second = _store(client)
    assert second.search_with_scores("w5 w6 w7 w8", top_k=3, passages=True) == expected
    second.compact()
    assert second.index_stats()["segments"] == 1
    third = _store(client)
    assert third.search_with_scores("w5 w6 w7 w8", top_k=3, passages=True) == expected
    assert third.index_stats()["vectors"] == first.index_stats()["vectors"]


def test_a_cached_store_catches_up_on_chunked_writes():
    client = FakeMinioClient()
    reader, writer = _store(client), _store(client)
    reader.add_document("seed", "seed")
    writer.add_document("long", words(12))
    assert reader.search_with_scores("w8 w9 w10 w11", top_k=1, passages=True)[0]["passage"] == "w8 w9 w10 w11"
    writer.delete_document("long")
    assert reader.index_stats()["vectors"] == 1


def test_chunking_off_embeds_one_vector():
    store = _store(chunk_overlap=None)
    store.add_document("long", words(20))
    assert store.index_stats()["vectors"] == 1


def test_an_embedder_without_chunk_spans_means_one_vector():
    class Plain(FakeEmbedder):
        chunk_spans = None

    store = VectorStore(FakeMinioClient(), Plain(), compaction="inline")
    store.add_document("long", words(20))
    assert store.index_stats()["vectors"] == 1


def test_segments_from_before_chunking_still_read():
    """A record with four fields (no spans) is one vector, as before."""
    dim = 3
    vec = np.array([[1, 0, 0]], dtype="<f4")
    header = json.dumps({"dim": dim, "records": [[5, "old", "text", {"a": 1}], [6, "gone", None, None]]}).encode()
    blob = core.SEGMENT_MAGIC + struct.pack(">Q", len(header)) + header + vec.tobytes()
    recs = core._decode_segment(blob, "k", dim, 1)
    assert [r[:4] for r in recs] == [(5, "old", "text", {"a": 1}), (6, "gone", None, None)]
    assert recs[0][5] is None and recs[1][4] is None
    block, row = recs[0][4]
    assert block[row].tolist() == [1, 0, 0]


def test_segment_codec_round_trips_chunked_records():
    rng = np.random.default_rng(0)
    dim = 4
    block = rng.normal(size=(6, dim)).astype(np.float32)
    records = [
        (None, "a", "x", {}, (block, 0), None),
        (3, "b", "y y y", {"n": 1}, (block, 1), [[0, 1], [2, 3], [4, 5]]),
        (None, "dead", None, None, None, None),
        (None, "c", "zz", {}, (block, 4), [[0, 1], [1, 2]]),
    ]
    back = core._decode_segment(core._encode_segment(dim, records), "k", dim, 9)
    assert [r[:4] for r in back] == [(9, "a", "x", {}), (3, "b", "y y y", {"n": 1}), (9, "dead", None, None), (9, "c", "zz", {})]
    assert [r[5] for r in back] == [None, [[0, 1], [2, 3], [4, 5]], None, [[0, 1], [1, 2]]]
    got = core._gather([back[0][4], back[1][4], back[3][4]], dim, [1, 3, 2])
    assert np.array_equal(got, block)


@pytest.mark.parametrize("seed", range(20))
def test_chunked_search_is_exact_max_over_chunks(seed):
    """Against brute force: a doc's score is its best chunk, top_k distinct docs."""
    rng = random.Random(seed)
    store = _store(window=rng.randint(2, 5))
    docs = {f"d{i}": words(rng.randint(1, 14), rng.choice("abc")) for i in range(rng.randint(1, 25))}
    tags = {d: rng.choice(["x", "y"]) for d in docs}
    store.add_documents([(d, t, {"tag": tags[d]}) for d, t in docs.items()])
    embed = store._embedder
    query = words(rng.randint(1, 4), rng.choice("abc"))
    qv = embed.encode([query])[0]
    use_filter = rng.random() < 0.5
    best = {}
    for d, text in docs.items():
        if use_filter and tags[d] != "x":
            continue
        vecs = embed.encode([text[a:b] for a, b in embed.chunk_spans(text, core.DEFAULT_CHUNK_OVERLAP)])
        best[d] = float((vecs @ qv).max())
    top_k = rng.randint(1, 8)
    want = sorted(best.values(), reverse=True)[:top_k]
    got = store.search_with_scores(query, top_k=top_k, filter={"tag": "x"} if use_filter else None)
    assert len({h["doc_id"] for h in got}) == len(got)
    assert [h["score"] for h in got] == pytest.approx(want, abs=1e-5)
    for h in got:
        assert h["score"] == pytest.approx(best[h["doc_id"]], abs=1e-5)


def test_a_chunk_label_collision_is_rejected(store, monkeypatch):
    store.add_document("first", "one")
    real = core.chunk_label

    def clashing(doc_id, chunk):
        # Pretend "long"'s chunk 1 hashes to the same id as "first".
        return core.object_key_id("first") if (doc_id, chunk) == ("long", 1) else real(doc_id, chunk)

    monkeypatch.setattr(core, "chunk_label", clashing)
    with pytest.raises(ValueError, match="collides with 'first'"):
        store.add_document("long", words(12))
    assert store.get_document("long") is None
