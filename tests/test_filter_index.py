"""FilterIndex must select exactly the documents `matches` accepts."""

import random

import pytest

from tests.fakes import FakeEmbedder, FakeMinioClient
from vectorstore.core import VectorStore
from vectorstore.filters import FilterIndex, matches, validate_filter

# Values chosen to catch equality traps: bools vs 0/1, 1 vs 1.0, -0.0,
# 2**60 as an int and as a float, and 2**60 + 1, which no float equals.
VALUES = ["a", "ab", "abc", "b", "", "é", True, False, 0, 1, 1.0, -1, 2.5, -0.0, 2**60, float(2**60), 2**60 + 1]
NUMBERS = [0, 1, 1.0, -1, 2.5, -0.0, 2**60, float(2**60), 2**60 + 1, -1e9]


def _metadata(rng):
    md = {}
    for field in "xyz":
        r = rng.random()
        if r < 0.3:
            continue
        md[field] = [rng.choice(VALUES) for _ in range(rng.randint(0, 3))] if r < 0.5 else rng.choice(VALUES)
    return md


def _filter(rng, depth=0):
    if depth < 2 and rng.random() < 0.25:
        return {rng.choice(["$and", "$or"]): [_filter(rng, depth + 1) for _ in range(rng.randint(1, 3))]}
    field = rng.choice("xyzw")  # w is never set
    op = rng.choice(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists", "$startsWith", "bare"])
    if op == "bare":
        flt = {field: rng.choice(VALUES)}
    elif op in ("$in", "$nin"):
        flt = {field: {op: [rng.choice(VALUES) for _ in range(rng.randint(0, 3))]}}
    elif op == "$exists":
        flt = {field: {op: rng.random() < 0.5}}
    elif op == "$startsWith":
        flt = {field: {op: rng.choice(["", "a", "ab", "b", "z", "é"])}}
    elif op in ("$gt", "$gte", "$lt", "$lte"):
        cond = {op: rng.choice(NUMBERS)}
        if rng.random() < 0.3:
            cond[rng.choice(["$gt", "$gte", "$lt", "$lte"])] = rng.choice(NUMBERS)
        flt = {field: cond}
    else:
        flt = {field: {op: rng.choice(VALUES)}}
    if rng.random() < 0.3:
        flt[rng.choice("xyz")] = rng.choice(VALUES)  # a second field: ANDed
    return flt


@pytest.mark.parametrize("seed", range(40))
def test_select_agrees_with_matches_through_changes(seed):
    rng = random.Random(seed)
    docs: dict[int, dict] = {}
    index = FilterIndex(docs)
    for _ in range(80):
        doc = rng.randint(0, 30)
        if doc in docs and rng.random() < 0.4:
            index.remove(doc, docs.pop(doc))
        else:
            if doc in docs:
                index.remove(doc, docs[doc])
            docs[doc] = _metadata(rng)
            index.add(doc, docs[doc])
        flt = _filter(rng)
        validate_filter(flt)
        assert index.select(flt) == {d for d, md in docs.items() if matches(flt, md)}, flt


@pytest.mark.parametrize(
    ("flt", "want"),
    [
        ({"f": 1}, {"one", "one-float"}),  # numbers compare by value
        ({"f": True}, {"true"}),  # ...but booleans aren't numbers
        ({"f": {"$ne": 1}}, {"true", "big", "big-float", "big-plus", "missing", "str"}),
        ({"f": 2**60}, {"big", "big-float"}),
        ({"f": 2**60 + 1}, {"big-plus"}),  # no float equals it
        ({"f": {"$gt": 2**60}}, {"big-plus"}),
        ({"f": {"$gte": 1, "$lte": 1}}, {"one", "one-float"}),
        ({"f": {"$lt": 1}}, set()),
        ({"f": {"$exists": False}}, {"missing"}),
        ({"f": {"$startsWith": "ab"}}, {"str"}),
        ({"f": {"$in": [True, "abc"]}}, {"true", "str"}),
        ({"f": {"$nin": [1]}}, {"true", "big", "big-float", "big-plus", "missing", "str"}),
    ],
)
def test_equality_traps(flt, want):
    docs = {
        "one": {"f": 1}, "one-float": {"f": 1.0}, "true": {"f": True}, "big": {"f": 2**60},
        "big-float": {"f": float(2**60)}, "big-plus": {"f": 2**60 + 1}, "missing": {}, "str": {"f": "abc"},
    }
    assert FilterIndex(docs).select(flt) == want
    assert {d for d, md in docs.items() if matches(flt, md)} == want


def test_fields_are_indexed_only_once_filtered_on():
    docs = {i: {"a": i, "b": [i, i + 1]} for i in range(10)}
    index = FilterIndex(docs)
    assert index.entries == 0
    assert index.select({"a": {"$gte": 5}}) == {5, 6, 7, 8, 9}
    assert index.entries == 10  # just "a"
    index.select({"b": 3})
    assert index.entries == 30
    docs[10] = {"a": 10, "b": [1]}
    index.add(10, docs[10])
    assert index.entries == 32
    assert index.select({"b": 1}) == {0, 1, 10}


def test_the_store_answers_filters_from_the_index():
    store = VectorStore(FakeMinioClient(), FakeEmbedder(), compaction="inline")
    store.add_documents([(f"d{i}", f"text {i}", {"n": i, "even": i % 2 == 0}) for i in range(40)])
    flt = {"n": {"$gte": 10, "$lt": 20}, "even": True}
    assert sorted(store.search("text 12", top_k=40, filter=flt)) == sorted(f"d{i}" for i in range(10, 20, 2))
    state = store._cache["default"].state
    assert state.filters is not None and state.filters.entries == 80
    # Later writes and deletes keep it current.
    store.add_document("d12", "text 12", {"n": 99, "even": True})
    store.delete_document("d14")
    assert sorted(store.search("text 12", top_k=40, filter=flt)) == ["d10", "d16", "d18"]
    stats_bytes = store.cache_stats()["bytes"]
    assert stats_bytes > 0
