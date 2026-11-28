import pytest

from vectorstore.filters import matches, validate_filter, validate_metadata

DOC = {"lang": "en", "year": 2021, "score": 4.5, "draft": False, "tags": ["ml", "search"]}


@pytest.mark.parametrize(
    "flt,expected",
    [
        ({}, True),
        ({"lang": "en"}, True),
        ({"lang": "de"}, False),
        ({"lang": {"$eq": "en"}}, True),
        ({"lang": {"$ne": "en"}}, False),
        ({"missing": {"$ne": "x"}}, True),
        ({"year": {"$gte": 2020, "$lt": 2022}}, True),
        ({"year": {"$gt": 2021}}, False),
        ({"score": {"$lte": 4.5}}, True),
        ({"year": 2021.0}, True),  # int and float compare as numbers
        ({"lang": {"$in": ["de", "en"]}}, True),
        ({"lang": {"$nin": ["de", "en"]}}, False),
        ({"tags": "ml"}, True),  # list field: any element
        ({"tags": {"$in": ["x", "search"]}}, True),
        ({"tags": {"$ne": "ml"}}, False),  # list field: no element may equal
        ({"tags": {"$nin": ["x", "y"]}}, True),
        ({"missing": {"$exists": False}}, True),
        ({"lang": {"$exists": False}}, False),
        ({"missing": "x"}, False),
        ({"missing": {"$gt": 1}}, False),
        ({"lang": {"$gt": 1}}, False),  # range ops ignore non-numbers
        ({"draft": False}, True),
        ({"draft": 0}, False),  # booleans are not numbers
        ({"year": True}, False),
        ({"lang": "en", "year": 1999}, False),  # fields are ANDed
        ({"$and": [{"lang": "en"}, {"year": 2021}]}, True),
        ({"$or": [{"lang": "de"}, {"year": 2021}]}, True),
        ({"$or": [{"lang": "de"}, {"year": 1}]}, False),
        ({"$or": [{"$and": [{"lang": "en"}, {"draft": True}]}, {"tags": "search"}]}, True),
    ],
)
def test_matches(flt, expected):
    validate_filter(flt)
    assert matches(flt, DOC) is expected


@pytest.mark.parametrize(
    "flt",
    [
        [],
        {"$nor": []},
        {"$and": []},
        {"$or": {"a": 1}},
        {"a": {"$regex": "x"}},
        {"a": {}},
        {"a": {"$gt": "2020"}},
        {"a": {"$gt": float("nan")}},
        {"a": {"$in": "x"}},
        {"a": {"$in": [{"b": 1}]}},
        {"a": {"$exists": 1}},
        {"a": [1, 2]},
        {"a": None},
        {"a": {"$eq": [1]}},
    ],
)
def test_rejects_malformed_filters(flt):
    with pytest.raises(ValueError):
        validate_filter(flt)


def test_rejects_too_deep_filter():
    flt = {"a": 1}
    for _ in range(10):
        flt = {"$and": [flt]}
    with pytest.raises(ValueError):
        validate_filter(flt)


def test_metadata_valid_round_trip():
    assert validate_metadata(DOC) == DOC
    assert validate_metadata(None) == {}


@pytest.mark.parametrize(
    "md",
    [
        "x",
        {"": 1},
        {"$a": 1},
        {"a" * 65: 1},
        {"a": None},
        {"a": {"b": 1}},
        {"a": [1, [2]]},
        {"a": float("inf")},
        {"a": list(range(65))},
        {f"k{i}": i for i in range(33)},
        {"a": "x" * 5000},
    ],
)
def test_rejects_invalid_metadata(md):
    with pytest.raises(ValueError):
        validate_metadata(md)
