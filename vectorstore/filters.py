"""Document metadata and the filter language for searching on it.

Metadata is a flat JSON object per document:
    {"lang": "en", "year": 2024, "draft": False, "tags": ["a", "b"]}
Values are strings, numbers, booleans, or lists of those.

Filters follow the MongoDB-style operators S3 Vectors uses:
    {"lang": "en"}                                   # implicit $eq
    {"year": {"$gte": 2020, "$lt": 2025}}            # several ops on a field: AND
    {"$or": [{"lang": "en"}, {"tags": {"$in": ["a"]}}]}
Operators: $eq $ne $gt $gte $lt $lte $in $nin $exists $startsWith, combined
with $and, $or, or by listing several fields (AND). For a list-valued field,
$eq/$in/$startsWith match if *any* element matches, and $ne/$nin if *no*
element does. Range operators compare numbers only; $startsWith matches
strings that begin with the given prefix. A missing field matches only $ne,
$nin and {"$exists": false}.

Booleans are kept distinct from numbers (True does not equal 1), which
Python's `==` would otherwise conflate.
"""

import bisect
import json
import math

MAX_METADATA_BYTES = 4096
MAX_METADATA_KEYS = 32
MAX_KEY_LENGTH = 64
MAX_LIST_LENGTH = 64
MAX_FILTER_DEPTH = 8

_FIELD_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists", "$startsWith"}
_RANGE_OPS = {"$gt", "$gte", "$lt", "$lte"}
_LOGICAL_OPS = {"$and", "$or"}


def _is_scalar(v) -> bool:
    if isinstance(v, (bool, str)):
        return True
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and not math.isfinite(v))
    return False


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_metadata(metadata) -> dict:
    """Return a clean copy of `metadata`, or raise ValueError."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        # ValueError, not TypeError: callers map it to "400 bad request".
        raise ValueError("metadata must be a JSON object")  # noqa: TRY004
    if len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata may have at most {MAX_METADATA_KEYS} keys")
    clean = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        _require_unicode(key, value)
        if key.startswith("$"):
            raise ValueError(f"metadata key {key!r} must not start with '$'")
        if len(key) > MAX_KEY_LENGTH:
            raise ValueError(f"metadata key {key[:20]!r}... is longer than {MAX_KEY_LENGTH} characters")
        if isinstance(value, list):
            if len(value) > MAX_LIST_LENGTH:
                raise ValueError(f"metadata list {key!r} may have at most {MAX_LIST_LENGTH} elements")
            if not all(_is_scalar(v) for v in value):
                raise ValueError(f"metadata list {key!r} may only contain strings, finite numbers and booleans")
            clean[key] = list(value)
        elif _is_scalar(value):
            clean[key] = value
        else:
            raise ValueError(
                f"metadata value for {key!r} must be a string, finite number, boolean, or a list of those"
            )
    if len(json.dumps(clean).encode()) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata must serialise to at most {MAX_METADATA_BYTES} bytes of JSON")
    return clean


def _require_unicode(key: str, value) -> None:
    # Lone surrogates (valid in JSON, not in UTF-8) would make the stored
    # JSON unreadable for other tools; reject them up front.
    for s in [key, *(value if isinstance(value, list) else [value])]:
        if isinstance(s, str):
            try:
                s.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"metadata {key!r} must be valid Unicode (it contains a lone surrogate)") from None


def validate_filter(flt, _depth: int = 0) -> None:
    """Raise ValueError unless `flt` is a well-formed filter."""
    if _depth > MAX_FILTER_DEPTH:
        raise ValueError(f"filter is nested more than {MAX_FILTER_DEPTH} levels deep")
    if not isinstance(flt, dict):
        raise ValueError("a filter must be a JSON object")  # noqa: TRY004 -- see validate_metadata
    for key, cond in flt.items():
        if key in _LOGICAL_OPS:
            if not isinstance(cond, list) or not cond:
                raise ValueError(f"{key} takes a non-empty list of filters")
            for sub in cond:
                validate_filter(sub, _depth + 1)
        elif key.startswith("$"):
            raise ValueError(f"unknown top-level operator {key!r}")
        elif isinstance(cond, dict):
            if not cond:
                raise ValueError(f"condition for {key!r} must not be empty")
            for op, operand in cond.items():
                _validate_op(key, op, operand)
        elif not _is_scalar(cond):
            raise ValueError(f"value for {key!r} must be a scalar or an operator object")


def _validate_op(field: str, op: str, operand) -> None:
    if op not in _FIELD_OPS:
        raise ValueError(f"unknown operator {op!r} for {field!r}")
    if op in ("$eq", "$ne"):
        if not _is_scalar(operand):
            raise ValueError(f"{op} on {field!r} takes a string, number or boolean")
    elif op in _RANGE_OPS:
        if not _is_number(operand) or not math.isfinite(operand):
            raise ValueError(f"{op} on {field!r} takes a finite number")
    elif op in ("$in", "$nin"):
        if not isinstance(operand, list) or not all(_is_scalar(v) for v in operand):
            raise ValueError(f"{op} on {field!r} takes a list of strings, numbers or booleans")
    elif op == "$exists" and not isinstance(operand, bool):
        raise ValueError(f"$exists on {field!r} takes true or false")
    elif op == "$startsWith" and not isinstance(operand, str):
        raise ValueError(f"$startsWith on {field!r} takes a string")


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def _values(metadata: dict, field: str):
    """(present, [values]) -- a scalar field is treated as a one-element list."""
    if field not in metadata:
        return False, []
    v = metadata[field]
    return True, (v if isinstance(v, list) else [v])


def _match_op(present: bool, values: list, op: str, operand) -> bool:
    if op == "$exists":
        return present == operand
    if op == "$eq":
        return any(_eq(v, operand) for v in values)
    if op == "$ne":
        return not any(_eq(v, operand) for v in values)
    if op == "$in":
        return any(_eq(v, o) for v in values for o in operand)
    if op == "$nin":
        return not any(_eq(v, o) for v in values for o in operand)
    if op == "$startsWith":
        return any(isinstance(v, str) and v.startswith(operand) for v in values)
    nums = [v for v in values if _is_number(v)]
    if op == "$gt":
        return any(v > operand for v in nums)
    if op == "$gte":
        return any(v >= operand for v in nums)
    if op == "$lt":
        return any(v < operand for v in nums)
    if op == "$lte":
        return any(v <= operand for v in nums)
    raise AssertionError(op)  # unreachable for validated filters


def filter_fields(flt: dict) -> set[str]:
    """The metadata fields a (validated) filter refers to."""
    fields = set()
    for key, cond in flt.items():
        if key in _LOGICAL_OPS:
            for sub in cond:
                fields |= filter_fields(sub)
        else:
            fields.add(key)
    return fields


def count_constraints(flt: dict) -> int:
    """Constraints in a (validated) filter, counted as S3 Vectors does: one
    per operand value, so {"$in": [a, b, c]} counts three."""
    n = 0
    for key, cond in flt.items():
        if key in _LOGICAL_OPS:
            n += sum(count_constraints(sub) for sub in cond)
        elif isinstance(cond, dict):
            n += sum(len(v) if isinstance(v, list) else 1 for v in cond.values())
        else:
            n += 1
    return n


def matches(flt: dict, metadata: dict) -> bool:
    """Whether `metadata` satisfies the (already validated) filter `flt`."""
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(sub, metadata) for sub in cond):
                return False
        elif key == "$or":
            if not any(matches(sub, metadata) for sub in cond):
                return False
        else:
            present, values = _values(metadata, key)
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            if not all(_match_op(present, values, op, operand) for op, operand in ops.items()):
                return False
    return True


# -- an inverted index for evaluating filters over many documents ----------------


def _key(v):
    """Dict key for a scalar under `_eq`: booleans are tagged so True isn't 1.
    Numbers are their own keys: Python's 1 == 1.0, with equal hashes, which
    is exactly `_eq` for numbers (exact even for large ints vs floats)."""
    return ("b", v) if isinstance(v, bool) else v


class _FieldIndex:
    """One metadata field: which documents hold which values.

    `eq` maps a value key to the documents with that value (any element, for
    a list); `present` holds every document with the field. Sorted lists of
    the distinct numbers and strings answer ranges and prefixes by bisection.
    """

    __slots__ = ("eq", "nums", "present", "strs")

    def __init__(self):
        self.eq: dict = {}
        self.present: set[int] = set()
        self.nums: list = []
        self.strs: list[str] = []

    def add(self, doc: int, value) -> None:
        self.present.add(doc)
        for v in value if isinstance(value, list) else (value,):
            k = _key(v)
            docs = self.eq.get(k)
            if docs is None:
                docs = self.eq[k] = set()
                if isinstance(v, str):
                    bisect.insort(self.strs, v)
                elif _is_number(v):
                    bisect.insort(self.nums, v)
            docs.add(doc)

    def remove(self, doc: int, value) -> None:
        self.present.discard(doc)
        for v in value if isinstance(value, list) else (value,):
            k = _key(v)
            docs = self.eq.get(k)
            if docs is None:
                continue
            docs.discard(doc)
            if not docs:
                del self.eq[k]
                # Each key has exactly one entry in its sorted list (1 and
                # 1.0 are one key), so the first equal entry is it.
                sorted_values = self.strs if isinstance(v, str) else self.nums if _is_number(v) else None
                if sorted_values is not None:
                    del sorted_values[bisect.bisect_left(sorted_values, v)]

    def _union(self, keys) -> set[int]:
        out: set[int] = set()
        for k in keys:
            docs = self.eq.get(k)
            if docs:
                out |= docs
        return out

    def _number_range(self, lo, lo_open: bool, hi, hi_open: bool) -> set[int]:
        nums = self.nums
        i = 0 if lo is None else (bisect.bisect_right(nums, lo) if lo_open else bisect.bisect_left(nums, lo))
        j = len(nums) if hi is None else (bisect.bisect_left(nums, hi) if hi_open else bisect.bisect_right(nums, hi))
        return self._union(nums[i:j])

    def match(self, op: str, operand, everyone) -> set[int]:
        """The documents for which `field op operand` holds (see `_match_op`).
        `everyone()` is the set of all documents, for the negations."""
        if op == "$exists":
            return set(self.present) if operand else everyone() - self.present
        if op == "$eq":
            return set(self.eq.get(_key(operand), ()))
        if op == "$ne":
            return everyone() - self.eq.get(_key(operand), set())
        if op == "$in":
            return self._union(_key(o) for o in operand)
        if op == "$nin":
            return everyone() - self._union(_key(o) for o in operand)
        if op == "$startsWith":
            strs = self.strs
            i = bisect.bisect_left(strs, operand)
            j = i
            while j < len(strs) and strs[j].startswith(operand):
                j += 1
            return self._union(strs[i:j])
        if op == "$gt":
            return self._number_range(operand, True, None, False)
        if op == "$gte":
            return self._number_range(operand, False, None, False)
        if op == "$lt":
            return self._number_range(None, False, operand, True)
        if op == "$lte":
            return self._number_range(None, False, operand, False)
        raise AssertionError(op)  # unreachable for validated filters


class FilterIndex:
    """Evaluates filters over a changing set of documents with set algebra,
    instead of running `matches` on every document.

    A field is indexed the first time a filter mentions it (one pass over
    the documents), and kept up to date from then on by `add`/`remove`, so
    fields nobody filters on cost nothing: no memory, no load time.
    `select` returns exactly the documents `matches` would accept (the tests
    check the two against each other on random metadata and filters).

    `docs` is the live {doc id: value} mapping, and `metadata(value)` a
    value's metadata (by default the value itself); the owner calls `add`
    and `remove` as it changes, and must serialise all calls (it holds a lock).
    """

    def __init__(self, docs: dict, metadata=None):
        self._docs = docs
        self._metadata = metadata or (lambda value: value)
        self._fields: dict[str, _FieldIndex] = {}
        self.entries = 0  # (document, value) pairs indexed, for memory estimates

    def add(self, doc: int, metadata: dict) -> None:
        for field, fi in self._fields.items():
            if field in metadata:
                fi.add(doc, metadata[field])
                self.entries += _entry_count(metadata[field])

    def remove(self, doc: int, metadata: dict) -> None:
        for field, fi in self._fields.items():
            if field in metadata:
                fi.remove(doc, metadata[field])
                self.entries -= _entry_count(metadata[field])

    def _field(self, field: str) -> _FieldIndex:
        fi = self._fields.get(field)
        if fi is None:
            fi = _FieldIndex()
            for doc, value in self._docs.items():
                metadata = self._metadata(value)
                if field in metadata:
                    fi.add(doc, metadata[field])
                    self.entries += _entry_count(metadata[field])
            self._fields[field] = fi
        return fi

    def select(self, flt: dict) -> set[int]:
        """The ids of the documents matching the (validated) filter `flt`."""
        cache: list[set[int]] = []

        def everyone() -> set[int]:  # built once, and only if a negation needs it
            if not cache:
                cache.append(set(self._docs))
            return cache[0]

        return self._select(flt, everyone)

    def _select(self, flt: dict, everyone) -> set[int]:
        parts = []
        for key, cond in flt.items():
            if key == "$and":
                parts.extend(self._select(sub, everyone) for sub in cond)
            elif key == "$or":
                out: set[int] = set()
                for sub in cond:
                    out |= self._select(sub, everyone)
                parts.append(out)
            else:
                fi = self._field(key)
                ops = cond if isinstance(cond, dict) else {"$eq": cond}
                parts.extend(fi.match(op, operand, everyone) for op, operand in ops.items())
        if not parts:
            return set(everyone())  # an empty filter matches everything, as `matches` does
        parts.sort(key=len)
        result = parts[0]
        for p in parts[1:]:
            if not result:
                break
            result &= p
        return result


def _entry_count(value) -> int:
    return len(value) if isinstance(value, list) else 1
