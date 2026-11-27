# SPDX-License-Identifier: MIT
"""merge_view_partials merges every kind of partial column of a view row."""

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.full]

pytest.importorskip("pyarrow")

from .pandas_reference import _partial_agg_columns, merge_view_partials, partial_arrow_view_groupby

FULL = ["dur"]
SUMS = ["size"]
MINS = ["lo"]
MAXS = ["hi"]
SETS = [("name", None)]
SET_COLS = ["name"]


def _flatten(cells):
    out = set()
    for cell in cells:
        out |= set(cell)
    return frozenset(out)


def _partial(view, dur, size, lo, hi, names):
    """One partition's partial row per view row, as partial_arrow_view_groupby names them."""
    n = len(view)
    return pd.DataFrame(
        {
            "dur_sum": dur,
            "dur_count": [2] * n,
            "dur_min": dur,
            "dur_max": dur,
            "dur_m2": [0.0] * n,
            "dur_mean_hi": dur,
            "dur_mean_lo": [0.0] * n,
            "size_sum": size,
            "lo_min": lo,
            "hi_max": hi,
            "name_unique": [frozenset(x) for x in names],
        },
        index=pd.Index(view, name="view"),
    )


def _two_partitions():
    a = _partial(["x", "y"], [1.0, 10.0], [1, 2], [5, 6], [7, 8], [{"a"}, {"c"}])
    b = _partial(["x", "y"], [3.0, 10.0], [10, 20], [4, 9], [9, 3], [{"b"}, {"c", "d"}])
    return pd.concat([a, b])


def test_sum_min_max_and_set_columns_merge_by_their_own_rule():
    merged = merge_view_partials(_two_partitions(), FULL, SUMS, MINS, MAXS, SET_COLS, _flatten)
    assert merged.loc["x", "size_sum"] == 11 and merged.loc["y", "size_sum"] == 22
    assert merged.loc["x", "lo_min"] == 4 and merged.loc["y", "lo_min"] == 6
    assert merged.loc["x", "hi_max"] == 9 and merged.loc["y", "hi_max"] == 8
    assert merged.loc["x", "name_unique"] == frozenset({"a", "b"})
    assert merged.loc["y", "name_unique"] == frozenset({"c", "d"})


def test_full_columns_still_use_the_pairwise_formula():
    merged = merge_view_partials(_two_partitions(), FULL, SUMS, MINS, MAXS, SET_COLS, _flatten)
    n = merged.loc["x", "dur_count"]
    mean = merged.loc["x", "dur_mean_hi"] + merged.loc["x", "dur_mean_lo"]
    assert n == 4 and mean == pytest.approx(2.0)
    assert merged.loc["x", "dur_m2"] == pytest.approx(np.var([1.0, 1.0, 3.0, 3.0]) * 4)


def test_the_result_has_the_partials_column_order():
    merged = merge_view_partials(_two_partitions(), FULL, SUMS, MINS, MAXS, SET_COLS, _flatten)
    assert list(merged.columns) == list(_partial_agg_columns(FULL, SUMS, MINS, MAXS, SETS, lambda c: None))


def test_one_row_per_view_row_whatever_the_partition_count():
    parts = pd.concat([_two_partitions(), _two_partitions(), _two_partitions()])
    merged = merge_view_partials(parts, FULL, SUMS, MINS, MAXS, SET_COLS, _flatten)
    assert list(merged.index) == ["x", "y"]
    assert merged.loc["x", "size_sum"] == 33


def test_a_column_no_partial_holds_is_skipped():
    merged = merge_view_partials(_two_partitions(), FULL, ["absent"], MINS, MAXS, SET_COLS, _flatten)
    assert "absent_sum" not in merged.columns and "lo_min" in merged.columns


def test_only_full_columns_need_no_other_arguments():
    merged = merge_view_partials(
        _two_partitions()[[c for c in _two_partitions().columns if c.startswith("dur_")]], FULL
    )
    assert list(merged.columns)[:2] == ["dur_sum", "dur_count"]


def test_set_columns_without_a_flatten_function_are_refused():
    with pytest.raises(ValueError, match="flatten_fn"):
        merge_view_partials(_two_partitions(), FULL, SUMS, MINS, MAXS, SET_COLS)


def _records(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "v": rng.integers(0, 6, n),
            "x": 1e9 + rng.normal(0, 2.0, n),
            "y": rng.normal(50.0, 5.0, n),
            "s": rng.integers(0, 100, n),
            "m": rng.normal(0, 1, n),
        }
    )


def _arrow_partials(frame, parts=3):
    pieces = np.array_split(np.arange(len(frame)), parts)
    return pd.concat(
        [
            partial_arrow_view_groupby(frame.iloc[ix], "v", ["x", "y"], ["s"], [], ["m"], [], lambda s: s)
            for ix in pieces
        ]
    )


def test_the_merge_keeps_every_dtype_of_the_partials():
    partials = _arrow_partials(_records())
    merged = merge_view_partials(partials, ["x", "y"])
    for name in merged.columns:
        assert merged[name].dtype == partials[name].dtype, name


def test_an_all_null_column_stays_its_type_after_replace():
    frame = _records()
    frame["y"] = np.nan  # null in every partition
    partials = _arrow_partials(frame)
    merged = merge_view_partials(partials, ["x", "y"])
    out = merged.replace(0, pd.NA)
    for name in ("y_min", "y_max", "y_sum"):
        assert merged[name].dtype == partials[name].dtype, name
        assert out[name].dtype == partials[name].dtype, name
        assert out[name].dtype != object
