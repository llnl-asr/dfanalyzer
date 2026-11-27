"""The set union the analyzer groups with on a DaskFrame: the cells of a column
as lists of their values, joined by `set_union(typed=True)` per group across
partitions."""

import numpy as np
import pandas as pd
import pytest
from dftracer.utils import DataFrame, col
from dftracer.utils.dask import DaskFrame

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _atoms(cell) -> list:
    """The values of a set cell, or of one value, as a list; a missing value is an empty list."""
    if isinstance(cell, (set, frozenset, list, tuple)):
        return [v.item() if hasattr(v, "item") else v for v in cell if not pd.isna(v)]
    return [] if pd.isna(cell) else [cell.item() if hasattr(cell, "item") else cell]


def union_by(frame: pd.DataFrame, keys, column: str = "col", npartitions: int = 2) -> pd.Series:
    """The set of the `column` cells of each `keys` group, over `npartitions` row slices."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    data = {k: frame[k].tolist() for k in keys}
    data[column] = [_atoms(v) for v in frame[column]]
    native = DataFrame.from_dict(data)
    step = max(1, -(-len(native) // npartitions))
    slices = [native.slice(start, step) for start in range(0, len(native), step)] or [native]
    parts = DaskFrame.from_frames(slices).shuffle(keys)
    grouped = parts.map_partitions(lambda p: p.group_by(keys).agg(**{column: col(column).set_union(typed=True)}))
    return grouped.to_frame().to_pandas(index=keys)[column]


def test_unique_set_scalar_column_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "col": 1},
            {"g": "a", "col": 2},
            {"g": "a", "col": 2},
            {"g": "b", "col": 3},
            {"g": "b", "col": 4},
            {"g": "b", "col": 4},
        ]
    )
    res = union_by(df, "g")
    assert set(res.loc["a"]) == {1, 2}
    assert set(res.loc["b"]) == {3, 4}


def test_unique_set_flatten_grouped_two_stage_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "p": "x", "col": 1},
            {"g": "a", "p": "x", "col": 2},
            {"g": "a", "p": "y", "col": 2},
            {"g": "b", "p": "y", "col": 3},
            {"g": "b", "p": "y", "col": 4},
            {"g": "b", "p": "y", "col": 4},
        ]
    )
    stage = union_by(df, ["g", "p"]).reset_index()
    res = union_by(stage, ["p"])
    assert set(res.loc["x"]) == {1, 2}
    assert set(res.loc["y"]) == {2, 3, 4}


def test_unique_set_flatten_grouped_three_stage_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "p": "x", "q": "z", "col": 1},
            {"g": "a", "p": "x", "q": "z", "col": 2},
            {"g": "a", "p": "y", "q": "w", "col": 2},
            {"g": "b", "p": "y", "q": "w", "col": 3},
            {"g": "b", "p": "y", "q": "w", "col": 4},
            {"g": "b", "p": "y", "q": "z", "col": 4},
        ]
    )
    first = union_by(df, ["g", "p", "q"]).reset_index()
    second = union_by(first, ["p", "q"]).reset_index()
    res = union_by(second, ["q"])
    assert set(res.loc["z"]) == {1, 2, 4}
    assert set(res.loc["w"]) == {2, 3, 4}


def test_unique_set_flatten_handles_strings_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "p": "x", "col": "foo"},
            {"g": "a", "p": "x", "col": "bar"},
            {"g": "a", "p": "y", "col": "bar"},
            {"g": "b", "p": "y", "col": "baz"},
        ]
    )
    stage = union_by(df, ["g", "p"]).reset_index()
    res = union_by(stage, ["p"])
    assert set(res.loc["x"]) == {"foo", "bar"}
    assert set(res.loc["y"]) == {"bar", "baz"}


def test_unique_set_handles_missing_values_via_dask():
    df = pd.DataFrame(
        [
            {"g": "a", "col": 1},
            {"g": "a", "col": np.nan},
            {"g": "a", "col": 2},
            {"g": "b", "col": pd.NA},
            {"g": "b", "col": 3},
            {"g": "b", "col": np.nan},
        ]
    )
    df["col"] = df["col"].astype("Int64")
    res = union_by(df, "g")
    assert set(res.loc["a"]) == {1, 2}
    assert set(res.loc["b"]) == {3}


def test_unique_set_empty_dataframe_returns_empty_series():
    df = pd.DataFrame({"g": pd.Series(dtype="object"), "col": pd.Series(dtype="object")})
    res = union_by(df, "g")
    assert len(res) == 0
