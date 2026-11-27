# SPDX-License-Identifier: MIT
"""The view aggregation's std does not lose its digits when the mean is large
next to the spread: per-partition partials merged by the pairwise formula."""

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.full]

pytest.importorskip("pyarrow")

from .pandas_reference import (
    finalize_view_partials,
    merge_view_partials,
    partial_arrow_view_groupby,
)

REL = 1e-9


def run(frame, parts, view="v", cols=("x",)):
    """partial per partition -> merge -> finalize, the analyzer's pipeline."""
    pieces = np.array_split(np.arange(len(frame)), parts)
    partials = [
        partial_arrow_view_groupby(frame.iloc[ix], view, list(cols), [], [], [], [], lambda s: s)
        for ix in pieces
        if len(ix)
    ]
    merged = merge_view_partials(pd.concat(partials), list(cols))
    return finalize_view_partials(merged, list(cols))


def truth(frame, view="v", col="x"):
    """numpy's two-pass std per group (pandas' own groupby std loses about 4e-9
    at an offset of 1e9, so it is no reference)."""
    stds = {
        k: np.std(g[col].dropna().to_numpy(), ddof=1) if g[col].count() > 1 else np.nan for k, g in frame.groupby(view)
    }
    return pd.DataFrame({"std": pd.Series(stds)})


@pytest.mark.parametrize("parts", [1, 2, 5, 16])
@pytest.mark.parametrize("offset,spread", [(0.0, 1.0), (1e6, 1.0), (1e9, 3.0), (1e12, 5.0)])
def test_std_matches_a_two_pass_reference(offset, spread, parts):
    rng = np.random.default_rng(parts)
    n = 4000
    frame = pd.DataFrame({"v": rng.integers(0, 7, n), "x": offset + rng.normal(0, spread, n)})
    got = run(frame, parts)
    want = truth(frame)
    std = got["x_std"].astype("float64")
    assert np.allclose(std.sort_index().to_numpy(), want["std"].sort_index().to_numpy(), rtol=REL, atol=0)


def test_the_sum_of_squares_route_would_lose_every_digit():
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(0, 1, 20000)
    sq_formula = (np.sum(x * x) - np.sum(x) ** 2 / len(x)) / (len(x) - 1)
    exact = np.var(x, ddof=1)
    assert abs(sq_formula - exact) / exact > 1e-3  # the failure this change removes
    got = run(pd.DataFrame({"v": 0, "x": x}), 4)
    assert abs(got["x_std"].astype("float64").iloc[0] ** 2 - exact) / exact < REL


def test_keys_missing_from_some_partitions_and_single_rows():
    frame = pd.DataFrame(
        {
            "v": ["a"] * 6 + ["b"] * 3 + ["c"],
            "x": [1e9 + 0.5, 1e9 + 1.5, 1e9, 1e9 + 2, 1e9 + 3, 1e9 + 4, 5.0, 7.0, 6.0, 42.0],
        }
    )
    got = run(frame, 5)  # "c" has one row: its std is undefined
    want = truth(frame)
    std = got["x_std"].astype("float64")
    for key in ("a", "b"):
        assert std[key] == pytest.approx(want.loc[key, "std"], rel=REL)
    assert np.isnan(std["c"])
    assert got.loc["a", "x_sum"] == pytest.approx(frame[frame.v == "a"].x.sum())


def test_nulls_are_skipped():
    rng = np.random.default_rng(3)
    x = pd.Series(1e8 + rng.normal(0, 2, 3000))
    x[::9] = np.nan
    frame = pd.DataFrame({"v": rng.integers(0, 4, 3000), "x": x})
    got = run(frame, 6)
    assert np.allclose(
        got["x_std"].astype("float64").sort_index().to_numpy(),
        truth(frame)["std"].sort_index().to_numpy(),
        rtol=REL,
    )
