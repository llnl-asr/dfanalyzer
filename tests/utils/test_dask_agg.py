import numpy as np
import pandas as pd
import dask.dataframe as dd
import pytest

from dftracer.analyzer.utils.dask_agg import quantile_stats, unique_set, unique_set_flatten


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _build_ddf(values, groups=None, npartitions=2):
    if groups is None:
        groups = ["A"] * len(values)
    pdf = pd.DataFrame({"g": groups, "val": values})
    return dd.from_pandas(pdf, npartitions=npartitions)


def _get_stats_cell(df: pd.DataFrame, group_key: str, col_name: str):
    # df has MultiIndex columns: ("val", "qX_qY_stats")
    return df.loc[group_key, ("val", col_name)]


def _expected_quantile_stats(values, qmin, qmax):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]
    if arr.size == 0:
        return [np.nan, np.nan, np.nan]
    q_low, q_high = np.quantile(arr, [qmin, qmax])
    mask = (arr >= q_low) & (arr <= q_high)
    arr_f = arr[mask]
    if arr_f.size == 0:
        return [np.nan, np.nan, np.nan]
    return [float(np.mean(arr_f)), float(np.std(arr_f)), int(arr_f.size)]


def test_quantile_stats_basic_excludes_zeros_and_nan():
    # Zeros should be excluded in chunk (replaced with NaN and dropped)
    values = [0, 1, 2, 3, 4, 5, np.nan]
    ddf = _build_ddf(values, groups=["A"] * len(values), npartitions=3)

    out = ddf.groupby("g").agg({"val": [quantile_stats(0.01, 0.99), quantile_stats(0.25, 0.75)]}).compute()

    stats_all = _get_stats_cell(out, "A", "q1_q99_stats")
    stats_iqr = _get_stats_cell(out, "A", "q25_q75_stats")

    assert isinstance(stats_all, list) and len(stats_all) == 3
    assert isinstance(stats_iqr, list) and len(stats_iqr) == 3

    exp_all = _expected_quantile_stats(values, 0.01, 0.99)
    exp_iqr = _expected_quantile_stats(values, 0.25, 0.75)
    for got, exp in zip(stats_all, exp_all):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert pytest.approx(got, rel=1e-6) == exp
    for got, exp in zip(stats_iqr, exp_iqr):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert pytest.approx(got, rel=1e-6) == exp


def test_quantile_stats_empty_after_filter_returns_nan_triplet():
    # All values are zeros -> replaced to NaN -> empty list
    values = [0, 0, 0]
    ddf = _build_ddf(values, groups=["A", "A", "A"], npartitions=1)

    out = ddf.groupby("g").agg({"val": [quantile_stats(0.01, 0.99)]}).compute()
    stats_all = _get_stats_cell(out, "A", "q1_q99_stats")
    assert isinstance(stats_all, list) and len(stats_all) == 3
    assert np.isnan(stats_all[0]) and np.isnan(stats_all[1]) and np.isnan(stats_all[2])


def test_quantile_stats_multiple_groups_and_partitions():
    values = [0, 1, 2, 0, 10, 20]
    groups = ["A", "A", "A", "B", "B", "B"]
    ddf = _build_ddf(values, groups=groups, npartitions=2)

    out = ddf.groupby("g").agg({"val": [quantile_stats(0.05, 0.95)]}).compute()

    stats_a = _get_stats_cell(out, "A", "q5_q95_stats")
    stats_b = _get_stats_cell(out, "B", "q5_q95_stats")

    exp_a = _expected_quantile_stats([0, 1, 2], 0.05, 0.95)
    exp_b = _expected_quantile_stats([0, 10, 20], 0.05, 0.95)
    for got, exp in zip(stats_a, exp_a):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert pytest.approx(got, rel=1e-6) == exp
    for got, exp in zip(stats_b, exp_b):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert pytest.approx(got, rel=1e-6) == exp


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
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
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
    ddf = dd.from_pandas(df, npartitions=2)
    res = (
        ddf.groupby(["g", "p"])
        .agg({"col": unique_set()})
        .groupby(["p"])
        .agg({"col": unique_set_flatten()})
        .compute()["col"]
    )
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
    ddf = dd.from_pandas(df, npartitions=2)
    res = (
        ddf.groupby(["g", "p", "q"])
        .agg({"col": unique_set()})
        .groupby(["p", "q"])
        .sum()
        .groupby(["q"])
        .agg({"col": unique_set_flatten()})
        .compute()["col"]
    )
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
    ddf = dd.from_pandas(df, npartitions=2)
    res = (
        ddf.groupby(["g", "p"])
        .agg({"col": unique_set()})
        .groupby(["p"])
        .agg({"col": unique_set_flatten()})
        .compute()["col"]
    )
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
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
    assert set(res.loc["a"]) == {1, 2}
    assert set(res.loc["b"]) == {3}


def test_unique_set_empty_dataframe_returns_empty_series():
    df = pd.DataFrame({"g": pd.Series(dtype="object"), "col": pd.Series(dtype="object")})
    ddf = dd.from_pandas(df, npartitions=2)
    res = ddf.groupby("g").agg({"col": unique_set()}).compute()["col"]
    assert len(res) == 0
