"""Parity of the utils-frame analysis steps with the pandas path.

The baseline in tests/data/parity was captured from the pandas path by
tests/data/parity/capture.py: for the dlio, posix and ai traces with view
types [proc_name, time_range], every `_compute_view` input and output, every
main view input and output and the flat views.
"""

import itertools as it
import json
import math
import pathlib
from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from betterframe import BetterFrame
from betterset import BetterSet
from dask.distributed import Client, LocalCluster
from dftracer.analyzer.analysis_utils import build_view_rename_map
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.checkpoint_table import read_table
from dftracer.analyzer.config import (
    AnalyzerPresetConfigAI,
    AnalyzerPresetConfigDLIO,
    AnalyzerPresetConfigPOSIX,
)
from dftracer.analyzer.constants import VIEW_TYPES
from dftracer.analyzer.frames.flat_metrics import _total, process_flat_metrics, set_additional_metrics
from dftracer.analyzer.frames.main_view import condition_masks, group_main_view, reconcile
from dftracer.analyzer.frames.view import aggregate_view
from dftracer.analyzer.utils.expr_utils import extract_numerator_and_denominators
from dftracer.utils import DataFrame, DType, Series
from dftracer.utils.dask import DaskFrame

from . import test_hybrid_profiles as hybrid
from .pandas_reference import (
    derive_call_stats,
    finalize_view_partials,
    fix_dtypes,
    fix_std_cols,
    merge_view_partials,
    partial_arrow_view_groupby,
    set_cross_layer_metrics,
    set_layer_metrics,
    set_main_metrics,
    set_unique_counts,
    set_view_metrics,
)

PARITY_DIR = pathlib.Path(__file__).parent / "data" / "parity"
DATASETS = {"dftracer-dlio": 24, "dftracer-ai": 28, "dftracer-posix": 2}
MAIN_VIEWS = {"dftracer-dlio": 12, "dftracer-ai": 14, "dftracer-posix": 1}
FLAT_VIEWS = 2

# Relative tolerance of every numeric comparison. Library group-by sums add in a
# different order than pandas, so the last bits can differ.
RTOL = 1e-12


def coerce_arrow_numerics_to_pandas_native(df: pd.DataFrame) -> pd.DataFrame:
    """Map pd.ArrowDtype int/float columns to pandas Int64/Float64."""
    if df.empty:
        return df
    for c in df.columns:
        dt = df[c].dtype
        if isinstance(dt, pd.ArrowDtype):
            pa_type = dt.pyarrow_dtype
            if pa.types.is_floating(pa_type):
                df[c] = df[c].astype("Float64")
            elif pa.types.is_integer(pa_type):
                df[c] = df[c].astype("Int64")
    return df


def baseline(path) -> pd.DataFrame:
    """A saved pandas-path frame with the pandas dtypes it was saved with."""
    return read_table(str(path)).to_pandas()


def _cells_kind(values: pd.Series) -> str:
    present = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA]
    return "list" if any(isinstance(v, (set, frozenset, list, tuple, np.ndarray)) for v in present) else "string"


def dtype_family(dtype, values: pd.Series | None = None) -> str:
    """int, float, string, bool or list, for a library DType or a pandas dtype."""
    if isinstance(dtype, DType):
        if dtype == DType.LIST:
            return "list"
        if dtype == DType.BOOL:
            return "bool"
        return "int" if dtype.is_integer else "float" if dtype.is_float else "string" if dtype.is_text else dtype.name
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype) and dtype != object:
        return "string"
    if dtype == object:
        return _cells_kind(values) if values is not None else "string"
    return str(dtype)


def _families(frame) -> tuple[dict, dict]:
    """The dtype family of each column and of each index level of a library or pandas frame."""
    if isinstance(frame, DataFrame):
        levels = frame.index_names()
        columns = dict(zip(frame.columns, frame.dtypes))
        index = {n: frame.get_level_values(n).dtype for n in levels}
        return {c: dtype_family(t) for c, t in columns.items()}, {n: dtype_family(t) for n, t in index.items()}
    columns = {c: dtype_family(frame[c].dtype, frame[c]) for c in frame.columns}
    index = {}
    for i, n in enumerate(frame.index.names):
        if n is not None:
            level = frame.index.get_level_values(i)
            index[n] = dtype_family(level.dtype, pd.Series(level))
    return columns, index


def assert_frames_match(got, expected: pd.DataFrame, rtol: float = RTOL) -> None:
    """Same columns in order, index, dtype families and missing values; numbers within `rtol` relative.

    `got` is a library frame or a pandas frame. A library dtype matches the pandas dtype of its
    family (int, float, string, bool, list). A list cell matches the sorted values of a set cell.
    """
    got_families, got_index_families = _families(got)
    expected_families, expected_index_families = _families(expected)
    if isinstance(got, DataFrame):
        got = got.to_pandas(nullable=True)
    assert list(got.columns) == list(expected.columns)
    assert list(got.index.names) == list(expected.index.names)
    assert list(got.index) == list(expected.index)
    assert got_index_families == expected_index_families
    for name in expected.columns:
        g, e = got[name], expected[name]
        assert got_families[name] == expected_families[name], (
            f"{name}: dtype {got_families[name]} != {expected_families[name]} ({e.dtype})"
        )
        assert (g.isna().to_numpy() == e.isna().to_numpy()).all(), f"{name}: missing values differ"
        present = ~e.isna().to_numpy()
        if expected_families[name] == "list":
            assert [list(g.iloc[i]) for i in np.flatnonzero(present)] == [
                sorted(e.iloc[i]) for i in np.flatnonzero(present)
            ], name
        elif e.dtype == object:
            assert [g.iloc[i] for i in np.flatnonzero(present)] == [e.iloc[i] for i in np.flatnonzero(present)], name
        elif pd.api.types.is_numeric_dtype(e.dtype):
            gv = g.to_numpy(dtype="float64", na_value=np.nan)[present]
            ev = e.to_numpy(dtype="float64", na_value=np.nan)[present]
            np.testing.assert_allclose(gv, ev, rtol=rtol, atol=0, err_msg=name)
        else:
            assert (g.to_numpy()[present] == e.to_numpy()[present]).all(), name


def _missing_cell(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def _list_column(cells: list) -> Series:
    """A LIST column of the sorted values of each set cell; a missing cell stays missing.

    With no value in any cell the column is a typed LIST of strings, as the analyzer makes it.
    """
    if any(not _missing_cell(v) and len(v) for v in cells):
        return Series([None if _missing_cell(v) else sorted(v) for v in cells])
    lists = Series([None] * len(cells), dtype="string").str.split("\x00")
    empty = [not _missing_cell(v) for v in cells]
    return lists.where(~Series(empty), Series([[]] * len(cells))) if any(empty) else lists


def as_frame(df: pd.DataFrame, lists: Iterable[str] = ()) -> DataFrame:
    """A pandas frame as a library frame with its index.

    A column of set or list cells, and each column named in `lists`, becomes a LIST of sorted values.
    """
    sets = [c for c in df.columns if c in set(lists) or (df[c].dtype == object and _cells_kind(df[c]) == "list")]
    if not sets:
        return DataFrame.from_pandas(df)
    keys = [n for n in df.index.names if n is not None]
    frame = DataFrame.from_pandas(df.drop(columns=sets).reset_index(keys))
    frame = frame.with_columns(**{c: _list_column(list(df[c])) for c in sets}).select(*keys, *df.columns)
    return frame.set_index(keys) if keys else frame


def _views():
    for dataset, count in DATASETS.items():
        manifest = json.loads((PARITY_DIR / dataset / "manifest.json").read_text())
        assert len(manifest["views"]) == count
        for n, meta in enumerate(manifest["views"]):
            yield pytest.param(dataset, n, meta, id=f"{dataset}-{n:02d}-{meta['layer']}-{meta['view_type']}")


@pytest.mark.smoke
@pytest.mark.parametrize("dataset, n, meta", list(_views()))
def test_view_matches_pandas_path(dataset, n, meta):
    records = read_table(str(PARITY_DIR / dataset / f"view_{n:03d}_in.pfw.gz"))
    expected = baseline(PARITY_DIR / dataset / f"view_{n:03d}_out.pfw.gz")
    got = aggregate_view(
        records, meta["view_type"], meta["view_types"], meta["logical_views"], meta["layer"], meta["time_sliced"]
    )
    assert_frames_match(got, expected)


@pytest.fixture(scope="module")
def threaded_client():
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, silence_logs="error")
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.mark.smoke
@pytest.mark.parametrize("dataset, n, meta", list(_views()))
def test_partitioned_view_matches_pandas_path(threaded_client, dataset, n, meta):
    """The DaskFrame group-by on an in-process client, over three row partitions, gives the pandas view."""
    records = read_table(str(PARITY_DIR / dataset / f"view_{n:03d}_in.pfw.gz"))
    expected = baseline(PARITY_DIR / dataset / f"view_{n:03d}_out.pfw.gz")
    got = aggregate_view(
        records,
        meta["view_type"],
        meta["view_types"],
        meta["logical_views"],
        meta["layer"],
        meta["time_sliced"],
        client=threaded_client,
        npartitions=3,
    )
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_partitioned_view_closes_every_dask_frame(threaded_client, monkeypatch):
    """The scattered, shuffled and grouped frames each hold worker memory until closed."""
    made, closed, events = [], [], []
    init, close, map_partitions = DaskFrame.__init__, DaskFrame.close, DaskFrame.map_partitions

    def track_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(id(self))

    def track_close(self):
        closed.append(id(self))
        events.append(("close", id(self)))
        close(self)

    def track_map_partitions(self, *args, **kwargs):
        events.append(("map_partitions", id(self)))
        return map_partitions(self, *args, **kwargs)

    monkeypatch.setattr(DaskFrame, "__init__", track_init)
    monkeypatch.setattr(DaskFrame, "close", track_close)
    monkeypatch.setattr(DaskFrame, "map_partitions", track_map_partitions)
    manifest = json.loads((PARITY_DIR / "dftracer-dlio" / "manifest.json").read_text())
    meta = manifest["views"][18]
    records = read_table(str(PARITY_DIR / "dftracer-dlio" / "view_018_in.pfw.gz"))
    expected = baseline(PARITY_DIR / "dftracer-dlio" / "view_018_out.pfw.gz")
    got = aggregate_view(
        records,
        meta["view_type"],
        meta["view_types"],
        meta["logical_views"],
        meta["layer"],
        meta["time_sliced"],
        client=threaded_client,
        npartitions=3,
    )
    assert_frames_match(got, expected)
    assert made and set(made) <= set(closed)
    scattered = made[0]
    assert events.index(("close", scattered)) < min(
        i for i, (kind, _) in enumerate(events) if kind == "map_partitions"
    )


@pytest.mark.smoke
def test_view_without_a_client_groups_one_frame(monkeypatch):
    """With no client the partitions would shuffle in this process, which only adds copies."""

    def no_dask_frame(*args, **kwargs):
        raise AssertionError("DaskFrame built without a client")

    monkeypatch.setattr(DaskFrame, "from_frames", no_dask_frame)
    manifest = json.loads((PARITY_DIR / "dftracer-dlio" / "manifest.json").read_text())
    meta = manifest["views"][18]
    records = read_table(str(PARITY_DIR / "dftracer-dlio" / "view_018_in.pfw.gz"))
    expected = baseline(PARITY_DIR / "dftracer-dlio" / "view_018_out.pfw.gz")
    got = aggregate_view(
        records,
        meta["view_type"],
        meta["view_types"],
        meta["logical_views"],
        meta["layer"],
        meta["time_sliced"],
        npartitions=3,
    )
    assert_frames_match(got, expected)


def test_partitioned_view_on_a_cluster_matches_pandas_path():
    """The DaskFrame group-by on two worker processes gives the pandas view of every dlio view."""
    manifest = json.loads((PARITY_DIR / "dftracer-dlio" / "manifest.json").read_text())
    with (
        LocalCluster(n_workers=2, threads_per_worker=1, processes=True, silence_logs=40) as cluster,
        Client(cluster) as client,
    ):
        for n, meta in enumerate(manifest["views"]):
            records = read_table(str(PARITY_DIR / "dftracer-dlio" / f"view_{n:03d}_in.pfw.gz"))
            expected = baseline(PARITY_DIR / "dftracer-dlio" / f"view_{n:03d}_out.pfw.gz")
            got = aggregate_view(
                records,
                meta["view_type"],
                meta["view_types"],
                meta["logical_views"],
                meta["layer"],
                meta["time_sliced"],
                client=client,
                npartitions=2,
            )
            assert_frames_match(got, expected)


@pytest.mark.smoke
def test_baseline_is_complete():
    assert sum(DATASETS.values()) == 54
    for dataset in DATASETS:
        manifest = json.loads((PARITY_DIR / dataset / "manifest.json").read_text())
        assert len(manifest["mains"]) == MAIN_VIEWS[dataset]
        assert len(manifest["flat"]) == FLAT_VIEWS
        mains = [baseline(PARITY_DIR / dataset / f"main_{n:03d}_out.pfw.gz") for n in range(len(manifest["mains"]))]
        assert any(not m.empty for m in mains)
        for n in range(FLAT_VIEWS):
            assert not baseline(PARITY_DIR / dataset / f"flat_{n:03d}.pfw.gz").empty


def _pandas_view(records, view_type, view_types, logical_views, layer, time_sliced):
    """The partial, merge and finalize view path as it ran on a pandas frame."""
    frame = BetterFrame(records).mutable()
    ops = frame.ops
    logical = set(it.chain.from_iterable(logical_views.values()))
    keep_object_cols = set(VIEW_TYPES) | set(view_types) | logical
    records = frame.native
    drop_cols = []
    for name in records.columns:
        dtype_str = str(records[name].dtype)
        if dtype_str in ("string", "category"):
            records[name] = records[name].astype("object")
            dtype_str = "object"
        if (
            dtype_str == "object"
            and name not in keep_object_cols
            and not any(name.endswith(vt) for vt in keep_object_cols)
        ):
            drop_cols.append(name)
    if drop_cols:
        records = records.drop(columns=drop_cols, errors="ignore")
    frame = BetterFrame(records)

    view_types_diff = set(VIEW_TYPES).difference(view_types)
    local_view_types_diff = set(frame.index_names()).difference([view_type])
    view_agg = {}
    for name in records.columns:
        if "_bin_" in name:
            view_agg[name] = ["sum"]
        elif any(map(name.endswith, view_types_diff)) or name in logical:
            view_agg[name] = [ops.set_union_flatten()]
        elif name.endswith("_sq"):
            view_agg[name] = ["sum"]
        elif name.endswith("_call_min"):
            view_agg[name] = ["min"]
        elif name.endswith("_call_max"):
            view_agg[name] = ["max"]
        elif pd.api.types.is_numeric_dtype(records[name].dtype):
            view_agg[name] = ["sum", "min", "max", "mean", "std"]
        else:
            raise TypeError(f"Unsupported dtype '{records[name].dtype}' for column '{name}'")
    view_agg.update({name: [ops.set_union()] for name in local_view_types_diff})

    full_cols, sum_cols, min_cols, max_cols, set_cols_items = [], [], [], [], []
    for name, aggs in view_agg.items():
        if name not in records.columns:
            continue
        if all(isinstance(a, str) for a in aggs):
            s = set(aggs)
            if s == {"sum", "min", "max", "mean", "std"}:
                full_cols.append(name)
            elif s == {"sum"}:
                sum_cols.append(name)
            elif s == {"min"}:
                min_cols.append(name)
            elif s == {"max"}:
                max_cols.append(name)
        else:
            set_cols_items.append((name, aggs[0]))

    prepared = frame.apply(fix_std_cols, std_cols=list(full_cols))
    partials = prepared.apply(
        partial_arrow_view_groupby,
        view_type,
        full_cols,
        sum_cols,
        min_cols,
        max_cols,
        set_cols_items,
        BetterSet.flatten,
    )
    merged = partials.pipe(
        lambda df: merge_view_partials(
            df, full_cols, sum_cols, min_cols, max_cols, [c for c, _ in set_cols_items], BetterSet.flatten
        )
    )
    final = merged.apply(finalize_view_partials, full_cols)
    final = final.pipe(lambda df: df.rename(columns=build_view_rename_map(df.columns)).replace(0, pd.NA))
    final = final.apply(derive_call_stats)
    final = final.apply(set_unique_counts, layer=layer)
    final = final.apply(fix_dtypes, time_sliced=time_sliced)
    final = final.apply(coerce_arrow_numerics_to_pandas_native)
    return final.finalize()


def _synthetic(sets: list) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [
            pd.array(["a", "a", "b", "b", "c"], dtype="string"),
            pd.array([1, 2, 1, 2, 1], dtype="int64"),
        ],
        names=["proc_name", "time_range"],
    )
    return pd.DataFrame(
        {
            "count": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "time": pd.array([0.5, 1.5, pd.NA, 2.5, 0.0], dtype="Float64"),
            "time_sq": pd.array([0.25, 2.25, 1.0, 6.25, 0.0], dtype="Float64"),
            "time_call_min": pd.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="Float64"),
            "time_call_max": pd.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype="Float64"),
            "size_bin_0_4k": pd.array([1, 0, 2, 0, 0], dtype="Int64"),
            "file_name": pd.Series(sets, index=index, dtype=object),
        },
        index=index,
    )


@pytest.mark.smoke
def test_set_columns_match_pandas_path():
    sets = [{"x", "y"}, {"y"}, {"z"}, None, {"x"}]
    records = _synthetic(sets)
    args = ("proc_name", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    expected = _pandas_view(records.copy(), *args)
    got = aggregate_view(as_frame(records), *args)
    assert "file_name_nunique" in expected.columns
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_all_null_sets_match_pandas_path():
    records = _synthetic([None] * 5)
    args = ("proc_name", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    frame = as_frame(records, lists=["file_name"])
    assert frame["file_name"].dtype == DType.LIST
    assert_frames_match(aggregate_view(frame, *args), _pandas_view(records.copy(), *args))


@pytest.mark.smoke
def test_empty_input_matches_pandas_path():
    records = _synthetic([{"x"}] * 5).iloc[:0]
    args = ("proc_name", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    frame = as_frame(records, lists=["file_name"])
    assert frame["file_name"].dtype == DType.LIST
    got = aggregate_view(frame, *args)
    expected = _pandas_view(records.copy(), *args)
    assert got.empty and expected.empty
    assert "file_name_nunique" in expected.columns
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_view_keys_keep_their_dtype_and_nan_is_missing():
    records = _synthetic([{"x"}] * 5)
    records["time"] = pd.array([0.5, np.nan, 1.5, 2.5, 0.0], dtype="float64")
    args = ("proc_name", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    got = aggregate_view(as_frame(records), *args)
    assert got.index_names() == ["proc_name"] and got.get_level_values("proc_name").dtype == DType.STRING
    by_time = aggregate_view(
        as_frame(records), "time_range", ["proc_name", "time_range"], {"file_name": []}, "posix", False
    )
    assert by_time.index_names() == ["time_range"] and by_time.get_level_values("time_range").dtype == DType.INT64
    assert_frames_match(got, _pandas_view(records.copy(), *args))


@pytest.mark.smoke
def test_view_drops_a_nan_key_like_pandas():
    records = _synthetic([{"x"}, {"y"}, None, {"z"}, {"x"}]).reset_index("time_range")
    records["time_range"] = pd.array([1.0, None, 2.0, None, 1.0], dtype="Float64")
    records = records.set_index("time_range", append=True)
    frame = with_nan(as_frame(records), "time_range")
    assert frame.get_level_values("time_range").is_nan().sum() == 2
    args = ("time_range", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    assert_frames_match(aggregate_view(frame, *args), _pandas_view(records.copy(), *args))


@pytest.mark.smoke
def test_set_cells_as_lists_add_no_value_for_a_missing_one():
    """A missing value or NaN adds nothing to a set union: an empty list, or a missing list for text."""
    from dftracer.analyzer.frames.view import _as_lists

    assert _as_lists(Series([1.0, math.nan, None], dtype="float64")).to_list() == [[1.0], [], []]
    assert _as_lists(Series([1, None], dtype="int64")).to_list() == [[1], []]
    assert _as_lists(Series(["a", None, ""], dtype="string")).to_list() == [["a"], None, [""]]
    assert _as_lists(Series(["a\x00b", None], dtype="string")).to_list() == [["a\x00b"], []]


@pytest.mark.smoke
def test_view_std_of_a_group_with_an_infinite_value_is_missing_like_pandas():
    records = _synthetic([{"x"}] * 5)
    records["time"] = pd.array([np.inf, 1.0, 2.0, 2.5, 0.25], dtype="Float64")
    args = ("proc_name", ["proc_name", "time_range"], {"file_name": []}, "posix", False)
    got = aggregate_view(as_frame(records), *args)
    expected = _pandas_view(records.copy(), *args)
    assert np.isinf(got["time_sum"].to_list()[0]) and np.isinf(got["time_proc_mean"].to_list()[0])
    assert got["time_proc_std"].to_list()[0] is None
    assert_frames_match(got, expected)


PRESETS = {
    "dftracer-dlio": AnalyzerPresetConfigDLIO,
    "dftracer-ai": AnalyzerPresetConfigAI,
    "dftracer-posix": AnalyzerPresetConfigPOSIX,
}


def _mains():
    for dataset in DATASETS:
        manifest = json.loads((PARITY_DIR / dataset / "manifest.json").read_text())
        for n, meta in enumerate(manifest["mains"]):
            yield pytest.param(dataset, n, meta, id=f"{dataset}-{n:02d}-{meta['layer']}")


@pytest.mark.smoke
@pytest.mark.parametrize("dataset, n, meta", list(_mains()))
def test_main_view_matches_pandas_path(dataset, n, meta):
    hlm = read_table(str(PARITY_DIR / dataset / f"main_{n:03d}_in.pfw.gz"))
    expected = baseline(PARITY_DIR / dataset / f"main_{n:03d}_out.pfw.gz")
    # The Dask path that made these files wrote NaN in *_ops where count and time are missing; pandas gives NA.
    for name in expected.columns[expected.dtypes == "Float64"]:
        expected[name] = expected[name].mask(np.isnan(expected[name].to_numpy(dtype=float, na_value=np.nan)))
    stub = SimpleNamespace(preset=PRESETS[dataset](), time_sliced=meta["time_sliced"])
    got = Analyzer._compute_main_view(stub, meta["layer"], hlm, meta["view_types"])
    assert_frames_match(got, expected)


def _pandas_main_view(hlm: pd.DataFrame, view_types: list, time_sliced: bool) -> pd.DataFrame:
    """The group-by and the steps after it of the pandas `_compute_main_view`."""
    agg = {}
    diff = set(VIEW_TYPES).difference(view_types)
    for name in hlm.columns:
        if name in ("cat", "io_cat", "acc_pat", "func_name"):
            continue
        if any(name.endswith(v) for v in diff):
            agg[name] = lambda cells: BetterSet.flatten(cells)
        elif name.endswith("_call_min"):
            agg[name] = "min"
        elif name.endswith("_call_max"):
            agg[name] = "max"
        else:
            agg[name] = lambda cells: cells.sum(min_count=1)
    view = hlm.groupby(view_types).agg(agg)
    view = set_main_metrics(view).replace(0, pd.NA)
    return fix_dtypes(view, time_sliced=time_sliced)


def assert_main_matches(got: DataFrame, expected: pd.DataFrame, suffix: str = "file_name") -> None:
    """The full frames match in column order; a set cell is a LIST of its sorted values, the rest through `assert_frames_match`."""
    assert list(got.columns) == list(expected.columns)
    set_names = [c for c in expected.columns if c.endswith(suffix)]
    for name in set_names:
        assert got[name].dtype == DType.LIST, name
        assert got[name].to_list() == [sorted(v) for v in expected[name]], name
    assert_frames_match(got.drop(columns=set_names), expected.drop(columns=set_names))


def _hlm(extra: dict | None = None) -> pd.DataFrame:
    """An HLM as the analyzer holds it: a zero is a missing value."""
    na = pd.NA
    index = pd.MultiIndex.from_arrays(
        [
            pd.array(["a", "a", "a", "b", "b", "c"], dtype="string"),
            np.array([1, 1, 2, 1, 2, 1], dtype="int64"),
            pd.array(["posix"] * 6, dtype="string"),
            np.zeros(6, dtype="int64"),
            np.zeros(6, dtype="int64"),
            np.array(["read", "write", "read", "read", "read", "read"], dtype=object),
        ],
        names=["proc_name", "time_range", "cat", "io_cat", "acc_pat", "func_name"],
    )
    data = {
        "time": pd.array([1.0, 2.0, na, 4.0, na, na], dtype="Float64"),
        "count": pd.array([1, 2, na, 4, 5, na], dtype="Int64"),
        "size": pd.array([10, na, na, 40, na, na], dtype="Int64"),
        "time_call_min": pd.array([0.5, 1.0, na, 2.0, na, na], dtype="Float64"),
        "time_call_max": pd.array([0.5, 1.0, na, 2.0, na, na], dtype="Float64"),
        "read_count": pd.array([1, na, na, 4, 5, na], dtype="Int64"),
        "read_time": pd.array([1.0, na, na, 4.0, na, na], dtype="Float64"),
        "read_size": pd.array([10, na, na, 40, na, na], dtype="Int64"),
        "size_bin_0_4k": pd.array([1, 2, na, na, 3, na], dtype="Int64"),
    }
    data.update(extra or {})
    return pd.DataFrame(data, index=index)


@pytest.mark.smoke
@pytest.mark.parametrize("time_sliced", [False, True])
def test_main_view_group_by_matches_pandas_with_missing_values(time_sliced):
    hlm = _hlm()
    view_types = ["proc_name", "time_range"]
    expected = _pandas_main_view(hlm.copy(), view_types, time_sliced)
    assert_frames_match(group_main_view(as_frame(hlm), view_types, time_sliced), expected)


@pytest.mark.smoke
def test_main_view_group_by_one_view_type_matches_pandas():
    hlm = _hlm()
    assert_frames_match(
        group_main_view(as_frame(hlm), ["proc_name"], False), _pandas_main_view(hlm.copy(), ["proc_name"], False)
    )


@pytest.mark.smoke
def test_main_view_group_by_masks_infinite_rates_like_pandas():
    # time is 0 where the size is positive, so bandwidth is infinite and must become missing
    hlm = _hlm()
    hlm["time"] = pd.array([0.0, 2.0, 0.0, 4.0, 1.0, 0.0], dtype="Float64")
    hlm["count"] = pd.array([1, 2, 1, 4, 5, 1], dtype="Int64")
    hlm["size"] = pd.array([10, 5, 1, 40, 7, 3], dtype="Int64")
    expected = _pandas_main_view(hlm.copy(), ["proc_name", "time_range"], False)
    got = group_main_view(as_frame(hlm), ["proc_name", "time_range"], False)
    assert got["bw"].isna().any()
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_main_view_group_by_set_columns_match_pandas():
    sets = [{"x"}, {"y"}, None, {"x", "z"}, {"z"}, None]
    hlm = _hlm({"file_name": pd.Series(sets, index=_hlm().index, dtype=object)})
    view_types = ["proc_name", "time_range"]
    got = group_main_view(as_frame(hlm), view_types, False)
    expected = _pandas_main_view(hlm.copy(), view_types, False)
    assert "file_name" in expected.columns
    assert_main_matches(got, expected)


HLM_KEYS = ["proc_name", "time_range", "cat", "io_cat", "acc_pat", "func_name"]


def _pandas_reconcile(trace_hlm: pd.DataFrame, profile_hlm: pd.DataFrame, view_types: list):
    """`Analyzer._reconcile_hlm` as it ran on pandas frames."""
    keys = list(dict.fromkeys(view_types + ["cat", "io_cat", "acc_pat", "func_name"]))
    trace, profile = trace_hlm.reset_index(), profile_hlm.reset_index()
    for key in keys:
        if trace[key].dtype != profile[key].dtype:
            trace[key] = trace[key].astype(profile[key].dtype)
    present = trace[keys].drop_duplicates().assign(_trace_present=1)
    profile = profile.merge(present, how="left", on=keys)
    overlap = int(profile["_trace_present"].fillna(0).sum())
    profile_only = profile[profile["_trace_present"].isna()].drop(columns=["_trace_present"])
    combined = pd.concat([trace, profile_only], ignore_index=True)
    bins = [c for c in combined.columns if "_bin_" in c]
    agg = {"time": "sum", "count": "sum", "size": "sum", **dict.fromkeys(bins, "sum")}
    hlm = combined.groupby(keys).agg(agg).replace(0, pd.NA)
    if bins:
        hlm[bins] = hlm[bins].astype("Int32")
    return hlm, overlap


def _rows(rows: list, bins: bool = False, key_dtypes: dict | None = None) -> pd.DataFrame:
    """HLM rows of (proc_name, time_range, func_name, time, count, size[, bin])."""
    key_dtypes = key_dtypes or {}
    frame = pd.DataFrame(
        {
            "proc_name": pd.array([r[0] for r in rows], dtype=key_dtypes.get("proc_name", "string")),
            "time_range": np.array([r[1] for r in rows], dtype=key_dtypes.get("time_range", "int64")),
            "cat": pd.array(["posix"] * len(rows), dtype="string"),
            "io_cat": np.ones(len(rows), dtype="int64"),
            "acc_pat": np.zeros(len(rows), dtype="int64"),
            "func_name": np.array([r[2] for r in rows], dtype=object),
            "time": pd.array([r[3] for r in rows], dtype="Float64"),
            "count": pd.array([r[4] for r in rows], dtype="Int64"),
            "size": pd.array([r[5] for r in rows], dtype="Int64"),
        }
    )
    if bins:
        frame["size_bin_0_4k"] = pd.array([r[6] for r in rows], dtype="Int64")
    return frame.set_index(HLM_KEYS)


VIEW_TYPES_PT = ["proc_name", "time_range"]
TRACE_ROWS = [("a", 1, "read", 1.0, 1, 10), ("a", 2, "read", 2.0, 2, 20), ("b", 1, "read", 3.0, 3, 30)]
PROFILE_ROWS = [
    ("a", 1, "read", 5.0, 5, 50, 1),
    ("c", 1, "read", 6.0, 6, 60, 2),
    ("c", 2, "write", 0.0, 0, 0, 0),
    ("b", 2, "read", 7.0, 7, 70, 4),
]


def _assert_reconcile(trace: pd.DataFrame, profile: pd.DataFrame) -> None:
    expected, expected_overlap = _pandas_reconcile(trace.copy(), profile.copy(), VIEW_TYPES_PT)
    got, overlap = reconcile(as_frame(trace), as_frame(profile), VIEW_TYPES_PT)
    assert overlap == expected_overlap
    assert_frames_match(got, expected.sort_index())


@pytest.mark.smoke
def test_reconcile_keeps_profile_rows_the_trace_lacks():
    trace, profile = _rows(TRACE_ROWS), _rows(PROFILE_ROWS, bins=True)
    got, overlap = reconcile(as_frame(trace), as_frame(profile), VIEW_TYPES_PT)
    assert overlap == 1
    assert len(got) == 6
    _assert_reconcile(trace, profile)


@pytest.mark.smoke
def test_reconcile_without_overlap_or_bins():
    trace = _rows(TRACE_ROWS)
    profile = _rows([("z", 9, "read", 1.0, 1, 1)])
    _assert_reconcile(trace, profile)


@pytest.mark.smoke
def test_reconcile_with_empty_sides():
    trace, profile = _rows(TRACE_ROWS), _rows(PROFILE_ROWS, bins=True)
    _assert_reconcile(trace.iloc[:0], profile)
    _assert_reconcile(trace, profile.iloc[:0])


@pytest.mark.smoke
@pytest.mark.parametrize("empty", ["profile", "trace"])
def test_reconcile_keeps_a_set_column_only_one_side_has(empty):
    """The side without the set column is empty, so the concat must still find one column type."""
    from dftracer.analyzer.trace_hlm import HLMConfig, _to_hlm

    sets = [{"/a", "/b"}, {"/a"}, None]
    trace = _rows(TRACE_ROWS).assign(file_name=pd.Series(sets, dtype=object).to_numpy())
    blank = _to_hlm(None, VIEW_TYPES_PT, HLMConfig())
    with_sets = as_frame(trace)
    if empty == "trace":
        got, overlap = reconcile(blank, with_sets, VIEW_TYPES_PT)
        expected, expected_overlap = _pandas_reconcile(blank.to_pandas(nullable=True), trace.copy(), VIEW_TYPES_PT)
    else:
        got, overlap = reconcile(with_sets, blank, VIEW_TYPES_PT)
        expected, expected_overlap = _pandas_reconcile(trace.copy(), blank.to_pandas(nullable=True), VIEW_TYPES_PT)
    assert overlap == expected_overlap == 0
    assert got["file_name"].dtype == DType.LIST
    assert got["file_name"].to_list() == [sorted(v or []) for v in sets]
    assert_frames_match(got.drop(columns=["file_name"]), expected.sort_index())


@pytest.mark.smoke
def test_reconcile_aligns_key_dtypes_to_the_profile():
    trace = _rows(TRACE_ROWS, key_dtypes={"time_range": "int32"})
    profile = _rows(PROFILE_ROWS, bins=True)
    _assert_reconcile(trace, profile)


@pytest.fixture
def dask_client():
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, silence_logs="error")
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.mark.parametrize(
    "name, lines, preset",
    [
        ("hybrid", hybrid.TRACE_CONTENT, None),
        ("full-hybrid", hybrid.FULL_TRACE_CONTENT, AnalyzerPresetConfigAI()),
    ],
)
def test_reconcile_matches_pandas_path_on_hybrid_traces(dask_client, tmp_path, monkeypatch, name, lines, preset):
    captured = []
    original = Analyzer._reconcile_hlm

    def spy(self, trace_hlm, profile_hlm, view_types, layer=None):
        captured.append((trace_hlm.copy(), profile_hlm.copy(), list(view_types)))
        return original(self, trace_hlm, profile_hlm, view_types, layer)

    monkeypatch.setattr(Analyzer, "_reconcile_hlm", spy)
    path = tmp_path / f"{name}.pfw.gz"
    hybrid._write_trace_lines(path, lines)
    analyzer = hybrid.make_analyzer(tmp_path=tmp_path, time_granularity=5, preset=preset)
    analyzer.analyze_trace(trace_path=str(path), view_types=VIEW_TYPES_PT)
    assert captured
    for trace, profile, view_types in captured:
        expected, expected_overlap = _pandas_reconcile(
            trace.to_pandas(nullable=True), profile.to_pandas(nullable=True), view_types
        )
        got, overlap = reconcile(trace, profile, view_types)
        assert overlap == expected_overlap
        assert_frames_match(got, expected.sort_index())


def _preset_conditions() -> list:
    conditions = set()
    for preset in PRESETS.values():
        for layer_metrics in preset().derived_metrics.values():
            conditions.update(layer_metrics.values())
    return sorted(conditions)


def test_preset_conditions_are_counted():
    assert len(_preset_conditions()) == 16


@pytest.mark.smoke
@pytest.mark.parametrize("condition", _preset_conditions())
def test_preset_condition_mask_matches_pandas_on_real_hlms(condition):
    checked = 0
    for dataset in DATASETS:
        manifest = json.loads((PARITY_DIR / dataset / "manifest.json").read_text())
        for n in range(len(manifest["mains"])):
            hlm = read_table(str(PARITY_DIR / dataset / f"main_{n:03d}_in.pfw.gz"))
            if hlm.empty:
                continue
            expected = hlm.to_pandas().eval(condition).to_numpy(dtype=bool)
            got = condition_masks(hlm.reset_index(), [condition])[condition]
            assert (got.to_numpy().astype(bool) == expected).all()
            checked += 1
    assert checked


FUNC_NAMES = [
    "open", "close", "xclose", ".close", "fclose", "close_dir", "opendir", "readdir", "lseek", "seek", "stat",
    "__xstat", "fstat", "item", "get_sample", "x.get_sample", "init", "preprocess", ".preprocess", "read", "write",
]  # fmt: skip


@pytest.mark.smoke
@pytest.mark.parametrize("condition", _preset_conditions())
def test_preset_condition_mask_matches_pandas_on_edge_names(condition):
    names = [*FUNC_NAMES, *FUNC_NAMES, "A", "b", ""]
    frame = pd.DataFrame({"io_cat": np.arange(len(names), dtype="int64") % 8, "func_name": names})
    expected = frame.eval(condition).to_numpy(dtype=bool)
    got = condition_masks(DataFrame.from_pandas(frame), [condition])[condition]
    assert (got.to_numpy().astype(bool) == expected).all()


DERIVED = {
    "rd": "io_cat == 1",
    "wr": "io_cat == 2",
    "meta": "io_cat == 3 and func_name.str.contains(\"stat\") and ~func_name.str.contains(\"dir\")",
    "any_io": "io_cat == 1 or io_cat == 2",
}
SIZE_DERIVED = ["rd", "wr"]


def _pandas_layer_main_view(hlm: pd.DataFrame, view_types: list, time_sliced: bool) -> pd.DataFrame:
    derived = set_layer_metrics(hlm, DERIVED, SIZE_DERIVED)
    return _pandas_main_view(derived, view_types, time_sliced)


def _layer_hlm(sets: list | None = None) -> pd.DataFrame:
    hlm = _hlm()
    names = ["read", "write", "read", "stat", "fstat", "readdir"]
    io_cat = [1, 2, 1, 3, 3, 3]
    frame = hlm.reset_index()
    frame["io_cat"] = np.array(io_cat, dtype="int64")
    frame["func_name"] = np.array(names, dtype=object)
    if sets is not None:
        frame["file_name"] = pd.Series(sets, dtype=object)
    return frame.set_index(HLM_KEYS)


@pytest.mark.smoke
@pytest.mark.parametrize("time_sliced", [False, True])
def test_main_view_with_derived_metrics_matches_pandas(time_sliced):
    hlm = _layer_hlm()
    view_types = ["proc_name", "time_range"]
    expected = _pandas_layer_main_view(hlm.copy(), view_types, time_sliced)
    got = group_main_view(as_frame(hlm), view_types, time_sliced, DERIVED, SIZE_DERIVED)
    assert "rd_size" in got.columns and "meta_size" not in got.columns
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_main_view_with_derived_metrics_and_one_view_type_matches_pandas():
    hlm = _layer_hlm()
    expected = _pandas_layer_main_view(hlm.copy(), ["proc_name"], False)
    assert_frames_match(group_main_view(as_frame(hlm), ["proc_name"], False, DERIVED, SIZE_DERIVED), expected)


@pytest.mark.smoke
def test_main_view_with_derived_set_columns_matches_pandas():
    sets = [{"x"}, {"y"}, None, {"x", "z"}, {"z"}, {"w"}]
    hlm = _layer_hlm(sets)
    view_types = ["proc_name", "time_range"]
    got = group_main_view(as_frame(hlm), view_types, False, DERIVED, SIZE_DERIVED)
    expected = _pandas_layer_main_view(hlm.copy(), view_types, False)
    set_names = sorted(c for c in expected.columns if c.endswith("file_name"))
    assert set_names == ["any_io_file_name", "file_name", "meta_file_name", "rd_file_name", "wr_file_name"]
    assert_main_matches(got, expected)


@pytest.mark.smoke
@pytest.mark.parametrize("time_sliced", [False, True])
def test_main_view_with_derived_set_columns_and_time_slices_matches_pandas(time_sliced):
    sets = [{"x"}, {"y"}, None, {"x", "z"}, {"z"}, {"w"}]
    hlm = _layer_hlm(sets)
    view_types = ["proc_name", "time_range"]
    got = group_main_view(as_frame(hlm), view_types, time_sliced, DERIVED, SIZE_DERIVED)
    expected = _pandas_layer_main_view(hlm.copy(), view_types, time_sliced)
    assert_main_matches(got, expected)


@pytest.mark.smoke
def test_main_view_when_keys_are_columns_matches_pandas():
    sets = [{"x"}, {"y"}, None, {"x", "z"}, {"z"}, {"w"}]
    hlm = _layer_hlm(sets)
    view_types = ["proc_name", "time_range"]
    expected = _pandas_layer_main_view(hlm.copy(), view_types, False)
    columns_hlm = hlm.reset_index(view_types)
    got = group_main_view(as_frame(columns_hlm), view_types, False, DERIVED, SIZE_DERIVED)
    assert_main_matches(got, expected)


@pytest.mark.smoke
def test_main_view_with_one_frequent_key_and_set_columns_matches_pandas():
    n = 40
    names = ["a"] * 30 + ["b"] * 6 + ["c"] * 4
    index = pd.MultiIndex.from_arrays(
        [
            pd.array(names, dtype="string"),
            np.ones(n, dtype="int64"),
            pd.array(["posix"] * n, dtype="string"),
            np.zeros(n, dtype="int64"),
            np.zeros(n, dtype="int64"),
            np.array([f"f{i}" for i in range(n)], dtype=object),
        ],
        names=HLM_KEYS,
    )
    hlm = pd.DataFrame(
        {
            "time": pd.array(np.arange(1, n + 1, dtype="float64"), dtype="Float64"),
            "count": pd.array(np.arange(1, n + 1), dtype="Int64"),
            "size": pd.array(np.arange(10, 10 * n + 1, 10), dtype="Int64"),
            "file_name": pd.Series([{f"f{i % 3}"} for i in range(n)], index=index, dtype=object),
        },
        index=index,
    )
    view_types = ["proc_name", "time_range"]
    got = group_main_view(as_frame(hlm), view_types, False)
    assert_main_matches(got, _pandas_main_view(hlm.copy(), view_types, False))


def with_nan(frame: DataFrame, name: str) -> DataFrame:
    """`frame` with the missing values of float column or index level `name` as NaN; `from_pandas` makes NaN null."""
    keys = frame.index_names()
    out = frame.reset_index()
    out = out.with_columns(**{name: out[name].fill_null(math.nan)}).select(*keys, *frame.columns)
    return out.set_index(keys) if keys else out


@pytest.mark.smoke
@pytest.mark.parametrize("nan_key", [False, True])
@pytest.mark.parametrize("derived", [False, True])
def test_main_view_drops_missing_keys_like_pandas(derived, nan_key):
    hlm = _layer_hlm().reset_index()
    hlm.loc[[1, 4], "time_range"] = np.nan
    hlm.loc[[2], "proc_name"] = pd.NA
    hlm = hlm.set_index(HLM_KEYS)
    view_types = ["proc_name", "time_range"]
    frame = as_frame(hlm)
    if nan_key:
        frame = with_nan(frame, "time_range")
        assert frame.get_level_values("time_range").is_nan().sum() == 2
    if derived:
        expected = _pandas_layer_main_view(hlm.copy(), view_types, False)
        got = group_main_view(frame, view_types, False, DERIVED, SIZE_DERIVED)
    else:
        expected = _pandas_main_view(hlm.copy(), view_types, False)
        got = group_main_view(frame, view_types, False)
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_main_view_with_a_condition_on_a_key_matches_pandas():
    hlm = _layer_hlm()
    view_types = ["proc_name", "func_name"]
    derived = {"rd": "func_name == 'read'", "both": "func_name == 'read' and io_cat == 1"}
    expected = _pandas_main_view(set_layer_metrics(hlm.copy(), derived, []), view_types, False)
    assert_frames_match(group_main_view(as_frame(hlm), view_types, False, derived), expected)


@pytest.mark.smoke
def test_main_view_with_a_condition_on_no_column_matches_an_always_true_one():
    """pandas cannot apply a constant condition; it must give what a condition every row meets gives."""
    hlm = _layer_hlm()
    expected = _pandas_main_view(set_layer_metrics(hlm.copy(), {"all": "io_cat == io_cat"}, []), ["proc_name"], False)
    assert_frames_match(group_main_view(as_frame(hlm), ["proc_name"], False, {"all": "1 == 1"}), expected)


@pytest.mark.smoke
@pytest.mark.parametrize("derived", [False, True])
def test_main_view_with_an_int_set_column_matches_pandas(derived):
    """A column named after a view type the view does not group by is a set even when it holds numbers."""
    hlm = _layer_hlm().reset_index(["time_range"])
    hlm["time_range"] = pd.array([1, 1, 2, None, 2, 3], dtype="Int64")
    view_types = ["proc_name"]
    cases = DERIVED if derived else {}
    expected = _pandas_main_view(set_layer_metrics(hlm.copy(), cases, SIZE_DERIVED), view_types, False)
    expected = expected.assign(
        **{
            c: [sorted(int(x) for x in v if not pd.isna(x)) for v in expected[c]]
            for c in expected.columns
            if c.endswith("time_range")
        }
    )
    got = group_main_view(as_frame(hlm), view_types, False, cases, SIZE_DERIVED)
    for c in [c for c in expected.columns if c.endswith("time_range")]:
        assert got[c].dtype == DType.LIST and got[c].to_list() == expected[c].tolist(), c
    sets = [c for c in expected.columns if c.endswith("time_range")]
    assert_frames_match(got.drop(columns=sets), expected.drop(columns=sets))


@pytest.mark.smoke
def test_main_view_with_unsorted_index_levels_matches_pandas():
    hlm = _hlm()
    levels = [level[::-1] for level in hlm.index.levels]
    codes = [len(level) - 1 - code for level, code in zip(hlm.index.levels, hlm.index.codes)]
    hlm.index = pd.MultiIndex(levels=levels, codes=codes, names=hlm.index.names)
    view_types = ["proc_name", "time_range"]
    expected = _pandas_main_view(hlm.copy(), view_types, False)
    assert_frames_match(group_main_view(as_frame(hlm), view_types, False), expected)


@pytest.mark.smoke
def test_main_view_with_a_condition_no_row_meets_matches_pandas():
    hlm = _layer_hlm()
    derived = {"never": "io_cat == 99"}
    expected = _pandas_main_view(set_layer_metrics(hlm.copy(), derived, []), ["proc_name"], False)
    assert_frames_match(group_main_view(as_frame(hlm), ["proc_name"], False, derived), expected)


@pytest.mark.smoke
def test_main_view_keeps_the_value_type_of_a_set_column_no_row_fills():
    """A derived set column that no row meets is empty lists of the base column's value type."""
    hlm = _layer_hlm().reset_index(["time_range"])
    hlm["time_range"] = pd.array([1, 1, 2, None, 2, 3], dtype="Int64")
    got = group_main_view(as_frame(hlm), ["proc_name"], False, {"rd": "io_cat == 1", "never": "io_cat == 7"}, ["rd"])
    assert got["never_time_range"].child(0).dtype == DType.INT64
    assert got["never_time_range"].to_list() == [[]] * got.num_rows
    both = got.head(1).concat(got.tail(1).with_columns(never_time_range=got["rd_time_range"].tail(1)))
    assert both["never_time_range"].child(0).dtype == DType.INT64


@pytest.mark.smoke
def test_main_view_with_more_conditions_than_int64_bits_matches_pandas():
    hlm = _layer_hlm()
    view_types = ["proc_name", "time_range"]
    derived = {f"c{i}": f"io_cat == {1 + i % 3} or time_range == {i}" for i in range(70)}
    assert len(set(derived.values())) == 70
    expected = _pandas_main_view(set_layer_metrics(hlm.copy(), derived, ["c0"]), view_types, False)
    got = group_main_view(as_frame(hlm), view_types, False, derived, ["c0"])
    assert "c69_time" in got.columns and "c0_size" in got.columns
    assert_frames_match(got, expected)


def _layer_args(preset, proc_based: bool) -> dict:
    layers = list(preset.layer_defs)
    return {
        "layers": layers,
        "layer_deps": preset.layer_deps,
        "async_layers": preset.async_layers,
        "derived_metrics": preset.derived_metrics,
        "time_boundary_layer": layers[0],
        "is_view_process_based": proc_based,
    }


def _flat_views():
    for dataset in DATASETS:
        for n in range(FLAT_VIEWS):
            yield pytest.param(dataset, n, id=f"{dataset}-{n}")


@pytest.mark.smoke
@pytest.mark.parametrize("dataset, n", list(_flat_views()))
def test_flat_metrics_match_pandas_path(dataset, n):
    flat = read_table(str(PARITY_DIR / dataset / f"flat_{n:03d}.pfw.gz"))
    args = _layer_args(PRESETS[dataset](), flat.index_names()[-1] == "proc_name")
    expected = set_view_metrics(flat.to_pandas(), None, args["is_view_process_based"])
    expected = set_cross_layer_metrics(expected, **args).sort_index(axis=1)
    got = process_flat_metrics(flat, additional={}, time_interval=1, **args)
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_ops_percentile_of_an_infinite_slope_matches_pandas_path():
    """A count fraction of exactly 0 gives an infinite ops slope; both paths must rank it alike."""
    flat = pd.DataFrame(
        {
            "x_count_sum": pd.array([0, 2, 2], dtype="Int64"),
            "x_time_sum": pd.array([3.0, 1.0, 2.0], dtype="Float64"),
        },
        index=pd.Index(["a", "b", "c"], name="proc_name"),
    )
    args = {
        "is_view_process_based": True,
        "layers": [],
        "layer_deps": {},
        "async_layers": [],
        "derived_metrics": {},
        "time_boundary_layer": "x",
    }
    expected = set_view_metrics(flat.copy(), None, True)
    expected = set_cross_layer_metrics(expected, **args).sort_index(axis=1)
    got = process_flat_metrics(as_frame(flat), additional={}, time_interval=1.0, **args)
    assert expected["x_ops_slope"].isna().iloc[0]
    assert_frames_match(got, expected)


ADDITIONAL = {
    "ratio": "read_time_sum / (read_count_sum + {epsilon})",
    "per_interval": "read_count_sum / {time_interval}",
    "share": "write_time_sum / read_time_sum.fillna(0)",
}


def _pandas_additional(view: pd.DataFrame, additional: dict, time_metric: str, interval: float) -> pd.DataFrame:
    """The additional-metrics step as it ran on a pandas frame."""
    for metric, condition in additional.items():
        condition = condition.format(epsilon=1e-9, time_interval=interval, time_metric=time_metric)
        view = view.eval(f"{metric} = {condition}")
        _, denominators = extract_numerator_and_denominators(condition)
        if denominators:
            mask = " & ".join(f"({d}.isna() | {d} == 0)" for d in denominators)
            view[metric] = view[metric].mask(view.eval(mask), pd.NA)
    return view


@pytest.mark.smoke
def test_additional_metrics_match_pandas_path():
    na = pd.NA
    flat = pd.DataFrame(
        {
            "read_time_sum": pd.array([1.0, 2.0, na, 0.0, 5.0], dtype="Float64"),
            "read_count_sum": pd.array([2, 0, 3, 0, na], dtype="Int64"),
            "write_time_sum": pd.array([1.0, 1.0, 1.0, na, 2.0], dtype="Float64"),
        },
        index=pd.Index(["a", "b", "c", "d", "e"], name="proc_name", dtype="string"),
    )
    expected = _pandas_additional(flat.copy(), ADDITIONAL, "time_sum", 5)
    got = (
        set_additional_metrics(DataFrame.from_pandas(flat.reset_index()), ADDITIONAL, "time_sum", 5)
        .to_pandas(nullable=True)
        .drop(columns="proc_name")
    )
    assert list(got.columns) == list(expected.columns)
    for name in ADDITIONAL:
        assert (got[name].isna().to_numpy() == expected[name].isna().to_numpy()).all(), name
        present = ~expected[name].isna().to_numpy()
        np.testing.assert_allclose(
            got[name].to_numpy("float64", na_value=np.nan)[present],
            expected[name].to_numpy("float64", na_value=np.nan)[present],
            rtol=RTOL,
            atol=0,
        )


BIG = 9007199254740993


@pytest.mark.smoke
def test_additional_metrics_keep_the_expression_type_like_pandas():
    na = pd.NA
    additional = {
        "plus": "x_count_sum + 1",
        "greater": "x_count_sum > 1",
        "ratio": "x_time_sum / (x_count_sum + {epsilon})",
        "scaled": "x_time_sum * {time_interval}",
    }
    flat = pd.DataFrame(
        {
            "x_count_sum": pd.array([BIG, 0, 3, na, 2], dtype="Int64"),
            "x_time_sum": pd.array([1.0, 2.0, na, 1.0, 5.0], dtype="Float64"),
        },
        index=pd.Index(["a", "b", "c", "d", "e"], name="proc_name", dtype="string"),
    )
    expected = _pandas_additional(flat.copy(), additional, "time_sum", 5)
    got = set_additional_metrics(DataFrame.from_pandas(flat.reset_index()), additional, "time_sum", 5)
    got = got.to_pandas(nullable=True).drop(columns="proc_name").set_axis(expected.index)
    assert got["plus"].iloc[0] == BIG + 1
    assert_frames_match(got, expected)


@pytest.mark.smoke
def test_additional_metrics_with_a_form_the_native_eval_lacks_match_pandas():
    """An expression the native eval cannot transpile (`.isin`) goes through `expr.evaluate` like every other string."""
    additional = {"picked": "x_count_sum.isin([0, 3])"}
    flat = pd.DataFrame(
        {
            "x_count_sum": pd.array([1, 0, 3, pd.NA, 3], dtype="Int64"),
            "x_time_sum": pd.array([1.0, 2.0, 4.0, 1.0, pd.NA], dtype="Float64"),
        },
        index=pd.Index(["a", "b", "c", "d", "e"], name="proc_name", dtype="string"),
    )
    expected = _pandas_additional(flat.copy(), additional, "time_sum", 5)
    got = set_additional_metrics(DataFrame.from_pandas(flat.reset_index()), additional, "time_sum", 5)
    got = got.to_pandas(nullable=True).drop(columns="proc_name").set_axis(expected.index)
    assert [None if pd.isna(v) else bool(v) for v in got["picked"]] == [
        None if pd.isna(v) else bool(v) for v in expected["picked"]
    ]


def assert_flat_matches(got, expected: pd.DataFrame) -> None:
    """Every column within RTOL, except a percentile rank of a row whose slope ties with another row.

    The main-view group-by adds in a different order than pandas, so slopes that are
    mathematically equal differ by one ulp and the rank breaks the tie instead of averaging it.
    A differing rank must belong to a slope within RTOL of another row's slope.
    """
    ranks = [c for c in expected.columns if c.endswith("ops_percentile")]
    rest = [c for c in expected.columns if c not in ranks]
    assert list(got.columns) == list(expected.columns)
    assert_frames_match(got[rest], expected[rest])
    if isinstance(got, DataFrame):
        got = got.to_pandas(nullable=True)
    for name in ranks:
        g, e = got[name].to_numpy("float64", na_value=np.nan), expected[name].to_numpy("float64", na_value=np.nan)
        assert (np.isnan(g) == np.isnan(e)).all(), name
        differing = ~np.isclose(g, e, rtol=RTOL, atol=0, equal_nan=True)
        slope = expected[name.replace("ops_percentile", "ops_slope")].to_numpy("float64", na_value=np.nan)
        for i in np.flatnonzero(differing):
            tied = np.isclose(slope, slope[i], rtol=RTOL, atol=0)
            assert tied.sum() > 1, f"{name}[{i}] differs and is not tied"


@pytest.mark.parametrize(
    "dataset, preset", [("dftracer-posix", "posix"), ("dftracer-dlio", "dlio"), ("dftracer-ai", "ai")]
)
def test_flat_views_match_baseline_through_the_analyzer(dask_client, trace_copy, tmp_path, dataset, preset):
    from dftracer.analyzer import init_with_hydra

    overrides = [
        "analyzer=dftracer",
        f"analyzer/preset={preset}",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=False",
        f"cluster.scheduler_address={dask_client.scheduler.address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_copy(dataset)}",
        "view_types=[proc_name,time_range]",
    ]
    if preset in ("dlio", "ai"):
        overrides.append("analyzer.assign_epochs=True")
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
    finally:
        dfa.shutdown()
    assert len(result.flat_views) == FLAT_VIEWS
    for n, flat in enumerate(result.flat_views.values()):
        expected = baseline(PARITY_DIR / dataset / f"flat_{n:03d}.pfw.gz")
        assert_flat_matches(flat, expected)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "values",
    [
        [1.0, math.nan, None, 4.0],
        [1, None, 3, 4],
        [1.0, math.nan, 3.0, 4.0],
        [1, 2, 3, 4],
    ],
)
def test_nan_as_null_as_float_keeps_values_and_makes_nan_missing(values):
    from dftracer.analyzer.analysis_utils import nan_as_null

    got = nan_as_null(Series(values), as_float=True)
    assert got.dtype == DType.FLOAT64
    assert got.to_list() == [None if v is None or math.isnan(v) else float(v) for v in values]


def _numpy_total(values: list) -> float:
    a = np.array([math.nan if v is None else v for v in values], dtype="float64")
    return float(np.sum(a, where=~np.isnan(a)))


# NumPy 2.2 adds a masked sum in blocks of 8192 values and NumPy 2.5 does not, so their last
# bits differ past 8192 values. `_total` follows 2.5. The 2.3 cut is assumed; 2.2.6 and 2.5.5 are checked.
_NUMPY_BLOCKS = tuple(int(p) for p in np.__version__.split(".")[:2]) < (2, 3)


def _assert_numpy_total(got: float, values: list, where) -> None:
    want = _numpy_total(values)
    if _NUMPY_BLOCKS and len(values) > 8192:
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=0.0), where
    else:
        assert got == want, where


@pytest.mark.smoke
@pytest.mark.parametrize("seed", range(6))
def test_total_adds_in_numpy_order(seed):
    """Bit-equal to NumPy's masked sum, on arrays that cross every pairwise block size."""
    rng = np.random.default_rng(seed)
    for n in [0, 1, 7, 8, 9, 127, 128, 129, 1000, 8191, 8192, 8193, 20000, 50000]:
        values = (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8, n)).tolist()
        for missing in (0.0, 0.01, 0.3):
            holes = rng.random(n) < missing
            cells = [None if h and i % 2 else math.nan if h else v for i, (v, h) in enumerate(zip(values, holes))]
            _assert_numpy_total(_total(Series(cells, dtype="float64")), cells, (n, missing))


@pytest.mark.smoke
def test_total_adds_in_numpy_order_on_the_parity_flat_views():
    checked = 0
    for dataset in DATASETS:
        for n in range(FLAT_VIEWS):
            flat = read_table(str(PARITY_DIR / dataset / f"flat_{n:03d}.pfw.gz"))
            for name, dtype in zip(flat.columns, flat.dtypes):
                if dtype.is_numeric:
                    cells = flat[name].cast("float64").to_list()
                    _assert_numpy_total(_total(flat[name]), cells, (dataset, n, name))
                    checked += 1
    assert checked > 100
