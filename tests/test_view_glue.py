"""The analyzer glue between the main view, the views and the flat views on library frames."""

import os
import pathlib
import re

import pandas as pd
import pytest
from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.analysis_utils import set_file_dir, set_file_pattern
from dftracer.analyzer.analyzer import _outer_join_on_index, _rows_in
from dftracer.analyzer.constants import FILE_PATTERN_PLACEHOLDER
from dftracer.utils import DataFrame, DType, Series

pytestmark = pytest.mark.smoke

FILES = {
    ("app#h0#1#1", 0): ["/d/a.npz", "/d/b.npz"],
    ("app#h0#1#1", 1): ["/d/b.npz", "/d/c.npz"],
    ("app#h1#2#2", 0): ["/d/a.npz"],
    ("app#h1#2#2", 1): [],
}


def _main_view() -> DataFrame:
    """A posix main view by proc_name and time_range whose file_name cells hold several names."""
    keys = list(FILES)
    return DataFrame.from_dict(
        {
            "proc_name": Series([p for p, _ in keys]),
            "time_range": Series([t for _, t in keys], dtype="int64"),
            "count": Series([3, 4, 5, 6], dtype="int64"),
            "time": Series([1.0, 2.0, 3.0, 4.0]),
            "file_name": Series([FILES[k] for k in keys]),
        }
    ).set_index(["proc_name", "time_range"])


@pytest.fixture
def analyzer(tmp_path: pathlib.Path, trace_copy):
    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=posix",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            "cluster=none",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            f"trace_path={trace_copy('dftracer-posix')}",
            "view_types=[proc_name,time_range]",
        ]
    )
    yield dfa.analyzer
    dfa.shutdown()


@pytest.mark.parametrize("max_bytes", [0, 256 * 1024**2])
def test_views_keep_the_distinct_count_of_a_set_column(analyzer, max_bytes):
    analyzer.view_materialize_max_bytes = max_bytes
    main_view = _main_view()
    view_types = ["proc_name", "time_range"]
    views = analyzer.compute_views(layer="posix", main_view=main_view, view_types=view_types)
    views.update(
        analyzer.compute_logical_views(layer="posix", main_view=main_view, views=views, view_types=view_types)
    )

    by_proc = views[("proc_name",)].reset_index().to_dicts()
    expected = {}
    for (proc, _), names in FILES.items():
        expected.setdefault(proc, set()).update(names)
    assert {row["proc_name"]: row["file_name_nunique"] for row in by_proc} == {p: len(n) for p, n in expected.items()}

    by_host = views[("proc_name", "host_name")].reset_index().to_dicts()
    hosts = {}
    for (proc, _), names in FILES.items():
        hosts.setdefault(proc.split("#")[1], set()).update(names)
    assert {row["host_name"]: row["file_name_nunique"] for row in by_host} == {h: len(n) for h, n in hosts.items()}


@pytest.mark.parametrize("keys", [["a", "c"], ["a", None], [None], []])
def test_rows_in_matches_pandas_query(keys):
    values = ["a", None, "b", "c", None]
    frame = DataFrame.from_dict({"k": Series(values), "v": Series(list(range(5)), dtype="int64")}).set_index(["k"])
    got = _rows_in(frame, "k", Series(keys, dtype="string"))
    pdf = pd.DataFrame({"k": pd.array(values, dtype="string"), "v": list(range(5))}).set_index("k")
    key_index = pd.Index(pd.array(keys, dtype="string"))
    expected = pdf.query("k in @keys", local_dict={"keys": key_index})
    assert got["v"].to_list() == expected["v"].tolist()


def test_outer_join_on_index_matches_the_pandas_merge():
    left = pd.DataFrame(
        {"k": pd.array(["b", "a", None], dtype="string"), "x": pd.array([1, 2, 3], dtype="Int64")}
    ).set_index("k")
    right = pd.DataFrame(
        {
            "k": pd.array(["c", "a", None], dtype="string"),
            "y": pd.array([0.5, None, 1.5], dtype="Float64"),
            "z": pd.array([True, False, None], dtype="boolean"),
        }
    ).set_index("k")
    expected = left.merge(right, how="outer", left_index=True, right_index=True).sort_index()
    got = _outer_join_on_index(DataFrame.from_pandas(left), DataFrame.from_pandas(right)).sort_index()

    assert got.index_names() == ["k"]
    assert got.dtypes == [DType.INT64, DType.FLOAT64, DType.BOOL]
    got = got.to_pandas(nullable=True)
    pd.testing.assert_frame_equal(got, expected, check_index_type=False)


def test_set_file_dir_and_pattern_match_the_pandas_map():
    names = ["/d/run_01/f12.npz", "f3", "/x", "//y/z", None]
    frame = DataFrame.from_dict({"file_name": Series(names), "v": Series(list(range(5)), dtype="int64")}).set_index(
        ["file_name"]
    )
    assert set_file_dir(frame)["file_dir"].to_list() == [None if n is None else os.path.dirname(n) for n in names]
    assert set_file_pattern(frame)["file_pattern"].to_list() == [
        None if n is None else re.sub("[0-9]+", FILE_PATTERN_PLACEHOLDER, n) for n in names
    ]
    no_file = DataFrame.from_dict({"proc_name": Series(["p"]), "v": Series([1])}).set_index(["proc_name"])
    assert set_file_dir(no_file).columns == ["v"]


def test_read_memory_budget_counts_the_index():
    from dftracer.analyzer.analysis_utils import frame_nbytes
    from dftracer.analyzer.typed_read import _frames_nbytes

    plain = DataFrame.from_dict({"k": Series(["a" * 100, "b" * 100]), "v": Series([1, 2], dtype="int64")})
    indexed = plain.set_index(["k"])
    assert frame_nbytes(indexed) == frame_nbytes(plain)
    assert frame_nbytes(indexed) > indexed.memory_usage(index=False).sum()
    assert _frames_nbytes([{"events": indexed, "profiles": None}]) == frame_nbytes(indexed)


@pytest.mark.parametrize(
    "host, host_hash, expected",
    [
        ("node1", "h1", "app#node1#7#8"),
        ("", "h1", "app#h1#7#8"),
        (None, "h1", "app#h1#7#8"),
        ("", None, "app#unknown#7#8"),
        (None, "", "app#unknown#7#8"),
    ],
)
def test_proc_name_falls_back_like_the_pandas_read_path(host, host_hash, expected):
    """An empty or missing host name takes the host hash, then ``unknown``, as the pandas read path did."""
    from dftracer.analyzer.dftracer import DFTracerAnalyzer
    from dftracer.analyzer.typed_read import compose_proc_name

    pid, tid = Series([7], dtype="int64"), Series([8], dtype="int64")
    traces = DataFrame.from_dict(
        {
            "host_name": Series([host], dtype="string"),
            "host_hash": Series([host_hash], dtype="string"),
            "pid": pid,
            "tid": tid,
        }
    )
    assert DFTracerAnalyzer._set_proc_names(traces)["proc_name"].to_list() == [expected]
    events = DataFrame.from_dict(
        {
            "host_name": Series([host], dtype="string"),
            "hhash": Series([host_hash], dtype="string"),
            "pid": pid,
            "tid": tid,
        }
    )
    assert compose_proc_name(events).to_list() == [expected]


@pytest.mark.parametrize("item", ["large_string", "int32", "float32", "date32"])
def test_a_missing_set_cell_is_an_empty_list_of_any_item_type(item):
    import datetime

    pa = pytest.importorskip("pyarrow")
    from dftracer.analyzer.frames.main_view import _empty_lists

    value = {"large_string": "a", "int32": 1, "float32": 1.5, "date32": datetime.date(2026, 1, 1)}[item]
    values = Series.from_arrow(pa.array([[value], None], type=pa.list_(getattr(pa, item)())))
    assert _empty_lists(values).to_list() == [[value], []]
