"""The output writers on library flat views."""

import json
import math
import pathlib

import pytest
from dftracer.analyzer.checkpoint_table import read_table
from dftracer.analyzer.output import ConsoleOutput, FileOutput, JSONOutput
from dftracer.analyzer.types import AnalysisFact, AnalysisResult, FactScope, FactSeverity, FactWindow, RawStats
from dftracer.utils import DataFrame, Series

pytestmark = pytest.mark.smoke


def _result() -> AnalysisResult:
    by_proc = DataFrame.from_dict(
        {
            "proc_name": Series(["p0", "p1", "p2"]),
            "posix_time_sum": Series([1.5, math.inf, None]),
            "posix_count_sum": Series([3, None, 7], dtype="int64"),
            "posix_size_sum": Series([4096.0, math.nan, 0.0]),
            "posix_file_name_nunique": Series([2, 1, None], dtype="int64"),
            "posix_bw_mbps": Series([math.inf, 2.0, math.nan]),
        }
    ).set_index(["proc_name"])
    by_time = DataFrame.from_dict(
        {
            "time_range": Series([0, 1], dtype="int64"),
            "posix_time_proc_max": Series([0.5, None]),
            "posix_count_sum": Series([2, 8], dtype="int64"),
        }
    ).set_index(["time_range"])
    fact = AnalysisFact(
        fact_type="small_reads",
        window=FactWindow(run_id="r1", view_type="proc_name"),
        scope=FactScope(layer="posix"),
        evidence={"metrics": {"posix_time_sum": 1.5}},
        severity=FactSeverity(score=0.7, label="high"),
    )
    return AnalysisResult(
        additional_metrics={"proc_name": ["posix_bw_mbps"]},
        checkpoint_dir="",
        flat_views={("proc_name",): by_proc, ("time_range",): by_time},
        layers=["posix"],
        raw_stats=RawStats(
            job_time=10.0,
            time_granularity=1,
            time_resolution=1_000_000,
            trace_event_count=12,
            profile_event_count=0,
            total_event_count=12,
            unique_file_count=3,
            unique_host_count=1,
            unique_process_count=3,
        ),
        view_types=["proc_name", "time_range"],
        views={},
        _hlms={},
        _main_views={},
        _metric_boundaries={},
        analysis_facts={("proc_name",): [fact]},
    )


def test_file_output_bundle_reads_back(tmp_path: pathlib.Path) -> None:
    result = _result()
    FileOutput(path=str(tmp_path / "out")).handle_result(result)

    out = tmp_path / "out"
    for view_key, flat_view in result.flat_views.items():
        back = read_table(str(out / f"detail_view_{'_'.join(view_key)}.pfw.gz"))
        assert back.index_names() == flat_view.index_names()
        assert back.columns == flat_view.columns
        assert back.dtypes == flat_view.dtypes
        assert back.num_rows == flat_view.num_rows
        assert _same_cells(back, flat_view), view_key
    facts = [json.loads(line) for line in (out / "facts.jsonl").read_text().splitlines()]
    assert [f["fact_type"] for f in facts[0]["facts"]] == ["small_reads"]
    assert json.loads((out / "raw_stats.json").read_text())["total_event_count"] == 12


def _same_cells(a: DataFrame, b: DataFrame) -> bool:
    """Same row count and columns, row dicts equal, NaN equal to NaN."""
    if a.num_rows != b.num_rows or a.columns != b.columns:
        return False
    for ra, rb in zip(a.reset_index().to_dicts(), b.reset_index().to_dicts(), strict=True):
        for k in rb:
            va, vb = ra[k], rb[k]
            if isinstance(vb, float) and math.isnan(vb):
                if not (isinstance(va, float) and math.isnan(va)):
                    return False
            elif va != vb:
                return False
    return True


def test_json_output_has_no_nan_or_infinity(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "out.json"
    JSONOutput(file_path=str(path)).handle_result(_result())
    payload = json.loads(path.read_text())
    proc = payload["views"]["proc_name"]
    assert proc["summary"]["layers"]["posix"]["time_s"] is None
    assert proc["summary"]["layers"]["posix"]["count"] == 10
    assert proc["summary"]["layers"]["posix"]["num_files"] == 2
    assert proc["additional_metrics"]["posix_bw_mbps"]["non_null"] == 1
    time = payload["views"]["time_range"]["summary"]["layers"]["posix"]
    assert time["time_s"] == 0.5 and time["count"] == 10 and time["ops_per_s"] == 20.0


def test_console_output_prints_library_views() -> None:
    ConsoleOutput().handle_result(_result())


def test_file_output_keeps_lists_missing_values_and_bools(tmp_path: pathlib.Path) -> None:
    lists = DataFrame.from_dict(
        {
            "file_name": Series(["/a", "/b"]),
            "posix_count_sum": Series([3, None], dtype="int64"),
            "posix_time_sum": Series([1.5, math.nan]),
            "posix_seen": Series([True, None]),
            "posix_host": Series(["h", None]),
            "posix_dirs": Series([["/x"], []]),
        }
    ).set_index(["file_name"])
    result = _result()
    result.flat_views[("file_name",)] = lists
    FileOutput(path=str(tmp_path / "out")).handle_result(result)
    back = read_table(str(tmp_path / "out" / "detail_view_file_name.pfw.gz"))
    assert back.index_names() == ["file_name"]
    assert back.columns == lists.columns and back.dtypes == lists.dtypes
    assert _same_cells(back, lists)


def _single_view_result(flat: DataFrame, additional=None) -> AnalysisResult:
    result = _result()
    result.flat_views = {("time_range",): flat}
    result.analysis_facts = {}
    result.additional_metrics = additional or {}
    return result


def test_summary_keeps_integer_counts_exact(tmp_path: pathlib.Path) -> None:
    big = 2**53 + 1
    flat = DataFrame.from_dict(
        {
            "time_range": Series([0, 1], dtype="int64"),
            "posix_time_proc_max": Series([1.0, 2.0]),
            "posix_count_sum": Series([big, 0], dtype="int64"),
        }
    ).set_index(["time_range"])
    result = _single_view_result(flat)
    summary = JSONOutput()._create_summary(result, ("time_range",))
    assert summary.layer_metrics["posix"].count == big
    path = tmp_path / "out.json"
    JSONOutput(file_path=str(path)).handle_result(result)
    assert json.loads(path.read_text())["views"]["time_range"]["summary"]["layers"]["posix"]["count"] == big


def test_summary_without_a_count_column_on_several_rows() -> None:
    flat = DataFrame.from_dict(
        {
            "time_range": Series([0, 1], dtype="int64"),
            "posix_time_proc_max": Series([1.0, 2.0]),
            "u_posix_time_proc_max": Series([0.5, 0.0]),
        }
    ).set_index(["time_range"])
    metrics = JSONOutput()._create_summary(_single_view_result(flat), ("time_range",)).layer_metrics["posix"]
    assert metrics.count == 0 and metrics.u_count == 0 and metrics.u_time == 0.5


def test_json_additional_metrics_with_an_overflowing_mean_stay_valid(tmp_path: pathlib.Path) -> None:
    flat = DataFrame.from_dict(
        {
            "time_range": Series([0, 1], dtype="int64"),
            "posix_time_proc_max": Series([1.0, 2.0]),
            "huge": Series([1.7e308, 1.7e308]),
        }
    ).set_index(["time_range"])
    path = tmp_path / "out.json"
    JSONOutput(file_path=str(path)).handle_result(_single_view_result(flat, {"time_range": ["huge"]}))
    payload = json.loads(path.read_text())["views"]["time_range"]["additional_metrics"]["huge"]
    assert payload["non_null"] == 2 and payload["max"] == 1.7e308 and payload["mean"] is None


def test_summary_of_an_empty_view_without_count_and_size_columns() -> None:
    flat = DataFrame.from_dict(
        {
            "time_range": Series([], dtype="int64"),
            "posix_time_proc_max": Series([], dtype="float64"),
            "u_posix_time_proc_max": Series([], dtype="float64"),
        }
    ).set_index(["time_range"])
    metrics = JSONOutput()._create_summary(_single_view_result(flat), ("time_range",)).layer_metrics["posix"]
    assert metrics.count == 0 and metrics.u_count == 0 and metrics.u_size == 0


def test_file_output_writes_string_view_columns(tmp_path: pathlib.Path) -> None:
    """The ai and dlio flat views hold Arrow string_view text columns."""
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "proc_name": pa.array(["p0", "p1"], type=pa.string_view()),
            "posix_host": pa.array(["h", None], type=pa.string_view()),
            "posix_files": pa.array([["/a"], []], type=pa.list_(pa.string_view())),
            "posix_time_sum": pa.array([1.5, None]),
        }
    )
    frame = DataFrame.from_arrow(table).set_index(["proc_name"])
    result = _single_view_result(frame)
    result.flat_views = {("proc_name",): frame}
    FileOutput(path=str(tmp_path / "out")).handle_result(result)
    back = read_table(str(tmp_path / "out" / "detail_view_proc_name.pfw.gz"))
    assert back.index_names() == ["proc_name"]
    assert back.dtypes == frame.dtypes
    assert _same_cells(back, frame)


def test_file_output_writes_large_list_columns(tmp_path: pathlib.Path) -> None:
    """A large_list column reads back with the same values, as a list column."""
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "proc_name": pa.array(["p0", "p1", "p2"]),
            "posix_files": pa.array([["/a", "/b"], [], None], type=pa.large_list(pa.string())),
        }
    )
    frame = DataFrame.from_arrow(table).set_index(["proc_name"])
    result = _single_view_result(frame)
    result.flat_views = {("proc_name",): frame}
    FileOutput(path=str(tmp_path / "out")).handle_result(result)
    back = read_table(str(tmp_path / "out" / "detail_view_proc_name.pfw.gz"))
    assert back.index_names() == ["proc_name"]
    assert back["posix_files"].to_list() == [["/a", "/b"], [], None]
    assert _same_cells(back, frame)
