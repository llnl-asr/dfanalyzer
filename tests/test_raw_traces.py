import gzip
import json

import pandas as pd
import pytest
from dftracer.analyzer import init_with_hydra
from dftracer.utils import DataFrame

from tests.test_preset_detection import write_flavor_trace

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def analyze(trace, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    analyzer = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=posix",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={run_dir}/checkpoints",
            "cluster=none",
            f"hydra.run.dir={run_dir}",
            f"hydra.runtime.output_dir={run_dir}",
            f"trace_path={trace}",
        ]
    )
    return analyzer, analyzer.analyze_trace()


def test_result_traces_are_the_raw_events(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "dlio")
    with gzip.open(tmp_path / "trace" / "trace-1.pfw.gz", "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    expected = sorted((e["ts"], e["dur"], e["name"]) for e in events if e["ph"] == "X")

    analyzer, result = analyze(trace, tmp_path / "run")
    raw = pd.concat([table.to_pandas() for table in result.traces.pages(page_size=5)])

    assert sorted(zip(raw["ts"], raw["dur"], raw["name"])) == expected
    analyzer.shutdown()


def test_aggregated_traces_keep_the_grouped_frame(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "posix")
    analyzer, result = analyze(trace, tmp_path / "run")
    grouped = result.aggregated_traces
    assert isinstance(grouped, DataFrame)

    raw_rows = sum(table.num_rows for table in result.traces.pages())

    assert "ts" not in grouped.columns
    assert int(grouped["count"].sum()) == raw_rows
    analyzer.shutdown()


@pytest.mark.parametrize("present", [False, True])
def test_postread_of_an_empty_trace_keeps_the_int64_columns(present):
    """A profile-only trace has no events; its empty frame still has int64 acc_pat and count."""
    from dftracer.analyzer.dftracer import PROFILE_OUTPUT_COLUMNS, DFTracerAnalyzer
    from dftracer.utils import DType, Series

    analyzer = object.__new__(DFTracerAnalyzer)
    analyzer.assign_epochs = False
    empty = DataFrame.from_dict({c: Series([], dtype=dt) for c, dt in PROFILE_OUTPUT_COLUMNS.items()})
    if present:
        empty = empty.with_columns(acc_pat=Series([], dtype="int64"), count=Series([], dtype="int64"))
    else:
        empty = empty.drop(columns=[c for c in ("acc_pat", "count") if c in empty.columns])
    traces = analyzer.postread_trace(empty, view_types=["proc_name"])
    assert traces["acc_pat"].dtype == DType.INT64
    assert traces["count"].dtype == DType.INT64
