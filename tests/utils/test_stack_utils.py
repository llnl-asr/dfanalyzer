import pandas as pd
import dask.dataframe as dd

from dftracer.analyzer.constants import COL_TIME, COL_TIME_END, COL_TIME_START
from dftracer.analyzer.utils.stack_utils import (
    add_stack_time_context,
    assign_hierarchy,
    compute_self_time,
    set_stack_metrics,
)


def test_assign_hierarchy_nested_stack():
    pdf = pd.DataFrame(
        {
            "pid": [0, 0, 0, 0, 0],
            "tid": [0, 0, 0, 0, 0],
            COL_TIME_START: [0, 1, 2, 6, 11],
            COL_TIME_END: [10, 8, 4, 7, 12],
        }
    )

    out = assign_hierarchy(pdf)

    # Event IDs are assigned after sorting; verify consistent stack structure.
    root_id = out.loc[0, "event_id"]
    assert out.loc[0, "depth"] == 0
    assert out.loc[0, "parent_id"] == ""
    assert out.loc[0, "root_id"] == root_id

    assert out.loc[1, "parent_id"] == root_id
    assert out.loc[1, "root_id"] == root_id
    assert out.loc[1, "depth"] == 1

    assert out.loc[2, "parent_id"] == out.loc[1, "event_id"]
    assert out.loc[2, "root_id"] == root_id
    assert out.loc[2, "depth"] == 2

    assert out.loc[3, "parent_id"] == out.loc[1, "event_id"]
    assert out.loc[3, "root_id"] == root_id
    assert out.loc[3, "depth"] == 2

    assert out.loc[4, "depth"] == 0
    assert out.loc[4, "parent_id"] == ""
    assert out.loc[4, "root_id"] == out.loc[4, "event_id"]


def test_compute_self_time_rollup():
    pdf = pd.DataFrame(
        {
            "event_id": ["r", "c1", "c2"],
            "parent_id": ["", "r", "r"],
            COL_TIME: [10.0, 3.0, 2.0],
        }
    )
    traces = dd.from_pandas(pdf, npartitions=1)
    result = compute_self_time(traces).compute()

    root = result.loc[result["event_id"] == "r"].iloc[0]
    child1 = result.loc[result["event_id"] == "c1"].iloc[0]
    child2 = result.loc[result["event_id"] == "c2"].iloc[0]

    assert root["child_time"] == 5.0
    assert root["self_time"] == 5.0
    assert child1["child_time"] == 0.0
    assert child1["self_time"] == 3.0
    assert child2["child_time"] == 0.0
    assert child2["self_time"] == 2.0


def test_add_stack_time_context():
    traces_pdf = pd.DataFrame(
        {
            "event_id": ["r", "c1"],
            "root_id": ["r", "r"],
            "depth": [0, 1],
            COL_TIME: [10.0, 4.0],
        }
    )
    traces = dd.from_pandas(traces_pdf, npartitions=1)

    main_view_pdf = pd.DataFrame(
        {
            "parent_id": ["", "r"],
            "root_id": ["r", "r"],
        }
    )
    main_view = dd.from_pandas(main_view_pdf, npartitions=1)
    out = add_stack_time_context(main_view, traces).compute()

    root_row = out.iloc[0]
    child_row = out.iloc[1]
    assert pd.isna(root_row["parent_time"])
    assert root_row["root_time"] == 10.0
    assert child_row["parent_time"] == 10.0
    assert child_row["root_time"] == 10.0


def test_set_stack_metrics_fractions():
    df = pd.DataFrame(
        {
            "time_sum": [5.0, 1.0],
            "self_time_sum": [3.0, 1.0],
            "child_time_sum": [2.0, 0.0],
            "parent_time": [10.0, 0.0],
            "root_time": [20.0, 0.0],
        }
    )

    out = set_stack_metrics(df, job_time=40.0)

    assert out.loc[0, "time_frac_parent"] == 0.5
    assert out.loc[0, "self_time_frac_parent"] == 0.3
    assert out.loc[0, "child_time_frac_parent"] == 0.2
    assert out.loc[0, "child_time_frac_self"] == 0.4
    assert out.loc[0, "time_frac_root"] == 0.25
    assert out.loc[0, "self_time_frac_root"] == 0.15
    assert out.loc[0, "child_time_frac_root"] == 0.1
    assert out.loc[0, "root_time_frac_job"] == 0.5
    assert out.loc[0, "time_frac_job"] == 0.125

    assert pd.isna(out.loc[1, "time_frac_parent"])
    assert pd.isna(out.loc[1, "self_time_frac_parent"])
    assert pd.isna(out.loc[1, "child_time_frac_parent"])
    assert pd.isna(out.loc[1, "time_frac_root"])
    assert pd.isna(out.loc[1, "self_time_frac_root"])
    assert pd.isna(out.loc[1, "child_time_frac_root"])