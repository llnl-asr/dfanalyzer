import dask.dataframe as dd
import pandas as pd
from typing import List

from ..constants import COL_TIME, COL_TIME_END, COL_TIME_START


def assign_hierarchy(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values(
        [COL_TIME_START, COL_TIME_END],
        ascending=[True, False],
    ).reset_index(drop=True)

    event_ids: List[str] = []
    parent_ids: List[str] = []
    root_ids: List[str] = []
    depths: List[int] = []

    stack: List[dict] = []
    for i, row in pdf.iterrows():
        event_id = f"{row['pid']}-{row['tid']}-{i}"
        start = row[COL_TIME_START]
        end = row[COL_TIME_END]

        while stack and start >= stack[-1]["end"]:
            stack.pop()

        if stack and end <= stack[-1]["end"]:
            parent = stack[-1]["id"]
            root = stack[-1]["root"]
            depth = stack[-1]["depth"] + 1
        else:
            parent = ""
            root = event_id
            depth = 0

        event_ids.append(event_id)
        parent_ids.append(parent)
        root_ids.append(root)
        depths.append(depth)

        stack.append({"end": end, "id": event_id, "root": root, "depth": depth})

    pdf = pdf.copy()
    pdf["event_id"] = event_ids
    pdf["parent_id"] = parent_ids
    pdf["root_id"] = root_ids
    pdf["depth"] = depths
    return pdf


def compute_self_time(traces: dd.DataFrame) -> dd.DataFrame:
    child_time = traces.groupby("parent_id")[COL_TIME].sum().rename("child_time")
    traces = traces.merge(
        child_time.to_frame(),
        left_on="event_id",
        right_index=True,
        how="left",
    )
    traces["child_time"] = traces["child_time"].fillna(0)
    traces["self_time"] = traces[COL_TIME] - traces["child_time"]
    return traces


def set_stack_metrics(df: pd.DataFrame, job_time: float) -> pd.DataFrame:
    df = df.copy()

    def pick(pref: list[str]) -> str:
        for name in pref:
            if name in df.columns:
                return name
        return ""

    def safe_divide(numer, denom):
        try:
            return numer / denom
        except ZeroDivisionError:
            return pd.Series([pd.NA] * len(numer), index=numer.index)

    time_col = pick(["time_sum", "time"])
    self_col = pick(["self_time_sum", "self_time"])
    child_col = pick(["child_time_sum", "child_time"])
    parent_time_col = pick(["parent_time_first", "parent_time"])
    root_time_col = pick(["root_time_first", "root_time"])

    with pd.option_context("mode.use_inf_as_na", True):
        parent_denom = (
            df[parent_time_col].where(df[parent_time_col] != 0, pd.NA) if parent_time_col else None
        )
        root_denom = df[root_time_col].where(df[root_time_col] != 0, pd.NA) if root_time_col else None
        if time_col and parent_denom is not None:
            df["time_frac_parent"] = safe_divide(df[time_col], parent_denom)
        if self_col and parent_denom is not None:
            df["self_time_frac_parent"] = safe_divide(df[self_col], parent_denom)
        if child_col and parent_denom is not None:
            df["child_time_frac_parent"] = safe_divide(df[child_col], parent_denom)
        if time_col and child_col:
            df["child_time_frac_self"] = df[child_col] / df[time_col]
        if time_col and root_denom is not None:
            df["time_frac_root"] = safe_divide(df[time_col], root_denom)
        if self_col and root_denom is not None:
            df["self_time_frac_root"] = safe_divide(df[self_col], root_denom)
        if child_col and root_denom is not None:
            df["child_time_frac_root"] = safe_divide(df[child_col], root_denom)
        if root_time_col and job_time != 0:
            df["root_time_frac_job"] = safe_divide(df[root_time_col], job_time)
        if time_col and job_time != 0:
            df["time_frac_job"] = safe_divide(df[time_col], job_time)
    return df


def add_stack_time_context(main_view: dd.DataFrame, traces: dd.DataFrame) -> dd.DataFrame:
    if "root_id" not in main_view.columns:
        main_view = main_view.reset_index()

    parent_time_df = traces[["event_id", COL_TIME]].rename(
        columns={"event_id": "parent_id", COL_TIME: "parent_time"}
    )
    root_time_df = (
        traces[traces["depth"] == 0][["root_id", COL_TIME]]
        .rename(columns={COL_TIME: "root_time"})
        .drop_duplicates(subset=["root_id"])
    )
    main_view = main_view.merge(parent_time_df, on="parent_id", how="left")
    main_view = main_view.merge(root_time_df, on="root_id", how="left")
    return main_view
