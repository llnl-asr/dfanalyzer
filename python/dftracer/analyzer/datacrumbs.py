import dask.dataframe as dd
import numpy as np
import pandas as pd
from typing import List

from .analysis_utils import fix_dtypes, fix_std_cols, set_unique_counts
from .constants import COL_COUNT, COL_PROC_NAME, COL_TIME
from .dftracer import DFTracerAnalyzer
from .metrics import set_main_metrics
from .types import ViewType
from .utils.dask_utils import flatten_column_names
from .utils.log_utils import log_block
from .utils.stack_utils import (
    add_stack_time_context,
    assign_hierarchy,
    compute_self_time,
    set_stack_metrics,
)


def _add_hierarchy_columns(traces: dd.DataFrame) -> dd.DataFrame:
    meta = traces._meta.copy()
    meta = meta.assign(
        event_id=pd.Series(dtype="string"),
        parent_id=pd.Series(dtype="string"),
        root_id=pd.Series(dtype="string"),
        depth=pd.Series(dtype="int64"),
    )
    return (
        traces.groupby(["pid", "tid"])
        .apply(assign_hierarchy, meta=meta)
        .reset_index(drop=True)
    )


class DataCrumbsAnalyzer(DFTracerAnalyzer):

    def read_trace(self, trace_path, extra_columns, extra_columns_fn):
        traces = super().read_trace(trace_path, extra_columns, extra_columns_fn)
        if self.preset.name == "dynamic":
            with log_block("set_dynamic_layer_defs"):
                layers = traces['cat'].unique().compute().tolist()
                self.layers = layers
                for layer in layers:
                    self.preset.derived_metrics[layer] = {}
                    self.preset.layer_defs[layer] = f"cat == '{layer}'"
                    self.preset.layer_deps[layer] = None
        return traces

    def postread_trace(self, traces, view_types):
        traces = super().postread_trace(traces, view_types)
        if self.preset.name == "stack":
            traces = _add_hierarchy_columns(traces)
            traces = compute_self_time(traces)
            self._stack_traces = traces
        return traces

    def _compute_high_level_metrics(
        self,
        traces: dd.DataFrame,
        view_types: list,
        partition_size: str,
    ) -> dd.DataFrame:
        if self.preset.name != "stack":
            return super()._compute_high_level_metrics(traces, view_types, partition_size)

        extra_cols = ["cat", "func_name", "parent_id", "root_id", "depth"]
        present_extra = [col for col in extra_cols if col in traces.columns]
        group_cols = list(dict.fromkeys(list(view_types) + present_extra))

        agg: dict[str, str] = {}
        if COL_TIME in traces.columns:
            agg[COL_TIME] = "sum"
        if "self_time" in traces.columns:
            agg["self_time"] = "sum"
        if "child_time" in traces.columns:
            agg["child_time"] = "sum"
        if COL_COUNT in traces.columns:
            agg[COL_COUNT] = "sum"
        if "size" in traces.columns:
            agg["size"] = "sum"
        for col in traces.columns:
            if col in group_cols:
                continue
            if "_bin_" in col:
                agg[col] = "sum"

        split_out = max(1, int(np.sqrt(traces.npartitions)))
        return (
            traces.groupby(group_cols)
            .agg(agg, split_out=split_out)
            .repartition(partition_size=partition_size)
        )

    def _compute_main_view(
        self,
        layer,
        hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
    ) -> dd.DataFrame:
        if self.preset.name != "stack":
            return super()._compute_main_view(layer, hlm, view_types, partition_size)

        with log_block("drop_and_set_metrics", layer=layer):
            hlm = hlm.map_partitions(self.set_layer_metrics, derived_metrics=self.preset.derived_metrics.get(layer, {}))

        stack_keys = ["parent_id", "root_id", "depth"]
        group_keys = list(view_types) + stack_keys
        main_view_agg = {}
        for col in hlm.columns:
            if col in group_keys:
                continue
            if pd.api.types.is_numeric_dtype(hlm[col].dtype):
                main_view_agg[col] = "sum"
            else:
                main_view_agg[col] = "first"

        main_view = hlm.groupby(group_keys).agg(main_view_agg, split_out=hlm.npartitions)
        if any(col.endswith("count") for col in main_view.columns) and any(
            col.endswith("size") for col in main_view.columns
        ):
            main_view = main_view.map_partitions(set_main_metrics)
        main_view = (
            main_view.replace(0, pd.NA)
            .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
            .persist()
        )
        return add_stack_time_context(main_view, traces=self._stack_traces)

    def _compute_view(
        self,
        layer,
        records: dd.DataFrame,
        view_key,
        view_type: str,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        if self.preset.name != "stack":
            return super()._compute_view(layer, records, view_key, view_type, view_types)

        stack_keys = ["parent_id", "root_id", "depth"]
        group_keys = [view_type] + stack_keys
        if not set(stack_keys).issubset(records.columns):
            records = records.reset_index()

        view_agg = {}
        for col in records.columns:
            if col in group_keys:
                continue
            if col in ["parent_time", "root_time"]:
                view_agg[col] = ["first"]
            elif pd.api.types.is_numeric_dtype(records[col].dtype):
                view_agg[col] = ["sum", "min", "max", "mean", "std"]
            else:
                view_agg[col] = ["first"]

        std_cols = [col for col, aggs in view_agg.items() if isinstance(aggs, list) and "std" in aggs]
        records = records.map_partitions(fix_std_cols, std_cols=std_cols)

        pre_view = records
        if view_type != COL_PROC_NAME and COL_PROC_NAME in pre_view.columns:
            pre_view = pre_view.groupby(group_keys + [COL_PROC_NAME]).sum().reset_index()

        view = pre_view.groupby(group_keys).agg(view_agg)
        view = flatten_column_names(view)

        stack_key_cols = {"parent_id", "root_id", "depth", "parent_id_first", "root_id_first", "depth_first"}

        def replace_zero_metrics(pdf: pd.DataFrame) -> pd.DataFrame:
            metric_cols = [
                c
                for c in pdf.columns
                if c not in stack_key_cols and pd.api.types.is_numeric_dtype(pdf[c])
            ]
            pdf = pdf.copy()
            pdf[metric_cols] = pdf[metric_cols].replace(0, pd.NA)
            return pdf

        job_time = self.get_job_time(self._stack_traces).compute()
        view = (
            view.map_partitions(replace_zero_metrics)
            .map_partitions(set_unique_counts, layer=layer)
            .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
            .map_partitions(set_stack_metrics, job_time=job_time)
            .persist()
        )
        return view

    