import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .types import Layer, MetricBoundaries, Score


INTENSITY_MIN = 1 / 1024
INTENSITY_MAX = 1 / 1024**3
INTENSITY_BINS = np.geomspace(INTENSITY_MAX, INTENSITY_MIN, num=5)
PERCENTAGE_BINS = [0, 0.25, 0.5, 0.75, 0.9]
SCORE_NAMES = [
    Score.TRIVIAL.value,
    Score.LOW.value,
    Score.MEDIUM.value,
    Score.HIGH.value,
    Score.CRITICAL.value,
]
SCORE_BINS = [1, 2, 3, 4, 5]
SLOPE_BINS = [
    np.tan(np.deg2rad(15)),  # ~0.27
    np.tan(np.deg2rad(30)),  # ~0.58
    np.tan(np.deg2rad(45)),  # 1.0
    np.tan(np.deg2rad(60)),  # ~1.73
    np.tan(np.deg2rad(75)),  # ~3.73
]


def _find_metric(metrics, suffix):
    return [m for m in metrics if m.endswith(suffix)]


def _find_metric_pairs(metrics: pd.MultiIndex, metric_type1: str, metric_type2: str, agg_type: str):
    map1 = {
        metric_name[: -len(metric_type1)]: (metric_name, agg)
        for metric_name, agg in metrics
        if metric_name.endswith(metric_type1) and agg == agg_type
    }
    map2 = {
        metric_name[: -len(metric_type2)]: (metric_name, agg)
        for metric_name, agg in metrics
        if metric_name.endswith(metric_type2) and agg == agg_type
    }
    common_prefixes = set(map1.keys()).intersection(map2.keys())
    return [(map1[prefix], map2[prefix]) for prefix in sorted(list(common_prefixes))]


def set_main_metrics(df: pd.DataFrame):
    count_cols = [col for col in df.columns if col.endswith('count')]
    size_cols = [col for col in df.columns if col.endswith('size')]

    for size_col in size_cols:
        bw_col = size_col.replace('size', 'bw')
        count_col = size_col.replace('size', 'count')
        intensity_col = size_col.replace('size', 'intensity')
        time_col = size_col.replace('size', 'time')
        df[size_col] = np.where(df[size_col] > 0, df[size_col], np.nan)
        df[bw_col] = np.where(df[size_col] > 0, df[size_col] / df[time_col], np.nan)
        df[intensity_col] = np.where(df[size_col] > 0, df[count_col] / df[size_col], np.nan)

    for count_col in count_cols:
        ops_col = count_col.replace('count', 'ops')
        time_col = count_col.replace('count', 'time')
        df[ops_col] = df[count_col] / df[time_col]

    return df.sort_index(axis=1)


def set_view_metrics(df: pd.DataFrame, is_view_process_based: bool, time_granularity: float):
    metrics = set(df.columns.get_level_values(0))

    std_cols = [(metric, 'std') for metric in metrics]
    min_cols = [(metric, 'min') for metric in metrics]
    max_cols = [(metric, 'max') for metric in metrics]

    for std_col, min_col, max_col in zip(std_cols, min_cols, max_cols):
        if std_col not in df.columns:
            continue
        df.loc[df[min_col] == df[max_col], std_col] = 0

    for metric in metrics:
        if metric.endswith('count') or metric.endswith('size'):
            df[(metric, 'per')] = df[(metric, 'sum')] / df[(metric, 'sum')].sum()
        elif metric.endswith('time'):
            if is_view_process_based:
                df[(metric, 'per')] = df[(metric, 'max')] / df[(metric, 'max')].sum()
                df[(metric, 'util')] = df[(metric, 'max')] / time_granularity
            else:
                df[(metric, 'per')] = df[(metric, 'sum')] / df[(metric, 'sum')].sum()
                df[(metric, 'util')] = df[(metric, 'sum')] / time_granularity

    for count_per_col, time_per_col in _find_metric_pairs(df.columns, 'count', 'time', 'per'):
        metric, _ = count_per_col
        ops_metric = metric.replace('count', 'ops')
        ops_slope = df[time_per_col] / df[count_per_col]
        df[(ops_metric, 'pct')] = ops_slope.rank(pct=True)
        df[(ops_metric, 'slope')] = ops_slope

    return df.sort_index(axis=1)


def set_cross_layer_metrics(
    df: pd.DataFrame,
    layer_defs: Dict[Layer, str],
    layer_deps: Dict[Layer, Optional[Layer]],
    async_layers: List[Layer],
    is_view_process_based: bool,
) -> pd.DataFrame:
    time_metric = 'time_sum' if is_view_process_based else 'time_max'
    compute_time_metric = f"compute_{time_metric}"

    metric_cols = []

    # Set overhead time metrics
    for layer, parent in layer_deps.items():
        if not parent:
            continue
        child_layers = [child for child, parent in layer_deps.items() if parent == layer]
        if not child_layers:
            continue
        overhead_time_col = f"{layer}_overhead_{time_metric}"
        child_times = sum(df[f"{child}_{time_metric}"].fillna(0) for child in child_layers)
        df[overhead_time_col] = np.maximum(df[f"{layer}_{time_metric}"] - child_times, 0)
        df[overhead_time_col] = df[overhead_time_col].astype('Float64')
        metric_cols.append(overhead_time_col)

    # Set unoverlapped times if there is compute time
    if compute_time_metric in df.columns:
        # Set unoverlapped time metrics
        for async_layer in async_layers:
            time_col = f"{async_layer}_{time_metric}"
            compute_times = df[compute_time_metric].fillna(0)
            time_series = df[time_col].astype('Float64')
            compute_series = compute_times.astype('Float64')
            unoverlapped_series = (time_series - compute_series).clip(lower=0)
            df[f"u_{time_col}"] = pd.array(unoverlapped_series, dtype='Float64')

    return df.replace([np.inf, -np.inf], pd.NA).sort_index(axis=1)
