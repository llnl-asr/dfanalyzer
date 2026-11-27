"""Flat-view metrics on dftracer-utils frames.

`set_view_metrics`, `set_cross_layer_metrics` and `set_additional_metrics` are the
steps of the same names in `metrics.py` and the additional-metrics step of the analyzer.
`process_flat_metrics` runs the three on a flat view.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from dftracer.utils import DataFrame, Series

from ..analysis_utils import nan_as_null
from ..metrics import _find_metric_pairs
from ..utils.expr_utils import extract_numerator_and_denominators
from .expr import evaluate

_PAIRWISE_BLOCK = 8
_PAIRWISE_LEAF = 128


def _pairwise(values: list[float], lo: int, n: int) -> float:
    """NumPy's pairwise sum of `values[lo:lo + n]`.

    Adapted from `pairwise_sum` in numpy/_core/src/umath/loops_utils.h.src
    (https://github.com/numpy/numpy, BSD-3-Clause).
    """
    if n < _PAIRWISE_BLOCK:
        total = 0.0
        for i in range(lo, lo + n):
            total += values[i]
        return total
    if n <= _PAIRWISE_LEAF:
        r = values[lo : lo + _PAIRWISE_BLOCK]
        end = lo + n - n % _PAIRWISE_BLOCK
        for i in range(lo + _PAIRWISE_BLOCK, end, _PAIRWISE_BLOCK):
            for j in range(_PAIRWISE_BLOCK):
                r[j] += values[i + j]
        total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(end, lo + n):
            total += values[i]
        return total
    half = n // 2
    half -= half % _PAIRWISE_BLOCK
    return _pairwise(values, lo, half) + _pairwise(values, lo + half, n - half)


def _total(series: Series) -> float:
    """`Series.sum()` as pandas adds it: `np.sum(a, where=~np.isnan(a))`, a missing value as NaN.

    The order of the additions decides the last digits of every fraction and the
    ties of the percentile ranks, so the library sum is not used. NumPy adds each
    run of values between two skipped ones as one pairwise sum.
    """
    # The library sum has no numpy pairwise order (dftracer-utils branch feat/analyzer-port-gaps: pairwise sum).
    values = series.cast("float64").to_list()
    total = 0.0
    start = 0
    for i, v in enumerate(values):
        if v is None or v != v:  # noqa: PLR0124
            if i > start:
                total += _pairwise(values, start, i - start)
            start = i + 1
    if len(values) > start:
        total += _pairwise(values, start, len(values) - start)
    return total


def _missing(like: Series) -> Series:
    return like.where(like != like)  # noqa: PLR0124


def _finite(series: Series) -> Series:
    return series.where((series != math.inf) & (series != -math.inf))


def _fraction(series: Series, total: float) -> Series:
    return series / total if total > 0 else _missing(series)


def _merge(frame: DataFrame, new: Mapping[str, Series]) -> tuple[DataFrame, list[str]]:
    if not new:
        return frame, []
    fixed = {name: _finite(v).astype("float64") for name, v in new.items()}
    overlap = [c for c in fixed if c in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    keys = list(frame.columns)
    return frame.with_columns(**fixed).select(*keys, *fixed), list(fixed)


def set_view_metrics(frame: DataFrame, is_view_process_based: bool) -> tuple[DataFrame, list[str]]:
    """The pandas path's `set_view_metrics` without the unused metric boundaries; returns the frame and the columns it set."""
    count_metric = "count_sum"
    size_proc = "size_sum" if is_view_process_based else "size_proc_max"
    time_proc = "time_sum" if is_view_process_based else "time_proc_max"
    names = list(frame.columns)
    new: dict[str, Series] = {}

    for name in names:
        for metric, target in (
            (count_metric, "count_frac_total"),
            (size_proc, "size_proc_frac_total"),
            (time_proc, "time_proc_frac_total"),
        ):
            if name.endswith(metric):
                new[name.replace(metric, target)] = _fraction(frame[name], _total(frame[name]))
                break
    for name in names:
        for metric, target in (("size_sum", "size_call_frac_total"), ("time_sum", "time_call_frac_total")):
            if name.endswith(metric):
                new[name.replace(metric, target)] = _fraction(frame[name], _total(frame[name]))
                break
    for count_frac, time_frac in _find_metric_pairs(list(new), "count_frac_total", "time_proc_frac_total"):
        slope = new[time_frac] / new[count_frac]
        # pandas ranks the infinite slopes it later hides, and a NaN slope is missing for the rank.
        ranked = slope.where(slope == slope)  # noqa: PLR0124
        new[count_frac.replace("count_frac_total", "ops_percentile")] = ranked.rank() / ranked.count()
        new[count_frac.replace("count_frac_total", "ops_slope")] = slope
    return _merge(frame, new)


def set_cross_layer_metrics(
    frame: DataFrame,
    layers: Sequence[str],
    layer_deps: Mapping[str, str | None],
    async_layers: Sequence[str],
    derived_metrics: Mapping[str, Mapping[str, str]],
    is_view_process_based: bool,
    time_boundary_layer: str,
) -> tuple[DataFrame, list[str]]:
    """The pandas path's `set_cross_layer_metrics` on a frame; returns the frame and the columns it set."""
    time_proc = "time_sum" if is_view_process_based else "time_proc_max"
    time_call = "time_sum"
    kinds = (("proc", time_proc), ("call", time_call))
    boundary = {kind: frame[f"{time_boundary_layer}_{metric}"] for kind, metric in kinds}
    new: dict[str, Series] = {}

    def total_fraction(name: str, series: Series) -> None:
        new[name] = _fraction(series, _total(series))

    for layer in layers:
        series = {kind: frame[f"{layer}_{metric}"] for kind, metric in kinds}
        for kind, _ in kinds:
            new[f"{layer}_time_{kind}_frac_{time_boundary_layer}"] = series[kind] / boundary[kind]
        children = [child for child, parent in layer_deps.items() if parent == layer]
        if not children:
            continue
        for kind, metric in kinds:
            child_sum = None
            for child in children:
                part = frame[f"{child}_{metric}"].fillna(0)
                child_sum = part if child_sum is None else child_sum + part
            overhead = (series[kind] - child_sum).clip(lower=0)
            new[f"o_{layer}_{metric}"] = overhead
            new[f"o_{layer}_time_{kind}_frac_{time_boundary_layer}"] = overhead / boundary[kind]
            new[f"o_{layer}_time_{kind}_frac_self"] = overhead / series[kind]
            total_fraction(f"o_{layer}_time_{kind}_frac_total", overhead)
        has_time = {kind: _total(series[kind]) > 0 for kind, _ in kinds}
        for child in children:
            for kind, metric in kinds:
                name = f"{child}_time_{kind}_frac_parent"
                new[name] = frame[f"{child}_{metric}"] / series[kind] if has_time[kind] else _missing(series[kind])

    for layer, metrics in derived_metrics.items():
        for derived in metrics:
            base = f"{layer}_{derived}"
            for kind, metric in kinds:
                column = f"{base}_{metric}"
                if column not in frame.columns:
                    continue
                series = frame[column]
                new[f"{base}_time_{kind}_frac_{time_boundary_layer}"] = series / boundary[kind]
                new[f"{base}_time_{kind}_frac_parent"] = series / frame[f"{layer}_{metric}"]
                total_fraction(f"{base}_time_{kind}_frac_total", series)

    if f"compute_{time_proc}" in frame.columns:
        compute = {kind: frame[f"compute_{metric}"].fillna(0) for kind, metric in kinds}
        for layer in async_layers:
            parent = layer_deps.get(layer)
            for kind, metric in kinds:
                series = frame[f"{layer}_{metric}"]
                unoverlapped = (series - compute[kind]).clip(lower=0)
                new[f"u_{layer}_{metric}"] = unoverlapped
                new[f"u_{layer}_time_{kind}_frac_self"] = unoverlapped / series
                new[f"u_{layer}_time_{kind}_frac_{time_boundary_layer}"] = unoverlapped / boundary[kind]
                total_fraction(f"u_{layer}_time_{kind}_frac_total", unoverlapped)
                if parent:
                    new[f"u_{layer}_time_{kind}_frac_parent"] = unoverlapped / frame[f"{parent}_{metric}"]
    return _merge(frame, new)


def set_additional_metrics(
    frame: DataFrame,
    additional: Mapping[str, str],
    time_metric: str,
    time_interval: float,
    epsilon: float = 1e-9,
) -> DataFrame:
    """The additional metrics of a preset: `metric = expression`, missing where a denominator is missing or zero."""
    for metric, expression in additional.items():
        expression = expression.format(epsilon=epsilon, time_interval=time_interval, time_metric=time_metric)
        frame = frame.with_columns(**{metric: evaluate(frame, expression)})
        _, denominators = extract_numerator_and_denominators(expression) or (None, [])
        if denominators:
            mask = evaluate(frame, " & ".join(f"({d}.isna() | {d} == 0)" for d in denominators))
            frame = frame.with_columns(**{metric: frame[metric].where(~mask)})
    return frame


def process_flat_metrics(
    flat: DataFrame,
    *,
    is_view_process_based: bool,
    layers: Sequence[str],
    layer_deps: Mapping[str, str | None],
    async_layers: Sequence[str],
    derived_metrics: Mapping[str, Mapping[str, str]],
    time_boundary_layer: str,
    additional: Mapping[str, str],
    time_interval: float,
) -> DataFrame:
    """The view, cross-layer and additional metrics of a flat view, indexed as `flat`, columns sorted by name."""
    keys = list(flat.index_names())
    frame = flat.reset_index()
    frame = frame.with_columns(**{c: nan_as_null(frame[c]) for c in frame.columns})
    frame, _ = set_view_metrics(frame, is_view_process_based)
    frame, _ = set_cross_layer_metrics(
        frame, layers, layer_deps, async_layers, derived_metrics, is_view_process_based, time_boundary_layer
    )
    time_metric = "time_sum" if is_view_process_based else "time_proc_max"
    frame = set_additional_metrics(frame, additional, time_metric, time_interval)
    frame = frame.select(*keys, *sorted(c for c in frame.columns if c not in keys))
    return frame.set_index(keys) if keys else frame
