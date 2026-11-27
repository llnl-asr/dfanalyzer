"""Tests for system event parsing and system metrics integration."""

import gzip
import json
import logging
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf


def _write_trace_lines(path, lines):
    """Write a list of trace text lines as gzipped JSONL (.pfw.gz)."""
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        for line in lines:
            if line in ("[", "]"):
                continue
            f.write(line + "\n")

from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX
from dftracer.analyzer.constants import COL_TIME_RANGE
from dftracer.analyzer.dftracer import (
    DFTracerAnalyzer,
    SYSTEM_OUTPUT_COLUMNS,
)

pytestmark = [pytest.mark.smoke, pytest.mark.full]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYSTEM_TRACE = "dftracer-system"


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def _make_analyzer(tmp_path, time_granularity=5):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        time_granularity=time_granularity,
        time_resolution=10**6,
        time_approximate=True,
        time_sliced=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------


class TestSystemEventParsing:
    """Test that cat='sys' ph='C' events are parsed as system events."""

    def test_read_trace_returns_system_metrics(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path)
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        assert result.system_metrics is not None
        sm = result.system_metrics
        assert not sm.empty

    def test_system_metrics_has_expected_columns(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path / "cols")
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics
        for col in SYSTEM_OUTPUT_COLUMNS:
            assert col in sm.columns, f"Missing column: {col}"

    def test_no_traces_or_profiles_from_system_only_file(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path / "noprof")
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        traces = result.traces
        assert traces.empty
        assert result.profiles is None


# ---------------------------------------------------------------------------
# Tests: standardization
# ---------------------------------------------------------------------------


class TestSystemMetricsStandardization:
    """Test that raw system events are correctly aggregated per time_range."""

    @pytest.fixture(autouse=True)
    def setup(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path / "std")
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        self.sm = result.system_metrics

    def test_rows_per_time_range(self):
        # 10.5s of data at 5s granularity => 3 time_range buckets
        assert len(self.sm) == 3

    def test_time_ranges_are_sequential(self):
        trs = self.sm[COL_TIME_RANGE].sort_values().values
        assert trs[1] == trs[0] + 1
        assert trs[2] == trs[1] + 1

    def test_host_hash_present(self):
        assert (self.sm["host_hash"] == "731340f16729098d").all()

    def test_cpu_percentages_in_range(self):
        for col in ["sys_cpu_iowait_pct", "sys_cpu_user_pct", "sys_cpu_system_pct", "sys_cpu_idle_pct"]:
            assert (self.sm[col] >= 0).all(), f"{col} has negative values"
            assert (self.sm[col] <= 100).all(), f"{col} exceeds 100%"

    def test_cpu_percentages_sum_reasonable(self):
        # user + system + iowait + idle should be close to 100%
        total = (
            self.sm["sys_cpu_user_pct"]
            + self.sm["sys_cpu_system_pct"]
            + self.sm["sys_cpu_iowait_pct"]
            + self.sm["sys_cpu_idle_pct"]
        )
        for val in total:
            assert 99.0 <= val <= 101.0, f"CPU pct sum out of range: {val}"

    def test_core_max_ge_aggregate(self):
        # Per-core max iowait should be >= aggregate mean iowait
        assert (self.sm["sys_core_iowait_pct_max"] >= self.sm["sys_cpu_iowait_pct"]).all()

    def test_core_p95_between_mean_and_max(self):
        assert (self.sm["sys_core_iowait_pct_p95"] >= self.sm["sys_cpu_iowait_pct"]).all()
        assert (self.sm["sys_core_iowait_pct_p95"] <= self.sm["sys_core_iowait_pct_max"]).all()

    def test_memory_available_positive(self):
        assert (self.sm["sys_mem_available"] > 0).all()


# ---------------------------------------------------------------------------
# Tests: mixed trace (system + I/O events)
# ---------------------------------------------------------------------------


class TestMixedTrace:
    """Test system metrics with a synthetic trace containing both I/O and system events."""

    @pytest.fixture(autouse=True)
    def setup(self, dask_client, tmp_path):
        self.trace_dir = tmp_path / "mixed"
        self.trace_dir.mkdir()

        events = [
            # Metadata
            {"name": "HH", "cat": "dftracer", "pid": 1, "tid": 1, "ph": "M",
             "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
            {"name": "FH", "cat": "dftracer", "pid": 1, "tid": 1, "ph": "M",
             "args": {"hhash": "h1", "name": "/tmp/file.bin", "value": "f1"}},
            # I/O trace event
            {"name": "read", "cat": "POSIX", "pid": 1, "tid": 1, "ph": "X",
             "ts": 5_000_100, "dur": 200,
             "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0}},
            # System events (same time window)
            {"name": "cpu", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 10.0, "system_pct": 5.0,
                       "iowait_pct": 3.0, "idle_pct": 82.0}},
            {"name": "cpu-0", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 20.0, "system_pct": 8.0,
                       "iowait_pct": 7.0, "idle_pct": 65.0}},
            {"name": "cpu-1", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "user_pct": 5.0, "system_pct": 2.0,
                       "iowait_pct": 1.0, "idle_pct": 92.0}},
            {"name": "memory", "cat": "sys", "pid": 0, "tid": 0, "ph": "C",
             "ts": 5_000_000,
             "args": {"hhash": "h1", "MemAvailable": 1000000, "Dirty": 0.5,
                       "Cached": 10.0}},
        ]

        trace_file = self.trace_dir / "mixed_trace.pfw.gz"
        _write_trace_lines(trace_file, [json.dumps(e) for e in events])

        analyzer = _make_analyzer(tmp_path / "mixed_analysis")
        self.result = analyzer.read_trace(
            trace_path=str(self.trace_dir),
            extra_columns=None,
            extra_columns_fn=None,
        )

    def test_traces_and_system_both_present(self):
        traces = self.result.traces
        assert not traces.empty, "Should have I/O trace events"
        assert self.result.system_metrics is not None, "Should have system metrics"

    def test_system_metrics_values(self):
        sm = self.result.system_metrics
        assert len(sm) == 1  # one time_range bucket
        row = sm.iloc[0]
        assert row["sys_cpu_user_pct"] == pytest.approx(10.0)
        assert row["sys_cpu_iowait_pct"] == pytest.approx(3.0)
        assert row["sys_core_iowait_pct_max"] == pytest.approx(7.0)  # cpu-0
        assert row["sys_mem_available"] == pytest.approx(1000000)
        assert row["sys_mem_dirty"] == pytest.approx(0.5)

    def test_profiles_not_contaminated(self):
        """System events should NOT appear as profiles."""
        assert self.result.profiles is None


# ---------------------------------------------------------------------------
# Tests: variable time granularity
# ---------------------------------------------------------------------------


class TestVariableTimeGranularity:
    """Test that system metrics aggregate correctly with different time_granularity."""

    def test_granularity_1s(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path / "g1", time_granularity=1)
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics
        # 10.5s at 1s granularity => ~11 buckets (some may have 0-1 samples)
        assert len(sm) >= 10
        assert len(sm) <= 12

    def test_granularity_10s(self, dask_client, tmp_path, trace_copy):
        analyzer = _make_analyzer(tmp_path / "g10", time_granularity=10)
        result = analyzer.read_trace(
            trace_path=trace_copy(SYSTEM_TRACE),
            extra_columns=None,
            extra_columns_fn=None,
        )
        sm = result.system_metrics
        # 10.5s at 10s granularity => 2 buckets
        assert len(sm) == 2


# ---------------------------------------------------------------------------
# Tests: parity with the pandas path
# ---------------------------------------------------------------------------


def _assert_matches_pandas(got, expected):
    """Same columns and rows; numbers within rtol 1e-12, NaN counted as missing."""
    import numpy as np
    import pandas as pd

    got = got.to_pandas()
    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    for col in expected.columns:
        g, e = got[col], expected[col]
        assert g.isna().tolist() == e.isna().tolist(), col
        if pd.api.types.is_numeric_dtype(e.dtype):
            assert pd.api.types.is_numeric_dtype(g.dtype), col
            np.testing.assert_allclose(
                g[g.notna()].to_numpy(dtype=float), e[e.notna()].to_numpy(dtype=float), rtol=1e-12, atol=0
            )
        else:
            assert g[g.notna()].tolist() == e[e.notna()].tolist(), col


class TestStandardizeSystemParity:
    def test_matches_pandas_on_the_system_trace(self, dask_client, tmp_path, trace_copy):
        from dftracer.analyzer.typed_read import read_frames

        from . import pandas_reference

        analyzer = _make_analyzer(tmp_path / "parity", time_granularity=1)
        analyzer.read_trace(trace_path=trace_copy(SYSTEM_TRACE), extra_columns=None, extra_columns_fn=None)
        frames = read_frames(analyzer._run, analyzer._read_settings(True))
        raw = frames["system"]
        args = (frames["system_time_origin"], analyzer.time_granularity, analyzer.time_resolution)
        got = DFTracerAnalyzer._standardize_system(raw, *args)
        _assert_matches_pandas(got, pandas_reference.standardize_system(raw.to_pandas(), *args))

    def test_matches_pandas_with_missing_samples_and_partial_groups(self):
        from dftracer.utils import DataFrame, Series

        from . import pandas_reference

        nan = float("nan")
        raw = DataFrame.from_dict(
            {
                "host_hash": ["h2", "h1", "h1", "h1", "h1", "h2", "h1", None],
                "name": ["cpu", "cpu", "cpu", "cpu-0", "cpu-1", "memory", "memory", "cpu"],
                "ts": Series([0, 0, 10, 0, 0, 20, 0, 0], dtype="int64"),
                "iowait_pct": [1.0, 2.0, nan, 3.0, 5.0, nan, nan, 9.0],
                "user_pct": [4.0, nan, 6.0, nan, nan, nan, nan, 1.0],
                "Dirty": [nan, nan, nan, nan, nan, 7.0, nan, nan],
            }
        )
        args = (0, 1.0, 10.0)
        got = DFTracerAnalyzer._standardize_system(raw, *args)
        _assert_matches_pandas(got, pandas_reference.standardize_system(raw.to_pandas(), *args))

    def test_matches_pandas_with_a_missing_time(self):
        from dftracer.utils import DataFrame, Series

        from . import pandas_reference

        raw = DataFrame.from_dict(
            {
                "host_hash": ["h1", "h1", "h1"],
                "name": ["cpu", "cpu", "cpu-0"],
                "ts": Series([0, None, None], dtype="int64"),
                "iowait_pct": [1.0, 2.0, 3.0],
                "user_pct": [4.0, 5.0, 6.0],
            }
        )
        args = (0, 1.0, 10.0)
        got = DFTracerAnalyzer._standardize_system(raw, *args)
        _assert_matches_pandas(got, pandas_reference.standardize_system(raw.to_pandas(nullable=True), *args))

    @pytest.mark.parametrize(
        "samples",
        [
            [1.0, float("inf")],
            [float("inf")],
            [float("-inf"), 1.0, 2.0],
            [float(i) for i in range(40)] + [float("inf")],
            [float(i) for i in range(39)] + [float("inf"), float("inf")],
            [float(i) for i in range(60)] + [float("-inf")],
        ],
        ids=["two", "one", "neg", "n41", "n41-two-inf", "n61-neg"],
    )
    def test_core_p95_with_infinities_matches_pandas(self, samples):
        from dftracer.utils import DataFrame, Series

        from . import pandas_reference

        n = len(samples)
        raw = DataFrame.from_dict(
            {
                "host_hash": ["h1"] * n + ["h1"] * 3,
                "name": [f"cpu-{i}" for i in range(n)] + ["cpu-0", "cpu-1", "cpu-2"],
                "ts": Series([0] * n + [10, 10, 10], dtype="int64"),
                "iowait_pct": samples + [1.0, 2.0, 4.0],
            }
        )
        args = (0, 1.0, 10.0)
        got = DFTracerAnalyzer._standardize_system(raw, *args)
        _assert_matches_pandas(got, pandas_reference.standardize_system(raw.to_pandas(), *args))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_core_p95_matches_pandas_on_random_groups(self, seed):
        """Group sizes 1 to 60 with +inf, -inf, NaN and ties, one group per time range."""
        import random

        from dftracer.utils import DataFrame, Series

        from . import pandas_reference

        rng = random.Random(seed)
        choices = [float("inf"), float("-inf"), float("nan"), 5.0]
        names, ts, values = [], [], []
        for group in range(120):
            n = rng.randint(1, 60)
            for i in range(n):
                names.append(f"cpu-{i}")
                ts.append(group * 10)
                values.append(rng.choice(choices) if rng.random() < 0.15 else rng.uniform(0, 100))
        raw = DataFrame.from_dict(
            {
                "host_hash": ["h1"] * len(ts),
                "name": names,
                "ts": Series(ts, dtype="int64"),
                "iowait_pct": values,
            }
        )
        args = (0, 1.0, 10.0)
        got = DFTracerAnalyzer._standardize_system(raw, *args)
        _assert_matches_pandas(got, pandas_reference.standardize_system(raw.to_pandas(), *args))
