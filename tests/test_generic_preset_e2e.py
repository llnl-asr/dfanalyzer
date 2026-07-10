"""End-to-end coverage for the `generic` preset's auto-discovered category layers."""
import gzip
import json
import pathlib

import pytest
from dask.distributed import LocalCluster

from dftracer.analyzer import init_with_hydra


pytestmark = [pytest.mark.smoke, pytest.mark.full]

HHASH = "abc123def456"
PID = TID = 1000
FHASH = "f00d"
FILE_NAME = "/scratch/input.bin"  # avoids POSIX_CAT_RULES suffixes (/data, /checkpoint, ...)

NESTED_CAT_EVENTS = {"ai": 4, "ai.data": 6, "ai.data.io": 20, "ai.compute": 5, "compute": 3}
FLAT_CAT_EVENTS = {"POSIX": 8, "MPI": 5}


@pytest.fixture(scope="session")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _write_trace(trace_dir: pathlib.Path, cat_events: dict, file_name: str = FILE_NAME) -> str:
    """Write a minimal dftracer trace. The cat="dftracer" start event is always
    present, so the boundary layer sees sum(cat_events) + 1 events."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[",
        json.dumps({"name": "HH", "cat": "dftracer", "pid": PID, "tid": TID, "ph": "M",
                    "args": {"hhash": HHASH, "name": "node0", "value": HHASH}}),
        json.dumps({"name": "FH", "cat": "dftracer", "pid": PID, "tid": TID, "ph": "M",
                    "args": {"hhash": HHASH, "name": file_name, "value": FHASH}}),
        json.dumps({"name": "PR", "cat": "dftracer", "pid": PID, "tid": TID, "ph": "M",
                    "args": {"hhash": HHASH, "name": "rank", "value": "0"}}),
        json.dumps({"id": 1, "name": "start", "cat": "dftracer", "pid": PID, "tid": TID,
                    "ts": 1000000, "dur": 0, "ph": "X", "args": {"hhash": HHASH, "ppid": 999}}),
    ]
    ts, eid = 1000100, 2
    for cat, count in cat_events.items():
        for _ in range(count):
            args = {"hhash": HHASH}
            if cat == "POSIX":
                args.update({"fhash": FHASH, "ret": 4096, "offset": 0})
            lines.append(json.dumps({"id": eid, "name": "read" if cat == "POSIX" else "op",
                                     "cat": cat, "pid": PID, "tid": TID, "ts": ts, "dur": 100,
                                     "ph": "X", "args": args}))
            ts += 110
            eid += 1
    with gzip.open(trace_dir / "trace-1_chunk0.pfw.gz", "wt") as f:
        f.write("\n".join(lines) + "\n")
    return str(trace_dir)


@pytest.fixture
def nested_cat_trace(tmp_path: pathlib.Path) -> str:
    return _write_trace(tmp_path / "nested", NESTED_CAT_EVENTS)


@pytest.fixture
def flat_cat_trace(tmp_path: pathlib.Path) -> str:
    return _write_trace(tmp_path / "flat", FLAT_CAT_EVENTS)


@pytest.fixture
def suffixed_posix_trace(tmp_path: pathlib.Path) -> str:
    # a /data path makes POSIX_CAT_RULES rewrite cat POSIX -> posix_reader
    return _write_trace(tmp_path / "suffixed", {"POSIX": 8}, file_name="/scratch/data/x.bin")


def _analyze(trace_path, tmp_path, dask_cluster):
    dfa = init_with_hydra(hydra_overrides=[
        "analyzer=dftracer",
        "analyzer/preset=generic",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={dask_cluster.scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        "view_types=[time_range]",
    ])
    return dfa, dfa.analyze_trace()


def _flat_view(result):
    return result.flat_views[next(iter(result.flat_views))]


def _layer_counts(result):
    flat_view = _flat_view(result)
    return {
        layer: int(flat_view[f"{layer}_count_sum"].sum())
        for layer in result.layers
        if f"{layer}_count_sum" in flat_view.columns
    }


def test_generic_preset_discovers_nested_category_layers(
    nested_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    dfa, result = _analyze(nested_cat_trace, tmp_path, dask_cluster)

    assert set(result.layers) == {"app", "ai", "ai_compute", "ai_data", "ai_data_io", "compute"}
    assert result.layers[0] == "app", "boundary layer must come first"

    # discovery mutates the analyzer's preset, not hydra's (instantiate deep-copies)
    layer_deps = dfa.analyzer.preset.layer_deps
    assert layer_deps["ai"] == "app"
    assert layer_deps["ai_data"] == "ai"
    assert layer_deps["ai_data_io"] == "ai_data"
    assert layer_deps["ai_compute"] == "ai"
    assert layer_deps["compute"] == "app"

    dfa.shutdown()


def test_generic_preset_parent_layers_contain_descendants(
    nested_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    dfa, result = _analyze(nested_cat_trace, tmp_path, dask_cluster)
    counts = _layer_counts(result)

    assert counts["ai"] == sum(NESTED_CAT_EVENTS[c] for c in ("ai", "ai.data", "ai.data.io", "ai.compute"))
    assert counts["ai_data"] == NESTED_CAT_EVENTS["ai.data"] + NESTED_CAT_EVENTS["ai.data.io"]
    assert counts["ai_data_io"] == NESTED_CAT_EVENTS["ai.data.io"]
    assert counts["ai_compute"] == NESTED_CAT_EVENTS["ai.compute"]
    # a flat `compute` cat must not be absorbed by `ai.compute`
    assert counts["compute"] == NESTED_CAT_EVENTS["compute"]
    assert counts["app"] == sum(NESTED_CAT_EVENTS.values()) + 1

    dfa.shutdown()


def test_generic_preset_computes_parent_relative_metrics(
    nested_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    """The nesting exists so cross-layer metrics resolve against the real parent."""
    dfa, result = _analyze(nested_cat_trace, tmp_path, dask_cluster)
    flat_view = _flat_view(result)

    for layer in ("ai", "ai_data", "ai_data_io", "ai_compute"):
        assert f"{layer}_time_proc_frac_parent" in flat_view.columns
    # `ai` has children, so it gets an overhead (self-time) column
    assert "o_ai_time_proc_frac_self" in flat_view.columns

    dfa.shutdown()


def test_generic_preset_omits_dftracer_metadata_category(
    nested_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    dfa, result = _analyze(nested_cat_trace, tmp_path, dask_cluster)

    assert "dftracer" not in result.layers
    # ...but its events still land in the catch-all boundary layer
    assert _layer_counts(result)["app"] == sum(NESTED_CAT_EVENTS.values()) + 1

    dfa.shutdown()


def test_generic_preset_flat_categories_hang_off_the_boundary(
    flat_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    dfa, result = _analyze(flat_cat_trace, tmp_path, dask_cluster)

    assert set(result.layers) == {"app", "posix", "mpi"}, "cats are lowercased"
    layer_deps = dfa.analyzer.preset.layer_deps
    assert layer_deps["posix"] == "app"
    assert layer_deps["mpi"] == "app"

    counts = _layer_counts(result)
    assert counts["posix"] == FLAT_CAT_EVENTS["POSIX"]
    assert counts["mpi"] == FLAT_CAT_EVENTS["MPI"]

    dfa.shutdown()


def test_generic_preset_marks_only_io_categories_as_size_layers(
    flat_cat_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    dfa, _ = _analyze(flat_cat_trace, tmp_path, dask_cluster)

    assert list(dfa.analyzer.preset.size_layers) == ["posix"], "mpi carries no transfer size"

    dfa.shutdown()


def test_generic_preset_discovers_posix_cat_rule_suffixes(
    suffixed_posix_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    """POSIX cats rewritten by POSIX_CAT_RULES are discovered under their new name."""
    dfa, result = _analyze(suffixed_posix_trace, tmp_path, dask_cluster)

    assert set(result.layers) == {"app", "posix_reader"}
    assert list(dfa.analyzer.preset.size_layers) == ["posix_reader"]

    dfa.shutdown()
