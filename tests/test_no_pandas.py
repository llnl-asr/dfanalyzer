"""The analyzer runs on dftracer-utils frames only and never loads pandas or pyarrow.

Each run is a subprocess, so the pandas import of the test process does not count.
numpy is not checked here because `distributed` loads it.
"""

import json
import pathlib
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.full]

SCRIPT = r"""
import json, sys
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from dftracer.utils import DataFrame

overrides, workers = json.loads(sys.argv[1]), int(sys.argv[2])
cluster = None
if workers:
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True)
    overrides += [
        "cluster=external",
        "cluster.restart_on_connect=False",
        f"cluster.scheduler_address={cluster.scheduler_address}",
    ]
else:
    overrides += ["cluster=none"]

def loaded():
    import sys
    return sorted(m for m in ("pandas", "pyarrow") if m in sys.modules)

dfa = init_with_hydra(hydra_overrides=overrides)
try:
    result = dfa.analyze_trace()
    dfa.output.handle_result(result)
    frames = {
        "flat_views": list(result.flat_views.values()),
        "views": [v for layer in result.views.values() for v in layer.values()],
        "hlms": list(result._hlms.values()),
        "main_views": list(result._main_views.values()),
        "read": [f for f in (result.aggregated_traces, result.profiles, result.system_metrics) if f is not None],
    }
    worker_loaded = sorted({m for ms in (dfa.client.run(loaded).values() if workers else []) for m in ms})
finally:
    dfa.shutdown()
    if cluster is not None:
        cluster.close()
print(json.dumps({
    "loaded": loaded(),
    "worker_loaded": worker_loaded,
    "kinds": {k: sorted({type(f).__module__ + "." + type(f).__name__ for f in fs}) for k, fs in frames.items()},
    "counts": {k: len(fs) for k, fs in frames.items()},
}))
"""

CASES = [
    ("dftracer-dlio", "ai", "console", 0),
    ("dftracer-posix", "posix", "json", 0),
    ("dftracer-dlio", "ai", "console", 2),
]


# Run as if numpy, pandas and pyarrow were not installed: an import of any of them raises ImportError.
BLOCK_THIRD_PARTY = "import sys\nfor _m in ('numpy', 'pandas', 'pyarrow'):\n    sys.modules[_m] = None\n"


def _run(tmp_path: pathlib.Path, trace_path: str, preset: str, output: str, workers: int, prelude: str = "") -> dict:
    overrides = [
        "analyzer=dftracer",
        f"analyzer/preset={preset}",
        "analyzer.checkpoint=True",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "facts.enabled=True",
        "logical_view_types=True",
        f"output={output}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        "view_types=[time_range,proc_name,file_name]",
    ]
    if output == "file":
        overrides.append(f"output.path={tmp_path}/out")
    run = subprocess.run(
        [sys.executable, "-c", prelude + SCRIPT, json.dumps(overrides), str(workers)],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(("trace", "preset", "output", "workers"), CASES)
def test_analysis_uses_library_frames_only(
    tmp_path: pathlib.Path, trace_copy, trace: str, preset: str, output: str, workers: int
) -> None:
    report = _run(tmp_path, trace_copy(trace), preset, output, workers)
    assert report["loaded"] == []
    assert report["worker_loaded"] == []
    for name in ("flat_views", "views", "hlms", "main_views", "read"):
        assert report["counts"][name] > 0, name
        assert report["kinds"][name] == ["dftracer.utils._dataframe.DataFrame"], (name, report["kinds"][name])


@pytest.mark.parametrize(
    ("trace", "preset", "output"), [("dftracer-dlio", "ai", "console"), ("dftracer-posix", "posix", "json")]
)
def test_analysis_runs_without_numpy_pandas_or_pyarrow(
    tmp_path: pathlib.Path, trace_copy, trace: str, preset: str, output: str
) -> None:
    report = _run(tmp_path, trace_copy(trace), preset, output, 0, prelude=BLOCK_THIRD_PARTY)
    for name in ("flat_views", "views", "hlms", "main_views", "read"):
        assert report["counts"][name] > 0, name


@pytest.mark.parametrize(("trace", "preset"), [("dftracer-posix", "posix"), ("dftracer-dlio", "ai")])
def test_file_output_runs_without_numpy_pandas_or_pyarrow(
    tmp_path: pathlib.Path, trace_copy, trace: str, preset: str
) -> None:
    report = _run(tmp_path, trace_copy(trace), preset, "file", 0, prelude=BLOCK_THIRD_PARTY)
    assert report["counts"]["flat_views"] > 0
    assert len(list((tmp_path / "out").glob("detail_view_*.pfw.gz"))) == report["counts"]["flat_views"]
