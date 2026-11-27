import gzip
import json
import os
import subprocess
import sys

import pytest
from dftracer.analyzer.utils.preset_utils import classify_categories, detect_preset, preset_is_explicit

pytestmark = [pytest.mark.smoke, pytest.mark.full]


@pytest.mark.parametrize(
    "categories, expected",
    [
        (["POSIX", "dftracer"], "posix"),
        (["STDIO"], "posix"),
        (["POSIX", "reader", "data_loader", "checkpoint"], "dlio"),
        (["POSIX", "ai_framework"], "dlio"),
        (["POSIX", "reader", "ai_root", "pipeline", "compute"], "ai"),
        (["sys"], "auto"),
        ([], "auto"),
    ],
)
def test_classify_categories(categories, expected):
    assert classify_categories(categories) == expected


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (["trace_path=/x"], False),
        (["trace_path=/x", "analyzer/preset=dlio"], True),
        (["+analyzer/preset=dlio"], True),
        (["~analyzer/preset"], True),
        (["analyzer.time_granularity=10"], False),
    ],
)
def test_preset_is_explicit(overrides, expected):
    assert preset_is_explicit(overrides) is expected


@pytest.mark.parametrize(
    "cats, expected",
    [
        (["POSIX", "dftracer"], "posix"),
        (["POSIX", "reader", "data_loader"], "dlio"),
        (["POSIX", "ai_root", "pipeline"], "ai"),
        (["sys"], "auto"),
    ],
)
def test_detect_preset_reads_trace_categories(tmp_path, cats, expected):
    events = [
        {"id": i, "name": f"ev{i}", "cat": cat, "pid": 1, "tid": 1, "ts": i, "dur": 1, "ph": "X", "args": {}}
        for i, cat in enumerate(cats, start=1)
    ]
    with gzip.open(tmp_path / "t.pfw.gz", "wt", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    assert detect_preset(str(tmp_path)) == expected


def test_detect_preset_without_traces_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_preset(str(tmp_path))


def test_library_logging_is_quiet_by_default_and_switchable():
    from dftracer.analyzer.config import Config
    from dftracer.analyzer.utils.log_utils import configure_library_logging
    from dftracer.utils.runtime import get_log_level

    assert Config.log_level == "warn"
    try:
        configure_library_logging("debug")
        assert get_log_level() == "debug"
        configure_library_logging("warn")
        assert get_log_level() in ("warn", "warning")
    finally:
        configure_library_logging("info")


HHASH = "abc123def456"
FHASH = "f00d"
PID = TID = 1000

# Each flavor lists (cat, name) pairs; POSIX events carry a file hash and a size.
TRACE_FLAVORS = {
    "posix": [("POSIX", "read"), ("POSIX", "write")],
    "dlio": [
        ("POSIX", "read"),
        ("reader", "NPZReader.get_sample"),
        ("data_loader", "TorchDataLoader.item"),
        ("checkpoint", "PyTorchCheckpointing.save"),
        ("ai_framework", "TorchFramework.compute"),
    ],
    "ai": [
        ("POSIX", "read"),
        ("reader", "NPZReader.get_sample"),
        ("ai_root", "ai_root"),
        ("pipeline", "train"),
        ("compute", "step"),
    ],
    "auto": [("MPI", "send"), ("MPI", "recv")],
}


def write_flavor_trace(trace_dir, flavor, repeat=4):
    trace_dir.mkdir(parents=True, exist_ok=True)
    meta = {"hhash": HHASH}
    lines = [
        json.dumps(
            {
                "name": "HH",
                "cat": "dftracer",
                "pid": PID,
                "tid": TID,
                "ph": "M",
                "args": {**meta, "name": "node0", "value": HHASH},
            }
        ),
        json.dumps(
            {
                "name": "FH",
                "cat": "dftracer",
                "pid": PID,
                "tid": TID,
                "ph": "M",
                "args": {**meta, "name": "/scratch/input.bin", "value": FHASH},
            }
        ),
        json.dumps(
            {
                "name": "PR",
                "cat": "dftracer",
                "pid": PID,
                "tid": TID,
                "ph": "M",
                "args": {**meta, "name": "rank", "value": "0"},
            }
        ),
        json.dumps(
            {
                "id": 1,
                "name": "start",
                "cat": "dftracer",
                "pid": PID,
                "tid": TID,
                "ts": 1_000_000,
                "dur": 0,
                "ph": "X",
                "args": {**meta, "ppid": 999},
            }
        ),
    ]
    ts, eid = 1_000_100, 2
    for _ in range(repeat):
        for cat, name in TRACE_FLAVORS[flavor]:
            args = dict(meta)
            if cat == "POSIX":
                args.update({"fhash": FHASH, "ret": 4096, "offset": 0})
            lines.append(
                json.dumps(
                    {
                        "id": eid,
                        "name": name,
                        "cat": cat,
                        "pid": PID,
                        "tid": TID,
                        "ts": ts,
                        "dur": 100,
                        "ph": "X",
                        "args": args,
                    }
                )
            )
            ts += 110
            eid += 1
    with gzip.open(trace_dir / "trace-1.pfw.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(trace_dir)


def run_cli(trace_path, run_dir, *overrides):
    run_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dftracer.analyzer",
            f"trace_path={trace_path}",
            "cluster=none",
            "analyzer.checkpoint=False",
            f"hydra.run.dir={run_dir}",
            *overrides,
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "FORCE_COLOR": "0", "TERM": "dumb"},
    )


def detected_presets(run_dir):
    found = []
    for log_file in run_dir.glob("*.log"):
        for line in log_file.read_text().splitlines():
            if "Detected analyzer preset" in line:
                found.append(json.loads(line)["preset"])
    return found


@pytest.mark.parametrize("flavor", ["posix", "dlio", "ai", "auto"])
def test_cli_picks_preset_from_trace(tmp_path, flavor):
    trace = write_flavor_trace(tmp_path / "trace", flavor)
    result = run_cli(trace, tmp_path / "run")
    assert result.returncode == 0, result.stderr[-2000:]
    assert detected_presets(tmp_path / "run") == [flavor]


def test_cli_explicit_preset_skips_detection(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "dlio")
    result = run_cli(trace, tmp_path / "run", "analyzer/preset=posix")
    assert result.returncode == 0, result.stderr[-2000:]
    assert detected_presets(tmp_path / "run") == []


def test_cli_library_logs_are_off_unless_asked(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "posix")
    quiet = run_cli(trace, tmp_path / "quiet")
    assert quiet.returncode == 0, quiet.stderr[-2000:]
    assert quiet.stderr.strip() == ""
    assert "Time Period Summary" in quiet.stdout

    loud = run_cli(write_flavor_trace(tmp_path / "trace2", "posix"), tmp_path / "loud", "log_level=info")
    assert loud.returncode == 0, loud.stderr[-2000:]
    assert "INFO" in loud.stderr

    debug = run_cli(write_flavor_trace(tmp_path / "trace3", "posix"), tmp_path / "debug", "debug=true")
    assert debug.returncode == 0, debug.stderr[-2000:]
    assert len(debug.stderr.splitlines()) > len(loud.stderr.splitlines())


def init_notebook_style(trace, run_dir, *overrides):
    from dftracer.analyzer import init_with_hydra

    run_dir.mkdir(parents=True, exist_ok=True)
    return init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={run_dir}/checkpoints",
            "cluster=none",
            f"hydra.run.dir={run_dir}",
            f"hydra.runtime.output_dir={run_dir}",
            f"trace_path={trace}",
            *overrides,
        ]
    )


@pytest.mark.parametrize("flavor", ["posix", "dlio", "ai", "auto"])
def test_init_with_hydra_picks_preset_from_trace(tmp_path, flavor):
    trace = write_flavor_trace(tmp_path / "trace", flavor)
    instance = init_notebook_style(trace, tmp_path / "run")
    assert instance.analyzer.preset.name == flavor
    instance.shutdown()


def test_init_with_hydra_explicit_preset_wins(tmp_path):
    trace = write_flavor_trace(tmp_path / "trace", "dlio")
    instance = init_notebook_style(trace, tmp_path / "run", "analyzer/preset=posix")
    assert instance.analyzer.preset.name == "posix"
    instance.shutdown()


def test_init_with_hydra_silences_library_logs(tmp_path):
    from dftracer.utils.runtime import get_log_level

    trace = write_flavor_trace(tmp_path / "trace", "posix")
    instance = init_notebook_style(trace, tmp_path / "run")
    assert get_log_level() in ("warn", "warning")
    instance.shutdown()
