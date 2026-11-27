"""Time and peak memory of one end-to-end CLI analysis, for one checkout.

usage: python bench_cli.py CHECKOUT TRACE_DIR PRESET RUNS [OVERRIDE ...]

Each run analyzes a fresh copy of TRACE_DIR (its index is rebuilt) in its own
process, with `PYTHONPATH=CHECKOUT/python`. The peak is the child's max RSS
from `wait4`. Prints one JSON line with every run, the median and the range.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


def _run(checkout: str, trace_dir: str, preset: str, overrides: list[str]) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as work:
        trace = os.path.join(work, os.path.basename(trace_dir.rstrip("/")))
        shutil.copytree(trace_dir, trace, ignore=shutil.ignore_patterns(".dftindex"))
        cmd = [
            sys.executable,
            "-m",
            "dftracer.analyzer",
            "cluster=none",
            f"analyzer/preset={preset}",
            "analyzer.checkpoint=False",
            f"trace_path={trace}",
            f"hydra.run.dir={work}/out",
            *overrides,
        ]
        env = {**os.environ, "PYTHONPATH": os.path.join(checkout, "python")}
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, status, usage = os.wait4(proc.pid, 0)
        elapsed = time.perf_counter() - start
        if os.waitstatus_to_exitcode(status) != 0:
            raise SystemExit(proc.stderr.read().decode()[-4000:])
        # ru_maxrss is bytes on macOS and kilobytes on Linux.
        scale = 1 if sys.platform == "darwin" else 1024
        return elapsed, usage.ru_maxrss * scale / 1e9


def main() -> None:
    checkout, trace_dir, preset, runs, *overrides = sys.argv[1:]
    samples = [_run(checkout, trace_dir, preset, overrides) for _ in range(int(runs))]
    times = [t for t, _ in samples]
    peaks = [p for _, p in samples]
    print(
        json.dumps(
            {
                "checkout": checkout,
                "trace": os.path.basename(trace_dir.rstrip("/")),
                "preset": preset,
                "time_s": [round(t, 3) for t in times],
                "time_median_s": round(statistics.median(times), 3),
                "time_range_s": [round(min(times), 3), round(max(times), 3)],
                "peak_gb": [round(p, 3) for p in peaks],
                "peak_median_gb": round(statistics.median(peaks), 3),
                "peak_range_gb": [round(min(peaks), 3), round(max(peaks), 3)],
            }
        )
    )


if __name__ == "__main__":
    main()
