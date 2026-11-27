"""Capture the pandas-path baseline for tests/test_utils_frame_parity.py.

usage: python tests/data/parity/capture.py <dataset> <preset> <out_dir>

Runs the real analyzer on tests/data/extracted/<dataset> with view_types
[proc_name, time_range] and saves, as checkpoint tables, every
DFTracerAnalyzer._compute_view call (input and output), every main view
(input and output) and the final flat views.
"""

import json
import pathlib
import shutil
import sys
import tarfile
import tempfile

import dask
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.checkpoint_table import write_table
from dftracer.analyzer.dftracer import DFTracerAnalyzer

VIEW_TYPES = ["proc_name", "time_range"]


def _pandas(x):
    return x.compute() if hasattr(x, "compute") else x.copy()


def main(dataset: str, preset: str, out: pathlib.Path) -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "data" / "extracted" / dataset
    if not src.exists():
        src.mkdir(parents=True)
        with tarfile.open(root / "data" / f"{dataset}.tar.gz") as tar:
            tar.extractall(src)
    tmp = pathlib.Path(tempfile.mkdtemp())
    trace = tmp / dataset
    shutil.copytree(src, trace, ignore=shutil.ignore_patterns(".dftindex"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"views": [], "mains": [], "flat": []}

    orig_view = DFTracerAnalyzer._compute_view
    orig_main = Analyzer._compute_main_view

    def view_wrap(self, layer, records, view_key, view_type, view_types):
        res = orig_view(self, layer, records, view_key, view_type, view_types)
        n = len(manifest["views"])
        write_table(_pandas(records), str(out / f"view_{n:03d}_in.pfw.gz"), "view_in")
        write_table(_pandas(res), str(out / f"view_{n:03d}_out.pfw.gz"), "view_out")
        manifest["views"].append(
            {
                "layer": str(layer),
                "view_key": list(view_key),
                "view_type": view_type,
                "view_types": list(view_types),
                "logical_views": {k: sorted(v) for k, v in self.logical_views.items()},
                "time_sliced": bool(self.time_sliced),
            }
        )
        return res

    def main_wrap(self, layer, hlm, view_types, partition_size):
        res = orig_main(self, layer, hlm, view_types, partition_size)
        n = len(manifest["mains"])
        write_table(_pandas(hlm), str(out / f"main_{n:03d}_in.pfw.gz"), "main_in")
        write_table(_pandas(res), str(out / f"main_{n:03d}_out.pfw.gz"), "main_out")
        manifest["mains"].append(
            {"layer": str(layer), "view_types": list(view_types), "time_sliced": bool(self.time_sliced)}
        )
        return res

    DFTracerAnalyzer._compute_view = view_wrap
    Analyzer._compute_main_view = main_wrap

    cluster = LocalCluster(processes=False, protocol="tcp")
    overrides = [
        "analyzer=dftracer",
        f"analyzer/preset={preset}",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=False",
        f"cluster.scheduler_address={cluster.scheduler_address}",
        f"hydra.run.dir={tmp}",
        f"hydra.runtime.output_dir={tmp}",
        f"trace_path={trace}",
        f"view_types=[{','.join(VIEW_TYPES)}]",
    ]
    if preset in ("dlio", "ai"):
        overrides.append("analyzer.assign_epochs=True")
    dfa = init_with_hydra(hydra_overrides=overrides)
    result = dfa.analyze_trace()
    for i, (key, flat) in enumerate(result.flat_views.items()):
        write_table(flat, str(out / f"flat_{i:03d}.pfw.gz"), "flat")
        manifest["flat"].append({"view_key": list(key)})
    dfa.shutdown()
    cluster.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(dataset, {k: len(v) for k, v in manifest.items()})


if __name__ == "__main__":
    dask.config.set({"distributed.worker.daemon": False})
    main(sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]))
