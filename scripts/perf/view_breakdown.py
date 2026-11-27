"""Attribute time inside the 'Compute views' phase.

Counts how many views and main views are built, times each, and reports how
much of the run is left for everything else.
"""

import os
import sys
import tempfile
import time
from collections import defaultdict


def main():
    trace_dir = os.path.abspath(sys.argv[1])
    preset = sys.argv[2]

    from dftracer.analyzer import init_with_hydra
    from dftracer.analyzer.analyzer import Analyzer
    from dftracer.analyzer.dftracer import DFTracerAnalyzer

    stats = defaultdict(float)
    counts = defaultdict(int)
    per_view = []

    orig_view = DFTracerAnalyzer._compute_view
    orig_main = Analyzer._compute_main_view

    def wrapped_view(self, *a, **kw):
        layer = kw.get("layer", a[0] if a else "?")
        view_key = kw.get("view_key", "?")
        t = time.perf_counter()
        out = orig_view(self, *a, **kw)
        el = time.perf_counter() - t
        stats["_compute_view"] += el
        counts["_compute_view"] += 1
        per_view.append((el, f"{layer}/{'.'.join(view_key) if isinstance(view_key, tuple) else view_key}"))
        return out

    def wrapped_main(self, *a, **kw):
        t = time.perf_counter()
        out = orig_main(self, *a, **kw)
        stats["_compute_main_view"] += time.perf_counter() - t
        counts["_compute_main_view"] += 1
        return out

    DFTracerAnalyzer._compute_view = wrapped_view
    Analyzer._compute_main_view = wrapped_main

    tmp = tempfile.mkdtemp(prefix="perfv-")
    dfa = init_with_hydra(hydra_overrides=[
        "analyzer=dftracer", f"analyzer/preset={preset}", "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp}/ck", "cluster=local", "cluster.n_workers=1",
        "cluster.processes=False", f"hydra.run.dir={tmp}", f"hydra.runtime.output_dir={tmp}",
        f"trace_path={trace_dir}", "view_types=[time_range,proc_name]",
    ])

    t0 = time.perf_counter()
    dfa.analyze_trace()
    total = time.perf_counter() - t0

    print(f"\n=== {os.path.basename(trace_dir)} | preset={preset} ===")
    print(f"total analyze_trace wall: {total:.2f}s")
    print(f"  _compute_main_view : {stats['_compute_main_view']:7.2f}s  "
          f"({stats['_compute_main_view']/total*100:5.1f}%)  n={counts['_compute_main_view']}")
    print(f"  _compute_view      : {stats['_compute_view']:7.2f}s  "
          f"({stats['_compute_view']/total*100:5.1f}%)  n={counts['_compute_view']}")
    rest = total - stats["_compute_view"] - stats["_compute_main_view"]
    print(f"  everything else    : {rest:7.2f}s  ({rest/total*100:5.1f}%)")
    if counts["_compute_view"]:
        print(f"  mean per view      : {stats['_compute_view']/counts['_compute_view']:.3f}s")
    per_view.sort(reverse=True)
    print("\n  slowest views:")
    for el, name in per_view[:8]:
        print(f"    {el:6.2f}s  {name}")

    dfa.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
