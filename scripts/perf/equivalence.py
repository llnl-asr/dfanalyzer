"""Equivalence harness for the view-computation optimisation.

  perf_equiv.py capture <trace_dir> <preset> <out_dir>
  perf_equiv.py compare <dir_a> <dir_b>

Captures every flat view plus raw_stats from a run, then compares two captures
exactly -- values, dtypes, column order, index -- so a change to the hot path
has to prove it produced identical output, not merely similar numbers.
"""

import math
import os
import pickle
import sys
import tempfile


def capture(trace_dir, preset, out_dir):
    from dftracer.analyzer import init_with_hydra
    import dask

    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="equiv-")
    dfa = init_with_hydra(hydra_overrides=[
        "analyzer=dftracer", f"analyzer/preset={preset}", "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp}/ck", "cluster=local", "cluster.n_workers=1",
        "cluster.processes=False", f"hydra.run.dir={tmp}", f"hydra.runtime.output_dir={tmp}",
        f"trace_path={trace_dir}", "view_types=[time_range,proc_name]",
    ])
    result = dfa.analyze_trace()
    raw = dask.compute(result.raw_stats)[0]
    payload = {
        "raw_stats": raw if isinstance(raw, dict) else raw.__dict__,
        "layers": list(result.layers),
        "view_types": list(result.view_types),
        # Library frames are stored as pandas, so a capture compares with one from a pandas build.
        "flat_views": {k: v.to_pandas(nullable=True) for k, v in result.flat_views.items()},
    }
    with open(os.path.join(out_dir, "capture.pkl"), "wb") as f:
        pickle.dump(payload, f)
    print(f"captured {len(payload['flat_views'])} flat views -> {out_dir}")
    for k, v in sorted(payload["flat_views"].items(), key=lambda kv: str(kv[0])):
        print(f"   {str(k):<28} {v.shape[0]:>6} rows x {v.shape[1]:>4} cols")
    dfa.shutdown()


_NULLABLE = {"b": "boolean", "i": "Int64", "u": "UInt64", "f": "Float64"}


def _is_text_or_missing(value) -> bool:
    import pandas as pd

    return isinstance(value, str) or value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _nullable(values):
    """`values` in the pandas nullable dtype of its family: an int stays an int and a float a float."""
    import pandas as pd

    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = values.dtype.categories
        kind = categories.dtype.kind
        text = kind in "OU" and all(isinstance(c, str) for c in categories)
        target = _NULLABLE.get(kind, "string" if text else None)
        return values.astype(object).astype(target) if target else values.astype(object)
    kind = values.dtype.kind
    if kind in _NULLABLE:
        return values.astype(_NULLABLE[kind])
    if kind in "OU" and all(_is_text_or_missing(v) for v in values):
        return values.astype("string")
    return values


def _normalized(frame):
    """`frame` with columns and index levels in the pandas nullable dtype of their kind, so a pandas-build
    capture (numpy dtypes, object index) and a library-build one compare by value and dtype family."""
    import pandas as pd

    out = pd.DataFrame({c: _nullable(frame[c]) for c in frame.columns}, index=frame.index)
    levels = [_nullable(pd.Series(frame.index.get_level_values(i))) for i in range(frame.index.nlevels)]
    if len(levels) > 1:
        out.index = pd.MultiIndex.from_arrays(levels, names=frame.index.names)
    else:
        out.index = pd.Index(levels[0], name=frame.index.name)
    return out


def compare(dir_a, dir_b):
    from pandas.testing import assert_frame_equal

    a = pickle.load(open(os.path.join(dir_a, "capture.pkl"), "rb"))
    b = pickle.load(open(os.path.join(dir_b, "capture.pkl"), "rb"))

    problems = []
    for field in ("layers", "view_types"):
        if a[field] != b[field]:
            problems.append(f"{field}: {a[field]} != {b[field]}")

    ra, rb = a["raw_stats"], b["raw_stats"]
    for k in sorted(set(ra) | set(rb)):
        if str(ra.get(k)) != str(rb.get(k)):
            problems.append(f"raw_stats.{k}: {ra.get(k)!r} != {rb.get(k)!r}")

    keys_a, keys_b = set(a["flat_views"]), set(b["flat_views"])
    if keys_a != keys_b:
        problems.append(f"view keys differ: only_a={sorted(keys_a - keys_b)} only_b={sorted(keys_b - keys_a)}")

    for k in sorted(keys_a & keys_b, key=str):
        fa, fb = _normalized(a["flat_views"][k]), _normalized(b["flat_views"][k])
        try:
            assert_frame_equal(fa, fb, check_dtype=True, check_like=False, check_exact=False, rtol=1e-9)
        except AssertionError as exc:
            first = str(exc).strip().splitlines()[0]
            problems.append(f"flat_view {k}: {first}")

    if problems:
        print(f"DIFFERENT -- {len(problems)} problem(s):")
        for p in problems[:25]:
            print(f"   {p}")
        sys.exit(1)
    print(f"IDENTICAL -- {len(keys_a)} flat views, raw_stats, layers, view_types all match")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.argv[1] == "capture":
        capture(os.path.abspath(sys.argv[2]), sys.argv[3], os.path.abspath(sys.argv[4]))
    else:
        compare(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
