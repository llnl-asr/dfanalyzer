"""Streamlit front end for DFAnalyzer.

Renders the same payload the JSON output backend produces, so the UI tracks the
analyzer's schema instead of reaching into internals that move underneath it.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from dftracer.analyzer import init_with_hydra

DEFAULT_TIME_GRANULARITY_IN_SECONDS = 5
TRACE_SUFFIXES = ("pfw.gz",)
PRESETS = ["posix", "dlio", "dlio-prev", "generic"]

# Total bytes accepted across one submission. Must match maxUploadSize in
# .streamlit/config.toml, which Streamlit enforces per *file* while this
# uploader accepts several; tests/test_streamlit_app.py asserts they agree.
#
# Set from measurement, not preference. Peak RSS is ~290 MB of fixed overhead
# plus 11-17 MB per 1,000 trace events, so against a Community Cloud allocation
# of 690 MB guaranteed / 2.7 GB ceiling:
#
#     ~450 KB gzipped  (~33k events)   fits the guaranteed allocation
#     ~2.7 MB gzipped  (~200k events)  needs the best-case ceiling
#     3.8 MB gzipped   (284k events)   measured at 3,499 MB -- cannot work
#
# The hard cap therefore sits at the best case: refuse what cannot complete,
# warn about what merely might not. Note the sibling wisio app uses 16 MB; its
# read path is far lighter (929 MB where this one peaks at 3,499 MB on the same
# trace), so the number does not carry across.
MAX_UPLOAD_MB = 2

# Above this, the run will likely exceed the guaranteed allocation but may still
# succeed if the host grants more. Warn rather than block.
WARN_UPLOAD_KB = 450

# Pinned rather than left to dask's autodetect, which derives a per-worker
# budget from the host's RAM and core count. In a container that is the host's
# numbers, not the container's, so the worker would get a limit far above the
# memory actually available and provide no protection at all -- on a 94 GiB /
# 40-core host it autodetects 2.35 GiB against a Community Cloud allocation of
# 690 MB. An explicit ceiling gives dask real spill thresholds.
WORKER_MEMORY_LIMIT_BYTES = 2 * 1024**3
VIEW_TYPE_MAPPING = {
    "File": "file_name",
    "Process": "proc_name",
    "Timeline": "time_range",
}
LAYER_COLUMNS = {
    "time_s": "I/O Time (s)",
    "count": "Operations",
    "size_bytes": "Size (bytes)",
    "ops_per_s": "Ops/s",
    "bandwidth_bps": "Bandwidth (B/s)",
}

st.set_page_config(
    page_title="DFAnalyzer",
    layout="centered",
    menu_items={
        "About": "https://dfanalyzer.readthedocs.io/en/latest/",
        "Report a bug": "https://github.com/LLNL/dfanalyzer/issues",
    },
)

st.write(
    r"""
    <style>
        [data-testid="stMainBlockContainer"] {max-width: 812px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DFAnalyzer")
st.markdown("Analyze, visualize, and understand I/O performance issues in HPC workflows.")


def _run_analysis(
    trace_files,
    preset: str,
    view_types: List[str],
    time_granularity: int,
    logical_view_types: bool,
    n_workers: int,
) -> Dict[str, Any]:
    """Analyze the uploaded traces and return the JSON output payload."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for trace_file in trace_files:
            with open(f"{temp_dir}/{trace_file.name}", "wb") as handle:
                handle.write(trace_file.getbuffer())

        output_path = Path(temp_dir) / "dfanalyzer_output.json"
        overrides = [
            "analyzer=dftracer",
            f"analyzer/preset={preset}",
            "analyzer.checkpoint=False",
            f"analyzer.time_granularity={time_granularity}",
            "cluster=local",
            # Pinned deliberately. With more than one worker the distributed read
            # can silently return a truncated result (LLNL/dfanalyzer#69), which
            # would show up here as plausible-looking but understated numbers.
            f"cluster.n_workers={n_workers}",
            # One in-process worker instead of a separate process. Measured on
            # this repo's fixtures, a process-based worker costs an extra
            # 120-210 MB because it re-imports pyarrow/pandas/dftracer, and with
            # the 0.078-2 cores Community Cloud allows it buys no parallelism.
            "cluster.processes=False",
            f"cluster.memory_limit={WORKER_MEMORY_LIMIT_BYTES}",
            f"hydra.run.dir={temp_dir}",
            f"hydra.runtime.output_dir={temp_dir}",
            f"logical_view_types={logical_view_types}",
            f"trace_path={temp_dir}",
            f"view_types=[{','.join(view_types)}]",
            "output=json",
            f"output.file_path={output_path}",
        ]

        dfa = init_with_hydra(hydra_overrides=overrides)
        try:
            result = dfa.analyze_trace()
            dfa.output.handle_result(result)
            with output_path.open() as handle:
                return json.load(handle)
        finally:
            try:
                dfa.shutdown()
            except Exception:  # noqa: BLE001 - shutdown must not mask a real error
                pass


def _layer_frame(summary: Dict[str, Any]) -> pd.DataFrame:
    """Per-layer breakdown as a table, dropping all-empty columns."""
    layers = summary.get("layers") or {}
    rows = []
    for layer, metrics in layers.items():
        row = {"Layer": layer}
        row.update({label: metrics.get(key) for key, label in LAYER_COLUMNS.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.dropna(axis=1, how="all") if not frame.empty else frame


with st.form("analysis_form"):
    trace_files = st.file_uploader(
        "Upload trace files",
        type=list(TRACE_SUFFIXES),
        accept_multiple_files=True,
        help="DFTracer traces (.pfw.gz). Uncompressed .pfw is not readable by the indexer.",
    )

    preset = st.selectbox(
        "Preset",
        options=PRESETS,
        index=0,
        help="Layer definitions to analyze the trace with.",
    )

    selected_views = st.multiselect(
        "Select perspectives to analyze",
        options=list(VIEW_TYPE_MAPPING),
        default=["Process", "Timeline"],
    )

    time_granularity = st.slider(
        "Set time granularity for analysis (in seconds)",
        min_value=1,
        max_value=100,
        value=DEFAULT_TIME_GRANULARITY_IN_SECONDS,
        step=1,
        help="This sets the granularity of time intervals for analysis.",
        disabled="Timeline" not in selected_views,
    )

    logical_view_types = st.checkbox(
        "Enable logical view types",
        value=False,
        help="Logical view types allow for more complex analysis but may take longer to compute.",
    )

    n_workers = st.number_input(
        "Dask workers",
        min_value=1,
        max_value=16,
        value=1,
        step=1,
        help=(
            "Left at 1 on purpose: multi-worker reads can silently drop events "
            "(LLNL/dfanalyzer#69). Raise only if you accept that risk."
        ),
    )

    submit = st.form_submit_button("Analyze")

if submit:
    if not trace_files:
        st.error("Please upload at least one trace file.")
        st.stop()
    if not selected_views:
        st.error("Please select at least one perspective.")
        st.stop()

    # Streamlit's maxUploadSize is per file, so a multi-file submission can clear
    # it and still exhaust the host. Fail here, naming the number, rather than
    # letting the container get OOM-killed with no explanation.
    total_bytes = sum(len(trace_file.getbuffer()) for trace_file in trace_files)
    if total_bytes > MAX_UPLOAD_MB * 1024**2:
        st.error(
            f"Uploaded {total_bytes / 1024**2:.1f} MB across {len(trace_files)} file(s), "
            f"over the {MAX_UPLOAD_MB} MB total limit. Analysis memory grows with event "
            "count, and a hosted instance has far less than a compute node. Run the CLI "
            "on the full trace instead:\n\n"
            "```\ndfanalyzer analyzer/preset=<preset> trace_path=<dir> view_types=[time_range]\n```"
        )
        st.stop()
    if total_bytes > WARN_UPLOAD_KB * 1024:
        st.warning(
            f"{total_bytes / 1024:.0f} KB uploaded. Past roughly {WARN_UPLOAD_KB} KB the run "
            "can exceed a hosted instance's guaranteed memory and be killed mid-analysis. "
            "It may still succeed; the CLI is the reliable path for traces this size."
        )

    view_types = [VIEW_TYPE_MAPPING[view] for view in selected_views]

    with st.status("Analyzing trace files", expanded=True) as status:
        st.write(f"Received {len(trace_files)} trace file(s).")
        st.write(f"Running the {preset} preset over: {', '.join(view_types)}.")
        try:
            payload = _run_analysis(
                trace_files=trace_files,
                preset=preset,
                view_types=view_types,
                time_granularity=int(time_granularity),
                logical_view_types=logical_view_types,
                n_workers=int(n_workers),
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            status.update(label="Analysis failed.", expanded=True, state="error")
            st.error(f"Analysis failed: {exc}")
            st.stop()

        st.session_state["payload"] = payload
        status.update(label="Analysis complete.", expanded=False, state="complete")

payload = st.session_state.get("payload")

if payload:
    raw_stats = payload.get("raw_stats", {})
    st.subheader("Analysis Results")

    # The JSON backend emits null for any stat it could not compute, so every
    # read below has to tolerate None rather than assume a number.
    def _count(key: str) -> str:
        value = raw_stats.get(key)
        return f"{value:,}" if value is not None else "n/a"

    job_time = raw_stats.get("job_time_s")
    granularity = raw_stats.get("time_granularity_s")

    col11, col12, col13 = st.columns(3)
    col11.metric("Runtime", f"{job_time:.2f} s" if job_time is not None else "n/a", border=True)
    col12.metric("Processes", _count("unique_process_count"), border=True)
    col13.metric("Files", _count("unique_file_count"), border=True)

    col21, col22, col23 = st.columns(3)
    col21.metric("Events", _count("total_event_count"), border=True)
    col22.metric("Hosts", _count("unique_host_count"), border=True)
    col23.metric("Granularity", f"{granularity:g} s" if granularity is not None else "n/a", border=True)

    if not raw_stats.get("total_event_count"):
        st.warning(
            "The trace loaded but produced zero events. The indexer reads .pfw.gz; "
            "an uncompressed .pfw parses to nothing."
        )

    views = payload.get("views") or {}
    if views:
        for tab, view_name in zip(st.tabs(list(views)), views):
            with tab:
                summary = views[view_name].get("summary", {})
                frame = _layer_frame(summary)
                if frame.empty:
                    st.info("No layer metrics for this perspective.")
                else:
                    st.dataframe(frame, hide_index=True, width="stretch")
                    if "I/O Time (s)" in frame.columns:
                        st.altair_chart(
                            alt.Chart(frame)
                            .mark_bar()
                            .encode(
                                x=alt.X("I/O Time (s)", title="I/O Time (s)"),
                                y=alt.Y("Layer", sort="-x", title=None),
                            ),
                            width="stretch",
                        )

                additional_metrics = views[view_name].get("additional_metrics") or {}
                if additional_metrics:
                    st.markdown("**Additional metrics**")
                    st.json(additional_metrics)
