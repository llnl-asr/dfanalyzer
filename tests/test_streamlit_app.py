"""Guard the Streamlit front end.

The app sat broken for 162 commits because nothing imported it: `rules.py` was
deleted, `XFER_SIZE_BIN_LABELS` was renamed, `AnalysisResult.characteristics`
went away, and `dask_jobqueue`'s import-time signal handler made the whole
package unimportable off the main thread. The render test below catches every
one of those in a couple of seconds.
"""

import pathlib
import re
import pytest

pytest.importorskip("streamlit", reason="streamlit is an optional extra: pip install .[web]")

from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = str(REPO_ROOT / "streamlit_app.py")
STREAMLIT_CONFIG = REPO_ROOT / ".streamlit" / "config.toml"
POSIX_TRACE_DIR = pathlib.Path("tests/data/extracted/dftracer-posix")

# Ground truth for the dftracer-posix fixture, matching tests/test_e2e.py.
EXPECTED_EVENTS = "2,056"
EXPECTED_PROCESSES = "1"


@pytest.mark.smoke
def test_streamlit_app_renders():
    """The app imports and lays out its form without raising."""
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()

    assert not at.exception, f"app raised on first render: {at.exception[0].value}"
    assert at.file_uploader, "expected a trace file uploader"
    assert at.button, "expected an Analyze button"

    # Uncompressed .pfw parses to nothing (see #63), so it must not be offered.
    assert at.file_uploader[0].allowed_type == [".pfw.gz"]


@pytest.mark.smoke
def test_upload_limit_matches_streamlit_config():
    """The per-file cap and the whole-submission cap must not drift apart.

    Streamlit enforces maxUploadSize per file; the app enforces MAX_UPLOAD_MB
    across a submission. Two copies of one number is exactly the kind of
    constant that rots silently.
    """
    assert STREAMLIT_CONFIG.is_file(), (
        f"{STREAMLIT_CONFIG} is missing, so Streamlit's 200 MB default applies"
    )
    config_match = re.search(r"^\s*maxUploadSize\s*=\s*(\d+)", STREAMLIT_CONFIG.read_text(), re.M)
    assert config_match, "maxUploadSize not set in .streamlit/config.toml"

    app_match = re.search(r"^MAX_UPLOAD_MB\s*=\s*(\d+)", pathlib.Path(APP_PATH).read_text(), re.M)
    assert app_match, "MAX_UPLOAD_MB not defined in streamlit_app.py"

    assert int(config_match.group(1)) == int(app_match.group(1)), (
        f"maxUploadSize={config_match.group(1)} MB in .streamlit/config.toml but "
        f"MAX_UPLOAD_MB={app_match.group(1)} MB in streamlit_app.py"
    )


@pytest.mark.full
def test_streamlit_app_analyzes_trace():
    """Uploading a real trace produces the same counts the e2e suite asserts."""
    traces = sorted(POSIX_TRACE_DIR.glob("*.pfw.gz"))
    assert traces, f"no traces in {POSIX_TRACE_DIR}; the conftest extraction should provide them"

    at = AppTest.from_file(APP_PATH, default_timeout=900).run()
    uploader = at.file_uploader[0]
    for trace in traces:
        uploader.upload(trace.name, trace.read_bytes())
    uploader.run(timeout=900)
    at.button[0].click().run(timeout=900)

    assert not at.exception, f"analysis raised: {at.exception[0].value}"
    assert not at.error, f"analysis reported an error: {[e.value for e in at.error]}"

    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("Events") == EXPECTED_EVENTS, (
        f"expected {EXPECTED_EVENTS} events, got {metrics.get('Events')}. A zero or short "
        "count means the trace was not read, not that the analysis is empty."
    )
    assert metrics.get("Processes") == EXPECTED_PROCESSES
    assert at.dataframe, "expected a per-layer breakdown table"
