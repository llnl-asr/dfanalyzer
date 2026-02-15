import glob
import json
import os
import pytest
import shutil
import subprocess
import tarfile
import time
import uuid


@pytest.fixture(scope="session", autouse=True)
def extract_test_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    _ensure_extracted_data(data_dir)


@pytest.fixture(scope="session")
def dftracer_ai_logging_posix_events():
    """Return up to 3 epochs extracted from a real trace file.

    Each epoch is a list of the original JSON lines (strings) captured from the
    trace until an epoch.block event is seen.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    _ensure_extracted_data(data_dir)

    ai_logging_dir = os.path.join(data_dir, "extracted", "dftracer-dlio-ai-logging")
    matches = sorted(glob.glob(os.path.join(ai_logging_dir, "*.pfw")))
    if not matches:
        raise FileNotFoundError(f"No .pfw files found in: {ai_logging_dir}")
    file_path = matches[0]

    if not os.path.exists(file_path):
        # If the test data isn't present, skip tests that request this fixture.
        raise FileNotFoundError(f"Real trace file not found: {file_path}")

    epochs = []
    events = None
    end_count = 0
    with open(file_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                # ignore non-json lines
                continue

            # begin collecting immediately to retain metadata before epoch.start
            if events is None:
                events = []

            if events is not None:
                # detect epoch.block; always include it and close the epoch
                if j.get("name") == "epoch.block":
                    events.append(line)
                    epochs.append(events)
                    events = None
                    end_count += 1
                    continue

                events.append(line)

            # stop when we have three epoch.block events
            if end_count >= 3:
                break

    return epochs


def _ensure_extracted_data(data_dir: str) -> None:
    """Ensure any tar.gz files under data_dir are extracted into data_dir/extracted.

    This is called by both the autouse fixture and by tests that need the
    extracted files (so CI can trigger extraction if the extracted folder is
    missing).
    """
    tar_files = glob.glob(os.path.join(data_dir, "*.tar.gz"))

    for tar_path in tar_files:
        tar_name = os.path.basename(tar_path)
        extract_folder_name = tar_name.replace(".tar.gz", "")
        extract_path = os.path.join(data_dir, "extracted", extract_folder_name)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)

        if not any(os.scandir(extract_path)):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_path)


def _mofka_available():
    try:
        import mochi.mofka.client as mofka  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@pytest.fixture(scope="module")
def bedrock_mofka():
    if not _mofka_available():
        pytest.skip("mochi.mofka is not installed")

    bedrock = shutil.which("bedrock")
    if not bedrock:
        pytest.skip("bedrock not found in PATH")

    tests_root = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(tests_root, "mofka.config.json")
    group_file = os.path.join(tests_root, "mofka.group.json")
    log_path = os.path.join(tests_root, "mofka_server.log")

    if os.path.exists(group_file):
        os.remove(group_file)

    proc = subprocess.Popen(
        [bedrock, "tcp", "-c", config_path, "-v", "trace"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=tests_root,
    )

    timeout = time.time() + 10
    while time.time() < timeout:
        if os.path.exists(group_file):
            break
        time.sleep(0.1)
    if not os.path.exists(group_file):
        proc.terminate()
        pytest.fail("mofka.group.json was not created")

    topic_name = f"dfanalyzer_test_{uuid.uuid4().hex}"
    subprocess.check_call(
        [
            "python",
            "-m",
            "mochi.mofka.mofkactl",
            "topic",
            "create",
            topic_name,
            "--groupfile",
            group_file,
        ]
    )
    subprocess.check_call(
        [
            "python",
            "-m",
            "mochi.mofka.mofkactl",
            "partition",
            "add",
            topic_name,
            "--type",
            "memory",
            "--rank",
            "0",
            "--groupfile",
            group_file,
        ]
    )

    yield group_file, topic_name

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if os.path.exists(group_file):
        os.remove(group_file)
