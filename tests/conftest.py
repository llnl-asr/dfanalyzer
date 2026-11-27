import glob
import os
import pytest
import shutil
import tarfile
import tempfile


@pytest.fixture(scope='session', autouse=True)
def extract_test_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    tar_files = glob.glob(os.path.join(data_dir, '*.tar.gz'))

    for tar_path in tar_files:
        tar_name = os.path.basename(tar_path)
        extract_folder_name = tar_name.replace('.tar.gz', '')
        extract_path = os.path.join(data_dir, 'extracted', extract_folder_name)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        if not any(os.scandir(extract_path)):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)


def _make_trace_copier(root):
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'extracted')

    def copy(name: str) -> str:
        dest = tempfile.mkdtemp(dir=str(root))
        target = os.path.join(dest, name)
        shutil.copytree(os.path.join(data_dir, name), target, ignore=shutil.ignore_patterns('.dftindex'))
        return target

    return copy


@pytest.fixture
def trace_copy(extract_test_data, tmp_path):
    """Return `copy(name)`, which gives a fresh copy of an extracted trace dir.

    The analyzer writes its index inside the trace dir, so tests must not share one.
    """
    return _make_trace_copier(tmp_path)


@pytest.fixture(scope='module')
def module_trace_copy(extract_test_data, tmp_path_factory):
    """Same as `trace_copy`, for module-scoped fixtures."""
    return _make_trace_copier(tmp_path_factory.mktemp('traces'))
