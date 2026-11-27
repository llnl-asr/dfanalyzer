import ast
import subprocess
import sys
import typing
from pathlib import Path

import pytest
from dftracer.utils import DataFrame

TYPES_PY = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer" / "types.py"


@pytest.mark.smoke
def test_types_py_has_no_module_level_dask_import():
    tree = ast.parse(TYPES_PY.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            module = getattr(node, "module", None) or ""
            assert not any("dask" in n for n in names + [module]), ast.dump(node)


def test_types_import_does_not_load_dask_dataframe():
    code = (
        "import sys\n"
        "import dftracer.analyzer.types\n"
        "assert 'dask.dataframe' not in sys.modules, 'dask.dataframe was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.smoke
def test_type_hints_resolve_at_run_time():
    from dftracer.analyzer.types import AnalysisResult, View

    assert View is DataFrame
    assert typing.get_type_hints(AnalysisResult)
