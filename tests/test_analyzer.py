"""Argument checks of `Analyzer.analyze_trace`."""

from types import SimpleNamespace

import pytest
from dftracer.analyzer.analyzer import Analyzer


@pytest.mark.smoke
def test_analyze_trace_refuses_a_view_type_it_cannot_build():
    with pytest.raises(ValueError, match="step"):
        Analyzer.analyze_trace(SimpleNamespace(), "unused", view_types=["proc_name", "step"])
