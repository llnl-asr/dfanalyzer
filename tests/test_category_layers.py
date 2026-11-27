"""Discovered category layers for the `auto` preset, plus their display names."""
from collections import Counter

import pytest
from dftracer.utils import DataFrame
from omegaconf import OmegaConf

from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.config import AnalyzerPresetConfigAuto, AnalyzerPresetConfigPOSIX
from dftracer.analyzer.constants import HUMANIZED_LAYERS
from dftracer.analyzer.dftracer import DFTracerAnalyzer
from dftracer.analyzer.frames.expr import query

pytestmark = [pytest.mark.smoke, pytest.mark.full]


NESTED_CATS = ["ai", "ai.compute", "ai.data", "ai.data.io", "compute", "posix"]


def _layers(cats):
    layer_defs, layer_deps, derived_metrics, size_layers = Analyzer._category_layers_from_cats(cats)
    return layer_defs, layer_deps, derived_metrics, size_layers


def _matched(df, condition):
    return df.num_rows if condition is None else query(df, condition).num_rows


# --- cat normalization -------------------------------------------------------


def test_normalize_cats_lowercases_and_sorts():
    assert Analyzer._normalize_cats(["POSIX", "MPI", "C_APP"]) == ["c_app", "mpi", "posix"]


def test_normalize_cats_omits_dftracer_metadata_category():
    assert Analyzer._normalize_cats(["POSIX", "dftracer", "DFTracer"]) == ["posix"]


def test_normalize_cats_drops_empty_and_none():
    assert Analyzer._normalize_cats(["posix", None, ""]) == ["posix"]


# --- layer key sanitization --------------------------------------------------


def test_layer_key_replaces_dots_with_underscores():
    assert Analyzer._category_layer_key("ai.data.io", set()) == "ai_data_io"


def test_layer_key_reserves_app_for_the_boundary_layer():
    assert Analyzer._category_layer_key("app", set()) == "app_cat"


def test_layer_key_dedupes_colliding_keys():
    taken = set()
    assert Analyzer._category_layer_key("x.y", taken) == "x_y"
    assert Analyzer._category_layer_key("x_y", taken) == "x_y_2"


# --- flat categories ---------------------------------------------------------


def test_flat_cats_become_children_of_the_app_boundary():
    layer_defs, layer_deps, _, _ = _layers(["mpi", "posix"])
    assert layer_defs["app"] is None  # boundary matches every event
    assert layer_deps == {"app": None, "mpi": "app", "posix": "app"}


def test_app_boundary_layer_is_first():
    # get_time_boundary_layer() takes list(layer_defs)[0]
    layer_defs, _, _, _ = _layers(NESTED_CATS)
    assert next(iter(layer_defs)) == "app"


def test_every_layer_gets_an_empty_derived_metrics_entry():
    layer_defs, _, derived_metrics, _ = _layers(NESTED_CATS)
    assert set(derived_metrics) == set(layer_defs)
    assert all(v == {} for v in derived_metrics.values())


# --- nested (dotted) categories ----------------------------------------------


def test_dotted_cats_build_a_parent_chain():
    _, layer_deps, _, _ = _layers(NESTED_CATS)
    assert layer_deps["ai"] == "app"
    assert layer_deps["ai_data"] == "ai"
    assert layer_deps["ai_data_io"] == "ai_data"
    assert layer_deps["ai_compute"] == "ai"
    assert layer_deps["compute"] == "app"


def test_missing_intermediate_prefix_is_synthesized():
    # only the leaf is present in the trace; `ai` and `ai.data` must still exist
    layer_defs, layer_deps, _, _ = _layers(["ai.data.io"])
    assert set(layer_defs) == {"app", "ai", "ai_data", "ai_data_io"}
    assert layer_deps["ai_data_io"] == "ai_data"
    assert layer_deps["ai_data"] == "ai"


def test_parent_layer_matches_itself_and_descendants():
    layer_defs, _, _, _ = _layers(NESTED_CATS)
    df = DataFrame.from_dict({"cat": ["ai", "ai.data", "ai.data.io", "ai.compute", "compute", "posix"]})
    assert _matched(df, layer_defs["app"]) == 6  # boundary: everything
    assert _matched(df, layer_defs["ai"]) == 4  # ai + ai.*
    assert _matched(df, layer_defs["ai_data"]) == 2  # ai.data + ai.data.io
    assert _matched(df, layer_defs["ai_data_io"]) == 1  # leaf


def test_leaf_layer_uses_exact_match():
    layer_defs, _, _, _ = _layers(["ai.data.io"])
    assert layer_defs["ai_data_io"] == 'cat == "ai.data.io"'


def test_sibling_prefix_is_not_over_captured():
    """`compute` must not absorb `ai.compute`, nor `ai` absorb a cat merely starting with 'ai'."""
    layer_defs, _, _, _ = _layers(["ai", "ai.compute", "compute", "airflow"])
    df = DataFrame.from_dict({"cat": ["ai", "ai.compute", "compute", "airflow"]})
    assert _matched(df, layer_defs["compute"]) == 1
    assert _matched(df, layer_defs["ai"]) == 2  # ai + ai.compute, NOT airflow
    assert _matched(df, layer_defs["airflow"]) == 1


# --- size layers -------------------------------------------------------------


@pytest.mark.parametrize("cat", ["posix", "stdio", "posix_reader_lustre", "stdio_checkpoint"])
def test_io_categories_are_size_layers(cat):
    _, _, _, size_layers = _layers([cat])
    assert size_layers == [Analyzer._category_layer_key(cat, set())]


@pytest.mark.parametrize("cat", ["mpi", "compute", "ai.data"])
def test_non_io_categories_are_not_size_layers(cat):
    _, _, _, size_layers = _layers([cat])
    assert size_layers == []


@pytest.mark.parametrize("cat", ["myposix", "not_stdio", "ai.posix"])
def test_categories_merely_containing_posix_are_not_size_layers(cat):
    """Size layers key off a prefix, not a substring."""
    _, _, _, size_layers = _layers([cat])
    assert size_layers == []


# --- query-expression safety -------------------------------------------------


@pytest.mark.parametrize("cat", ['we"ird', "back\\slash", "quo'te", "sp ace"])
def test_query_significant_characters_are_escaped(cat):
    """A raw cat interpolated into a query string could break parsing or inject."""
    layer_defs, _, _, _ = _layers([cat])
    key = Analyzer._category_layer_key(cat, set())
    df = DataFrame.from_dict({"cat": [cat, "other"]})
    assert _matched(df, layer_defs[key]) == 1


def test_escaped_parent_condition_still_matches_descendants():
    layer_defs, _, _, _ = _layers(['we"ird', 'we"ird.child'])
    parent_key = Analyzer._category_layer_key('we"ird', set())
    df = DataFrame.from_dict({"cat": ['we"ird', 'we"ird.child', "other"]})
    assert _matched(df, layer_defs[parent_key]) == 2


# --- preset wiring -----------------------------------------------------------


def test_category_layers_are_built_from_the_read_frame(tmp_path):
    analyzer = DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigAuto()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        time_granularity=1,
        time_resolution=10**6,
    )
    traces = DataFrame.from_dict({"cat": ["POSIX", None, "ai.data", "", "dftracer", "posix"]})
    analyzer._build_category_layers(traces)
    assert analyzer.layers == ["app", "ai", "ai_data", "posix"]
    assert analyzer.preset.layer_deps["ai_data"] == "ai"


def test_auto_preset_enables_category_discovery():
    assert AnalyzerPresetConfigAuto().auto_layers_by_category is True


def test_auto_preset_falls_back_to_a_catch_all_layer():
    preset = AnalyzerPresetConfigAuto()
    assert preset.layer_defs == {"app": None}  # used when no categories are found


def test_other_presets_do_not_enable_category_discovery():
    assert AnalyzerPresetConfigPOSIX().auto_layers_by_category is False


# --- display names -----------------------------------------------------------


@pytest.mark.parametrize(
    "cat,label",
    [
        ("mpi", "MPI"),
        ("mpiio", "MPI-IO"),
        ("c_app", "C App"),
        ("cpp_app", "C++ App"),
        ("ai_root", "AI Root"),
        ("dataloader", "Data Loader"),  # pydftracer ProfileCategory.DATALOADER
        ("data_loader", "DLIO Data Loader"),  # dlio layer
        ("posix", "POSIX - All"),
        ("posix_reader_lustre", "POSIX - Reader (Lustre)"),
    ],
)
def test_known_categories_have_display_names(cat, label):
    assert HUMANIZED_LAYERS[cat] == label


def test_dlio_and_pydftracer_loaders_are_distinguishable():
    # unet3d emits both cats; identical labels made the breakdown ambiguous
    assert HUMANIZED_LAYERS["data_loader"] != HUMANIZED_LAYERS["dataloader"]


def test_unknown_category_renders_as_is():
    assert HUMANIZED_LAYERS.get("ai_data_io", "ai_data_io") == "ai_data_io"


def test_no_ambiguous_display_names():
    """Two layers that can appear in one run must not share a label.

    `posix_reader`/`reader_posix` style pairs are aliases of the same concept
    across presets and never coexist, so they are exempt.
    """
    duplicates = {
        label: sorted(k for k in HUMANIZED_LAYERS if HUMANIZED_LAYERS[k] == label)
        for label, count in Counter(HUMANIZED_LAYERS.values()).items()
        if count > 1
    }
    ambiguous = {
        label: keys for label, keys in duplicates.items() if not all("posix" in k for k in keys)
    }
    assert ambiguous == {}
