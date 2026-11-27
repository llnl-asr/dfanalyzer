import pandas as pd
import pytest
from betterset import BetterSet as S

from tests.pandas_reference import is_set_like_series

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_is_set_like_series_detects_betterset_values() -> None:
    series = pd.Series([None, S(["a"]), S(["b"])], dtype="object")
    assert is_set_like_series(series) is True


def test_is_set_like_series_ignores_plain_strings() -> None:
    series = pd.Series(["a", "b", None], dtype="object")
    assert is_set_like_series(series) is False
