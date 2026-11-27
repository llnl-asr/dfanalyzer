"""Library frames for tests, built as the analyzer builds them."""

from dftracer.utils import DataFrame, Series


def flat_view(index_name, index, **columns) -> DataFrame:
    """A library frame indexed by `index_name` with float64 `columns`, as the analyzer builds a flat view."""
    data = {index_name: Series(index)}
    data.update({name: Series(values, dtype="float64") for name, values in columns.items()})
    return DataFrame.from_dict(data).set_index([index_name])
