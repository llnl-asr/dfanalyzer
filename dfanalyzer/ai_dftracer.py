from typing import Callable, Dict, List, Optional

import pandas as pd

from .constants import (
    COL_EPOCH,
    COL_FUNC_NAME,
    COL_PROC_NAME,
    COL_TIME_RANGE,
    AIDFTracer,
)
from .types import (
    ViewMetricBoundaries,
    ViewType,
)
from .dftracer import DFTracerAnalyzer


class AIDFTracerAnalyzer(DFTracerAnalyzer):
    def postread_trace(self, traces, view_types):
        traces = super().postread_trace(traces, view_types)
        epochs = (
            traces.query(AIDFTracer.get_epoch_query())
            .groupby([COL_PROC_NAME, COL_FUNC_NAME])
            .agg({COL_TIME_RANGE: list})
        )
        epochs[COL_EPOCH] = epochs[COL_TIME_RANGE].apply(lambda x: list(range(1, len(x) + 1)))
        epochs = (
            epochs.explode([COL_TIME_RANGE, COL_EPOCH])
            .groupby(COL_EPOCH)
            .min()
            .reset_index()
            .astype('uint64[pyarrow]')
        )
        traces = traces.map_partitions(self._set_epochs, epochs=epochs)
        traces[COL_EPOCH] = traces[COL_EPOCH].replace({0: pd.NA}).astype('uint64[pyarrow]')

        return traces