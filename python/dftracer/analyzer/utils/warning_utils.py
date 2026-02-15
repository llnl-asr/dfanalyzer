import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning, message=".*grouper.*")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="dask_expr._collection")
# TODO(izzet): Remove this once we have a proper fix for the sqrt warning
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in sqrt.*")
warnings.filterwarnings(action="ignore", category=UserWarning, message=".*pkg_resources is deprecated as an API.*")
