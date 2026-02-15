import dask
import dask.bag as db
import dask.dataframe as dd
import glob
import json
import math
import numpy as np
import os
import pandas as pd
import portion as I
import structlog
from dftracer.utils import Indexer, Reader
from dask.distributed import wait
from typing import Callable, Dict, List, Optional

from .analyzer import Analyzer
from .constants import (
    COL_ACC_PAT,
    COL_COUNT,
    COL_EPOCH,
    COL_FILE_HASH,
    COL_FILE_NAME,
    COL_FUNC_NAME,
    COL_HOST_HASH,
    COL_HOST_NAME,
    COL_IO_CAT,
    COL_PROC_NAME,
    COL_TIME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    POSIX_IO_CAT_MAPPING,
    POSIX_METADATA_FUNCTIONS,
    IOCategory,
)
from .types import ViewType
from .utils.log_utils import log_block

logger = structlog.get_logger()

CAT_POSIX = "POSIX"
CAT_STDIO = "STDIO"
IGNORED_FILE_PATTERNS = [
    "/dev/",
    "/etc/",
    "/gapps/python",
    "/lib/python",
    "/proc/",
    "/software/",
    "/sys/",
    "/usr/lib",
    "/usr/tce/backend",
    "/usr/tce/packages",
    "/venv",
    "__pycache__",
]
IGNORED_FUNC_NAMES = [
    "DLIOBenchmark.__init__",
    # 'DLIOBenchmark._train',
    "DLIOBenchmark.initialize",
    # 'DLIOBenchmark.run',
    "FileStorage.__init__",
    "TorchDataset.__init__",
    # "TorchDataset.worker_init",
]
IGNORED_FUNC_PATTERNS = [
    "Checkpointing.__init__",
    "Checkpointing.finalize",
    "Checkpointing.get_tensor",
    "DataLoader.__init__",
    "DataLoader.finalize",
    "DataLoader.get_tensor",
    "DataLoader.next",
    "Framework.get_loader",
    "Framework.init_loader",
    "Framework.is_nativeio_available",
    "Framework.trace_object",
    "Reader.__init__",
    "Reader.load_index",
    "Reader.next",
    "Reader.read_index",
    ".save_state",
    "checkpoint_end_",
    "checkpoint_start_",
]
TRACE_COL_MAPPING = {
    "dur": COL_TIME,
    "name": COL_FUNC_NAME,
    "te": COL_TIME_END,
    "trange": COL_TIME_RANGE,
    "ts": COL_TIME_START,
}


def create_index(filename):
    index_file = f"{filename}.idx"
    if not os.path.exists(index_file):
        indexer = Indexer(filename, index_file, checkpoint_size=32 * 1024 * 1024)
        indexer.build()
        logger.debug("Creating index", filename=filename)
    return filename


def generate_batches(filename, max_bytes):
    batch_size = 4 * 1024 * 1024  # 4 MB
    for start in range(0, max_bytes, batch_size):
        # this range is intended since DFTracerJsonLinesBytesReader do
        # line boundary algorithm internally to chop incomplete line
        end = min(start + batch_size, max_bytes)
        logger.debug("Created batch", filename=filename, start=start, end=end)
        yield filename, start, end


def get_size(filename):
    size = 0
    if filename.endswith(".pfw"):
        size = os.stat(filename).st_size
    elif filename.endswith(".pfw.gz"):
        index_file = f"{filename}.idx"
        indexer = Indexer(filename, index_file)
        size = indexer.get_max_bytes()
    logger.debug("File has size", filename=filename, size=size / 1024**3)
    return filename, int(size)


def get_io_cat(func_name: str):
    if func_name in POSIX_METADATA_FUNCTIONS:
        return IOCategory.METADATA.value
    if func_name in POSIX_IO_CAT_MAPPING:
        return POSIX_IO_CAT_MAPPING[func_name].value
    return IOCategory.OTHER.value


def io_columns():
    columns = {
        "file_hash": "string",
        "host_hash": "string",
        "image_id": "Int64",
        "io_cat": "Int8",
        "size": "Int64",
        "offset": "Int64",
    }
    return columns


def io_function(json_dict: dict):
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "args" in json_dict:
        if "fhash" in json_dict["args"]:
            d["file_hash"] = str(json_dict["args"]["fhash"])
        if "size_sum" in json_dict["args"]:
            d["size"] = int(json_dict["args"]["size_sum"])
        elif json_dict["cat"] in [CAT_POSIX, CAT_STDIO]:
            name = json_dict["name"]
            io_cat = get_io_cat(name)
            if "ret" in json_dict["args"]:
                size = int(json_dict["args"]["ret"])
                if size > 0:
                    if io_cat in [IOCategory.READ.value, IOCategory.WRITE.value]:
                        d["size"] = size
            if "offset" in json_dict["args"]:
                offset = int(json_dict["args"]["offset"])
                if offset >= 0:
                    d["offset"] = offset
            d[COL_IO_CAT] = io_cat
        else:
            if "image_idx" in json_dict["args"]:
                image_id = int(json_dict["args"]["image_idx"])
                if image_id > 0:
                    d["image_id"] = image_id
            # if "image_size" in json_object["args"]:
            #     name = json_object["name"].lower()
            #     # e.g. NPZReader.open image_size is not correct
            #     if 'reader.open' not in name:
            #         size = int(json_object["args"]["image_size"])
            #         if size > 0:
            #             d["size"] = size
    return d


def load_indexed_gzip_files(filename, start, end):
    index_file = f"{filename}.idx"
    reader = Reader(filename, index_file)
    json_lines = reader.read_line_bytes_json(start, end)
    logger.debug("Read json lines", filename=filename, start=start, end=end, num_lines=len(json_lines))
    return json_lines


def load_objects_dict(
    json_dict: dict,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    final_dict = {}
    logger.debug("Loading dict", json_dict=json_dict)
    if json_dict is not None:
        try:
            if "name" in json_dict:
                final_dict["name"] = json_dict["name"]
            if "cat" in json_dict:
                final_dict["cat"] = json_dict["cat"].lower()
            if "pid" in json_dict:
                final_dict["pid"] = json_dict["pid"]
            if "tid" in json_dict:
                final_dict["tid"] = json_dict["tid"]
            if "args" in json_dict:
                if "hhash" in json_dict["args"]:
                    final_dict["host_hash"] = str(json_dict["args"]["hhash"])
                # if "level" in val["args"]:
                #     d["level"] = int(val["args"]["level"])
                # if (
                #     "epoch" in val["args"]
                #     and val["args"]["epoch"] != "train"
                #     and val["args"]["epoch"] != "valid"
                # ):
                #     epoch = int(val["args"]["epoch"])
                #     if epoch > 0:
                #         d["epoch"] = epoch
                if "step" in json_dict["args"]:
                    step = int(json_dict["args"]["step"])
                    if step > 0:
                        final_dict["step"] = step
            if "M" == json_dict["ph"]:
                if final_dict["name"] == "FH":
                    final_dict["type"] = 1  # 1-> file hash
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "HH":
                    final_dict["type"] = 2  # 2-> hostname hash
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "SH":
                    final_dict["type"] = 3  # 3-> string hash
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "PR":
                    final_dict["type"] = 5  # 5-> process metadata
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                else:
                    final_dict["type"] = 4  # 4-> others
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["value"] = str(json_dict["args"]["value"])
            else:
                final_dict["type"] = 0  # 0->regular event
                if "dur" in json_dict:
                    if type(json_dict["dur"]) is not int:
                        json_dict["dur"] = int(json_dict["dur"])
                    if type(json_dict["ts"]) is not int:
                        json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                    final_dict["dur"] = json_dict["dur"]
                    final_dict["te"] = final_dict["ts"] + final_dict["dur"]
                    if not time_approximate:
                        final_dict["tinterval"] = I.to_string(
                            I.closed(json_dict["ts"], json_dict["ts"] + json_dict["dur"])
                        )
                final_dict.update(io_function(json_dict))
                if extra_columns and extra_columns_fn:
                    final_dict.update(extra_columns_fn(json_dict))
            # check if all extra columns are present
            if extra_columns and not all(col in final_dict for col in extra_columns):
                missing_cols = [col for col in extra_columns if col not in final_dict]
                raise ValueError(f"Missing extra columns: {missing_cols}")
            logger.debug("Built a dictionary for dict", final_dict=final_dict)
            yield final_dict
        except ValueError as error:
            logger.error("Processing dict failed", dict=json_dict, error=error)
    return {}


def load_objects_str(
    line: str,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    if line is not None and line != "" and len(line) > 0 and "[" != line[0] and "]" != line[0] and line != "\n":
        try:
            unicode_line = "".join([i if ord(i) < 128 else "#" for i in line])
            json_dict = json.loads(unicode_line, strict=False)
            yield from load_objects_dict(json_dict, time_approximate, extra_columns, extra_columns_fn)
        except ValueError as error:
            logger.error("Processing line failed", line=line, error=error)
    return {}


class DFTracerAnalyzer(Analyzer):
    def __init__(self, preset, assign_epochs=False, **kwargs):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs

    def read_trace(self, trace_path, extra_columns, extra_columns_fn):
        with log_block("glob_files"):
            pfw_pattern, pfw_gz_pattern = [], []
            if os.path.isdir(trace_path):
                pfw_pattern = glob.glob(os.path.join(trace_path, "*.pfw"))
                pfw_gz_pattern = glob.glob(os.path.join(trace_path, "*.pfw.gz"))
            elif trace_path.endswith(".pfw.gz"):
                pfw_gz_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            elif trace_path.endswith(".pfw"):
                pfw_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            all_files = pfw_pattern + pfw_gz_pattern
            if not all_files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")
        logger.debug("Processing files", files=all_files)
        if len(pfw_gz_pattern) > 0:
            with log_block("create_index"):
                db.from_sequence(pfw_gz_pattern).map(create_index).compute()
                logger.info("Created index for files", num_files=len(pfw_gz_pattern))
        with log_block("sum_total_size"):
            sizes = db.from_sequence(all_files).map(get_size).compute()
            total_size = sum(size for _, size in sizes)
            logger.info("Total size of all files", total_size=total_size)
        gz_bag = None
        pfw_bag = None
        if len(pfw_gz_pattern) > 0:
            with log_block("gzip_index_and_batches"):
                logger.debug("Max bytes per file", sizes=sizes)
                json_line_delayed = []
                total_lines = 0
                for filename, max_bytes in sizes:
                    total_lines += max_bytes
                    for _, start, end in generate_batches(filename, max_bytes):
                        json_line_delayed.append((filename, start, end))

                logger.info(
                    "Loading batches",
                    num_batches=len(json_line_delayed),
                    num_files=len(pfw_gz_pattern),
                    total_lines=total_lines,
                )
                json_line_bags = []
                for filename, start, end in json_line_delayed:
                    json_line_bags.append(dask.delayed(load_indexed_gzip_files)(filename, start, end))
                json_lines = db.concat(json_line_bags)
            with log_block("parse_gzip_json_lines"):
                gz_bag = (
                    json_lines.map(
                        load_objects_dict,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        main_bag = None
        if len(pfw_pattern) > 0:
            with log_block("parse_json_lines"):
                pfw_bag = (
                    db.read_text(pfw_pattern)
                    .map(
                        load_objects_str,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        if len(pfw_gz_pattern) > 0 and len(pfw_pattern) > 0:
            main_bag = db.concat([pfw_bag, gz_bag])
        elif len(pfw_gz_pattern) > 0:
            main_bag = gz_bag
        elif len(pfw_pattern) > 0:
            main_bag = pfw_bag
        if main_bag:
            self._columns = self._get_columns(extra_columns)
            with log_block("to_dataframe"):
                raw_traces = main_bag.to_dataframe(meta=self._columns)
            with log_block("_handle_metadata"):
                traces = self._handle_metadata(raw_traces)
            self._npartitions = math.ceil(total_size / (128 * 1024**2))
            logger.debug(f"Number of partitions used are {self._npartitions}")
            with log_block("repartition+persist"):
                traces = traces.repartition(npartitions=self._npartitions).persist()
            with log_block("_fix_time+persist"):
                traces = self._fix_time(traces).persist()
            with log_block("wait_all"):
                if isinstance(traces, dd.DataFrame):
                    wait([traces, self._file_hashes, self._host_hashes, self._string_hashes, self._metadata])
        else:
            logger.error("Unable to load traces")
            exit(1)
        # ===============================================
        return self._rename_columns(traces)

    def read_zmq(self, trace_address, extra_columns, extra_columns_fn):
        trace_stream = super().read_zmq(
            trace_address=trace_address,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
        )
        return trace_stream.map(
            lambda line: next(
                load_json(
                    line,
                    time_approximate=self.time_approximate,
                    extra_columns=extra_columns,
                    extra_columns_fn=extra_columns_fn,
                )
            )
        )

    def postread_trace(self, traces, view_types):
        # print("Post-reading trace", traces)
        # print("Post-reading trace columns", traces.columns)
        is_dask = isinstance(traces, dd.DataFrame)

        if not is_dask and traces.empty:
            logger.warning("No traces found for postread_trace")
            return traces

        # Ignore redundant files
        with log_block("filter_files"):
            if COL_FILE_NAME in traces.columns:
                traces = traces[
                    traces[COL_FILE_NAME].isna()
                    | ~traces[COL_FILE_NAME].str.contains("|".join(IGNORED_FILE_PATTERNS), na=False)
                ]
            else:
                traces[COL_FILE_NAME] = traces[COL_FILE_HASH].astype(str).replace("nan", "")

        # Ignore redundant function calls
        with log_block("filter_functions"):
            traces = traces[~traces[COL_FUNC_NAME].isin(IGNORED_FUNC_NAMES)]
            traces = traces[~traces[COL_FUNC_NAME].str.contains("|".join(IGNORED_FUNC_PATTERNS))]

        # Set epochs
        with log_block("assign_epochs"):
            if self.assign_epochs:
                if "epoch" not in self.preset.layer_defs:
                    raise ValueError("Epoch layer definition is missing")
                epochs = traces.query(self.preset.layer_defs["epoch"])
                if is_dask:
                    epochs = epochs.compute()
                epochs_with_index = epochs.sort_values(["pid", "time_start"]).reset_index(drop=True)
                epochs_with_index["epoch"] = epochs_with_index.groupby("pid").cumcount() + 1
                epoch_boundaries = epochs_with_index[["pid", "time_start", "time_end", "epoch"]]
                if is_dask:
                    traces = traces.map_partitions(self._set_epochs, epoch_boundaries=epoch_boundaries)
                else:
                    traces = self._set_epochs(traces, epoch_boundaries=epoch_boundaries)

        with log_block("wait"):
            if is_dask:
                wait(traces)

        with log_block("set_basic_columns"):
            traces[COL_ACC_PAT] = 0
            traces[COL_COUNT] = 1

        if is_dask:
            traces = (
                traces.map_partitions(self._set_proc_names)
                .map_partitions(self._fix_file_posix_category)
                .map_partitions(self._sanitize_size)
            )
        else:
            traces = self._set_proc_names(traces)
            traces = self._fix_file_posix_category(traces)
            traces = self._sanitize_size(traces)

        return traces

    def postread_zmq(self, trace_stream, view_types, extra_columns, extra_columns_fn):
        columns = self._get_columns(extra_columns)
        return (
            trace_stream.map(lambda traces: pd.DataFrame(traces, columns=columns))
            .map(self._handle_metadata)
            .map(self._fix_time)
            .map(self._rename_columns)
            .map(lambda df: df.assign(time_range=1))
            .map(self.postread_trace, view_types=view_types)
        )

    def normalize_stream_event(
        self,
        event: dict,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ) -> dict:
        logger.debug(
            "stream.normalize_input",
            name=event.get("name"),
            ph=event.get("ph"),
            args_name=event.get("args", {}).get("name"),
        )
        normalized_event = next(
            load_objects_dict(
                event,
                time_approximate=self.time_approximate,
                extra_columns=extra_columns,
                extra_columns_fn=extra_columns_fn,
            )
        )
        return normalized_event

    def handle_stream_events(
        self,
        events: List[dict],
        view_types: List[ViewType],
        extra_columns: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        columns = self._get_columns(extra_columns)
        traces = pd.DataFrame(events, columns=columns)
        traces = self._handle_metadata(traces)
        traces = self._fix_time(traces)
        traces = self._rename_columns(traces)
        traces = traces.assign(time_range=1)
        traces = self.postread_trace(traces=traces, view_types=view_types)
        return traces

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_unique_file_count(self, traces: dd.DataFrame):
        return traces["file_hash"].nunique()

    def get_unique_host_count(self, traces: dd.DataFrame):
        return traces["host_hash"].nunique()

    def get_unique_process_count(self, traces: dd.DataFrame):
        return traces["pid"].nunique()

    @staticmethod
    def _fix_file_posix_category(df: pd.DataFrame):
        base_condition = df["cat"].str.contains("posix|stdio") & ~df["file_name"].isna()

        # Step 1: Map file purpose suffixes first
        purpose_updates = {"/data": "_reader", "/checkpoint": "_checkpoint"}

        for path, suffix in purpose_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        # Step 2: Map filesystem suffixes
        filesystem_updates = {"/lustre": "_lustre", "/ssd": "_ssd"}

        for path, suffix in filesystem_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        return df

    def _fix_time(self, traces: dd.DataFrame) -> dd.DataFrame:
        traces["ts"] = traces["ts"] - traces["ts"].min()
        traces["te"] = traces["ts"] + traces["dur"]
        traces["trange"] = traces["ts"] // (self.time_granularity * self.time_resolution)
        traces["ts"] = traces["ts"].astype("Int64")
        traces["te"] = traces["te"].astype("Int64")
        traces["trange"] = traces["trange"].astype("Int16")
        traces["dur"] = traces["dur"] / self.time_resolution
        logger.debug(
            "Fixed time columns",
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
        )
        return traces

    def _get_columns(self, extra_columns: Optional[Dict[str, str]]):
        columns = {
            "name": "string",
            "cat": "string",
            "type": "Int8",
            "pid": "Int64",
            "tid": "Int64",
            "ts": "Int64",
            "te": "Int64",
            "dur": "Int64",
            "tinterval": "Int64" if self.time_approximate else "string",
            "trange": "Int64",
            "level": "Int8",
        }
        metadata_columns = {
            "hash": "string",
            "host_hash": "string",
            "value": "string",
        }
        columns.update(io_columns())
        columns.update(metadata_columns)
        columns.update(extra_columns or {})
        logger.debug("get_columns", columns=columns)
        return columns

    def _handle_metadata(self, raw_traces: dd.DataFrame) -> dd.DataFrame:
        is_dask = isinstance(raw_traces, dd.DataFrame)
        traces = raw_traces.query("type == 0")
        file_hashes = raw_traces.query("type == 1")[["name", "hash"]].groupby("hash").first()
        host_hashes = raw_traces.query("type == 2")[["name", "hash"]].groupby("hash").first()
        string_hashes = raw_traces.query("type == 3")[["name", "hash"]].groupby("hash").first()
        metadata = raw_traces.query("type == 4")[["name", "value"]]
        file_hashes.index = file_hashes.index.astype(str)
        host_hashes.index = host_hashes.index.astype(str)
        string_hashes.index = string_hashes.index.astype(str)
        if is_dask:
            file_hashes = file_hashes.persist()
            host_hashes = host_hashes.persist()
            string_hashes = string_hashes.persist()
            metadata = metadata.persist()
        traces = traces.merge(
            file_hashes.rename(columns={"name": COL_FILE_NAME}),
            how="left",
            left_on="file_hash",
            right_index=True,
        )
        traces = traces.merge(
            host_hashes.rename(columns={"name": COL_HOST_NAME}),
            how="left",
            left_on="host_hash",
            right_index=True,
        )
        self._file_hashes = file_hashes
        self._host_hashes = host_hashes
        self._string_hashes = string_hashes
        self._metadata = metadata
        logger.debug("Handled metadata")
        return traces

    @staticmethod
    def _rename_columns(traces: dd.DataFrame) -> dd.DataFrame:
        return traces.rename(columns=TRACE_COL_MAPPING)

    @staticmethod
    def _sanitize_size(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, pd.NA)
        return df

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _set_proc_names(df: pd.DataFrame):
        df[COL_PROC_NAME] = (
            "app#" + df[COL_HOST_NAME].astype(str) + "#" + df["pid"].astype(str) + "#" + df["tid"].astype(str)
        )
        return df

    @staticmethod
    def _set_steps(df: pd.DataFrame, step_time_ranges: pd.DataFrame):
        mapped_traces = df.copy()

        for pid in df["pid"].unique():
            pid_trace_cond = mapped_traces["pid"] == pid
            pid_traces = mapped_traces[pid_trace_cond]
            pid_step_ranges = step_time_ranges[step_time_ranges["pid"] == pid]

            # Sort step ranges by start timestamp
            pid_step_ranges_sorted = pid_step_ranges.sort_values("ts")

            # Create bins and labels
            bins = pid_step_ranges_sorted["ts"].tolist()
            if len(bins) > 0:
                bins.append(pid_step_ranges_sorted["te"].max())
            # print(pid, bins)
            steps = pid_step_ranges_sorted["step"].tolist()

            # Use np.digitize to find bin indices
            bin_indices = np.digitize(pid_traces["ts"], bins=bins) - 1

            # Map indices to steps, leaving as None for out-of-range timestamps
            mapped_traces.loc[pid_trace_cond, "step"] = [
                steps[idx] if 0 <= idx < len(steps) else pd.NA for idx in bin_indices
            ]

        return mapped_traces
