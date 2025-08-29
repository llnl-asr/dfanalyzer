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
import sys
import zindex_py as zindex
from dask.distributed import wait
from typing import Callable, Dict, List, Optional

from .analyzer import Analyzer
from .constants import (
    COL_ACC_PAT,
    COL_COUNT,
    COL_EPOCH,
    COL_FILE_NAME,
    COL_FUNC_NAME,
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
    Layer,
)
from .types import ViewType
from .utils.log_utils import console_block, log_block

logger = structlog.get_logger()

CAT_POSIX = "POSIX"
CAT_STDIO = "STDIO"
COND_CHECKPOINT = {
    "cat": {"checkpoint"},
    "name": {"TFCheckpointing.checkpoint"},
}
COND_COMPUTE = {
    "cat": {"compute"},
    "name": {"TFFramework.compute", "compute", "cpu"},
}
COND_READ = {
    "cat": {"IO"},
    "name": {
        "TFReader._parse_image",
        "TorchDataset.__getitem__",
    },
}
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
    "IndexedBinaryMMapReader.__init__",
    "IndexedBinaryMMapReader.load_index",
    "IndexedBinaryMMapReader.next",
    "IndexedBinaryMMapReader.read_index",
    "NPZReader.__init__",
    "NPZReader.next",
    "NPZReader.read_index",
    "PyTorchCheckpointing.__init__",
    "PyTorchCheckpointing.finalize",
    "PyTorchCheckpointing.get_tensor",
    "SCRPyTorchCheckpointing.__init__",
    "SCRPyTorchCheckpointing.finalize",
    "SCRPyTorchCheckpointing.get_tensor",
    "TFCheckpointing.__init__",
    "TFCheckpointing.finalize",
    "TFCheckpointing.get_tensor",
    "TFDataLoader.__init__",
    "TFDataLoader.finalize",
    "TFDataLoader.next",
    "TFDataLoader.read",
    "TFFramework.get_loader",
    "TFFramework.init_loader",
    "TFFramework.is_nativeio_available",
    "TFFramework.trace_object",
    "TFReader.__init__",
    "TFReader.next",
    "TFReader.read_index",
    "TorchDataLoader.__init__",
    "TorchDataLoader.finalize",
    "TorchDataLoader.next",
    "TorchDataLoader.read",
    "TorchDataset.__init__",
    # 'TorchDataset.worker_init',
    "TorchFramework.get_loader",
    "TorchFramework.init_loader",
    "TorchFramework.is_nativeio_available",
    "TorchFramework.trace_object",
]
IGNORED_FUNC_PATTERNS = [
    '.save_state',
    'checkpoint_end_',
    'checkpoint_start_',
]
TRACE_COL_MAPPING = {
    'dur': COL_TIME,
    'name': COL_FUNC_NAME,
    'te': COL_TIME_END,
    'trange': COL_TIME_RANGE,
    'ts': COL_TIME_START,
}

def create_index(filename):
    index_file = f"{filename}.zindex"
    if not os.path.exists(index_file):
        status = zindex.create_index(
            filename,
            index_file=f"file:{index_file}",
            regex="id:\b([0-9]+)",
            numeric=True,
            unique=True,
            debug=False,
            verbose=False,
        )
        logger.debug("Creating index", filename=filename, status=status)
    return filename


def generate_line_batches(filename, max_line):
    batch_size = 1024 * 16
    for start in range(0, max_line, batch_size):
        end = min((start + batch_size - 1), (max_line - 1))
        logger.debug("Created batch", filename=filename, start=start, end=end)
        yield filename, start, end


def get_linenumber(filename):
    index_file = f"{filename}.zindex"
    line_number = zindex.get_max_line(
        filename,
        index_file=index_file,
        debug=False,
        verbose=False,
    )
    logger.debug("File has lines", filename=filename, line_number=line_number)
    return (filename, line_number)


def get_size(filename):
    if filename.endswith(".pfw"):
        size = os.stat(filename).st_size
    elif filename.endswith(".pfw.gz"):
        index_file = f"{filename}.zindex"
        line_number = zindex.get_max_line(
            filename,
            index_file=index_file,
            debug=False,
            verbose=False,
        )
        size = line_number * 256
    logger.debug("File has size", filename=filename, size=size / 1024**3)
    return int(size)


def get_io_cat(func_name: str):
    if func_name in POSIX_METADATA_FUNCTIONS:
        return IOCategory.METADATA.value
    if func_name in POSIX_IO_CAT_MAPPING:
        return POSIX_IO_CAT_MAPPING[func_name].value
    return IOCategory.OTHER.value


def io_columns():
    columns = {
        "fhash": "string[pyarrow]",
        "hhash": "string[pyarrow]",
        "image_id": "uint64[pyarrow]",
        "io_cat": "uint8[pyarrow]",
        "size": "uint64[pyarrow]",
        "offset": "uint64[pyarrow]",
    }
    return columns


def io_function(json_dict: dict):
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "args" in json_dict:
        if "fhash" in json_dict["args"]:
            d["fhash"] = str(json_dict["args"]["fhash"])
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


def is_pyarrow_dtype_supported() -> bool:
    return sys.version_info >= (3, 9)


def load_indexed_gzip_files(filename, start, end):
    index_file = f"{filename}.zindex"
    json_lines = zindex.zquery(
        filename,
        index_file=index_file,
        raw=f"select a.line from LineOffsets a where a.line >= {start} AND a.line <= {end};",
        debug=False,
        verbose=False,
    )
    logger.debug("Read json lines", filename=filename, start=start, end=end, num_lines=len(json_lines))
    return json_lines


def load_objects(
    line: str,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    final_dict = {}
    if line is not None and line != "" and len(line) > 0 and "[" != line[0] and "]" != line[0] and line != "\n":
        json_dict = {}
        try:
            unicode_line = "".join([i if ord(i) < 128 else "#" for i in line])
            json_dict = json.loads(unicode_line, strict=False)
            logger.debug("Loading dict", json_dict=json_dict)
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
                    final_dict["hhash"] = str(json_dict["args"]["hhash"])
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
                    json_dict["dur"] = int(json_dict["dur"])
                    json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                    final_dict["dur"] = json_dict["dur"]
                    final_dict["te"] = final_dict["ts"] + final_dict["dur"]
                    if not time_approximate:
                        final_dict["tinterval"] = I.to_string(
                            I.closed(json_dict["ts"], json_dict["ts"] + json_dict["dur"])
                        )
                final_dict.update(io_function(json_dict))
                final_dict.update(extra_columns_fn(json_dict) if extra_columns_fn else {})
            # check if all extra columns are present
            if extra_columns and not all(col in final_dict for col in extra_columns):
                missing_cols = [col for col in extra_columns if col not in final_dict]
                raise ValueError(f"Missing extra columns: {missing_cols}")
            logger.debug("Built a dictionary for line", final_dict=final_dict)
            yield final_dict
        except ValueError as error:
            logger.error("Processing line failed", line=line, error=error)
    return {}


class DFTracerAnalyzer(Analyzer):
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
            total_size = db.from_sequence(all_files).map(get_size).sum().compute()
            logger.info("Total size of all files", total_size=total_size)
        gz_bag = None
        pfw_bag = None
        if len(pfw_gz_pattern) > 0:
            with log_block("gzip_index_and_batches"):
                max_line_numbers = db.from_sequence(pfw_gz_pattern).map(get_linenumber).compute()
                logger.debug("Max lines per file", max_line_numbers=max_line_numbers)
                json_line_delayed = []
                total_lines = 0
                for filename, max_line in max_line_numbers:
                    total_lines += max_line
                    for _, start, end in generate_line_batches(filename, max_line):
                        json_line_delayed.append((filename, start, end))

                logger.info(
                    "Loading batches",
                    num_batches=len(json_line_delayed),
                    num_files=len(pfw_gz_pattern),
                    total_lines=total_lines,
                )
                json_line_bags = []
                for filename, start, end in json_line_delayed:
                    num_lines = end - start + 1
                    json_line_bags.append(dask.delayed(load_indexed_gzip_files, nout=num_lines)(filename, start, end))
                json_lines = db.concat(json_line_bags)
            with log_block("parse_gzip_json_lines"):
                gz_bag = (
                    json_lines.map(
                        load_objects,
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
                        load_objects,
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
            if is_pyarrow_dtype_supported():
                columns = {
                    "name": "string[pyarrow]",
                    "cat": "string[pyarrow]",
                    "type": "uint8[pyarrow]",
                    "pid": "uint64[pyarrow]",
                    "tid": "uint64[pyarrow]",
                    "ts": "uint64[pyarrow]",
                    "te": "uint64[pyarrow]",
                    "dur": "uint64[pyarrow]",
                    "tinterval": "uint64[pyarrow]",
                    "trange": "uint64[pyarrow]",
                    "level": "uint8[pyarrow]",
                }
                if self.time_approximate:
                    columns["tinterval"] = "string[pyarrow]"
            columns.update(io_columns())
            file_hash_columns = {
                "name": "string",
                "hash": "string",
                "pid": "Int64",
                "tid": "Int64",
                "hhash": "string",
            }
            hostname_hash_columns = {
                "name": "string",
                "hash": "string",
                "pid": "Int64",
                "tid": "Int64",
                "hhash": "string",
            }
            string_hash_columns = {
                "name": "string",
                "hash": "string",
                "pid": "Int64",
                "tid": "Int64",
                "hhash": "string",
            }
            other_metadata_columns = {
                "name": "string",
                "value": "string",
                "pid": "Int64",
                "tid": "Int64",
                "hhash": "string",
            }
            if is_pyarrow_dtype_supported():
                file_hash_columns = {
                    'name': "string[pyarrow]",
                    'hash': "string[pyarrow]",
                    'pid': "uint64[pyarrow]",
                    'tid': "uint64[pyarrow]",
                    'hhash': "string[pyarrow]",
                }
                hostname_hash_columns = {
                    'name': "string[pyarrow]",
                    'hash': "string[pyarrow]",
                    'pid': "uint64[pyarrow]",
                    'tid': "uint64[pyarrow]",
                    'hhash': "string[pyarrow]",
                }
                string_hash_columns = {
                    'name': "string[pyarrow]",
                    'hash': "string[pyarrow]",
                    'pid': "uint64[pyarrow]",
                    'tid': "uint64[pyarrow]",
                    'hhash': "string[pyarrow]",
                }
                other_metadata_columns = {
                    'name': "string[pyarrow]",
                    'value': "string[pyarrow]",
                    'pid': "uint64[pyarrow]",
                    'tid': "uint64[pyarrow]",
                    'hhash': "string[pyarrow]",
                }
            columns.update(file_hash_columns)
            columns.update(hostname_hash_columns)
            columns.update(string_hash_columns)
            columns.update(other_metadata_columns)
            columns.update(extra_columns or {})
            with log_block("to_dataframe+filter_events"):
                self.all_events = main_bag.to_dataframe(meta=columns)
                events = self.all_events.query("type == 0")
            self.file_hash = (
                self.all_events.query("type == 1")[list(file_hash_columns.keys())].groupby("hash").first().persist()
            )
            self.host_hash = (
                self.all_events.query("type == 2")[list(hostname_hash_columns.keys())]
                .groupby("hash")
                .first()
                .persist()
            )
            self.string_hash = (
                self.all_events.query("type == 3")[list(string_hash_columns.keys())].groupby("hash").first().persist()
            )
            self.metadata = self.all_events.query("type == 4")[list(other_metadata_columns.keys())].persist()
            with log_block("repartition+persist+indexing"):
                self.n_partition = math.ceil(total_size / (128 * 1024**2))
                logger.debug("Number of partitions used", n_partition=self.n_partition)
                self.events = events.repartition(npartitions=self.n_partition).persist()
                _ = wait(self.events)
                self.events["ts"] = self.events["ts"] - self.events["ts"].min()
                self.events["te"] = self.events["ts"] + self.events["dur"]
                self.events["trange"] = self.events["ts"] // (self.time_granularity * self.time_resolution)
                if is_pyarrow_dtype_supported():
                    self.events["ts"] = self.events["ts"].astype("uint64[pyarrow]")
                    self.events["te"] = self.events["te"].astype("uint64[pyarrow]")
                    self.events["trange"] = self.events["trange"].astype("uint16[pyarrow]")
                else:
                    self.events["ts"] = self.events["ts"].astype("Int64")
                    self.events["te"] = self.events["te"].astype("Int64")
                    self.events["trange"] = self.events["trange"].astype("Int16")
                self.events = self.events.persist()
                _ = wait(
                    [
                        self.file_hash,
                        self.host_hash,
                        self.string_hash,
                        self.metadata,
                        self.events,
                    ]
                )
        else:
            logger.error("Unable to load traces")
            exit(1)
        # ===============================================
        with log_block("normalize_durations"):
            self.events["dur"] = self.events["dur"] / self.time_resolution

        file_hashes = self.file_hash[["name"]].rename(columns={"name": COL_FILE_NAME})
        host_hhash_empty = self.host_hash["hhash"].isna().all().compute()
        if host_hhash_empty:
            host_hashes = self.host_hash[["name"]].rename(columns={"name": COL_HOST_NAME})
        else:
            host_hashes = self.host_hash.set_index("hhash")[["name"]].rename(columns={"name": COL_HOST_NAME})

        with log_block("join_hashes"):
            self.events = (
                self.events.merge(
                    file_hashes,
                    how="left",
                    left_on="fhash",
                    right_index=True,
                )
                .merge(
                    host_hashes,
                    how="left",
                    left_on="hhash",
                    right_index=True,
                )
                .drop(columns=["fhash", "hhash"])
            )

        with log_block("finalize+rename_columns+persist"):
            self.events = self.events.rename(columns=TRACE_COL_MAPPING).persist()
            _ = wait(self.events)
            return self.events

    def postread_trace(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        with log_block("filter_files"):
            traces = traces[
                traces[COL_FILE_NAME].isna() | ~traces[COL_FILE_NAME].str.contains("|".join(IGNORED_FILE_PATTERNS))
            ]

        # Ignore redundant function calls
        with log_block("filter_functions"):
            traces = traces[~traces[COL_FUNC_NAME].isin(IGNORED_FUNC_NAMES)]
            traces = traces[~traces[COL_FUNC_NAME].str.contains("|".join(IGNORED_FUNC_PATTERNS))]

        with log_block("wait"):
            _ = wait(traces)

        with log_block("set_basic_columns"):
            traces[COL_ACC_PAT] = 0
            traces[COL_COUNT] = 1

        # drop columns that are not needed
        # if COL_FILE_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_FILE_NAME], errors='ignore')
        # if COL_HOST_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_HOST_NAME], errors='ignore')

        # Set batches
        # traces['batch'] = traces.groupby(['func_name', 'step']).cumcount() + 1
        # batch_counts = traces['batch'].value_counts()
        # last_valid_batch = batch_counts[batch_counts > 1].index.max()
        # traces['batch'] = traces['batch'].mask(
        #     traces['batch'] > last_valid_batch, pd.NA
        # )

        # pytorch reads images instead of batches
        # e.g. 4 workers = 0..4 images = who starts/finishes first

        # epoch and step make sense in dlio layer

        # to put step back, target variable = previous compute + my io

        # Set steps depending on time ranges
        # step_time_ranges = traces.groupby(['pid', 'epoch', 'step']).agg({'ts': min, 'te': max})
        # traces = traces.map_partitions(
        #     self._set_steps, step_time_ranges=step_time_ranges.reset_index()
        # )

        return (
            traces.map_partitions(self._set_proc_names)
            .map_partitions(self._fix_file_posix_category)
            .map_partitions(self._sanitize_size_offset)
        )

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_unique_process_count(self, traces: dd.DataFrame):
        return traces["pid"].nunique()

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epochs: pd.DataFrame):
        return df.assign(epoch=np.digitize(df["time_range"], bins=epochs["time_range"], right=False))

    @staticmethod
    def _fix_file_posix_category(df: pd.DataFrame):
        base_condition = (df["cat"].str.contains("posix|stdio") & ~df["file_name"].isna())
    
        # Step 1: Map file purpose suffixes first
        purpose_updates = {
            "/data": "_reader",
            "/checkpoint": "_checkpoint"
        }
        
        for path, suffix in purpose_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix
        
        # Step 2: Map filesystem suffixes
        filesystem_updates = {
            "/lustre": "_lustre",
            "/ssd": "_ssd"
        }
        
        for path, suffix in filesystem_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        return df

    @staticmethod
    def _sanitize_size_offset(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, np.nan)
        if "offset" in df.columns:
            df["offset"] = df["offset"].replace(0, np.nan)
        return df

    @staticmethod
    def _set_proc_names(df: pd.DataFrame):
        df[COL_PROC_NAME] = (
                "app#"
                + df[COL_HOST_NAME].astype(str)
                + "#"
                + df["pid"].astype(str)
                + "#"
                + df["tid"].astype(str)
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
