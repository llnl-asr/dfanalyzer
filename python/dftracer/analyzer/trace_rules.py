"""The DFTracer domain rules: ignored files and functions, and POSIX category suffixes."""

from __future__ import annotations

from dataclasses import dataclass

from dftracer.utils import DataFrame, Series

from .constants import COL_FILE_NAME, COL_FUNC_NAME


def _contains_any(values: Series, patterns: tuple[str, ...]) -> Series:
    """True where `values` contains one of the plain-text `patterns`; a null value never matches."""
    found = values.str.contains(patterns[0], regex=False, na=False)
    for pattern in patterns[1:]:
        found = found | values.str.contains(pattern, regex=False, na=False)
    return found


def drop_matching_files(df: DataFrame, patterns: tuple[str, ...], column: str = COL_FILE_NAME) -> DataFrame:
    """Drop rows whose `column` contains one of the plain-text `patterns`.

    Rows without a file are kept. A file folded to a bucket name matches the
    same way as its full path, so a pattern passed as a bucket still drops it.
    """
    if not patterns or column not in df.columns or df.num_rows == 0:
        return df
    return df.filter(~_contains_any(df[column].astype("string"), patterns))


@dataclass(frozen=True)
class TraceRules:
    """Rule lists and the frame transforms that apply them to read frames.

    Patterns are plain text matched anywhere in the value, except
    `ignored_func_names`, which match exactly. Every view and the file count
    match them the same way. `posix_cat_rules` apply in order
    and add up, so a path matching two rules gets both suffixes.
    """

    posix_cat_rules: tuple[tuple[str, str], ...] = ()
    ignored_file_patterns: tuple[str, ...] = ()
    ignored_func_names: tuple[str, ...] = ()
    ignored_func_patterns: tuple[str, ...] = ()

    def drop_ignored(self, df: DataFrame) -> DataFrame:
        """Drop rows of ignored files and functions. Rows without a file or function are kept."""
        df = drop_matching_files(df, self.ignored_file_patterns)
        if COL_FUNC_NAME in df.columns:
            if self.ignored_func_names:
                df = df.filter(~df[COL_FUNC_NAME].isin(list(self.ignored_func_names)))
            if self.ignored_func_patterns:
                df = df.filter(~_contains_any(df[COL_FUNC_NAME], self.ignored_func_patterns))
        return df

    def suffix_posix_cat(self, df: DataFrame) -> DataFrame:
        """Add the rule suffixes to `cat` of posix and stdio rows with a file. Changes `df` in place."""
        if self.posix_cat_rules:
            df["cat"] = apply_cat_suffix(df["cat"], df[COL_FILE_NAME], self.posix_cat_rules)
        return df


def apply_cat_suffix(cat: Series, path: Series, rules: tuple[tuple[str, str], ...]) -> Series:
    """`cat` with the suffix of every rule whose substring is in `path`, on posix and stdio rows with a path."""
    if not rules:
        return cat
    base = cat.str.contains("posix|stdio", na=False) & path.notna()
    out = cat
    p = path.fillna("")
    for sub, suffix in rules:
        out = out.mask(base & p.str.contains(sub, regex=False, na=False), out.str.cat(suffix))
    return out


DEFAULT_TRACE_RULES = TraceRules(
    posix_cat_rules=(
        ("/data", "_reader"),
        ("/checkpoint", "_checkpoint"),
        ("/lustre", "_lustre"),
        ("/ssd", "_ssd"),
    ),
    ignored_file_patterns=(
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
    ),
    ignored_func_names=(
        "DLIOBenchmark.__init__",
        "DLIOBenchmark.initialize",
        "FileStorage.__init__",
        "TorchDataset.__init__",
    ),
    ignored_func_patterns=(
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
    ),
)
