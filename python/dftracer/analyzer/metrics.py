def _find_metric_pairs(metrics: list[str], metric_type1: str, metric_type2: str):
    """
    Find pairs of metrics with a common prefix, one ending with metric_type1 and one with metric_type2.
    Example:
        metrics = ["foo_count_per", "foo_time_per", "bar_count_per", "bar_time_per"]
        _find_metric_pairs(metrics, "count_per", "time_per")
        -> [("foo_count_per", "foo_time_per"), ("bar_count_per", "bar_time_per")]
    """
    map1 = {
        metric_name[: -len(metric_type1)]: metric_name for metric_name in metrics if metric_name.endswith(metric_type1)
    }
    map2 = {
        metric_name[: -len(metric_type2)]: metric_name for metric_name in metrics if metric_name.endswith(metric_type2)
    }

    common_prefixes = set(map1.keys()).intersection(map2.keys())
    return [(map1[prefix], map2[prefix]) for prefix in sorted(common_prefixes)]


