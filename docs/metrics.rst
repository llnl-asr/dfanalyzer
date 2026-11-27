.. _metrics:

Metrics Reference
=================

DFAnalyzer produces flat views, which are dftracer-utils DataFrames where each row represents a group (e.g., a process, a time range, a file) and each column is a metric. This document explains how to read column names, what each metric family measures, and how to use the two-track statistics system for diagnosing parallel I/O issues.

.. contents::
   :local:
   :depth: 2

Column Naming Convention
------------------------

Every flat view column follows a structured naming pattern::

   {layer}_{metric_family}_{statistic}

For example::

   posix_read_time_proc_mean
   │     │    │    │    └── statistic: mean across processes
   │     │    │    └─────── track: per-process (proc)
   │     │    └──────────── base metric: time (seconds)
   │     └───────────────── derived metric: read operations
   └─────────────────────── layer: posix

**Layer** is the execution layer (e.g., ``posix``, ``reader``, ``epoch``, ``compute``). Layers are defined by the preset configuration and form a hierarchy.

**Metric family** is the optional derived metric prefix (e.g., ``read_``, ``write_``, ``data_``, ``metadata_``, ``open_``, ``close_``, ``seek_``, ``stat_``, ``sync_``). When absent, the metric covers all operations in that layer.

**Base metric** is one of: ``time`` (seconds), ``count`` (number of operations), ``size`` (bytes).

**Track and statistic** identify which distribution the value describes and the aggregation function. See :ref:`two-track-statistics` for details.


.. _two-track-statistics:

Two-Track Statistics: Proc vs Call
----------------------------------

DFAnalyzer computes two parallel sets of statistics for ``time`` and ``size`` metrics, each answering a different question about the data.

Per-Process Statistics (``proc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question**: How are process-level totals distributed across processes?

The ``_proc_`` statistics describe the distribution of **per-process aggregate values**. When creating a view (e.g., grouped by ``time_range``), DFAnalyzer first computes each process's total for that group, then takes the min, max, mean, and standard deviation across processes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column suffix
     - Meaning
   * - ``time_proc_min``
     - Minimum per-process total time across all processes in this group
   * - ``time_proc_max``
     - Maximum per-process total time (the **bottleneck** process)
   * - ``time_proc_mean``
     - Average per-process total time
   * - ``time_proc_std``
     - Standard deviation of per-process totals (measures **inter-process imbalance**)

Per-Call Statistics (``call``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question**: How are individual call durations/sizes distributed?

The ``_call_`` statistics describe the distribution of **individual trace events** (calls) within each group. These are computed from sum-of-squares support columns, allowing exact aggregation across processes without storing every individual value.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column suffix
     - Meaning
   * - ``time_call_min``
     - Minimum individual call duration across all calls in this group
   * - ``time_call_max``
     - Maximum individual call duration (the **slowest single call**)
   * - ``time_call_mean``
     - Average individual call duration
   * - ``time_call_std``
     - Standard deviation of individual call durations (measures **tail latency**)

Why Two Tracks?
~~~~~~~~~~~~~~~

Consider an 8-process job where each process makes 1,000 I/O calls:

- ``time_proc_mean = 0.33s``, ``time_proc_std = 0.25s`` — process totals vary moderately. Some processes do more I/O than others.
- ``time_call_mean = 0.012s``, ``time_call_std = 0.051s`` — individual calls average 12ms, but with a standard deviation of 51ms. The coefficient of variation (std/mean ≈ 4x) indicates heavy-tailed call latency.

The process-level view tells you about **load balance**. The call-level view tells you about **individual call behavior**. A system can be well-balanced across processes yet have problematic tail latency on individual calls — or vice versa. You need both to diagnose effectively.

.. note::

   **Why no proc/call split for** ``count``? Each raw trace event has ``count=1``, so ``count_call_std`` would always be 0 — every "call" has the same count. The ``count`` metric only has aggregate statistics (``count_sum``, ``count_proc_min``, ``count_proc_max``, etc.).


Aggregate Metrics
-----------------

These are the raw aggregated values before any derived ratios.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column suffix
     - Meaning
   * - ``time_sum``
     - Total time across all processes and calls in the group
   * - ``count_sum``
     - Total number of operations
   * - ``size_sum``
     - Total bytes transferred


Derived Performance Metrics
---------------------------

Computed from the base metrics above. These are always per-process statistics (derived from the aggregated per-process values).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column suffix
     - Meaning
   * - ``bw_sum``
     - Bandwidth (``size_sum / time_sum``), bytes/second
   * - ``bw_proc_{min,max,mean,std}``
     - Per-process bandwidth distribution
   * - ``ops_sum``
     - Operations per second (``count_sum / time_sum``)
   * - ``ops_proc_{min,max,mean,std}``
     - Per-process ops/sec distribution
   * - ``intensity_sum``
     - I/O intensity (``count_sum / size_sum``), operations/byte
   * - ``intensity_proc_{min,max,mean,std}``
     - Per-process intensity distribution

.. note::

   ``bw``, ``ops``, and ``intensity`` are only meaningful when their denominators are non-zero. Columns are set to ``NA`` when the denominator is zero or when size is not applicable (e.g., non-POSIX layers).


Fractional Metrics
------------------

Fractional metrics express a group's share of a total. These are where the proc/call distinction becomes most actionable.

``frac_total`` — Share of Column Total
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column suffix
     - Meaning
   * - ``count_frac_total``
     - This row's ``count_sum`` as a fraction of the column total
   * - ``time_proc_frac_total``
     - This row's share of **bottleneck process time** across all rows
   * - ``time_call_frac_total``
     - This row's share of **aggregate time** across all rows
   * - ``size_proc_frac_total``
     - This row's share of **bottleneck process data volume** across all rows
   * - ``size_call_frac_total``
     - This row's share of **aggregate data volume** across all rows

**How the tracks differ for non-process views:**

In a ``proc_name`` view (process-based), both tracks use ``time_sum`` and produce identical values. In a ``time_range`` or ``file_name`` view (non-process-based):

- ``proc_frac_total`` uses ``time_proc_max`` (the bottleneck process's value per group)
- ``call_frac_total`` uses ``time_sum`` (the aggregate across all processes per group)

**Interpreting divergence:**

If ``time_proc_frac_total`` is much higher than ``time_call_frac_total`` for a particular time range, the bottleneck process is disproportionately loaded during that period — a straggler signal. If the values are similar everywhere, the workload is well-balanced.

``frac_boundary`` — Share of Top-Layer Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These express a layer's time relative to the top layer (the "boundary" layer, typically ``app`` or ``epoch``).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column suffix
     - Meaning
   * - ``time_proc_frac_{boundary}``
     - Layer's bottleneck time / boundary layer's bottleneck time
   * - ``time_call_frac_{boundary}``
     - Layer's aggregate time / boundary layer's aggregate time

For example, ``posix_time_proc_frac_epoch = 0.8`` means the bottleneck process spends 80% of its epoch time in POSIX I/O.

**Interpreting divergence:**

If ``posix_time_proc_frac_epoch = 0.8`` but ``posix_time_call_frac_epoch = 0.4``, the straggler is spending twice as much of its time in POSIX I/O compared to the fleet average — a strong signal that the straggler's bottleneck is I/O-specific.

``frac_parent`` — Share of Parent Layer Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These express a child layer's time relative to its parent in the layer hierarchy.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column suffix
     - Meaning
   * - ``time_proc_frac_parent``
     - Child's bottleneck time / parent's bottleneck time
   * - ``time_call_frac_parent``
     - Child's aggregate time / parent's aggregate time

``ops_slope`` and ``ops_percentile``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column suffix
     - Meaning
   * - ``ops_slope``
     - Ratio of ``time_proc_frac_total / count_frac_total``. Values > 1 mean this group uses more than its share of time relative to its share of operations — indicating slower-than-average operations.
   * - ``ops_percentile``
     - Percentile rank of ``ops_slope`` within the view. High percentiles indicate groups with disproportionately slow operations.


Cross-Layer Metrics
-------------------

Cross-layer metrics quantify relationships between layers in the hierarchy. They are computed by ``set_cross_layer_metrics`` and produce both proc and call variants.

Overhead Metrics (``o_``)
~~~~~~~~~~~~~~~~~~~~~~~~~

Overhead is the time a parent layer spends in its own logic, excluding time attributed to child layers.

::

   o_{layer}_time = max(layer_time - sum(child_times), 0)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column
     - Meaning
   * - ``o_{layer}_{time_metric}``
     - Overhead time value (base column). For proc track, ``time_metric`` is ``time_proc_max`` (non-process views) or ``time_sum`` (process views). For call track, always ``time_sum``.
   * - ``o_{layer}_time_proc_frac_self``
     - Overhead as fraction of the layer's own bottleneck time
   * - ``o_{layer}_time_call_frac_self``
     - Overhead as fraction of the layer's own aggregate time
   * - ``o_{layer}_time_proc_frac_{boundary}``
     - Overhead as fraction of top-layer bottleneck time
   * - ``o_{layer}_time_call_frac_{boundary}``
     - Overhead as fraction of top-layer aggregate time
   * - ``o_{layer}_time_proc_frac_total``
     - This row's share of total overhead across all rows (proc track)
   * - ``o_{layer}_time_call_frac_total``
     - This row's share of total overhead across all rows (call track)

**Interpreting divergence:**

If ``o_epoch_time_proc_frac_self`` is high but ``o_epoch_time_call_frac_self`` is low, the bottleneck process has more overhead than typical. This points to a process-specific issue (e.g., the straggler is doing extra work in the epoch's own logic rather than in its child layers).

Unoverlapped Metrics (``u_``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unoverlapped time measures how much of an async layer's time does **not** overlap with compute. This quantifies I/O time that is actually stalling the application (cannot be hidden behind computation).

::

   u_{layer}_time = max(layer_time - compute_time, 0)

These metrics are only produced for layers marked as ``async_layers`` in the preset configuration, and only when a ``compute`` layer exists.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column
     - Meaning
   * - ``u_{layer}_{time_metric}``
     - Unoverlapped time value (base column)
   * - ``u_{layer}_time_proc_frac_self``
     - Unoverlapped time as fraction of layer's own bottleneck time
   * - ``u_{layer}_time_call_frac_self``
     - Unoverlapped time as fraction of layer's own aggregate time
   * - ``u_{layer}_time_proc_frac_{boundary}``
     - Unoverlapped time as fraction of top-layer bottleneck time
   * - ``u_{layer}_time_call_frac_{boundary}``
     - Unoverlapped time as fraction of top-layer aggregate time
   * - ``u_{layer}_time_proc_frac_parent``
     - Unoverlapped time as fraction of parent layer's bottleneck time
   * - ``u_{layer}_time_call_frac_parent``
     - Unoverlapped time as fraction of parent layer's aggregate time
   * - ``u_{layer}_time_proc_frac_total``
     - This row's share of total unoverlapped time (proc track)
   * - ``u_{layer}_time_call_frac_total``
     - This row's share of total unoverlapped time (call track)

**Interpreting divergence:**

If ``u_posix_time_proc_frac_self = 0.9`` but ``u_posix_time_call_frac_self = 0.5``, the bottleneck process has 90% unoverlapped I/O while the average is 50%. The straggler is not benefiting from compute overlap — possibly because its I/O is serialized or its compute phase is shorter.


Other Metric Families
---------------------

Size Bins
~~~~~~~~~

Columns matching ``size_bin_{range}_sum`` count how many operations fall into each transfer size bucket. These are useful for characterizing the I/O size distribution.

The bins are::

   0-4KiB, 4-16KiB, 16-64KiB, 64-256KiB, 256KiB-1MiB,
   1-4MiB, 4-16MiB, 16-64MiB, 64-256MiB, 256MiB-1GiB,
   1-4GiB, 4GiB+

Unique Counts
~~~~~~~~~~~~~

Columns ending in ``_nunique`` count distinct values within a group:

- ``file_name_nunique`` — number of distinct files accessed
- ``proc_name_nunique`` — number of distinct processes (in non-process views)
- ``time_range_nunique`` — number of distinct time ranges spanned


Interpretation Guide
--------------------

Diagnosing Load Imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~

Compare ``time_proc_std`` against ``time_proc_mean``. A high coefficient of variation (std/mean > 0.5) suggests significant inter-process imbalance.

In non-process views, compare ``time_proc_frac_total`` vs ``time_call_frac_total``:

- **Proc > Call**: The bottleneck process is disproportionately loaded here — straggler behavior
- **Proc ≈ Call**: Workload is evenly distributed
- **Proc < Call**: Many processes contribute aggregate time, but no single one dominates

Diagnosing Tail Latency
~~~~~~~~~~~~~~~~~~~~~~~~

Compare ``time_call_std`` against ``time_call_mean``. A coefficient of variation > 2x indicates heavy-tailed call durations — some individual I/O calls are much slower than average. This is common with:

- Shared file contention
- Metadata-heavy workloads
- Network filesystem latency spikes

Diagnosing Compute/I/O Overlap Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look at ``u_{layer}_time_proc_frac_self`` (or the call variant):

- **Near 1.0**: Almost no overlap with compute — I/O is fully stalling the application
- **Near 0.0**: I/O is well-hidden behind compute
- **Proc much higher than call**: The bottleneck process specifically fails to overlap I/O with compute

Diagnosing Layer-Specific Bottlenecks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``time_proc_frac_{boundary}`` to identify which layers consume the most time:

1. Start with the top-level boundary fractions to find the dominant layer
2. Use ``time_proc_frac_parent`` to drill down into child layers
3. Use ``o_{layer}_time_proc_frac_self`` to see how much time the parent spends in its own logic vs delegating to children

Compare proc vs call variants at each level to determine whether the bottleneck is straggler-specific or systemic.

Process-Based vs Non-Process Views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proc/call distinction only produces different values in **non-process-based views** (``time_range``, ``file_name``). In process-based views (``proc_name``), both tracks use ``time_sum`` and produce identical values. This is by design: when the view is already per-process, "bottleneck process" and "aggregate" are the same concept.
