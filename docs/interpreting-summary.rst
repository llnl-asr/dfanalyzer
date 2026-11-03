.. _interpreting-summary:

Interpreting the Output Summary
===============================

DFAnalyzer prints a Summary table and a Layer Breakdown for each requested view.
Use the Summary to check the scope of the recorded workload, then use the
breakdown to understand where time and I/O activity occur. This guide covers
``analyzer/preset=posix`` and ``analyzer/preset=ai``.
See :doc:`getting-started` for running the analyzer and :doc:`metrics` for the
underlying DataFrame columns.

.. contents::
   :local:
   :depth: 2

Choose the Preset and Read the View Title
-----------------------------------------

You can select either preset explicitly:

.. code-block:: bash

   dfanalyzer trace_path=/path/to/traces analyzer/preset=posix
   dfanalyzer trace_path=/path/to/traces analyzer/preset=ai

The POSIX preset selects events whose categories contain ``posix`` or ``stdio``.
It reports a single ``POSIX - All`` layer. The AI preset adds application,
training, compute, data loading, and checkpoint layers when matching
instrumentation is present. Selecting AI does not create missing application
events from POSIX calls.

The title, such as ``Time Period Summary``, identifies the view used to calculate
the layer values. The default time view groups activity into time ranges.
Process and file views group the same workload differently. Global rows such as
``Job Time`` repeat across views; layer time and rates can change with grouping.

Read the Summary Rows
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Row
     - Interpretation
   * - Job Time
     - Latest trace end minus earliest trace start, in seconds. This is the
       observed trace span, including gaps between events; it need not cover the
       entire application run.
   * - Trace Count
     - Number of loaded trace event rows. This includes application events when
       present, so it is not necessarily a count of POSIX operations.
   * - Profile Count
     - Number of operations represented by loaded aggregate profiles, using
       their recorded counts. Zero means no profile operations were included.
   * - Total Count
     - Trace Count plus Profile Count.
   * - Total Files
     - Number of distinct non-null file names in the loaded traces. This counts
       recorded names, not opens or the dataset's total files on disk.
   * - Total Nodes
     - Number of distinct recorded host names in the traces.
   * - Total Processes
     - Number of distinct ``proc_name`` values in the traces. DFTracer normally
       constructs these from host, process ID, and thread ID, so this can count
       multiple threads within one OS process.
   * - <Layer> Count
     - Sum of operation counts for that layer in this view, including applicable
       profile contributions. With ordinary trace events, each event contributes
       one operation.
   * - <Layer> Size
     - Sum of recorded bytes for that POSIX layer, displayed in binary megabytes.
   * - <Layer> Bandwidth
     - Layer Size divided by the layer time used in the breakdown below.
   * - <Layer> Avg Transfer Size
     - Layer Size divided by Layer Count. The denominator includes all operations
       in the layer, including metadata calls, rather than only reads and writes.

The global rows come from loaded input statistics, while the per-layer rows come
from the selected view. Filtering and layer selection can therefore make the
layer counts smaller than Total Count. AI layers also share events, so adding
their counts does not reproduce Total Count.

Layers with zero operations are omitted. Size, Bandwidth, and Avg Transfer Size
rows appear only when the layer has a positive recorded size. Missing size rows
do not imply that the layer had no operations.

.. note::

   The console labels sizes as ``MB`` and rates as ``MB/s``, but divides bytes by
   1,048,576 (2**20). Read these as MiB and MiB/s when comparing with other tools.
   All displayed times are seconds.

Understand the Layer Breakdown
-------------------------------

The breakdown puts each layer's time, operations, and bytes on one row:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Interpretation
   * - Time (s)
     - For a process view: the maximum per-process total duration. For a
       non-process view, including time and file views: the sum of each group's
       maximum per-process duration (``time_proc_max``).
   * - Ops
     - Total operation count across all groups and processes in the layer.
   * - Ops/sec
     - Ops divided by the displayed layer time.
   * - Size (MB)
     - Total recorded bytes across all groups and processes in the POSIX layer.
   * - Bandwidth (MB/s)
     - Size divided by the displayed layer time, using the same binary units as
       the Summary.

Layer time is accumulated event duration, not the union of occupied wall-clock
intervals. Nested or concurrent calls can make it exceed Job Time. In a time
view, the process contributing the maximum can also change between ranges.
Consequently, the reported time is not a measured application critical path.
Changing ``analyzer.time_granularity`` can change the maxima and resulting rates.

Ops and Size aggregate activity across processes, but Time uses the process
maxima described above. Treat the resulting rates as summary indicators for
this grouping. To calculate bytes per second of observed job elapsed time,
divide Size by Job Time separately.

POSIX: Read the I/O Totals
--------------------------

With ``posix``, begin with ``POSIX - All Count``, Size, and Avg Transfer Size to
characterize the recorded I/O. ``POSIX - All`` includes both data transfers and
metadata operations; it does not mean reads alone. Repeated reads and writes
contribute repeatedly to Size, so Size is not the unique dataset size or a
measurement of physical storage traffic below caches.

For example, suppose a process-view summary has Job Time = 20 seconds and the
breakdown shows POSIX Time = 4 seconds, Ops = 1,000, and Size = 200 MB. Then:

* Bandwidth is 200 / 4 = 50 MB/s.
* Ops/sec is 1,000 / 4 = 250.
* Avg Transfer Size is 200 / 1,000 = 0.200 MB, including metadata calls in the
  denominator.
* Bytes per job elapsed second would be 200 / 20 = 10 MB/s.

These are illustrative values, not benchmark targets. A low average size can
reflect small transfers, many metadata operations, or both. Use the read, write,
and metadata metric families in :doc:`metrics` to distinguish them. The POSIX
preset alone cannot tell how I/O relates to training or compute phases.

AI: Read the Application Context
--------------------------------

With ``ai``, read the application layers before interpreting the POSIX rates:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layer
     - What its events represent
   * - App, Training, Epoch
     - Instrumented application root, training, and epoch scopes.
   * - Compute
     - Events categorized as compute; their durations are recorded scopes, not
       a GPU utilization percentage.
   * - Fetch Data
     - ``fetch.iter`` events, representing the instrumented data-fetch scope.
   * - Checkpoint
     - Checkpoint scope, which can include more than its underlying POSIX calls.
   * - Communication, Device
     - Events categorized as communication or device activity, when present.
   * - DLIO Data Loader, DLIO Data Loader Fork, Reader
     - Data-loader categories, POSIX fork events, and reader/preprocessing scopes.
   * - POSIX - All
     - All matching POSIX/stdio events, including those in the subsets below.
   * - POSIX - Reader, POSIX - Checkpoint
     - POSIX/stdio events whose categories also contain ``_reader`` or
       ``_checkpoint``. Attribution depends on these recorded category tags.

For example, checkpoint scope time can include serialization as well as writes.
Reader scope time can include preprocessing as well as reads. Compare each scope
with its POSIX subset to decide what to investigate next, without treating the
times as disjoint pieces of Job Time. Likewise, do not add ``POSIX - Reader``
Size to ``POSIX - All`` Size: those reader bytes are already included in All.

Understand the Parenthesized Overlap Percentages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple layers are present, the breakdown title becomes
``Layer Breakdown (w/ overlap %)``. A cell such as ``23.415 (45%)`` contains
the original value followed by an estimated overlap percentage.

For configured asynchronous layers with compute metrics available, DFAnalyzer
first calculates unoverlapped time within each group:

.. code-block:: text

   unoverlapped_time = max(layer_time - compute_time, 0)
   displayed_overlap = max(0, 1 - summary_unoverlapped_time / summary_layer_time)

Summary unoverlapped time uses the same maximum-or-sum rule as layer time.
The percentage compares grouped durations with compute; it is not the layer's
share of its parent and is not calculated from exact intersections of event
intervals. A high value suggests that compute may hide asynchronous work, but
does not establish that the work was fully hidden in execution.

The Ops and Size percentages use a different calculation. They report the
fraction of operations or bytes belonging to groups where computed unoverlapped
time is zero. They do not measure the exact fraction of individual calls or
bytes that overlapped compute. Their values depend on the view and grouping.

``(----)`` means the percentage is unavailable or the displayed total is zero;
it does not mean zero overlap. A dash in a value cell can mean zero or an
inapplicable quantity, such as Size for a compute layer. Rates may display
``nan`` when their denominator is zero or their inputs are unavailable.

Use the Summary to Decide Where to Look Next
--------------------------------------------

* Check Job Time and the input counts first to confirm the expected trace scope.
* For POSIX workloads, inspect operation mix and transfer sizes before judging
  bandwidth. Compare runs with the same preset, view, and time granularity.
* For AI workloads, compare Fetch Data and Checkpoint scopes with their POSIX
  subsets, then inspect compute and estimated overlap for possible exposed I/O.
* Follow up in time, process, or file views to locate expensive groups. Use
  per-process statistics for imbalance and per-call statistics for latency
  variation, as described in :doc:`metrics`.
