.. _configuration:

Configuration
=============

This reference guide documents all the configuration parameters available in
DFAnalyzer. These parameters control the behavior of analyzers, outputs, and
cluster configurations.

Using Configuration Parameters
------------------------------

DFAnalyzer uses `Hydra <https://hydra.cc>`_ for configuration management, which
provides a flexible way to organize and override parameters. You can specify
parameters in several ways:

1. **Command line overrides**:

   .. code-block:: bash

      dfanalyzer trace_path=path/to/traces analyzer/preset=ai

2. **Group selection**:

   .. code-block:: bash

      dfanalyzer analyzer=dftracer cluster=slurm output=csv

3. **Nested parameters**:

   .. code-block:: bash

      dfanalyzer output.compact=true cluster.n_workers=4

Core Parameters
---------------

.. list-table::
   :widths: 25 15 20 40
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``trace_path``
     - string
     - Required
     - Path to the I/O trace data for analysis.
   * - ``view_types``
     - list[str]
     - ``["time_range"]``
     - A list of perspectives to analyze the data from.
   * - ``debug``
     - bool
     - ``false``
     - Enable debug mode.
       Logs everything to the log file and sets ``log_level`` to ``debug``.
   * - ``log_level``
     - string
     - ``warn``
     - Level of the dftracer-utils logger on the terminal.
       Use ``info`` or ``debug`` to see the index and scan steps when something goes wrong.
       The dfanalyzer log file in the run directory always has the full detail.
   * - ``verbose``
     - bool
     - ``false``
     - Enable verbose information display.

Analyzer Configuration
----------------------

DFAnalyzer supports multiple analyzers for different trace formats. You can
select an analyzer using the ``analyzer=<type>`` parameter. The default analyzer
is **dftracer**.

Common Analyzer Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 30 30
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``analyzer.checkpoint``
     - bool
     - ``true``
     - Enable checkpointing of analysis state.
   * - ``analyzer.checkpoint_dir``
     - string
     - ``${hydra:runtime.output_dir}/checkpoints``
     - Directory for saving checkpoints.
   * - ``analyzer.time_approximate``
     - bool
     - ``true``
     - Use approximate time for analysis.
   * - ``analyzer.preset``
     - group
     - ``posix``
     - Select a preset configuration for the analyzer.
   * - ``analyzer.time_granularity``
     - float
     - Varies
     - Time granularity for analysis (in seconds). Defaults vary by analyzer.
   * - ``analyzer.time_resolution``
     - float
     - Varies
     - Time resolution for the analyzer (in nanoseconds). Defaults vary by analyzer.
   * - ``analyzer.view_materialize_max_bytes``
     - int
     - 268435456
     - Layers whose main view is smaller than this are computed in this process rather than on Dask, which avoids building a task graph for frames that are often only a few dozen rows. Output is identical either way, so a wrong value costs speed, never correctness. Set to ``0`` to keep everything on Dask.

Analyzer Presets (``analyzer/preset``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Presets provide pre-configured settings for different analysis scenarios,
making it easier to analyze common workloads without manual configuration.
They define how raw trace data is structured into layers and what metrics
are derived.

When you do not set ``analyzer/preset``, DFAnalyzer picks one from the ``cat`` values in the trace.
It reads them from the trace index, so it does not scan the events.
The choice is written to the log file as ``Detected analyzer preset``.
A preset that you set always wins.

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Preset
     - For
     - Picked when the trace has
   * - **ai**
     - AI training workloads instrumented with DFTracer's AI logging helpers.
       Its layers key off the ``ai_root`` root event and the ``pipeline`` category.
     - ``ai_root`` or ``pipeline``
   * - **dlio**
     - Workloads from the `DLIO benchmark <https://github.com/argonne-lcf/dlio_benchmark>`_.
       Its layers key off ``DLIOBenchmark.*`` function names.
     - ``reader``, ``data_loader``, ``checkpoint``, ``ai_framework`` or ``dlio_benchmark``
   * - **posix**
     - General POSIX I/O workloads.
     - ``POSIX`` or ``STDIO``
   * - **auto**
     - Any other trace.
       It adds an ``app`` boundary layer and one layer for each distinct ``cat``.
     - none of the above

The first row that matches wins.
Set a preset yourself like this:

.. code-block:: bash

   dfanalyzer analyzer/preset=ai

DFTracer Analyzer (``analyzer=dftracer``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For analyzing DFTracer trace files. This is the **default** analyzer.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``analyzer.time_granularity``
     - float
     - 1
     - Time granularity for DFTracer (in seconds).
   * - ``analyzer.time_resolution``
     - float
     - 1e6
     - Time resolution for DFTracer (in nanoseconds).

Output Configuration
--------------------

Control how analysis results are presented and stored. You can select an output
format using ``output=<type>``.

Common Output Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``output.compact``
     - bool
     - ``false``
     - Use compact output format.
   * - ``output.name``
     - string
     - ""
     - Custom name for the output.
   * - ``output.root_only``
     - bool
     - ``true``
     - Only show output on the root process.

Console Output (``output=console``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prints the analysis summary directly to the console. This is the **default** output.

JSON Output (``output=json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saves a JSON summary report to disk. This output includes raw stats, per-view
summary metrics, per-layer metrics, and additional metrics statistics.

.. list-table::
   :widths: 25 15 30 30
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``output.file_path``
     - string
     - ""
     - JSON output file path. If empty, writes to ``<hydra.runtime.output_dir>/dfanalyzer_output.json``.

CSV Output (``output=csv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saves the analysis results to a set of CSV files in the output directory.

SQLite Output (``output=sqlite``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saves the analysis results to a SQLite database file.

.. list-table::
   :widths: 25 15 30 30
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``output.run_db_path``
     - string
     - ""
     - Path to the SQLite database file. If empty, a new file is created.

Cluster Configuration
---------------------

Configure how DFAnalyzer executes: on a Dask cluster, or in the calling process.
Select a cluster type with ``cluster=<type>``.

Automatic (``cluster=auto``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **default**. Analyses the trace in the calling process when it is small
enough, and starts a local cluster when it is not.

The trace is scanned here first and the aggregated result measured against
``cluster.max_bytes``. Under the budget, that scan is used as-is and no cluster
is ever built, which avoids roughly 2.2 seconds of setup and teardown. Over it,
the scan is discarded, a ``LocalCluster`` is started, and the read is redone
across its workers. Results are identical either way.

The budget is compared against the *aggregated* frames (columns and index), not the size of the
traces on disk. The C++ indexer aggregates before Python sees anything, and how
far it collapses the trace depends on time buckets, processes and categories
rather than bytes — across the bundled fixtures the ratio between the two varies
about twentyfold, so no threshold on file size predicts it.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``cluster.max_bytes``
     - int
     - 268435456
     - Bytes of the aggregated frames that may be analysed in process. Above this, a local cluster is started.
   * - ``cluster.n_workers``
     - int
     - null
     - Workers to start *if* a cluster turns out to be needed.
   * - ``cluster.memory_limit``
     - int
     - null
     - Memory limit per worker, if a cluster is started.

.. note::

   The budget bounds what is *processed* in this process, not what is read into
   it. An oversized trace is aggregated here once before being handed to the
   cluster, because checking part-way through the scan would mean scanning each
   file group separately — several times slower, since each group is a full
   scan of the index. Traces whose files are more than fifty times the budget
   skip the attempt entirely. If you already know a trace is large, naming
   ``cluster=local`` or a job-queue cluster avoids the wasted scan.

Local Cluster (``cluster=local``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always runs a Dask cluster on the local machine, whatever the trace size.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``cluster.n_workers``
     - int
     - null
     - Number of Dask workers. Defaults to the number of CPU cores.
   * - ``cluster.memory_limit``
     - int
     - null
     - Memory limit per worker (e.g., "8GB").

In-Process (``cluster=none``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs the analysis in the calling process, with no scheduler and no workers,
unconditionally — unlike ``cluster=auto``, it never falls back to a cluster.

Layers small enough to materialise are already computed in this process rather than on
Dask (see ``analyzer.view_materialize_max_bytes``), so for a small trace the
cluster is mostly orchestration. On the ``dftracer-ai`` fixture, cluster setup
and teardown alone account for roughly 2.2 seconds of a 14.4 second run.

.. code-block:: bash

   dfanalyzer analyzer=dftracer trace_path=/path/to/traces cluster=none

Results are identical to a clustered run — the same code path is used, with
this process acting as the single worker. Prefer this for small traces,
interactive exploration, and hosted deployments where spawning worker
processes is undesirable.

.. warning::

   Not suitable for traces too large to aggregate in a single process. The
   whole trace is read here rather than staying spread across workers, so
   memory is bounded by one machine. Use ``cluster=local`` (the default) or a
   job-queue cluster at scale.

Note that this mode consolidates memory rather than reducing it: a local
cluster keeps its workers in separate processes, so their memory does not
count against the driver's.

Takes no parameters beyond ``cluster.local_directory``.

Slurm Cluster (``cluster=slurm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For environments using the Slurm workload manager.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``cluster.processes``
     - int
     - 1
     - Number of Slurm jobs (nodes) to request.
   * - ``cluster.cores``
     - int
     - 16
     - Cores per job.
   * - ``cluster.memory``
     - string
     - null
     - Memory per job (e.g., "24GB").
   * - ``cluster.job_extra_directives``
     - list[str]
     - []
     - Additional directives for the Slurm job script (e.g., ``["--account=<name>", "--partition=<name>"]``).

LSF Cluster (``cluster=lsf``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For environments using the LSF workload manager.

PBS Cluster (``cluster=pbs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
