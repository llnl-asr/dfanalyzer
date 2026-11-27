.. _interactive-analysis:

Interactive Analysis
====================

DFAnalyzer provides a Python API for interactive analysis, allowing for detailed exploration of I/O traces within environments like Jupyter notebooks. This guide walks through a typical interactive analysis workflow.

.. contents::
   :local:

Prepare Environment
-------------------

First, ensure DFAnalyzer is installed in your Python environment. For detailed instructions, please refer to the :doc:`getting-started` guide.

Prepare Trace Data
------------------

Next, ensure your trace data is accessible. You can use the sample datasets located in the ``tests/data`` directory. For this example, we extract a sample trace archive.

.. code-block:: bash

   !mkdir -p ./data
   !tar -xzf ../../tests/data/dftracer-ai.tar.gz -C ./data

Run Analysis
------------

With the environment and data ready, you can run the analysis.

Initialize DFAnalyzer
~~~~~~~~~~~~~~~~~~~~~

Initialize DFAnalyzer using ``init_with_hydra``, providing configuration overrides as needed. This sets up the analyzer, such as ``dftracer`` with a specific preset like ``ai``.
Leave out ``analyzer/preset`` to let DFAnalyzer pick the preset from the trace, as the command line does.
The terminal shows only your own output.
Add ``log_level=info`` or ``debug=true`` to the overrides when something goes wrong.

.. code-block:: python

   from dftracer.analyzer import init_with_hydra

   run_dir = f"./unet3d_v100_hdf5"
   time_granularity = 5  # 5 seconds
   trace_path = f"./data/dftracer-ai"
   view_types = ["time_range", "proc_name"]

   dfa = init_with_hydra(
       hydra_overrides=[
           'analyzer=dftracer',
           'analyzer/preset=ai',
           'analyzer.checkpoint=False',
           f"analyzer.time_granularity={time_granularity}",
           f"hydra.run.dir={run_dir}",
           f"trace_path={trace_path}",
       ]
   )

You can inspect the execution client and the preset configuration:

.. code-block:: python

   # The client work is submitted through. With the default ``cluster=auto``
   # this is an ``AutoClient``, which runs work in this process until the
   # trace proves too large and then starts a cluster; with ``cluster=local``
   # or ``cluster=external`` it is a Dask ``Client``.
   dfa.client

   # View the preset configuration
   dict(dfa.analyzer.preset.layer_defs)


Execute the Analysis
~~~~~~~~~~~~~~~~~~~~

Run the trace analysis using the ``analyze_trace`` method.

.. code-block:: python

   result = dfa.analyze_trace(view_types=view_types)

The results can then be passed to the output handler to display a summary.

.. code-block:: python

   dfa.output.handle_result(result)


Result Exploration
------------------

The ``result`` object (of type ``AnalysisResult``) contains detailed views of the analyzed data as dftracer-utils ``DataFrame`` objects.
Call ``.to_pandas()`` on a frame to get a pandas DataFrame with the same index and columns.
The ``AnalysisResult`` provides convenient methods to access different aspects of the analysis results.

AnalysisResult Class
~~~~~~~~~~~~~~~~~~~~

The ``AnalysisResult`` dataclass encapsulates all the results from a DFAnalyzer analysis run. It provides both direct attribute access and convenience methods for exploring the data.

**Key Distinction**: Most users should primarily use ``flat_views``.
The other views are exposed for advanced use.
Every view is a dftracer-utils ``DataFrame``, also when the layer was computed on the Dask cluster (see ``analyzer.view_materialize_max_bytes``).
Set columns, such as the files a process touched, are list columns.

Key Attributes:

- ``layers``: List of layer names available in the analysis
- ``view_types``: List of view types used in the analysis  
- ``flat_views``: Dictionary of flattened frames for quick access to aggregated metrics (recommended for most users)
- ``views``: Nested dictionary of frames organized by layer and view type
- ``raw_stats``: Basic statistics about the trace data
- ``checkpoint_dir``: Directory where analysis checkpoints are stored

Primary Method (Recommended for most users):

View aggregated metrics across all layers, grouped by time intervals:

.. code-block:: python

   result.get_flat_view('time_range').head(10)

Get the same view as a pandas DataFrame (needs pandas installed):

.. code-block:: python

   result.get_flat_view('time_range').to_pandas()

List all the layers available for detailed analysis:

.. code-block:: python

   result.layers

Advanced Methods:

Show the high-level metrics for a specific layer:

.. code-block:: python

   result.get_hlm('app').head()

Display a layered main view for a specific layer:

.. code-block:: python

   result.get_main_view('reader_posix_lustre').head()

Access a specific view for a layer, grouped by a particular dimension:

.. code-block:: python

   result.get_layer_view('reader_posix_lustre', 'time_range').head()

Read the raw events, one row per event (``result.traces`` is a lazy ``DaskTraceViewer`` that reads on the analyzer's Dask client, so nothing is loaded until you ask):

.. code-block:: python

   result.traces.page(10).table.to_pandas()

Filter the events first, and go through all of them page by page when they do not fit in memory:

.. code-block:: python

   for table in result.traces.filter('cat == "POSIX"').pages():
       df = table.to_pandas()

Display the events grouped by category, function, process, file and time bucket:

.. code-block:: python

   result.aggregated_traces.head()
