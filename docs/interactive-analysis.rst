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
   !tar -xzf ../../tests/data/dftracer-dlio.tar.gz -C ./data

Run Analysis
------------

With the environment and data ready, you can run the analysis.

Initialize DFAnalyzer
~~~~~~~~~~~~~~~~~~~~~

Initialize DFAnalyzer using ``init_with_hydra``, providing configuration overrides as needed. This sets up the analyzer, such as ``dftracer`` with a specific preset like ``dlio``.

.. code-block:: python

   from dfanalyzer import init_with_hydra

   run_dir = f"./unet3d_v100_hdf5"
   time_granularity = 5  # 5 seconds
   trace_path = f"./data/dftracer-dlio"
   view_types = ["time_range", "proc_name"]

   dfa = init_with_hydra(
       hydra_overrides=[
           'analyzer=dftracer',
           'analyzer/preset=dlio',
           'analyzer.checkpoint=False',
           f"analyzer.time_granularity={time_granularity}",
           f"hydra.run.dir={run_dir}",
           f"trace_path={trace_path}",
       ]
   )

You can inspect the Dask client and the preset configuration:

.. code-block:: python

   # Access the Dask client
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

The ``result`` object (of type ``AnalyzerResultType``) contains detailed views of the analyzed data, which you can explore using pandas DataFrames. The ``AnalyzerResultType`` provides convenient methods to access different aspects of the analysis results.

AnalyzerResultType Class
~~~~~~~~~~~~~~~~~~~~~~~~

The ``AnalyzerResultType`` dataclass encapsulates all the results from a DFAnalyzer analysis run. It provides both direct attribute access and convenience methods for exploring the data.

**Key Distinction**: Most users should primarily use ``flat_views`` (pandas DataFrames) for interactive analysis. The other views are Dask DataFrames exposed for advanced users who need distributed processing capabilities.

Key Attributes:

- ``layers``: List of layer names available in the analysis
- ``view_types``: List of view types used in the analysis  
- ``flat_views``: Dictionary of flattened pandas DataFrames for quick access to aggregated metrics (recommended for most users)
- ``views``: Nested dictionary of Dask DataFrames organized by layer and view type (for advanced distributed processing)
- ``raw_stats``: Basic statistics about the trace data
- ``checkpoint_dir``: Directory where analysis checkpoints are stored

Primary Method (Recommended for most users):

View aggregated metrics across all layers, grouped by time intervals (returns pandas DataFrame):

.. code-block:: python

   result.get_flat_view('time_range').head(10)

List all the layers available for detailed analysis:

.. code-block:: python

   result.layers

Advanced Methods (Dask DataFrames - for distributed processing):

Show the high-level metrics for a specific layer (returns Dask DataFrame):

.. code-block:: python

   result.get_hlm('app').head()

Display a layered main view for a specific layer (returns Dask DataFrame):

.. code-block:: python

   result.get_main_view('reader_posix_lustre').head()

Access a specific view for a layer, grouped by a particular dimension (returns Dask DataFrame):

.. code-block:: python

   result.get_layer_view('reader_posix_lustre', 'time_range').head()

Display the raw trace data, showing individual I/O events (returns Dask DataFrame):

.. code-block:: python

   result._traces.head()
