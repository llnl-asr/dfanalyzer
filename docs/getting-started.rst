Getting Started
===============

Installation
------------

To install DFAnalyzer through ``pip`` (recommended for most users):

.. code-block:: bash

   pip install dftracer-analyzer

To install DFAnalyzer from source (for developers or custom builds):

.. code-block:: bash

   # 1. Install Python build dependencies:
   python -m pip install --upgrade pip setuptools wheel

   # 2. Install DFAnalyzer from the root of this repository:
   pip install -e .

   # (Optional) Install dependencies for running tests if you plan to contribute or run local tests:
   # pip install -r tests/requirements.txt

Usage
-----

Here's an example of how to run DFAnalyzer using sample data included in the repository:

.. code-block:: bash

   # Before running, ensure the sample data is extracted.
   # For example, to extract the 'dftracer-ai' sample used below:
   # mkdir -p tests/data/extracted
   # tar -xzf tests/data/dftracer-ai.tar.gz -C tests/data/extracted
   dfanalyzer trace_path=tests/data/extracted/dftracer-ai

This command picks the ``ai`` preset from the trace, analyzes it and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

.. code-block:: none

                                                        Time Period Summary
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
   ┃ Metric                                                                         ┃ Unit              ┃                Value ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
   │ Job Time                                                                       │ seconds           │               86.998 │
   │ Trace Count                                                                    │ count             │              125,669 │
   │ Profile Count                                                                  │ count             │                    0 │
   │ Total Count                                                                    │ count             │              125,669 │
   │ Total Files                                                                    │ count             │                   47 │
   │ Total Nodes                                                                    │ count             │                    1 │
   │ Total Processes                                                                │ count             │                    1 │
   │ App Count                                                                      │ count             │                    1 │
   │ Training Count                                                                 │ count             │                    5 │
   │ Compute Count                                                                  │ count             │                   50 │
   │ Fetch Data Count                                                               │ count             │                   50 │
   │ Checkpoint Count                                                               │ count             │                    3 │
   │ DLIO Data Loader Count                                                         │ count             │                  302 │
   │ DLIO Data Loader Fork Count                                                    │ count             │                   10 │
   │ Reader Count                                                                   │ count             │                  800 │
   │ POSIX - All Count                                                              │ count             │              124,025 │
   │ POSIX - All Size                                                               │ MB                │            28757.953 │
   │ POSIX - All Bandwidth                                                          │ MB/s              │             1228.160 │
   │ POSIX - All Avg Transfer Size                                                  │ MB                │                0.232 │
   │ POSIX - Reader Count                                                           │ count             │              124,004 │
   │ POSIX - Reader Size                                                            │ MB                │            28757.942 │
   │ POSIX - Reader Bandwidth                                                       │ MB/s              │             1234.529 │
   │ POSIX - Reader Avg Transfer Size                                               │ MB                │                0.232 │
   │ POSIX - Checkpoint Count                                                       │ count             │                   10 │
   │ POSIX - Checkpoint Size                                                        │ MB                │                0.011 │
   │ POSIX - Checkpoint Bandwidth                                                   │ MB/s              │                1.043 │
   │ POSIX - Checkpoint Avg Transfer Size                                           │ MB                │                0.001 │
   └────────────────────────────────────────────────────────────────────────────────┴───────────────────┴──────────────────────┘

For explanations of these rows, units, and rates, see :doc:`interpreting-summary`.

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, including estimated overlap with compute for supported asynchronous layers:

.. code-block:: none

                                                  Layer Breakdown (w/ overlap %)
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ Layer                     ┃          Time (s) ┃               Ops ┃    Ops/sec ┃           Size (MB) ┃   Bandwidth (MB/s) ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
   │ App                       │     85.534 (----) │          1 (----) │      0.012 │                   - │                  - │
   │ Training                  │     85.384 (----) │          5 (----) │      0.059 │                   - │                  - │
   │ Compute                   │     68.079 (----) │         50 (----) │      0.734 │                   - │                  - │
   │ Fetch Data                │     15.428 (----) │         50 (----) │      3.241 │                   - │                  - │
   │ Checkpoint                │      0.078 (----) │          3 (----) │     38.240 │                   - │                  - │
   │ DLIO Data Loader          │    102.992 ( 66%) │        302 (  0%) │      2.932 │                   - │                  - │
   │ DLIO Data Loader Fork     │      0.109 (  0%) │         10 (  0%) │     91.467 │                   - │                  - │
   │ Reader                    │     56.889 ( 57%) │        800 ( 59%) │     14.063 │                   - │                  - │
   │ POSIX - All               │     23.415 ( 45%) │    124,025 ( 63%) │   5296.712 │    28757.953 ( 63%) │           1228.160 │
   │ POSIX - Reader            │     23.295 ( 45%) │    124,004 ( 63%) │   5323.278 │    28757.942 ( 63%) │           1234.529 │
   │ POSIX - Checkpoint        │      0.010 (----) │         10 (----) │    957.671 │        0.011 (----) │              1.043 │
   └───────────────────────────┴───────────────────┴───────────────────┴────────────┴─────────────────────┴────────────────────┘
