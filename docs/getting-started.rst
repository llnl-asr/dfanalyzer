Getting Started
===============

Installation
------------

To install DFAnalyzer through ``pip`` (recommended for most users):

.. code-block:: bash

   # Ensure runtime dependencies for optional features (e.g., Darshan, Recorder) are installed.
   # This might involve using your system's package manager or a tool like Spack.
   # Example using Spack to prepare the environment:
   # spack -e tools install
   pip install dfanalyzer

To install DFAnalyzer from source (for developers or custom builds):

.. code-block:: bash

   # 1. Install system dependencies:
   #    Refer to the "Install system dependencies" step in .github/workflows/ci.yml
   #    (e.g., build-essential, cmake, libarrow-dev, libhdf5-dev, ninja-build, etc.).
   #    Alternatively, tools like Spack can help manage these:
   #    # spack -e tools install
   module load ninja

   # 2. Install Python build dependencies:
   python -m pip install --upgrade pip meson-python setuptools wheel

   # 3. Install DFAnalyzer from the root of this repository:
   #    The following command includes optional C++ components (tests and tools).
   #    The --prefix argument is optional and specifies the installation location.
   pip install -e . \
     -Csetup-args="--prefix=$HOME/.local" \
     -Csetup-args="-Denable_tests=true" \
     -Csetup-args="-Denable_tools=true"

   # (Optional) Install dependencies for running tests if you plan to contribute or run local tests:
   # pip install -r tests/requirements.txt

Usage
-----

Here's an example of how to run DFAnalyzer with the ``recorder`` analyzer using sample data included in the repository:

.. code-block:: bash

   # Before running, ensure the sample data is extracted.
   # For example, to extract the 'dftracer-dlio' sample used below:
   # mkdir -p tests/data/extracted
   # tar -xzf tests/data/dftracer-dlio.tar.gz -C tests/data/extracted
   dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio view_types=[time_range]

This command analyzes the traces and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

.. code-block:: none

                                                   Time Period Summary
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ Metric                                                                        ┃ Unit           ┃              Value ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
   │ Job Time                                                                      │ seconds        │             56.695 │
   │ Total Count                                                                   │ count          │             15,901 │
   │ Total Files                                                                   │ count          │                 87 │
   │ Total Nodes                                                                   │ count          │                  0 │
   │ Total Processes                                                               │ count          │                 23 │
   │ App Count                                                                     │ count          │                  8 │
   │ Training Count                                                                │ count          │                 40 │
   │ Compute Count                                                                 │ count          │                200 │
   │ Fetch Data Count                                                              │ count          │                160 │
   │ Data Loader Count                                                             │ count          │                808 │
   │ Data Loader Fork Count                                                        │ count          │                 96 │
   │ Reader Count                                                                  │ count          │              4,008 │
   │ Reader POSIX (Lustre) Count                                                   │ count          │             10,432 │
   │ Reader POSIX (Lustre) Size                                                    │ MB             │         111833.161 │
   │ Reader POSIX (Lustre) Bandwidth                                               │ MB/s           │            874.982 │
   │ Reader POSIX (Lustre) Avg Transfer Size                                       │ MB             │             10.720 │
   │ Checkpoint Count                                                              │ count          │                  8 │
   │ Checkpoint POSIX (Lustre) Count                                               │ count          │                 45 │
   │ Checkpoint POSIX (Lustre) Size                                                │ MB             │              0.011 │
   │ Checkpoint POSIX (Lustre) Bandwidth                                           │ MB/s           │              0.791 │
   │ Checkpoint POSIX (Lustre) Avg Transfer Size                                   │ MB             │              0.000 │
   │ Other POSIX Count                                                             │ count          │                 96 │
   └───────────────────────────────────────────────────────────────────────────────┴────────────────┴────────────────────┘

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, which includes the percentage of time each layer overlaps with its parent layer:

.. code-block:: none

                                             Layer Breakdown (w/ overlap %)
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
   ┃ Layer                       ┃         Time (s) ┃            Ops ┃   Ops/sec ┃          Size (MB) ┃ Bandwidth (MB/s) ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
   │ App                         │   441.967 (----) │       8 (----) │     0.018 │                  - │                - │
   │ Training                    │   439.442 (----) │      40 (----) │     0.091 │                  - │                - │
   │ Compute                     │   272.356 (----) │     200 (----) │     0.734 │                  - │                - │
   │ Fetch Data                  │   126.179 ( 16%) │     160 ( 25%) │     1.268 │                  - │                - │
   │ Data Loader                 │   151.471 ( 45%) │     808 ( 46%) │     5.334 │                  - │                - │
   │ Data Loader Fork            │     2.392 (  0%) │      96 (  0%) │    40.135 │                  - │                - │
   │ Reader                      │   299.992 ( 40%) │   4,008 ( 51%) │    13.360 │                  - │                - │
   │ Reader POSIX (Lustre)       │   127.812 ( 45%) │  10,432 ( 48%) │    81.620 │  111833.161 ( 46%) │          874.982 │
   │ Checkpoint                  │     0.014 (  0%) │       8 (  0%) │   571.551 │                  - │                - │
   │ Checkpoint POSIX (Lustre)   │     0.014 (  0%) │      45 (  0%) │  3268.686 │       0.011 (  0%) │            0.791 │
   │ Other POSIX                 │     2.392 (  0%) │      96 (  0%) │    40.135 │       0.000 (----) │                - │
   └─────────────────────────────┴──────────────────┴────────────────┴───────────┴────────────────────┴──────────────────┘
