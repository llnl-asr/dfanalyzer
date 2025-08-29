# Data Flow Analyzer

<!-- ![Build and Test](https://github.com/LLNL/dfanalyzer/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/dfanalyzer?label=PyPI)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/dfanalyzer?label=Wheel)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dfanalyzer?label=Python) -->

## Overview

DFAnalyzer is an open-source tool for analyzing performance data from large-scale workflows on distributed systems. It presents a hierarchical, layer-by-layer summary of an application's execution, from high-level application events down to low-level POSIX calls. For each layer, DFAnalyzer quantifies time, operation counts, and data volume, and calculates key performance metrics like bandwidth and operations per second. It also visualizes the overlap between different layers, helping to characterize and understand complex I/O and compute patterns.

## Installation

To install DFAnalyzer through `pip` (recommended for most users):

```bash
# Ensure runtime dependencies for optional features (e.g., Darshan, Recorder) are installed.
# This might involve using your system's package manager or a tool like Spack.
# Example using Spack to prepare the environment:
# spack -e tools install
pip install dfanalyzer
```

To install DFAnalyzer from source (for developers or custom builds):

```bash
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
```

## Usage

Here's an example of how to run DFAnalyzer using sample data included in the repository:

```bash
# Before running, ensure the sample data is extracted.
# For example, to extract the 'dftracer-dlio' sample used below:
# mkdir -p tests/data/extracted
# tar -xzf tests/data/dftracer-dlio.tar.gz -C tests/data/extracted
dfanalyzer analyzer/preset=dlio trace_path=tests/data/extracted/dftracer-dlio view_types=[time_range]
```

This command analyzes the traces and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

```bash
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
```

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, which includes the percentage of time each layer overlaps with its parent layer:

```bash
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
```

## Further Information

For more details, to report issues, or to contribute to DFAnalyzer, please refer to the following resources:

- **[Official DFAnalyzer Documentation](https://dfanalyzer.readthedocs.io/)**: For detailed usage, configuration options, and information about analyzers.
- **[Issue Tracker](https://github.com/LLNL/dfanalyzer/issues)**: To report bugs or suggest new features.
- **[Contributing Guidelines](./CONTRIBUTING.md)**: For information on how to contribute to the project, including setting up a development environment and coding standards.
- **[Citation File](./CITATION.cff)**: If you use DFAnalyzer in your research, please cite it using the information in this file.

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under the DOE Early Career Research Program (LLNL-CONF-862440). Also, this research is supported in part by the National Science Foundation (NSF) under Grants OAC-2104013, OAC-2313154, and OAC-2411318.
