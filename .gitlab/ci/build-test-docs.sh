#!/bin/bash
# Runs ON the allocated compute node (via
#   flux proxy <jobid> flux run -N 1 bash .gitlab/ci/build-test-docs.sh)
# inside podman containers mirroring the GitHub Actions python environment.
# TEST_TYPE is computed by the CI job (tags/full-run pipelines => full).
set -ex

TEST_TYPE=${TEST_TYPE:-smoke}

PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"
PODMAN="podman --root $PODMAN_STORE --runroot $PODMAN_RUNROOT"

# --user 0:0: container root maps to the host user under rootless podman, so
# the bind-mounted checkout stays readable even for images with a non-root USER.

# Install + test suite + external-cluster check, all in one container.
# -e USER: the dask local_directory default interpolates ${oc.env:USER}
# (python/dftracer/analyzer/config.py) and podman does not propagate USER, so
# omegaconf fails with InterpolationResolutionError in every cluster test.
$PODMAN run --rm --user 0:0 -v "$PWD:/ws" -w /ws -e TEST_TYPE="$TEST_TYPE" \
  -e USER="${USER:-root}" \
  docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip setuptools wheel
  pip install --quiet -r tests/requirements.txt
  pip install --quiet .
  pytest -m "$TEST_TYPE" --verbose --cov=dftracer.analyzer --cov-report=xml
  bash .gitlab/ci/cluster-check.sh
'

# Docs.
# NOTE(gitlab-migration): docs/requirements.txt pins (sphinx 5.0.2, babel 2.10)
# fail on newer Pythons (stdlib `cgi` removed), so unpinned equivalents are
# installed here; requirements.txt is left untouched for ReadTheDocs.
$PODMAN run --rm --user 0:0 -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet sphinx sphinx-rtd-theme sphinxcontrib-mermaid
  sphinx-build -b html docs public
'
