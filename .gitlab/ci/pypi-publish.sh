#!/bin/bash
# Manual PyPI upload from the dist/ artifact, inside podman on the runner.
# Requires the PYPI_TOKEN GitLab CI/CD variable.
set -ex

PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"

podman --root "$PODMAN_STORE" --runroot "$PODMAN_RUNROOT" run --rm \
  --user 0:0 \
  -v "$PWD:/ws" -w /ws -e PYPI_TOKEN docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip twine
  twine upload -u __token__ -p "$PYPI_TOKEN" dist/*
'
