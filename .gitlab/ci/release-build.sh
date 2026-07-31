#!/bin/bash
# Manual release build (sdist + pure-python wheel) inside podman.
set -ex

PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"
PODMAN="podman --root $PODMAN_STORE --runroot $PODMAN_RUNROOT"

$PODMAN run --rm -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip build
  python3 -m build .
'
