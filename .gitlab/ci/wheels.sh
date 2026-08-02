#!/bin/bash
# Builds the sdist and the pure-python wheel into wheelhouse/.
#
# Nothing here needs the workspace packages: dfanalyzer declares dftracer-utils
# as a runtime dependency, and a wheel build only records that as metadata. Only
# installing resolves it.
set -eo pipefail
cd "$CI_PROJECT_DIR"

# Same .postN.dev0 scheme as dftracer, pydftracer and dftracer-utils: .postN
# outranks the tag, .dev0 keeps it a pre-release so only --pre reaches it.
described=$(git describe --tags --long --match 'v*')
tag=${described%-*-g*}
distance=${described#"$tag"-}
distance=${distance%-g*}
tag=${tag#v}
if [ "$distance" = "0" ]; then
  VERSION="$tag"
else
  VERSION="$tag.post$distance.dev0"
fi
echo "building version $VERSION"

PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"

rm -rf wheelhouse
podman --root "$PODMAN_STORE" --runroot "$PODMAN_RUNROOT" run --rm --user 0:0 \
  -v "$PWD:/ws" -w /ws \
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DFTRACER_ANALYZER="$VERSION" \
  docker.io/library/python:3.11 bash -ec '
    pip install --quiet --upgrade pip build
    python3 -m build --outdir wheelhouse .
  '
ls -1 wheelhouse
