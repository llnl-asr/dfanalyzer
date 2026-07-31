# Migration Plan: dfanalyzer

Selected: 2026-07-30. Source: git@github.com:llnl/dfanalyzer.git (develop). Target: ssh://git@czgitlab.llnl.gov:7999/dftracer/dfanalyzer.git (repo exists on czgitlab).

## Steps (subagent fills findings + status)

1. [x] Latest develop (dfanalyzer freshly cloned; utils pulled)
2. [x] gitlab remote added + verified
3. [x] Convert .github/workflows → .gitlab-ci.yml on corona batch runner (.corona-batch template); publish jobs manual
4. [x] pages job for Sphinx docs (develop + temp gitlab-migration rule)
5. [x] In-place GitLab URL switch for deps — **none needed** (see findings)
6. [x] Local tests (YAML, sphinx, smoke)
7. [x] Commit on gitlab-migration; sync .migration/ into repo
8. [x] Push develop, tags, gitlab-migration
9. [x] Pipeline: <https://czgitlab.llnl.gov/dftracer/dfanalyzer/-/pipelines>
10. [ ] User merges after green pipeline

## Findings

- `.github/workflows/ci.yml` (build-and-test, ubuntu x py3.10–3.13 matrix, smoke/full pytest + external-cluster dfanalyzer run) and `cd.yml` (release-build sdist/wheel + pypi-publish on release) converted into one `.gitlab-ci.yml` with the exact `.corona-batch` template from dftracer-agents.
- Matrix collapsed to the `python/3.13.2` module toolchain (comment in file).
- `build-and-test`: TEST_TYPE=full on v*.*.* tags or web pipeline with RUN_FULL_TESTS=true (maps workflow_dispatch input), else smoke. Test commands kept verbatim, incl. the external-cluster facts run. Coverage artifacts kept (codecov upload dropped — GitHub-only).
- `release-build` + `pypi-publish`: `when: manual`, gated on `v*.*.*` tags; required CI/CD variable `PYPI_TOKEN` listed in header comment; publish uses twine (replaces pypa/gh-action-pypi-publish).
- `pages`: corona batch venv; **docs/requirements.txt pins (sphinx 5.0.2, babel 2.10) are py3.10-era and fail on py3.13 (`cgi` removed)** — pages job installs unpinned `sphinx sphinx-rtd-theme sphinxcontrib-mermaid` instead, with a NOTE(gitlab-migration) comment; requirements.txt untouched for ReadTheDocs. Rules: develop + gitlab-migration (temp).
- Dependency URL scan: only dftracer-group dep is `dftracer-utils>=0.0.12` in pyproject.toml — a PyPI version spec, no GitHub URL. `[project.urls]` github links are self-referential metadata. **No in-place source changes made.**

## Local test results (2026-07-30)

- `yaml.safe_load(.gitlab-ci.yml)` — OK.
- Sphinx: pinned docs/requirements.txt fails on py3.13 (`ModuleNotFoundError: No module named 'cgi'` via babel 2.10); unpinned venv build: `build succeeded, 1 warning.`
- pytest (venv, tests/requirements.txt + `pip install .`): `pytest tests/ --collect-only -q -m smoke` → `171/197 tests collected (26 deselected)`.

## Executed changes (what to undo on revert)

- Added `gitlab` remote (ssh://git@czgitlab.llnl.gov:7999/dftracer/dfanalyzer.git).
- Branch `gitlab-migration` from develop: commit(s) adding `.gitlab-ci.yml` and `.migration/{REVERT.md,dfanalyzer.md}` (SHAs in status log).
- Pushed to gitlab: develop, tags, gitlab-migration. Nothing pushed to origin (GitHub).
- **In-place changes: NONE** — revert is fully additive-only for dfanalyzer.

## Status log

- 2026-07-30: plan created.
- 2026-07-30: executed — .gitlab-ci.yml written/validated, docs build + pytest smoke collection verified locally, no in-place dep changes needed, committed on gitlab-migration and pushed develop/tags/gitlab-migration to czgitlab. Awaiting pipeline + user merge.
- 2026-07-30: CI switched to corona flux-allocation flow, single allocation per pipeline; MR opened.
- 2026-07-30: Flux allocation made global via allocate/.flux-jobid artifact/release-allocation jobs; wait-event timeout removed.
- 2026-07-30: branch rebuilt onto merged develop; allocate switched to flux alloc --bg.
- 2026-07-30: CI now runs inside podman containers (python:3.11) on the allocated node via flux run; cluster-check extracted to .gitlab/ci/cluster-check.sh; TEST_TYPE passed with flux run --env. Pattern validated on cpp-logger.
- 2026-07-30: fixed allocation-id race — 'flux job last' is user-global and concurrent pipelines cancelled each other's allocations; now uses a unique per-job name (<proj>-$CI_PIPELINE_ID-$CI_JOB_ID) with 'flux jobs --name' lookup, and cleanup only cancels a non-empty .flux-jobid.
- 2026-07-30: test container now gets -e USER — dask local_directory interpolates ${oc.env:USER} and podman does not propagate it, which failed 16 tests + 6 errors with InterpolationResolutionError.
