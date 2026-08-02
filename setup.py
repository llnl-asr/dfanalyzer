from setuptools import setup
from setuptools_scm import ScmVersion


def myversion_func(version: ScmVersion) -> str:
    """Version a commit as the tag it follows, plus its distance from it.

    An exact tag gives 0.1.0; N commits later gives 0.1.0.postN.dev0. The .postN
    orders the build after the tag and the .dev0 keeps it a pre-release, so only
    `pip install --pre` reaches a develop build. setuptools-scm's own
    "post-release" scheme is final, and would be installed without --pre.
    """
    from setuptools_scm.version import only_version

    if version.distance and version.distance > 0:
        return version.format_next_version(
            only_version, fmt="{tag}.post{distance}.dev0"
        )
    return version.format_next_version(only_version, fmt="{tag}")


# Metadata lives in pyproject.toml; this exists only to register the version
# scheme, which cannot be named there without a registered entry point.
setup(use_scm_version={"version_scheme": myversion_func})
