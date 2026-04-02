"""Version information for rtnn."""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))


def get_version():
    """Return the version string."""
    return __version__
