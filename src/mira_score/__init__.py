# src/mira_score/__init__.py

from .mira import get_device, mira, mira_bootstrap

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0-dev"

__all__ = (
    "get_device",
    "mira",
    "mira_bootstrap",
    "__version__",
)