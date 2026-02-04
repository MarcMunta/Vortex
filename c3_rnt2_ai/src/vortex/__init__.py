from __future__ import annotations

"""Vortex package (compat wrapper around `c3rnt2`)."""

try:  # pragma: no cover
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("vortex")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
