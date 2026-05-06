from __future__ import annotations

import os
from urllib.parse import urlparse

from .configuration import (
    DEFAULT_PROFILE,
    DEFAULT_SETTINGS_PATH,
    load_settings,
    load_settings_document,
    normalize_settings,
    resolve_profile,
    resolve_settings_sources,
    resolve_web_allowlist,
    resolve_web_strict,
    validate_profile,
)

__all__ = [
    "DEFAULT_PROFILE",
    "DEFAULT_SETTINGS_PATH",
    "load_settings",
    "load_settings_document",
    "normalize_settings",
    "resolve_profile",
    "resolve_settings_sources",
    "resolve_web_allowlist",
    "resolve_web_strict",
    "validate_profile",
    "_is_local_base_url",
]


def _truthy(val: object) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return bool(val)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_local_base_url(raw: object | None) -> bool:
    if raw is None:
        return False
    try:
        parsed = urlparse(str(raw).strip())
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    if host in {"127.0.0.1", "localhost", "::1", "host.docker.internal", "gateway.docker.internal"}:
        return True
    if _truthy(os.getenv("C3RNT2_ASSUME_DOCKER_READY")):
        if host.endswith(".docker.internal"):
            return True
        if host and "." not in host:
            return True
    return False
