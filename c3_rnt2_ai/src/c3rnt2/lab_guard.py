from __future__ import annotations

import ipaddress
import re
from typing import Iterable
from urllib.parse import urlparse


_OFFENSIVE_TERMS = (
    "attack",
    "exploit",
    "bypass",
    "phish",
    "payload",
    "dump",
    "harvest",
    "exfil",
    "reverse shell",
    "shellcode",
    "lateral movement",
    "privilege escalation",
)

_HIGH_RISK_PATTERNS = (
    re.compile(r"\b(extract|steal|harvest|dump|exfiltrat\w*)\b.{0,24}\b(credentials?|tokens?|passwords?|cookies?)", re.IGNORECASE),
    re.compile(r"\b(phish|credential\s+harvest|session\s+hijack)\b", re.IGNORECASE),
)

_IP_RE = re.compile(r"\b(?:(?:\d{1,3}\.){3}\d{1,3})\b")
_URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)


def _user_text(messages: Iterable[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _looks_offensive(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in _OFFENSIVE_TERMS)


def _guard_text(text: str) -> str:
    marker = "Objetivo principal:"
    if marker not in text:
        return text
    objective = text.split(marker, 1)[1]
    for end_marker in ("\n\nContexto reciente:", "\n\nPermisos locales:"):
        if end_marker in objective:
            objective = objective.split(end_marker, 1)[0]
    return objective.strip() or text


def _contains_high_risk_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in _HIGH_RISK_PATTERNS)


def _is_private_host(value: str) -> bool:
    raw = str(value or "").strip().strip("[]")
    if not raw:
        return False
    host = raw.split(":")[0]
    if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(addr.is_private or addr.is_loopback or addr.is_link_local)


def _looks_like_file_host(host: str) -> bool:
    suffix = host.rsplit(".", 1)[-1].lower() if "." in host else ""
    return suffix in {
        "dart",
        "py",
        "ts",
        "tsx",
        "js",
        "jsx",
        "json",
        "md",
        "txt",
        "yaml",
        "yml",
        "toml",
        "lock",
    }


def _extract_targets(text: str) -> tuple[list[str], list[str]]:
    private_targets: list[str] = []
    public_targets: list[str] = []

    for match in _IP_RE.findall(text):
        target_list = private_targets if _is_private_host(match) else public_targets
        if match not in target_list:
            target_list.append(match)

    for raw_url in _URL_RE.findall(text):
        parsed = urlparse(raw_url)
        host = parsed.netloc or parsed.path
        if not host:
            continue
        target_list = private_targets if _is_private_host(host) else public_targets
        if host not in target_list:
            target_list.append(host)

    for host in _HOST_RE.findall(text):
        if _looks_like_file_host(host):
            continue
        if host.endswith((".local", ".test")) or host == "localhost":
            if host not in private_targets:
                private_targets.append(host)
            continue
        if host not in public_targets:
            public_targets.append(host)

    return private_targets, public_targets


def evaluate_lab_request(messages: Iterable[dict], settings: dict) -> dict[str, str]:
    local_lab = settings.get("local_lab", {}) or {}
    if not bool(local_lab.get("guardrails_enabled", False)):
        return {"action": "allow"}

    text = _guard_text(_user_text(messages)).strip()
    if not text:
        return {"action": "allow"}

    confirmation_token = str(local_lab.get("lab_confirmation_token") or "LAB_CONFIRMED")
    private_targets, public_targets = _extract_targets(text)
    offensive = _looks_offensive(text)

    if _contains_high_risk_request(text):
        return {
            "action": "block",
            "message": (
                "This local lab only supports defensive work and exercises against systems you own. "
                "Requests to steal or extract credentials, tokens, or session material are blocked."
            ),
        }

    if offensive and public_targets:
        return {
            "action": "block",
            "message": (
                "This profile is locked to lab-only security work. "
                "I will not help target public IPs, real domains, third-party systems, or production environments."
            ),
        }

    if offensive and private_targets and confirmation_token.lower() not in text.lower():
        return {
            "action": "confirm_lab",
            "message": (
                f"Before I help with offensive-style security steps, confirm this target belongs to your isolated lab "
                f"by replying with `{confirmation_token}` and keeping the scope inside your private lab."
            ),
        }

    return {"action": "allow"}
