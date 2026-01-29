from __future__ import annotations

from dataclasses import dataclass
from time import time
from urllib.parse import urlparse


@dataclass
class WebPolicy:
    allowlist: list[str]
    rate_limit_per_min: int = 30
    _last_reset: float = 0.0
    _count: int = 0

    def __post_init__(self) -> None:
        self.allowlist = [a.lower().strip() for a in self.allowlist]

    def allow_url(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return False
        # Strip userinfo and port
        domain = domain.split("@")[-1]
        if domain.startswith("[") and domain.endswith("]"):
            domain = domain[1:-1]
        domain = domain.split(":")[0]
        return any(domain == a or domain.endswith("." + a) for a in self.allowlist)

    def check_rate(self) -> bool:
        now = time()
        if now - self._last_reset > 60:
            self._last_reset = now
            self._count = 0
        if self._count >= self.rate_limit_per_min:
            return False
        self._count += 1
        return True
