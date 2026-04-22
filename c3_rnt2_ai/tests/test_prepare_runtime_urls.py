from __future__ import annotations

from c3rnt2 import config as config_mod
from c3rnt2 import prepare as prepare_mod


def test_is_local_base_url_accepts_docker_local_hosts(monkeypatch) -> None:
    monkeypatch.delenv("C3RNT2_ASSUME_DOCKER_READY", raising=False)

    assert prepare_mod._is_local_base_url("http://host.docker.internal:30000") is True
    assert prepare_mod._is_local_base_url("http://sglang-runtime:30000") is False
    assert config_mod._is_local_base_url("http://host.docker.internal:30000") is True
    assert config_mod._is_local_base_url("http://sglang-runtime:30000") is False

    monkeypatch.setenv("C3RNT2_ASSUME_DOCKER_READY", "1")

    assert prepare_mod._is_local_base_url("http://sglang-runtime:30000") is True
    assert prepare_mod._is_local_base_url("http://vortex-api:8000") is True
    assert config_mod._is_local_base_url("http://sglang-runtime:30000") is True
    assert config_mod._is_local_base_url("http://vortex-api:8000") is True
