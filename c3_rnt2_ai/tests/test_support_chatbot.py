from __future__ import annotations

from pathlib import Path

import pytest

from c3rnt2.support_chatbot import (
    SupportChatbot,
    SupportChunk,
    es_respuesta_valida,
    manejar_pregunta,
    preprocesar_input,
)


def _write_support_docs(root: Path) -> Path:
    docs = root / "documentos"
    docs.mkdir()
    (docs / "faq.txt").write_text(
        "\n".join(
            [
                "FAQ",
                "Para reiniciar el router:",
                "1. Desconectar alimentacion.",
                "2. Esperar 30 segundos.",
                "3. Volver a conectar.",
                "Si no tienes internet, reinicia el router y comprueba las luces.",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "incidencias.txt").write_text(
        "\n".join(
            [
                "Escalar a humano si el usuario lleva varios dias sin servicio.",
                "Escalar si nadie responde o si pide reclamacion.",
                "Crear incidencia con nombre, telefono, direccion y descripcion.",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "soporte_red.txt").write_text(
        "\n".join(
            [
                "Soporte de red",
                "Si el WiFi no funciona, reiniciar el router.",
                "Si internet sigue sin funcionar despues del reinicio, abrir una incidencia tecnica.",
            ]
        ),
        encoding="utf-8",
    )
    return docs


def test_preprocesado_corrige_errores_de_input() -> None:
    assert preprocesar_input("  No teng internat  ") == "no tengo internet"
    assert preprocesar_input("el ruter no funksiona") == "el router no funciona"


def test_chatbot_responde_faq_con_rag(tmp_path: Path) -> None:
    docs = _write_support_docs(tmp_path)
    bot = SupportChatbot.from_documents(
        docs,
        index_path=tmp_path / "support.sqlite",
        escalations_path=tmp_path / "escalations.jsonl",
        index_backend="none",
    )

    result = bot.manejar_pregunta("Como reinicio el router?")

    assert result.escalated is False
    assert result.intent == "soporte_red"
    assert result.chunks
    assert "router" in result.answer.lower()


def test_chatbot_maneja_errores_y_muestra_chunks(tmp_path: Path) -> None:
    docs = _write_support_docs(tmp_path)
    result = manejar_pregunta(
        "el ruter no funksiona",
        docs_dir=docs,
        index_path=tmp_path / "support.sqlite",
        escalations_path=tmp_path / "escalations.jsonl",
    )

    assert result["escalated"] is False
    assert result["clean_input"] == "el router no funciona"
    assert result["chunks"]
    assert any("soporte_red.txt" in str(chunk["source"]) for chunk in result["chunks"])


def test_chatbot_escala_caso_complejo_y_registra_log(tmp_path: Path) -> None:
    docs = _write_support_docs(tmp_path)
    log_path = tmp_path / "escalations.jsonl"
    bot = SupportChatbot.from_documents(
        docs,
        index_path=tmp_path / "support.sqlite",
        escalations_path=log_path,
        index_backend="none",
    )

    result = bot.manejar_pregunta("llevo 3 dias sin internet y nadie responde")

    assert result.escalated is True
    assert result.reason == "human_escalation_intent"
    assert "agente" in result.answer.lower()
    assert log_path.exists()
    assert "nadie responde" in log_path.read_text(encoding="utf-8")


def test_chatbot_escala_si_no_hay_contexto_relevante(tmp_path: Path) -> None:
    docs = _write_support_docs(tmp_path)
    bot = SupportChatbot.from_documents(
        docs,
        index_path=tmp_path / "support.sqlite",
        escalations_path=tmp_path / "escalations.jsonl",
        index_backend="none",
    )

    result = bot.manejar_pregunta("quiero cambiar el color de mi coche")

    assert result.escalated is True
    assert result.reason == "invalid_or_missing_context"


def test_chatbot_escala_si_faltan_documentos(tmp_path: Path) -> None:
    docs = tmp_path / "documentos"
    docs.mkdir()

    result = manejar_pregunta(
        "tengo internet?",
        docs_dir=docs,
        index_path=tmp_path / "support.sqlite",
        escalations_path=tmp_path / "escalations.jsonl",
    )

    assert result["escalated"] is True
    assert result["reason"] == "no_documents"


def test_validacion_rechaza_respuesta_corta() -> None:
    chunk = SupportChunk(text="Para reiniciar el router, esperar 30 segundos.", score=0.9, source="faq.txt")

    assert es_respuesta_valida("ok", [chunk]) is False


def test_support_chatbot_api_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    docs = _write_support_docs(tmp_path)

    class DummyModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return "Respuesta local basada en el contexto del soporte tecnico."

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda _settings, _base_dir, _backend: DummyModel())
    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda settings, base_dir=None: {
            "ok": True,
            "offline_ready": True,
            "engine_ready": True,
            "engine_kind": "vortex",
            "engine_base_url": None,
            "model_ready": True,
            "active_backend": "core",
            "active_model": "core",
            "training_ready": True,
            "web_disabled": True,
            "docker_ready": True,
            "degraded_reason": None,
            "offline_reason": "offline_ready",
            "engine_reason": "engine_ready",
            "model_reason": "model_ready",
            "training_reason": "training_ready",
            "docker_reason": "docker_not_required",
            "wsl_ready": True,
            "wsl_reason": "wsl_not_required",
            "ollama_ready": None,
            "ollama_reason": None,
        },
    )
    app = server_mod.create_app(
        {
            "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
            "rag": {"enabled": False},
            "support_chatbot": {"index_backend": "none"},
        },
        base_dir=tmp_path,
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/support-chatbot/ask",
        json={"question": "no teng internat", "docs_dir": str(docs), "use_llm": False},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["clean_input"] == "no tengo internet"
    assert data["escalated"] is False
    assert data["chunks"]
    assert data["llm_used"] is False

    llm_resp = client.post(
        "/v1/support-chatbot/ask",
        json={"question": "Como reinicio el router?", "docs_dir": str(docs), "use_llm": True},
    )
    assert llm_resp.status_code == 200
    assert llm_resp.json()["llm_used"] is True
