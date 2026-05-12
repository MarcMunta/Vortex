from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .continuous.knowledge_store import KnowledgeChunk, KnowledgeStore


DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[2] / "documentos"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "support_chatbot" / "support.sqlite"
DEFAULT_ESCALATIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "support_chatbot" / "escalations.jsonl"
)

TYPO_FIXES = {
    "internat": "internet",
    "internt": "internet",
    "ruter": "router",
    "routher": "router",
    "funksiona": "funciona",
    "funcionaa": "funciona",
    "teng": "tengo",
    "tngo": "tengo",
}

STOPWORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "no",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "un",
    "una",
    "y",
}

HUMAN_ESCALATION_PATTERNS = (
    "3 dias",
    "tres dias",
    "nadie responde",
    "reclamacion",
    "queja",
    "denuncia",
    "compensacion",
    "darme de baja",
    "incidencia repetida",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _tokens(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(_strip_accents(text).lower()) if tok not in STOPWORDS}


def preprocesar_input(texto: str) -> str:
    cleaned = _strip_accents(str(texto or "").lower())
    cleaned = " ".join(cleaned.split())
    for wrong, right in TYPO_FIXES.items():
        cleaned = re.sub(rf"\b{re.escape(wrong)}\b", right, cleaned)
    return cleaned


@dataclass
class SupportChunk:
    text: str
    score: float
    source: str

    @classmethod
    def from_knowledge(cls, chunk: KnowledgeChunk, score: float | None = None) -> "SupportChunk":
        return cls(
            text=chunk.text,
            score=float(chunk.score if score is None else score),
            source=chunk.source_ref,
        )

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "score": round(float(self.score), 4), "source": self.source}


@dataclass
class SupportChatbotResult:
    question: str
    clean_input: str
    answer: str
    escalated: bool
    intent: str
    reason: str
    chunks: list[SupportChunk] = field(default_factory=list)
    prompt: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "clean_input": self.clean_input,
            "answer": self.answer,
            "escalated": self.escalated,
            "intent": self.intent,
            "reason": self.reason,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "prompt": self.prompt,
        }


@dataclass
class SupportChatbotConfig:
    docs_dir: Path = DEFAULT_DOCS_DIR
    index_path: Path = DEFAULT_INDEX_PATH
    escalations_path: Path = DEFAULT_ESCALATIONS_PATH
    top_k: int = 3
    min_score: float = 0.08
    embedding_backend: str = "hash"
    index_backend: str = "auto"
    rebuild_index: bool = True


class SupportChatbot:
    def __init__(self, config: SupportChatbotConfig, store: KnowledgeStore, indexed_docs: list[str]):
        self.config = config
        self.store = store
        self.indexed_docs = indexed_docs

    @classmethod
    def from_documents(
        cls,
        docs_dir: Path | str = DEFAULT_DOCS_DIR,
        *,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        escalations_path: Path | str = DEFAULT_ESCALATIONS_PATH,
        top_k: int = 3,
        min_score: float = 0.08,
        embedding_backend: str = "hash",
        index_backend: str = "auto",
        rebuild_index: bool = True,
    ) -> "SupportChatbot":
        config = SupportChatbotConfig(
            docs_dir=Path(docs_dir),
            index_path=Path(index_path),
            escalations_path=Path(escalations_path),
            top_k=int(top_k),
            min_score=float(min_score),
            embedding_backend=embedding_backend,
            index_backend=index_backend,
            rebuild_index=bool(rebuild_index),
        )
        config.index_path.parent.mkdir(parents=True, exist_ok=True)
        if config.rebuild_index and config.index_path.exists():
            try:
                config.index_path.unlink()
            except PermissionError:
                stamp = int(time.time() * 1000)
                config.index_path = config.index_path.with_name(
                    f"{config.index_path.stem}-{stamp}{config.index_path.suffix}"
                )
        store = KnowledgeStore(
            config.index_path,
            embedding_backend=config.embedding_backend,
            index_backend=config.index_backend,
        )
        indexed_docs: list[str] = []
        if config.docs_dir.exists():
            for doc_path in sorted(config.docs_dir.glob("*.txt")):
                text = doc_path.read_text(encoding="utf-8")
                if text.strip():
                    store.ingest_text("support_doc", doc_path.name, text, quality=0.95)
                    indexed_docs.append(doc_path.name)
        return cls(config, store, indexed_docs)

    def recuperar_chunks(self, input_limpio: str) -> list[SupportChunk]:
        raw = self.store.retrieve(
            input_limpio,
            top_k=max(self.config.top_k * 4, self.config.top_k),
            min_quality=0.0,
        )
        seen = {(chunk.text, chunk.source_ref) for chunk in raw}
        for chunk in self.store.sample_chunks(limit=100, min_quality=0.0):
            key = (chunk.text, chunk.source_ref)
            if key not in seen:
                raw.append(chunk)
                seen.add(key)
        qtokens = _tokens(input_limpio)
        ranked: list[SupportChunk] = []
        for chunk in raw:
            ctokens = _tokens(chunk.text)
            lexical = (len(qtokens & ctokens) / max(1, len(qtokens))) if qtokens else 0.0
            combined = max(float(chunk.score) * 0.75, lexical)
            if lexical <= 0.0:
                combined = min(combined, 0.02)
            if combined >= self.config.min_score:
                ranked.append(SupportChunk.from_knowledge(chunk, score=combined))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.config.top_k]

    def build_prompt(self, contexto: Iterable[SupportChunk], pregunta_limpia: str) -> str:
        ctx = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in contexto)
        return (
            "Responde como agente de soporte tecnico.\n"
            "Usa SOLO la informacion del contexto.\n"
            "Si no puedes responder con seguridad, indica que escalaras el caso.\n\n"
            f"Contexto:\n{ctx}\n\n"
            f"Pregunta: {pregunta_limpia}\n"
            "Respuesta:"
        )

    def generar_respuesta(
        self,
        contexto: list[SupportChunk],
        pregunta_limpia: str,
        *,
        llm: Callable[[str], str] | None = None,
    ) -> tuple[str, str]:
        prompt = self.build_prompt(contexto, pregunta_limpia)
        if llm is not None:
            try:
                generated = str(llm(prompt) or "").strip()
                if generated:
                    return generated, prompt
            except Exception:
                pass
        if not contexto:
            return "No tengo informacion suficiente para responder con seguridad.", prompt
        answer = "Segun la documentacion interna: " + " ".join(contexto[0].text.split())
        return answer[:900], prompt

    def es_respuesta_valida(self, respuesta: str, contexto: list[SupportChunk]) -> bool:
        if not contexto:
            return False
        if max(chunk.score for chunk in contexto) < self.config.min_score:
            return False
        lower = _strip_accents(str(respuesta or "").lower())
        invalid_markers = (
            "no tengo informacion",
            "no tengo suficiente informacion",
            "no puedo responder",
            "escalare",
            "escalar el caso",
        )
        if any(marker in lower for marker in invalid_markers):
            return False
        return len(str(respuesta or "").strip()) >= 20

    def escalar_a_humano(self, pregunta: str, input_limpio: str, reason: str) -> str:
        self.config.escalations_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "question": pregunta,
            "clean_input": input_limpio,
            "reason": reason,
        }
        with self.config.escalations_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "No he podido resolver tu problema con la documentacion disponible. Te paso con un agente."

    def detectar_intencion(self, input_limpio: str) -> str:
        tokens = _tokens(input_limpio)
        if {"router", "internet", "conexion", "wifi"} & tokens:
            return "soporte_red"
        if {"factura", "pago", "cobro"} & tokens:
            return "facturacion"
        if {"incidencia", "averia", "ticket"} & tokens:
            return "incidencia"
        return "general"

    def requiere_humano(self, input_limpio: str) -> bool:
        return any(pattern in input_limpio for pattern in HUMAN_ESCALATION_PATTERNS)

    def manejar_pregunta(
        self,
        pregunta: str,
        *,
        llm: Callable[[str], str] | None = None,
        include_prompt: bool = True,
    ) -> SupportChatbotResult:
        input_limpio = preprocesar_input(pregunta)
        intent = self.detectar_intencion(input_limpio)
        chunks = self.recuperar_chunks(input_limpio) if input_limpio else []
        respuesta, prompt = self.generar_respuesta(chunks, input_limpio, llm=llm)
        reason = "answered"
        escalated = False
        if not input_limpio:
            reason = "empty_question"
            escalated = True
        elif not self.indexed_docs:
            reason = "no_documents"
            escalated = True
        elif self.requiere_humano(input_limpio):
            reason = "human_escalation_intent"
            escalated = True
        elif not self.es_respuesta_valida(respuesta, chunks):
            reason = "invalid_or_missing_context"
            escalated = True
        if escalated:
            respuesta = self.escalar_a_humano(pregunta, input_limpio, reason)
        return SupportChatbotResult(
            question=pregunta,
            clean_input=input_limpio,
            answer=respuesta,
            escalated=escalated,
            intent=intent,
            reason=reason,
            chunks=chunks,
            prompt=prompt if include_prompt else None,
        )


def _default_bot(
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    index_path: Path | str = DEFAULT_INDEX_PATH,
    escalations_path: Path | str = DEFAULT_ESCALATIONS_PATH,
) -> SupportChatbot:
    return SupportChatbot.from_documents(
        docs_dir=docs_dir,
        index_path=index_path,
        escalations_path=escalations_path,
    )


def recuperar_chunks(
    input_limpio: str,
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> list[dict[str, object]]:
    return [chunk.to_dict() for chunk in _default_bot(docs_dir, index_path).recuperar_chunks(input_limpio)]


def generar_respuesta(
    contexto: list[SupportChunk],
    input_limpio: str,
    llm: Callable[[str], str] | None = None,
) -> str:
    bot = _default_bot()
    return bot.generar_respuesta(contexto, input_limpio, llm=llm)[0]


def es_respuesta_valida(respuesta: str, contexto: list[SupportChunk]) -> bool:
    if not contexto:
        return False
    if max(chunk.score for chunk in contexto) < 0.08:
        return False
    lower = _strip_accents(str(respuesta or "").lower())
    invalid_markers = (
        "no tengo informacion",
        "no tengo suficiente informacion",
        "no puedo responder",
        "escalare",
        "escalar el caso",
    )
    if any(marker in lower for marker in invalid_markers):
        return False
    return len(str(respuesta or "").strip()) >= 20


def escalar_a_humano(pregunta: str) -> str:
    bot = _default_bot()
    return bot.escalar_a_humano(pregunta, preprocesar_input(pregunta), "manual")


def manejar_pregunta(
    pregunta: str,
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    index_path: Path | str = DEFAULT_INDEX_PATH,
    escalations_path: Path | str = DEFAULT_ESCALATIONS_PATH,
    llm: Callable[[str], str] | None = None,
) -> dict[str, object]:
    bot = _default_bot(docs_dir=docs_dir, index_path=index_path, escalations_path=escalations_path)
    return bot.manejar_pregunta(pregunta, llm=llm).to_dict()
