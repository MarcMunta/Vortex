from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any

from .voice_models import extract_voice_intent, normalize_transcript


class VoiceService:
    def __init__(self, *, settings: dict[str, Any], base_dir: Path) -> None:
        self.settings = settings
        self.base_dir = Path(base_dir)
        cfg = settings.get("voice", {}) or {}
        raw_output_dir = cfg.get("output_dir") or "data/multimodal/voice"
        self.output_dir = Path(raw_output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (self.base_dir / self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._whisper_model = None
        self._tts_model = None

    def _import_faster_whisper(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore

            return WhisperModel
        except Exception:
            return None

    def _import_tts(self):
        try:
            from TTS.api import TTS  # type: ignore

            return TTS
        except Exception:
            return None

    def _voice_cfg(self) -> dict[str, Any]:
        return self.settings.get("voice", {}) or {}

    def restart(self) -> dict[str, Any]:
        self._whisper_model = None
        self._tts_model = None
        return {"ok": True, "restarted": True, "ts": float(time.time())}

    def status(self) -> dict[str, Any]:
        cfg = self._voice_cfg()
        whisper_import = self._import_faster_whisper()
        tts_import = self._import_tts()
        return {
            "ok": True,
            "enabled": bool(cfg.get("enabled", False)),
            "push_to_talk": bool(cfg.get("push_to_talk", True)),
            "vad_enabled": bool(cfg.get("vad_enabled", True)),
            "whisper_model": str(cfg.get("whisper_model") or "small"),
            "tts_model": str(cfg.get("tts_model") or ""),
            "asr_backend": "faster_whisper" if whisper_import is not None else "unavailable",
            "tts_backend": "coqui" if tts_import is not None else "browser_fallback",
            "asr_available": whisper_import is not None,
            "tts_available": tts_import is not None,
            "output_dir": str(self.output_dir),
        }

    def _load_whisper(self):
        cfg = self._voice_cfg()
        whisper_cls = self._import_faster_whisper()
        if whisper_cls is None:
            return None
        if self._whisper_model is None:
            model_name = str(cfg.get("whisper_model") or "small")
            compute_type = str(cfg.get("compute_type") or "int8")
            device = str(cfg.get("device") or "auto")
            self._whisper_model = whisper_cls(model_name, device=device, compute_type=compute_type)
        return self._whisper_model

    def _load_tts(self):
        cfg = self._voice_cfg()
        tts_cls = self._import_tts()
        if tts_cls is None:
            return None
        if self._tts_model is None:
            model_name = str(cfg.get("tts_model") or "tts_models/en/ljspeech/tacotron2-DDC")
            self._tts_model = tts_cls(model_name)
        return self._tts_model

    def transcribe(
        self,
        *,
        raw_audio: bytes | None = None,
        content_type: str | None = None,
        text_hint: str | None = None,
        language: str | None = None,
        session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transcript = normalize_transcript(text_hint)
        detected_language = language or None
        stored_path = None
        if not transcript and raw_audio:
            suffix = ".wav"
            content_label = str(content_type or "").lower()
            if "webm" in content_label:
                suffix = ".webm"
            elif "ogg" in content_label:
                suffix = ".ogg"
            sample_path = self.output_dir / f"voice-{uuid.uuid4().hex[:10]}{suffix}"
            sample_path.write_bytes(raw_audio)
            stored_path = str(sample_path)
            whisper_model = self._load_whisper()
            if whisper_model is None:
                return {
                    "ok": False,
                    "error": "voice_asr_unavailable",
                    "stored_path": stored_path,
                }
            segments, info = whisper_model.transcribe(
                str(sample_path),
                language=str(language or "").strip() or None,
                vad_filter=bool(self._voice_cfg().get("vad_enabled", True)),
                beam_size=1,
            )
            transcript = normalize_transcript(" ".join(str(segment.text or "").strip() for segment in segments))
            detected_language = getattr(info, "language", None) or language or None
        if not transcript:
            return {"ok": False, "error": "voice_transcript_empty"}
        intent = extract_voice_intent(transcript, session=session)
        return {
            "ok": True,
            "transcript": transcript,
            "detected_language": detected_language,
            "intent": intent,
            "stored_path": stored_path,
        }

    def speak(self, *, text: str, language: str | None = None) -> dict[str, Any]:
        message = normalize_transcript(text)
        if not message:
            return {"ok": False, "error": "tts_text_required"}
        tts_model = self._load_tts()
        if tts_model is None:
            return {
                "ok": True,
                "fallback_browser_tts": True,
                "text": message,
                "language": language or None,
            }
        output_path = self.output_dir / f"tts-{uuid.uuid4().hex[:10]}.wav"
        tts_model.tts_to_file(text=message, file_path=str(output_path))
        return {
            "ok": True,
            "audio_path": str(output_path),
            "audio_url": f"/v1/voice/audio/{output_path.name}",
            "language": language or None,
        }

    def resolve_audio_file(self, file_name: str) -> Path | None:
        candidate = (self.output_dir / file_name).resolve()
        if candidate.parent != self.output_dir.resolve():
            return None
        if not candidate.exists():
            return None
        return candidate
