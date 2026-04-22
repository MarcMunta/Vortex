import React, { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Play, Save, Sparkles } from "lucide-react";
import { ObsidianStatus, VoiceStatus, VoiceTranscriptionResult } from "../../types";

type VoiceControlDockProps = {
  language: "es" | "en";
  voiceStatus: VoiceStatus | null;
  obsidianStatus: ObsidianStatus | null;
  transcript: string;
  ttsReady: boolean;
  vaultPath: string;
  onTranscript: (value: string, result?: VoiceTranscriptionResult | null) => void;
  onSpeakSummary: () => void;
  onSaveToObsidian: () => void;
  onManualCommand: (value: string) => void;
  onRecord: (blob: Blob) => Promise<VoiceTranscriptionResult | null>;
  onVaultPathChange: (value: string) => void;
  onSaveVaultPath: () => void;
};

const VoiceControlDock: React.FC<VoiceControlDockProps> = ({
  language,
  voiceStatus,
  obsidianStatus,
  transcript,
  ttsReady,
  vaultPath,
  onTranscript,
  onSpeakSummary,
  onSaveToObsidian,
  onManualCommand,
  onRecord,
  onVaultPathChange,
  onSaveVaultPath,
}) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [manualCommand, setManualCommand] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        recorder.stream.getTracks().forEach((track) => track.stop());
        mediaRecorderRef.current = null;
        setBusy(true);
        try {
          const result = await onRecord(blob);
          if (result?.transcript) onTranscript(result.transcript, result);
        } catch (recordError) {
          setError(recordError instanceof Error ? recordError.message : "voice_record_failed");
        } finally {
          setBusy(false);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (recordError) {
      setError(recordError instanceof Error ? recordError.message : "voice_init_failed");
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;
    setIsRecording(false);
    recorder.stop();
  };

  const submitManualCommand = () => {
    const next = manualCommand.trim();
    if (!next) return;
    onManualCommand(next);
    setManualCommand("");
  };

  return (
    <div className="glass-card rounded-[1.4rem] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-black uppercase tracking-[0.14em] text-primary">
            {language === "es" ? "Voice dock" : "Voice dock"}
          </p>
          <p className="mt-2 text-sm font-bold tracking-tight text-foreground">
            {voiceStatus?.enabled
              ? (language === "es" ? "Push-to-talk local" : "Local push-to-talk")
              : (language === "es" ? "Voz en fallback" : "Voice fallback")}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {voiceStatus?.whisper_model || "faster-whisper"} {" - "} {voiceStatus?.tts_backend || "browser"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid="voice-record"
            onMouseDown={() => void startRecording()}
            onMouseUp={stopRecording}
            onMouseLeave={() => { if (isRecording) stopRecording(); }}
            onTouchStart={() => void startRecording()}
            onTouchEnd={stopRecording}
            disabled={busy}
            className={`inline-flex h-12 w-12 items-center justify-center rounded-full border transition-all ${
              isRecording
                ? "border-red-500/40 bg-red-500/15 text-red-200"
                : "border-primary/35 bg-primary/[0.12] text-primary"
            }`}
          >
            {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
          </button>
          <button
            type="button"
            data-testid="voice-speak"
            onClick={onSpeakSummary}
            className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-border/70 bg-background/85 text-foreground transition-all hover:border-primary/25 hover:text-primary"
          >
            <Play size={18} />
          </button>
          <button
            type="button"
            data-testid="voice-save-obsidian"
            onClick={onSaveToObsidian}
            className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-border/70 bg-background/85 text-foreground transition-all hover:border-primary/25 hover:text-primary"
          >
            <Save size={18} />
          </button>
        </div>
      </div>

      <div className="mt-4 rounded-[1rem] border border-white/10 bg-white/[0.03] px-4 py-3">
        <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
          {language === "es" ? "Ultima orden" : "Latest command"}
        </p>
        <p className="mt-2 min-h-[24px] text-sm leading-6 text-foreground">
          {transcript || (language === "es" ? "Sin orden todavia." : "No command yet.")}
        </p>
      </div>

      <div className="mt-4 flex gap-2">
        <input
          data-testid="voice-command-input"
          value={manualCommand}
          onChange={(event) => setManualCommand(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            submitManualCommand();
          }}
          placeholder={language === "es" ? "open this presentation here" : "open this presentation here"}
          className="h-11 flex-1 rounded-full border border-border/70 bg-background/80 px-4 text-sm text-foreground outline-none transition-all focus:border-primary/40"
        />
        <button
          type="button"
          data-testid="voice-command-run"
          onClick={submitManualCommand}
          className="inline-flex h-11 items-center gap-2 rounded-full border border-primary/35 bg-primary/[0.12] px-4 text-[11px] font-black uppercase tracking-[0.12em] text-primary"
        >
          <Sparkles size={14} /> {language === "es" ? "Lanzar" : "Run"}
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-[11px] font-black uppercase tracking-[0.12em]">
        <span className={`rounded-full border px-3 py-2 ${voiceStatus?.asr_available ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-200" : "border-border/70 bg-background/85 text-muted-foreground"}`}>
          ASR {voiceStatus?.asr_available ? "ON" : "OFF"}
        </span>
        <span className={`rounded-full border px-3 py-2 ${ttsReady ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-200" : "border-border/70 bg-background/85 text-muted-foreground"}`}>
          TTS {ttsReady ? "ON" : "FB"}
        </span>
        <span className={`rounded-full border px-3 py-2 ${obsidianStatus?.validated ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-200" : "border-border/70 bg-background/85 text-muted-foreground"}`}>
          Obsidian {obsidianStatus?.validated ? "OK" : "WAIT"}
        </span>
      </div>

      <div className="mt-4">
        <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
          {language === "es" ? "Vault Obsidian" : "Obsidian vault"}
        </p>
        <div className="mt-2 flex gap-2">
          <input
            data-testid="obsidian-vault-input"
            value={vaultPath}
            onChange={(event) => onVaultPathChange(event.target.value)}
            placeholder={language === "es" ? "D:\\Obsidian\\Vault" : "D:\\Obsidian\\Vault"}
            className="h-10 flex-1 rounded-full border border-border/70 bg-background/80 px-4 text-sm text-foreground outline-none transition-all focus:border-primary/40"
          />
          <button
            type="button"
            data-testid="obsidian-vault-save"
            onClick={onSaveVaultPath}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-border/70 bg-background/85 px-4 text-[11px] font-black uppercase tracking-[0.12em] text-foreground transition-all hover:border-primary/25 hover:text-primary"
          >
            <Save size={14} /> {language === "es" ? "Vault" : "Vault"}
          </button>
        </div>
      </div>

      {error ? (
        <p className="mt-3 text-xs text-amber-300">{error}</p>
      ) : null}
    </div>
  );
};

export default VoiceControlDock;
