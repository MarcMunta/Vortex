import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ControlStatus,
  MultimodalStatus,
  ObsidianStatus,
  SpatialPanelModel,
  SpatialSessionState,
  VoiceStatus,
  VoiceTranscriptionResult,
} from "../../types";
import { controlService } from "../../services/controlService";
import { vortexService } from "../../services/vortexService";
import CameraGestureLayer, { GestureSignal } from "./CameraGestureLayer";
import GestureDebugOverlay from "./GestureDebugOverlay";
import {
  buildObsidianPreviewPanel,
  clamp,
  createDefaultSpatialSession,
  createSpatialPanel,
  hitTestSpatialPanels,
} from "./SpatialPanelManager";
import SpatialToolbar from "./SpatialToolbar";
import TransformCanvas from "./TransformCanvas";
import VoiceControlDock from "./VoiceControlDock";

type SpatialWorkspaceViewProps = {
  language: "es" | "en";
  controlStatus: ControlStatus | null;
  onAddLog: (level: "INFO" | "LEARN" | "SEARCH" | "SYSTEM", message: string) => void;
  onSendPrompt: (
    content: string,
    useInternet?: boolean,
    selectedMode?: "ask" | "agent",
    useThinking?: boolean,
    autoTrain?: boolean,
    options?: { preserveView?: boolean },
  ) => Promise<void>;
};

const SpatialWorkspaceView: React.FC<SpatialWorkspaceViewProps> = ({
  language,
  controlStatus,
  onAddLog,
  onSendPrompt,
}) => {
  const sampleBrowserUrl = `data:text/html;charset=utf-8,${encodeURIComponent(
    "<!doctype html><html><head><meta charset='utf-8'><title>Vortex Browser Panel</title><style>body{margin:0;background:#0c1118;color:#e6edf5;font-family:Segoe UI,Arial,sans-serif;display:grid;place-items:center;height:100vh}main{max-width:520px;padding:32px;border:1px solid rgba(255,255,255,.12);border-radius:22px;background:linear-gradient(180deg,rgba(0,174,255,.16),rgba(255,255,255,.03))}h1{margin:0 0 12px;font-size:28px}p{margin:0;line-height:1.6;color:#afbccd}</style></head><body><main><h1>Browser-like panel</h1><p>Local iframe payload for spatial browsing, review, and focused agent work.</p></main></body></html>",
  )}`;
  const sampleImageUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
    "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='900' viewBox='0 0 1200 900'><defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'><stop stop-color='#00aeff'/><stop offset='1' stop-color='#12314f'/></linearGradient></defs><rect width='1200' height='900' fill='#09111b'/><rect x='70' y='70' width='1060' height='760' rx='48' fill='url(#g)' opacity='.35'/><circle cx='940' cy='240' r='170' fill='#00aeff' opacity='.18'/><text x='110' y='220' fill='#eef7ff' font-size='92' font-family='Segoe UI, Arial, sans-serif' font-weight='700'>Vortex Spatial</text><text x='115' y='310' fill='#c6d5e8' font-size='34' font-family='Segoe UI, Arial, sans-serif'>Local image panel for multimodal focus and review</text><text x='115' y='700' fill='#eef7ff' font-size='56' font-family='Segoe UI, Arial, sans-serif'>voice + gesture + panels + obsidian</text></svg>",
  )}`;
  const [session, setSession] = useState<SpatialSessionState>(createDefaultSpatialSession);
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus | null>(null);
  const [obsidianStatus, setObsidianStatus] = useState<ObsidianStatus | null>(null);
  const [multimodalStatus, setMultimodalStatus] = useState<MultimodalStatus | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [gestureEnabled, setGestureEnabled] = useState(true);
  const [selectionMode, setSelectionMode] = useState(false);
  const [perspectiveMode, setPerspectiveMode] = useState(false);
  const [trackerMode, setTrackerMode] = useState("simulated");
  const [cameraReady, setCameraReady] = useState(false);
  const [gestureLabel, setGestureLabel] = useState(language === "es" ? "Esperando gesto" : "Waiting gesture");
  const [gestureDetail, setGestureDetail] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [vaultPathDraft, setVaultPathDraft] = useState("");
  const gestureTickRef = useRef(0);
  const lastGestureTsRef = useRef(0);
  const syncTimerRef = useRef<number | null>(null);

  const clearScheduledSync = useCallback(() => {
    if (syncTimerRef.current) {
      window.clearTimeout(syncTimerRef.current);
      syncTimerRef.current = null;
    }
  }, []);

  const applyServerSession = useCallback((nextSession: SpatialSessionState) => {
    clearScheduledSync();
    setSession(nextSession);
  }, [clearScheduledSync]);

  const focusedPanelId = session.selected_object_id;
  const focusedPanel = useMemo(
    () => session.panels.find((panel) => panel.id === focusedPanelId) || null,
    [focusedPanelId, session.panels],
  );
  const voiceEnabled = Boolean(voiceStatus?.enabled);
  const ttsReady = Boolean(voiceStatus?.tts_available);
  const panelRegion = session.selected_region || { x: 150, y: 140, width: 380, height: 260 };

  const syncSession = useCallback(async (nextSession: SpatialSessionState) => {
    try {
      const result = await vortexService.updateSpatialSession(nextSession as unknown as Record<string, unknown>);
      if (result.ok && result.session) {
        applyServerSession(result.session);
      }
    } catch {
      // optimistic local state remains
    }
  }, [applyServerSession]);

  const publishSemanticEvent = useCallback(async (payload: Record<string, unknown>) => {
    try {
      await vortexService.publishSpatialEvent(payload);
    } catch {
      // semantic event best-effort only
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    const hydrate = async () => {
      try {
        const [serverSession, nextVoiceStatus, nextObsidianStatus, nextMultimodalStatus] = await Promise.all([
          vortexService.getSpatialSession(),
          vortexService.fetchVoiceStatus(),
          vortexService.fetchObsidianStatus(),
          controlService.getMultimodalStatus(),
        ]);
        if (disposed) return;
        if (serverSession.ok && serverSession.session) {
          applyServerSession(serverSession.session);
        } else {
          setSession(createDefaultSpatialSession());
        }
        setVoiceStatus(nextVoiceStatus);
        setObsidianStatus(nextObsidianStatus);
        setMultimodalStatus(nextMultimodalStatus);
        setVaultPathDraft(String(nextObsidianStatus?.vault_path || ""));
      } catch {
        if (!disposed) setSession(createDefaultSpatialSession());
      }
    };

    void hydrate();
    const unsubscribe = controlService.subscribeMultimodalStream((payload) => {
      setMultimodalStatus(payload.status);
      if (payload.status.spatial?.panels) applyServerSession(payload.status.spatial);
      if (payload.status.voice) setVoiceStatus(payload.status.voice);
      if (payload.status.obsidian) {
        setObsidianStatus(payload.status.obsidian);
        setVaultPathDraft(String(payload.status.obsidian.vault_path || ""));
      }
    });

    return () => {
      disposed = true;
      unsubscribe();
      clearScheduledSync();
    };
  }, [applyServerSession, clearScheduledSync]);

  const scheduleSync = useCallback((nextSession: SpatialSessionState) => {
    clearScheduledSync();
    syncTimerRef.current = window.setTimeout(() => {
      void syncSession(nextSession);
    }, 280);
  }, [clearScheduledSync, syncSession]);

  const patchSession = useCallback((
    updater: (prev: SpatialSessionState) => SpatialSessionState,
    persist: boolean = true,
  ) => {
    setSession((prev) => {
      const next = updater(prev);
      if (persist) scheduleSync(next);
      return next;
    });
  }, [scheduleSync]);

  const updatePanelLocal = useCallback((panelId: string, patch: Partial<SpatialPanelModel>, persist: boolean) => {
    patchSession((prev) => {
      const nextPanels = prev.panels.map((panel) => {
        if (panel.id !== panelId) return panel;
        return {
          ...panel,
          ...patch,
          transform: patch.transform ? { ...panel.transform, ...patch.transform } : panel.transform,
          source: patch.source ? { ...(panel.source || {}), ...patch.source } : panel.source,
          updated_at: Date.now(),
        };
      });
      return {
        ...prev,
        panels: nextPanels,
        selected_object_id: panelId,
        active_panel_ids: nextPanels.map((panel) => panel.id),
        updated_at: Date.now(),
      };
    }, persist);
  }, [patchSession]);

  const persistPanel = useCallback(async (panelId: string, patch: Partial<SpatialPanelModel>) => {
    try {
      const result = await vortexService.updateSpatialPanel(panelId, patch as unknown as Record<string, unknown>);
      if (result.ok && result.session) {
        applyServerSession(result.session);
      }
    } catch {
      // no-op
    }
  }, [applyServerSession]);

  const openPanel = useCallback(async (panel: SpatialPanelModel) => {
    try {
      const result = await vortexService.openSpatialPanel(panel as unknown as Record<string, unknown>);
      if (result.ok && result.session) {
        applyServerSession(result.session);
        return;
      }
    } catch {
      // fallback below
    }
    setSession((prev) => ({
      ...prev,
      panels: [...prev.panels, panel],
      selected_object_id: panel.id,
      active_panel_ids: [...prev.active_panel_ids, panel.id],
      updated_at: Date.now(),
    }));
  }, [applyServerSession]);

  const handleFocusPanel = useCallback((panelId: string | null) => {
    patchSession((prev) => ({
      ...prev,
      selected_object_id: panelId,
      focused_item: panelId ? { type: "panel", panel_id: panelId } : null,
      interaction_mode: panelId ? "panel_focus" : "inspect",
      updated_at: Date.now(),
    }), false);
    if (panelId) {
      void publishSemanticEvent({ kind: "focus", panel_id: panelId, ts: Date.now() });
    }
  }, [patchSession, publishSemanticEvent]);

  const handleRegionChange = useCallback((region: SpatialSessionState["selected_region"]) => {
    let nextSession: SpatialSessionState | null = null;
    clearScheduledSync();
    setSession((prev) => {
      nextSession = {
        ...prev,
        selected_region: region,
        interaction_mode: region ? "selection" : "inspect",
        updated_at: Date.now(),
      };
      return nextSession;
    });
    if (nextSession) {
      void syncSession(nextSession);
    }
    void publishSemanticEvent({ kind: "selection", region, ts: Date.now() });
  }, [clearScheduledSync, publishSemanticEvent, syncSession]);

  const saveToObsidian = useCallback(async () => {
    const title = `Spatial Session ${new Date().toISOString().slice(0, 16).replace("T", " ")}`;
    const lines = [
      `Focused panel: ${focusedPanel?.title || "none"}`,
      `Voice command: ${session.last_voice_command || "none"}`,
      `Gesture: ${gestureLabel}`,
      `Panels: ${session.panels.length}`,
      `Summary: ${session.recent_multimodal_summary || multimodalStatus?.fusion?.summary || "workspace active"}`,
    ];
    try {
      const result = await vortexService.saveObsidianNote({
        folder: "Sessions",
        title,
        content: lines.join("\n"),
        note_type: "session",
        metadata: {
          selected_object_id: session.selected_object_id,
          panel_ids: session.panels.map((panel) => panel.id),
        },
      });
      const savedPath = String(result?.path || "");
      onAddLog("LEARN", language === "es" ? `Obsidian guardado: ${savedPath || title}` : `Obsidian saved: ${savedPath || title}`);
      const next = await vortexService.fetchObsidianStatus();
      setObsidianStatus(next);
      setVaultPathDraft(String(next?.vault_path || savedPath || vaultPathDraft));
      if (savedPath) {
        await openPanel(
          buildObsidianPreviewPanel(title, lines.join("\n"), savedPath, session.selected_region || undefined),
        );
      }
    } catch (error) {
      onAddLog("SYSTEM", error instanceof Error ? error.message : "obsidian_save_failed");
    }
  }, [focusedPanel, gestureLabel, language, multimodalStatus?.fusion?.summary, onAddLog, openPanel, session, vaultPathDraft]);

  const speakSummary = useCallback(async () => {
    const text = focusedPanel
      ? `${focusedPanel.title}. ${focusedPanel.content || ""}`.trim()
      : (language === "es" ? "Workspace listo. Selecciona un panel." : "Workspace ready. Select a panel.");
    try {
      const result = await vortexService.speakText(text);
      const audioUrl = result?.audio_url ? String(result.audio_url) : "";
      if (audioUrl) {
        const audio = new Audio(audioUrl);
        void audio.play().catch(() => {});
      } else if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = language === "es" ? "es-ES" : "en-US";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }
    } catch {
      if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = language === "es" ? "es-ES" : "en-US";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
      }
    }
  }, [focusedPanel, language]);

  const applyVoiceResult = useCallback(async (text: string, result: VoiceTranscriptionResult | null) => {
    if (!result) return;
    if (result.transcript) {
      setTranscript(result.transcript);
      patchSession((prev) => ({
        ...prev,
        last_voice_command: result.transcript || prev.last_voice_command,
        updated_at: Date.now(),
      }));
    }
    if (result.action_result && typeof result.action_result === "object") {
      const actionSession = (result.action_result as { session?: SpatialSessionState }).session;
      if (actionSession) {
        applyServerSession(actionSession);
      }
      const savedPath = String((result.action_result as { path?: unknown }).path || "").trim();
      if (savedPath) {
        await openPanel(
          buildObsidianPreviewPanel(
            language === "es" ? "Nota guardada" : "Saved note",
            result.transcript || text,
            savedPath,
            session.selected_region || undefined,
          ),
        );
      }
    }
    if (result.intent?.kind === "chat_query") {
      await onSendPrompt(text, false, "agent", true, false, { preserveView: true });
    }
  }, [applyServerSession, language, onSendPrompt, openPanel, patchSession, session.selected_region]);

  const handleVoiceRecord = useCallback(async (blob: Blob): Promise<VoiceTranscriptionResult | null> => {
    const result = await vortexService.transcribeVoice(blob, { language });
    await applyVoiceResult(result?.transcript || transcript, result);
    return result;
  }, [applyVoiceResult, language, transcript]);

  const handleManualCommand = useCallback(async (value: string) => {
    setTranscript(value);
    const result = await vortexService.transcribeVoice(value, { language });
    await applyVoiceResult(value, result);
  }, [applyVoiceResult, language]);

  const handleNavigatePanel = useCallback(async (panelId: string, delta: number) => {
    setSession((prev) => ({
      ...prev,
      panels: prev.panels.map((panel) => {
        if (panel.id !== panelId) return panel;
        const nextIndex = clamp(panel.page_index + delta, 0, Math.max(0, panel.page_count - 1));
        return { ...panel, page_index: nextIndex, updated_at: Date.now() };
      }),
      active_presentation_id: panelId,
      updated_at: Date.now(),
    }));
    try {
      const result = await vortexService.navigateSpatialPanel(panelId, delta);
      if (result.ok && result.session) {
        applyServerSession(result.session);
      }
    } catch {
      // optimistic only
    }
  }, [applyServerSession]);

  const handleCameraStatusChange = useCallback((payload: { cameraReady: boolean; trackerMode: string; error?: string | null }) => {
    setCameraReady(payload.cameraReady);
    setTrackerMode(payload.trackerMode);
    patchSession((prev) => ({
      ...prev,
      camera_state: {
        enabled: cameraEnabled,
        ready: payload.cameraReady,
        tracker_mode: payload.trackerMode,
        error: payload.error,
        ts: Date.now(),
      },
      updated_at: Date.now(),
    }), false);
    void publishSemanticEvent({
      kind: "camera_status",
      enabled: cameraEnabled,
      ready: payload.cameraReady,
      trackerMode: payload.trackerMode,
      error: payload.error,
      gesture_state: {
        enabled: gestureEnabled,
        tracker_mode: payload.trackerMode,
      },
    });
    if (payload.error) {
      onAddLog("SYSTEM", payload.error);
    }
  }, [cameraEnabled, gestureEnabled, onAddLog, patchSession, publishSemanticEvent]);

  const saveVaultPath = useCallback(async () => {
    const nextPath = vaultPathDraft.trim();
    if (!nextPath) return;
    try {
      await controlService.configureObsidian({ enabled: true, vault_path: nextPath });
      const next = await vortexService.configureObsidian({ enabled: true, vault_path: nextPath });
      setObsidianStatus(next);
      onAddLog("INFO", language === "es" ? `Vault Obsidian: ${nextPath}` : `Obsidian vault: ${nextPath}`);
    } catch (error) {
      onAddLog("SYSTEM", error instanceof Error ? error.message : "obsidian_config_failed");
    }
  }, [language, onAddLog, vaultPathDraft]);

  const handleGesture = useCallback((event: GestureSignal) => {
    const now = Date.now();
    if (now - lastGestureTsRef.current < 42) return;
    lastGestureTsRef.current = now;
    gestureTickRef.current += 1;

    const positionLabel = event.position
      ? `${Math.round((event.position.x || 0) * 100)}%, ${Math.round((event.position.y || 0) * 100)}%`
      : null;
    setGestureLabel(event.gesture.replaceAll("_", " "));
    setGestureDetail(positionLabel ? `${event.trackerMode} - ${positionLabel}` : event.trackerMode);

    patchSession((prev) => ({
      ...prev,
      last_gesture_event: event as unknown as Record<string, unknown>,
      gesture_state: {
        gesture: event.gesture,
        confidence: event.confidence,
        tracker_mode: event.trackerMode,
        ts: event.ts,
        enabled: gestureEnabled,
      },
      interaction_mode: perspectiveMode ? "perspective" : "gesture",
      updated_at: Date.now(),
    }), gestureTickRef.current % 6 === 0);

    if (event.position) {
      const hit = hitTestSpatialPanels(session.panels, {
        x: event.position.x * 960,
        y: event.position.y * 640,
      });
      if (event.gesture === "point" && hit) {
        handleFocusPanel(hit.id);
      }
    }

    if (gestureTickRef.current % 6 === 0 || ["swipe_left", "swipe_right", "twist", "fist", "dwell"].includes(event.gesture)) {
      void publishSemanticEvent({
        kind: "gesture",
        ...event,
        panel_id: focusedPanel?.id || undefined,
        gesture_enabled: gestureEnabled,
        camera_state: {
          enabled: cameraEnabled,
          ready: cameraReady,
          tracker_mode: trackerMode,
        },
      });
    }

    if (!focusedPanel) return;

    if (event.gesture === "pinch_hold") {
      updatePanelLocal(focusedPanel.id, {
        transform: {
          x: focusedPanel.transform.x + (event.deltaX || 0) * 820,
          y: focusedPanel.transform.y + (event.deltaY || 0) * 580,
          tilt_x: perspectiveMode ? focusedPanel.transform.tilt_x - (event.deltaY || 0) * 42 : focusedPanel.transform.tilt_x,
          tilt_y: perspectiveMode ? focusedPanel.transform.tilt_y + (event.deltaX || 0) * 42 : focusedPanel.transform.tilt_y,
        } as SpatialPanelModel["transform"],
      }, true);
      return;
    }

    if (event.gesture === "two_hand_spread" || event.gesture === "two_hand_pinch") {
      updatePanelLocal(focusedPanel.id, {
        transform: {
          scale: clamp(focusedPanel.transform.scale + (event.spreadDelta || 0), 0.25, 3),
        } as SpatialPanelModel["transform"],
      }, true);
      return;
    }

    if (event.gesture === "twist") {
      updatePanelLocal(focusedPanel.id, {
        transform: {
          rotation: focusedPanel.transform.rotation + (event.twistDelta || 10),
        } as SpatialPanelModel["transform"],
      }, true);
      return;
    }

    if (event.gesture === "swipe_left" || event.gesture === "swipe_right") {
      if (focusedPanel.type === "presentation") {
        void handleNavigatePanel(focusedPanel.id, event.gesture === "swipe_right" ? 1 : -1);
      }
      return;
    }

    if (event.gesture === "perspective_mode_trigger") {
      setPerspectiveMode(true);
      return;
    }

    if (event.gesture === "fist") {
      updatePanelLocal(focusedPanel.id, { locked: true, selected: true }, true);
      return;
    }

    if (event.gesture === "dwell") {
      patchSession((prev) => ({
        ...prev,
        focused_item: { type: "panel", panel_id: focusedPanel.id, dwell: true },
        updated_at: Date.now(),
      }), false);
      return;
    }

    if (event.gesture === "cancel" || event.gesture === "open_palm") {
      setPerspectiveMode(false);
      patchSession((prev) => ({ ...prev, interaction_mode: "inspect", updated_at: Date.now() }), false);
    }
  }, [
    cameraEnabled,
    cameraReady,
    focusedPanel,
    gestureEnabled,
    handleFocusPanel,
    handleNavigatePanel,
    patchSession,
    perspectiveMode,
    publishSemanticEvent,
    session.panels,
    trackerMode,
    updatePanelLocal,
  ]);

  const workspaceSummary = multimodalStatus?.fusion?.summary
    || session.recent_multimodal_summary
    || (language === "es" ? "Workspace multimodal listo." : "Multimodal workspace ready.");

  return (
    <div className="mx-auto flex h-full w-full max-w-[1560px] flex-col gap-5 px-4 pb-8 pt-24 md:px-6 lg:px-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-3xl">
          <p className="text-[10px] font-black uppercase tracking-[0.18em] text-primary">
            {language === "es" ? "Spatial workspace" : "Spatial workspace"}
          </p>
          <h2 className="mt-3 text-3xl font-extrabold tracking-[-0.04em] text-foreground">
            {language === "es" ? "Camara, voz, gestos y paneles pseudo-3D en shell Vortex." : "Camera, voice, gestures, and pseudo-3D panels inside Vortex."}
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-muted-foreground">
            {workspaceSummary}
          </p>
        </div>
        <div className="glass-card rounded-[1.2rem] px-4 py-3 text-right">
          <p className="text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">
            {language === "es" ? "Stack" : "Stack"}
          </p>
          <p className="mt-2 text-sm font-bold tracking-tight text-foreground">
            {controlStatus?.ok ? (language === "es" ? "Control vivo" : "Control live") : (language === "es" ? "Control degradado" : "Control degraded")}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {voiceStatus?.asr_backend || "faster-whisper"} {" - "} {trackerMode} {" - "} {obsidianStatus?.vault_path || "obsidian"}
          </p>
        </div>
      </div>

      <SpatialToolbar
        language={language}
        cameraEnabled={cameraEnabled}
        gestureEnabled={gestureEnabled}
        selectionMode={selectionMode}
        perspectiveMode={perspectiveMode}
        voiceEnabled={voiceEnabled}
        onToggleCamera={() => setCameraEnabled((prev) => !prev)}
        onToggleGestures={() => setGestureEnabled((prev) => !prev)}
        onToggleSelectionMode={() => {
          setSelectionMode((prev) => {
            const next = !prev;
            if (next && !session.selected_region) {
              handleRegionChange({
                x: Math.round(panelRegion.x),
                y: Math.round(panelRegion.y),
                width: Math.round(panelRegion.width),
                height: Math.round(panelRegion.height),
              });
            }
            return next;
          });
        }}
        onTogglePerspectiveMode={() => setPerspectiveMode((prev) => !prev)}
        onOpenNotePanel={() => {
          void openPanel(createSpatialPanel("note", language === "es" ? "Nota spatial" : "Spatial note", workspaceSummary, session.selected_region));
        }}
        onOpenPresentationPanel={() => {
          void openPanel(
            createSpatialPanel(
              "presentation",
              language === "es" ? "Presentacion local" : "Local presentation",
              "Slide 1",
              session.selected_region,
              { pages: ["Open this presentation here", "Move this left", "Tilt this panel"] },
            ),
          );
        }}
        onOpenBrowserPanel={() => {
          void openPanel(
            createSpatialPanel(
              "browser",
              language === "es" ? "Browser local" : "Local browser",
              "",
              panelRegion,
              { url: sampleBrowserUrl },
            ),
          );
        }}
        onOpenPdfPanel={() => {
          void openPanel(
            createSpatialPanel(
              "pdf",
              language === "es" ? "PDF local" : "Local PDF",
              language === "es" ? "Documento PDF listo para revisar, resumir y mover en workspace." : "PDF document ready for review, summary, and spatial handling.",
              panelRegion,
              {},
            ),
          );
        }}
        onOpenImagePanel={() => {
          void openPanel(
            createSpatialPanel(
              "image",
              language === "es" ? "Poster local" : "Local poster",
              "",
              panelRegion,
              { imageUrl: sampleImageUrl },
            ),
          );
        }}
        onOpenSketchPanel={() => {
          void openPanel(
            createSpatialPanel(
              "sketch",
              language === "es" ? "Sketch libre" : "Free sketch",
              language === "es" ? "Canvas listo para ideas rapidas y anotacion visual." : "Canvas ready for quick ideas and visual annotation.",
              panelRegion,
            ),
          );
        }}
        onSaveToObsidian={() => { void saveToObsidian(); }}
      />

      <div className="grid min-h-0 flex-1 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="relative min-h-[720px]">
          <TransformCanvas
            language={language}
            session={session}
            focusedPanelId={focusedPanelId}
            selectionMode={selectionMode}
            onFocusPanel={handleFocusPanel}
            onPreviewPanelUpdate={(panelId, patch) => updatePanelLocal(panelId, patch, false)}
            onCommitPanelUpdate={(panelId, patch) => {
              updatePanelLocal(panelId, patch, true);
              void persistPanel(panelId, patch);
            }}
            onNavigatePanel={(panelId, delta) => { void handleNavigatePanel(panelId, delta); }}
            onRegionChange={handleRegionChange}
          />

          <GestureDebugOverlay
            language={language}
            trackerMode={trackerMode}
            cameraReady={cameraReady}
            gestureLabel={gestureLabel}
            detail={gestureDetail}
          />

          <CameraGestureLayer
            enabled={cameraEnabled}
            gestureEnabled={gestureEnabled}
            language={language}
            onGesture={handleGesture}
            onStatusChange={handleCameraStatusChange}
          />
        </div>

        <div className="flex flex-col gap-5">
          <VoiceControlDock
            language={language}
            voiceStatus={voiceStatus}
            obsidianStatus={obsidianStatus}
            transcript={transcript}
            ttsReady={ttsReady}
            vaultPath={vaultPathDraft}
            onTranscript={(value) => setTranscript(value)}
            onSpeakSummary={() => { void speakSummary(); }}
            onSaveToObsidian={() => { void saveToObsidian(); }}
            onManualCommand={(value) => { void handleManualCommand(value); }}
            onRecord={handleVoiceRecord}
            onVaultPathChange={setVaultPathDraft}
            onSaveVaultPath={() => { void saveVaultPath(); }}
          />

          <div className="glass-card rounded-[1.4rem] p-4">
            <p className="text-[10px] font-black uppercase tracking-[0.14em] text-primary">
              {language === "es" ? "Focus fusion" : "Focus fusion"}
            </p>
            <div className="mt-4 space-y-3 text-sm">
              <div className="rounded-[1rem] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                  {language === "es" ? "Panel activo" : "Focused panel"}
                </p>
                <p className="mt-2 font-bold tracking-tight text-foreground">
                  {focusedPanel?.title || (language === "es" ? "Ninguno" : "None")}
                </p>
                <p className="mt-2 text-xs leading-6 text-muted-foreground">
                  {focusedPanel?.content || (language === "es" ? "Point para focus, pinch para mover, twist para rotar." : "Point to focus, pinch to move, twist to rotate.")}
                </p>
              </div>
              <div className="rounded-[1rem] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                  {language === "es" ? "Region seleccionada" : "Selected region"}
                </p>
                <p className="mt-2 text-xs leading-6 text-muted-foreground">
                  {session.selected_region
                    ? `${Math.round(session.selected_region.x)}, ${Math.round(session.selected_region.y)} - ${Math.round(session.selected_region.width)}x${Math.round(session.selected_region.height)}`
                    : (language === "es" ? "Activa Region y arrastra para marcar zona." : "Enable Region and drag to mark a drop zone.")}
                </p>
              </div>
              <div className="rounded-[1rem] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                  {language === "es" ? "Camera / gesture" : "Camera / gesture"}
                </p>
                <p className="mt-2 text-xs leading-6 text-muted-foreground">
                  {cameraReady ? (language === "es" ? "Camara lista" : "Camera ready") : (language === "es" ? "Camara pausada" : "Camera paused")} {" - "} {trackerMode} {" - "} {gestureLabel}
                </p>
              </div>
              <div className="rounded-[1rem] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                  {language === "es" ? "Comandos utiles" : "Useful commands"}
                </p>
                <ul className="mt-2 space-y-2 text-xs leading-6 text-muted-foreground">
                  <li>"open this presentation here"</li>
                  <li>"move this left"</li>
                  <li>"tilt this panel"</li>
                  <li>"talk to me about this"</li>
                  <li>"save this to Obsidian"</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpatialWorkspaceView;
