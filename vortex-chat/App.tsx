import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Globe } from "lucide-react";
import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "framer-motion";
import Sidebar from "./components/Sidebar";
import ChatInput from "./components/ChatInput";
import type { SettingsTab } from "./components/SettingsModal";
import VirtualizedMessageList from "./components/VirtualizedMessageList";
import { BrowserAction, ChatSession, Message, Role, ViewType, LogEntry, AppMode, Source } from "./types";
import { isLikelyTruncatedCode, vortexService } from "./services/vortexService";
import type { StreamChunk } from "./services/vortexService";
import { translations } from "./translations";
import { AppHeader } from "./app/AppHeader";
import { ChatHomeState } from "./app/ChatHomeState";
import { useSystemStatus } from "./app/useSystemStatus";
import { useWorkspaceState } from "./app/useWorkspaceState";
import { createEmptySession, permissionsFromProject, repairMojibakeText, VIEW_INDEX } from "./app/shellUtils";

const CommandPalette = lazy(() => import("./components/CommandPalette"));
const SettingsModal = lazy(() => import("./components/SettingsModal"));
const HelpModal = lazy(() => import("./components/HelpModal"));
const ReasoningDrawer = lazy(() => import("./components/ReasoningDrawer"));
const SpatialWorkspaceView = lazy(() => import("./components/spatial/SpatialWorkspaceView"));
const ModificationExplorerModal = lazy(() => import("./components/ModificationExplorerModal"));

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ViewType>("chat");
  const [prevView, setPrevView] = useState<ViewType>("chat");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<SettingsTab>("general");
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [mode, setMode] = useState<AppMode>("ask");
  const [isComposerFocused, setIsComposerFocused] = useState(false);
  const [hasComposerDraft, setHasComposerDraft] = useState(false);
  const [headerVisible, setHeaderVisible] = useState(true);
  const [footerVisible, setFooterVisible] = useState(true);
  const [activeModificationFiles, setActiveModificationFiles] = useState<{ path: string; diff: string }[] | null>(null);
  const [isReasoningOpen, setIsReasoningOpen] = useState(false);
  const [activeThoughtMessageId, setActiveThoughtMessageId] = useState<string | null>(null);

  const addLog = useCallback((level: LogEntry["level"], message: string) => {
    const newLog: LogEntry = { id: Math.random().toString(36).slice(2, 11), timestamp: Date.now(), level, message };
    setLogs((prev) => [...prev.slice(-149), newLog]);
  }, []);

  const {
    accounts,
    currentAccount,
    currentAccountId,
    currentSession,
    currentSessionId,
    handleClearHistory: clearHistory,
    handleCreateAccount: createAccount,
    handleDeleteSession: deleteSession,
    handleNewChat: newChat,
    handleSelectAccount: selectAccount,
    isDarkMode,
    setCurrentSessionId,
    setSessions,
    setSettings,
    sessions,
    settings,
  } = useWorkspaceState({ isLoading });

  const { controlStatus, operationalStatus } = useSystemStatus({ addLog, language: settings.language });

  const inactivityTimerRef = useRef<number | null>(null);
  const isAutoScrollingRef = useRef(false);
  const shouldStickToBottomRef = useRef(true);
  const lastScrollYRef = useRef(0);
  const abortControllerRef = useRef(false);
  const mainScrollRef = useRef<HTMLDivElement>(null);
  const autoContinueDepthRef = useRef(0);
  const lastAutoContinueSignatureRef = useRef("");

  const AUTO_APPLY_SELF_EDITS = false;
  const { scrollY } = useScroll({ container: mainScrollRef });
  const t = translations[settings.language];
  const hasMessages = Boolean(currentSession?.messages?.length);
  const activeProject = useMemo(
    () => settings.projects.find((project) => project.id === settings.activeProjectId) || null,
    [settings.activeProjectId, settings.projects],
  );
  const scopedSessions = useMemo(() => {
    const activeProjectId = activeProject?.id || null;
    return sessions.filter((session) => (session.projectId || null) === activeProjectId);
  }, [activeProject?.id, sessions]);
  const internetAllowlist = controlStatus?.internet?.allowlist || [];
  const stackReady = Boolean(operationalStatus?.ok);
  const rawStatusReason = operationalStatus?.chat_block_reason
    || operationalStatus?.degraded_reason
    || operationalStatus?.engine_reason
    || operationalStatus?.model_reason
    || operationalStatus?.docker_reason
    || operationalStatus?.offline_reason;
  const chatReady = Boolean(operationalStatus?.chat_ready ?? operationalStatus?.ok);
  const chatMode = operationalStatus?.chat_mode || (chatReady ? "primary" : "unavailable");
  const canUseInternet = Boolean(controlStatus?.ok);
  const degradedChatAvailable = !stackReady && chatReady;
  const chatStatusReason = chatReady
    ? undefined
    : rawStatusReason || (settings.language === "es" ? "Chat local no listo." : "Local chat not ready.");
  const sendDisabledReason = undefined;
  const activeModelLabel = operationalStatus?.active_model || (settings.language === "es" ? "Modelo base pendiente" : "Base model pending");
  const activeEngineLabel = (operationalStatus?.engine_kind || "local").toUpperCase();
  const readyLabel = stackReady
    ? (settings.language === "es" ? "Listo" : "Ready")
    : degradedChatAvailable || chatMode === "fallback_degraded"
      ? (settings.language === "es" ? "Degradado" : "Degraded")
      : (settings.language === "es" ? "Pendiente" : "Pending");
  const lazyPanelFallback = (
    <div className="flex h-full items-center justify-center px-6 text-xs font-black uppercase tracking-[0.14em] text-muted-foreground">
      {settings.language === "es" ? "Cargando vista..." : "Loading view..."}
    </div>
  );
  const statusHeadline = stackReady
    ? (settings.language === "es" ? "Chat local listo." : "Local chat ready.")
    : degradedChatAvailable || chatMode === "fallback_degraded"
      ? (settings.language === "es" ? "Chat degradado disponible." : "Degraded chat is available.")
      : (settings.language === "es" ? "Chat local pendiente." : "Local chat pending.");
  const statusBody = stackReady
    ? (settings.language === "es"
      ? "Chat y agente listos. Memoria Obsidian disponible si esta configurada."
      : "Chat and agent are ready. Obsidian memory is used when configured.")
    : degradedChatAvailable || chatMode === "fallback_degraded"
      ? (settings.language === "es"
        ? "El chat sigue disponible mientras el runtime principal se recupera."
        : "Chat stays available while the primary runtime recovers.")
      : rawStatusReason || chatStatusReason;


  const activeThought = useMemo(() => {
    if (!activeThoughtMessageId || !currentSessionId) return undefined;
    return currentSession?.messages.find((message) => message.id === activeThoughtMessageId)?.thought;
  }, [activeThoughtMessageId, currentSession, currentSessionId]);

  const isCurrentThoughtStreaming = useMemo(() => {
    if (!isLoading || !currentSession || !activeThoughtMessageId) return false;
    const lastMessage = currentSession.messages[currentSession.messages.length - 1];
    return lastMessage?.id === activeThoughtMessageId;
  }, [activeThoughtMessageId, currentSession, isLoading]);

  useEffect(() => {
    if (sessions.length === 0) return;
    const expectedProjectId = activeProject?.id || null;
    if (currentSession && (currentSession.projectId || null) === expectedProjectId) return;
    const candidate = [...scopedSessions].sort((left, right) => right.updatedAt - left.updatedAt)[0];
    if (candidate) {
      setCurrentSessionId(candidate.id);
      return;
    }
    const freshSession = createEmptySession(settings.language, activeProject);
    setSessions((prev) => [freshSession, ...prev]);
    setCurrentSessionId(freshSession.id);
  }, [activeProject, currentSession, scopedSessions, sessions.length, setCurrentSessionId, setSessions, settings.language]);

  const openSettings = useCallback((tab: SettingsTab = "general") => {
    setSettingsInitialTab(tab);
    setIsSettingsOpen(true);
  }, []);

  const extractDiffBlocks = useCallback((content: string): string => {
    const blocks: string[] = [];
    const regex = /```(?:diff|patch)\n([\s\S]*?)```/gi;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(content)) !== null) {
      if (match[1]) blocks.push(match[1].trim());
    }
    return blocks.filter(Boolean).join("\n\n");
  }, []);

  const acceptAndApplyProposal = useCallback(async (proposalId: string) => {
    const acceptResponse = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/accept`, { method: "POST" });
    if (!acceptResponse.ok) return { ok: false, stage: "accept" };
    const applyResponse = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/apply`, { method: "POST" });
    if (!applyResponse.ok) return { ok: false, stage: "apply" };
    return { ok: true };
  }, []);

  const suggestPatchFromMessage = useCallback(async (messageId: string, reason: string) => {
    const session = sessions.find((entry) => entry.id === currentSessionId);
    const message = session?.messages.find((entry) => entry.id === messageId);
    if (!message?.content) {
      addLog("SYSTEM", settings.language === "es" ? "No hay contenido para sugerir parche." : "No content available to suggest a patch.");
      return;
    }
    const diffText = extractDiffBlocks(message.content);
    if (!diffText) {
      addLog("SYSTEM", settings.language === "es" ? "No se detectó un bloque diff en la respuesta." : "No diff block detected in the response.");
      return;
    }
    const title = settings.language === "es" ? "Parche sugerido (frontend)" : "Suggested patch (frontend)";
    const summary = settings.language === "es" ? `Generado desde chat · ${reason}` : `Generated from chat · ${reason}`;
    const proposal = await vortexService.proposeSelfEditFromDiff(diffText, title, summary);
    if (!proposal.ok || !proposal.id) {
      addLog("SYSTEM", settings.language === "es" ? `No se pudo crear propuesta: ${proposal.error || "error"}` : `Failed to create proposal: ${proposal.error || "error"}`);
      return;
    }
    addLog("LEARN", settings.language === "es" ? `Propuesta creada: ${proposal.id}` : `Proposal created: ${proposal.id}`);
    if (AUTO_APPLY_SELF_EDITS) {
      const applyResult = await acceptAndApplyProposal(proposal.id);
      if (applyResult.ok) {
        addLog("LEARN", settings.language === "es" ? `Parche aplicado: ${proposal.id}` : `Patch applied: ${proposal.id}`);
      } else {
        addLog("SYSTEM", settings.language === "es" ? `No se pudo aplicar: ${proposal.id}` : `Failed to apply: ${proposal.id}`);
      }
    }
  }, [acceptAndApplyProposal, addLog, currentSessionId, extractDiffBlocks, sessions, settings.language]);

  const openBrowserActions = useCallback((actions: BrowserAction[]) => {
    if (settings.permissions.level !== "full" || settings.permissions.actionMode !== "full") return;
    const openedTargets = new Set<string>();
    for (const action of actions) {
      const target = repairMojibakeText(String(action?.target || "").trim());
      if (!target || openedTargets.has(target) || !/^https?:\/\//i.test(target)) continue;
      openedTargets.add(target);
      try {
        const handle = window.open(target, "_blank", "noopener,noreferrer");
        if (handle) {
          addLog("SYSTEM", settings.language === "es" ? `Navegador abierto: ${target}` : `Browser opened: ${target}`);
        } else {
          addLog("SYSTEM", settings.language === "es" ? `Apertura bloqueada por el navegador: ${target}` : `Browser blocked opening: ${target}`);
        }
      } catch {
        addLog("SYSTEM", settings.language === "es" ? `No se pudo abrir el navegador: ${target}` : `Could not open browser: ${target}`);
      }
    }
  }, [addLog, settings.language, settings.permissions.actionMode, settings.permissions.level]);

  const resetInactivityTimer = useCallback(() => {
    if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current);
    if (activeModificationFiles) return;
    if (isLoading || isSearching) {
      setFooterVisible(true);
      return;
    }
    if (!hasMessages && activeView === "chat") {
      setFooterVisible(true);
      return;
    }
    if (isComposerFocused || hasComposerDraft) {
      setFooterVisible(true);
      return;
    }
    if (hasMessages) setFooterVisible(true);
    inactivityTimerRef.current = window.setTimeout(() => {
      if (activeView === "chat") setFooterVisible(false);
    }, 6000);
  }, [activeModificationFiles, activeView, hasComposerDraft, hasMessages, isComposerFocused, isLoading, isSearching]);

  useEffect(() => {
    resetInactivityTimer();
    return () => {
      if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current);
    };
  }, [resetInactivityTimer]);



  useEffect(() => {
    if (activeView === "chat" && currentSession?.messages.length) {
      const container = mainScrollRef.current;
      if (!container) return;
      const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
      const isAtBottom = distanceFromBottom <= 120;
      if (shouldStickToBottomRef.current || isAtBottom) {
        isAutoScrollingRef.current = true;
        container.scrollTo({ top: container.scrollHeight, behavior: isLoading ? "auto" : "smooth" });
        const timer = window.setTimeout(() => {
          isAutoScrollingRef.current = false;
        }, 200);
        return () => window.clearTimeout(timer);
      }
    }
  }, [activeView, currentSession, isLoading, isSearching, sessions]);

  useEffect(() => {
    const container = mainScrollRef.current;
    if (!container) return;
    const handleScroll = () => {
      if (isAutoScrollingRef.current) return;
      const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
      shouldStickToBottomRef.current = distanceFromBottom <= 160;
    };
    const handleWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) shouldStickToBottomRef.current = false;
    };
    container.addEventListener("scroll", handleScroll, { passive: true });
    container.addEventListener("wheel", handleWheel, { passive: true });
    handleScroll();
    return () => {
      container.removeEventListener("scroll", handleScroll);
      container.removeEventListener("wheel", handleWheel);
    };
  }, []);

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (activeModificationFiles) return;
    const container = mainScrollRef.current;
    if (!container || isAutoScrollingRef.current) return;
    const diff = latest - lastScrollYRef.current;
    lastScrollYRef.current = latest;
    if (latest < 10) {
      if (hasMessages) setHeaderVisible(true);
      return;
    }
    if (Math.abs(diff) < 10) return;
    if (diff > 15) setHeaderVisible(false);
    else if (diff < -20) setHeaderVisible(true);
  });

  useEffect(() => {
    const handleGlobalActivity = (event: MouseEvent) => {
      if (activeModificationFiles) return;
      if (event.clientY < 80 && lastScrollYRef.current < 10 && !headerVisible) {
        setHeaderVisible(true);
      }
      if (event.clientY > window.innerHeight - 120) {
        if (!footerVisible) setFooterVisible(true);
        resetInactivityTimer();
      }
    };
    window.addEventListener("mousemove", handleGlobalActivity);
    return () => window.removeEventListener("mousemove", handleGlobalActivity);
  }, [activeModificationFiles, footerVisible, headerVisible, resetInactivityTimer]);

  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      resetInactivityTimer();
      if (!footerVisible) setFooterVisible(true);
      if (event.altKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
        return;
      }
      if (event.key === "Escape") {
        if (activeModificationFiles) setActiveModificationFiles(null);
        else if (isSettingsOpen) setIsSettingsOpen(false);
        else if (isCommandPaletteOpen) setIsCommandPaletteOpen(false);
        else if (isHelpOpen) setIsHelpOpen(false);
        else if (isReasoningOpen) setIsReasoningOpen(false);
        else openSettings("general");
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [activeModificationFiles, footerVisible, isCommandPaletteOpen, isHelpOpen, isReasoningOpen, isSettingsOpen, openSettings, resetInactivityTimer]);

  const handleSelectAccount = useCallback((accountId: string) => {
    selectAccount(accountId);
    setHeaderVisible(true);
    setFooterVisible(true);
  }, [selectAccount]);

  const handleCreateAccount = useCallback((draft: { name: string; email: string; handle?: string }) => {
    createAccount(draft);
    setHeaderVisible(true);
    setFooterVisible(true);
  }, [createAccount]);

  const handleSelectView = useCallback((newView: ViewType) => {
    setActiveView((prev) => {
      setPrevView(prev);
      return newView;
    });
    setHeaderVisible(true);
  }, []);

  const handleSelectProject = useCallback((projectId: string | null) => {
    const project = settings.projects.find((item) => item.id === projectId) || null;
    const projectSession = [...sessions]
      .filter((session) => (session.projectId || null) === (project?.id || null))
      .sort((left, right) => right.updatedAt - left.updatedAt)[0];
    setSettings({
      ...settings,
      activeProjectId: project?.id || null,
      permissions: permissionsFromProject(project),
    });
    if (projectSession) {
      setCurrentSessionId(projectSession.id);
    } else {
      const freshSession = createEmptySession(settings.language, project);
      setSessions((prev) => [freshSession, ...prev]);
      setCurrentSessionId(freshSession.id);
    }
  }, [sessions, setCurrentSessionId, setSessions, setSettings, settings]);

  const handleNewChat = useCallback(() => {
    newChat(activeProject);
    handleSelectView("chat");
    setHeaderVisible(false);
    setFooterVisible(false);
    addLog("SYSTEM", settings.language === "es" ? "Sincronización de núcleo completada." : "Kernel sync complete.");
  }, [activeProject, addLog, handleSelectView, newChat, settings.language]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    if (scopedSessions.length <= 1) {
      handleSelectView("chat");
      setHeaderVisible(false);
      setFooterVisible(false);
    }
    deleteSession(sessionId);
  }, [deleteSession, handleSelectView, scopedSessions.length]);

  const handleClearHistory = useCallback(() => {
    clearHistory();
    handleSelectView("chat");
    setHeaderVisible(false);
    setFooterVisible(false);
    addLog("SYSTEM", settings.language === "es" ? "Historial purgado. Conversación reiniciada." : "History cleared. Conversation reset.");
  }, [addLog, clearHistory, handleSelectView, settings.language]);

  const handleNavigateToChat = useCallback((sessionId: string, messageId?: string) => {
    setCurrentSessionId(sessionId);
    setActiveView("chat");
    if (messageId) {
      window.setTimeout(() => {
        document.getElementById(messageId)?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 300);
    }
  }, [setCurrentSessionId]);

  const handleLoadDemo = useCallback(() => {
    if (!currentSessionId) return;
    const mockSources: Source[] = [
      { url: "https://react.dev", title: t.analysis_library + ": React Hooks", domain: "react.dev", kind: "web", index: 0 },
      { url: "https://fastapi.tiangolo.com", title: t.analysis_library + ": FastAPI", domain: "fastapi.tiangolo.com", kind: "web", index: 1 },
      { url: "https://developer.mozilla.org/es/docs/Web/API/Element/scrollIntoView", title: "MDN: Element.scrollIntoView()", domain: "developer.mozilla.org", kind: "web", index: 2 },
    ];

    const demoContentEs = `Protocolo activado. He analizado la estructura actual y optimizado los parámetros del kernel.

### Análisis de Componentes:
* **Motor de Animación**: Optimización de constantes de resorte (stiffness) para mayor fluidez.
* **Gestión de Estado**: Reducción de latencia en el ciclo de renderizado virtualizado.
* **Seguridad**: Verificación de firmas de integridad en parches dinámicos.

### Parámetros de Configuración Actualizados:
\`\`\`typescript
const VORTEX_CONFIG = {
  neuralPrecision: 0.98,
  latencyThreshold: "45ms",
  autoSync: true,
  engineVersion: "v2.5.0-beta",
  activeModules: ["Search", "PatchExplorer", "NeuralReasoning"]
};
\`\`\`

### Modificaciones de Archivo Propuestas:
\`\`\`file:App.tsx
- const timer = 100;
+ const timer = 60;
\`\`\`

\`\`\`file:components/Sidebar.tsx
- stiffness: 400;
+ stiffness: 500;
\`\`\``;

    const demoContentEn = `Protocol activated. I have analyzed the current structure and optimized kernel parameters.

### Component Analysis:
* **Animation Engine**: Optimization of spring constants (stiffness) for greater fluidity.
* **State Management**: Latency reduction in the virtualized rendering cycle.
* **Security**: Integrity signature verification in dynamic patches.

### Updated Configuration Parameters:
\`\`\`typescript
const VORTEX_CONFIG = {
  neuralPrecision: 0.98,
  latencyThreshold: "45ms",
  autoSync: true,
  engineVersion: "v2.5.0-beta",
  activeModules: ["Search", "PatchExplorer", "NeuralReasoning"]
};
\`\`\`

### Proposed File Modifications:
\`\`\`file:App.tsx
- const timer = 100;
+ const timer = 60;
\`\`\`

\`\`\`file:components/Sidebar.tsx
- stiffness: 400;
+ stiffness: 500;
\`\`\``;

    const demoMessages: Message[] = [
      { id: "demo-1", role: Role.USER, content: settings.language === "es" ? "Activar protocolo de demostración." : "Activate demo protocol.", timestamp: Date.now() - 60000 },
      {
        id: "demo-2",
        role: Role.AI,
        content: settings.language === "es" ? demoContentEs : demoContentEn,
        thought: settings.language === "es"
          ? "Análisis completado. Se han identificado cuellos de botella en la renderización y se han ajustado las físicas del sidebar para una respuesta táctil superior."
          : "Analysis complete. Rendering bottlenecks identified and sidebar physics adjusted for superior tactile response.",
        sources: mockSources,
        fileChanges: [
          { path: "App.tsx", diff: "- const timer = 100;\n+ const timer = 60;" },
          { path: "components/Sidebar.tsx", diff: "- stiffness: 400;\n+ stiffness: 500;" },
        ],
        timestamp: Date.now() - 30000,
      },
    ];
    setSessions((prev) => prev.map((session) => (
      session.id === currentSessionId ? { ...session, messages: demoMessages, updatedAt: Date.now() } : session
    )));
    setHeaderVisible(true);
    setFooterVisible(true);
    addLog("SYSTEM", settings.language === "es" ? "Carga de demostración completada." : "Demo load complete.");
  }, [addLog, currentSessionId, setSessions, settings.language, t.analysis_library]);

  const buildInitialAssistantStatus = useCallback((
    selectedMode: AppMode,
    useInternet: boolean,
  ): string => {
    if (settings.language === "es") {
      return selectedMode === "agent"
        ? [
            "Preparando agente local.",
            "- Revisare contexto del repo y permisos activos.",
            "- Aplicare cambios si la tarea lo requiere.",
            "- Validare con comandos disponibles.",
          ].join("\n")
        : [
            "Preparando consulta.",
            `- Contexto: ${useInternet ? "repo + busqueda web permitida" : "repo/local sin web"}.`,
            "- Devolvere diagnostico breve, codigo si aplica y validacion.",
          ].join("\n");
    }
    return selectedMode === "agent"
      ? [
          "Preparing local agent.",
          "- I will inspect repo context and active permissions.",
          "- I will apply changes when the task requires them.",
          "- I will validate with available commands.",
        ].join("\n")
      : [
          "Preparing query.",
          `- Context: ${useInternet ? "repo + allowed web search" : "repo/local without web"}.`,
          "- I will return brief diagnosis, code when useful, and validation.",
        ].join("\n");
  }, [settings.language]);

  const resolveSendPermissions = useCallback((selectedMode: AppMode, project?: typeof activeProject) => {
    if (selectedMode === "agent" && project) return permissionsFromProject(project);
    if (selectedMode === "agent") return undefined;
    return settings.permissions;
  }, [settings.permissions]);

  const mergeFileChangesLocal = (
    left: { path: string; diff: string }[] = [],
    right: { path: string; diff: string }[] = [],
  ) => {
    const merged: { path: string; diff: string }[] = [];
    const seen = new Set<string>();
    for (const change of [...left, ...right]) {
      const path = String(change?.path || "").trim();
      const diff = String(change?.diff || "").trim();
      if (!path || !diff) continue;
      const key = `${path}\n${diff}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push({ path, diff });
    }
    return merged;
  };

  const trimContinuationOverlap = (previous: string, addition: string): string => {
    const cleanAddition = String(addition || "").trim();
    if (!cleanAddition) return "";
    const cleanPrevious = String(previous || "").trimEnd();
    if (!cleanPrevious) return cleanAddition;
    const maxOverlap = Math.min(4000, cleanPrevious.length, cleanAddition.length);
    for (let size = maxOverlap; size >= 80; size -= 1) {
      if (cleanPrevious.endsWith(cleanAddition.slice(0, size))) {
        return cleanAddition.slice(size).trimStart();
      }
    }
    if (cleanPrevious.includes(cleanAddition)) return "";
    return cleanAddition;
  };

  const detectAutoContinuationReason = (chunk: StreamChunk | null, selectedMode: AppMode): string => {
    const finalText = chunk?.text || "";
    if (chunk?.finishReason === "length") return "finish_reason_length";
    if (isLikelyTruncatedCode(finalText)) return "truncated_code";
    if (
      selectedMode === "agent"
      && /stopped_by_context_compaction_limit|stopped_by_wall_time_limit|stream_first_chunk_timeout|stream_connect_timeout|No se pudo completar la ejecucion del agente|Agent run could not complete/i.test(finalText)
    ) {
      return "agent_incomplete";
    }
    return "";
  };

  const looksLikeBadContinuation = (text: string): boolean => {
    const normalized = String(text || "").trim().toLowerCase();
    if (!normalized) return true;
    return /please provide|could you please|to get started|necessary information|happy to help|what programming language|what task|can you clarify|provide the necessary/i.test(normalized);
  };

  const buildHiddenContinuationPrompt = (
    originalTask: string,
    previousText: string,
    selectedMode: AppMode,
    reason: string,
  ): string => {
    const tail = previousText.slice(-6000);
    if (settings.language === "es") {
      return [
        "CONTINUA. Mismo mensaje. Mismo chat. Misma tarea.",
        "IDIOMA OBLIGATORIO: ESPANOL.",
        "No saludes. No digas que necesitas mas informacion. No preguntes el lenguaje ni la tarea: ya estan abajo.",
        "Escribe solo la continuacion literal de la respuesta anterior. No repitas texto ya emitido.",
        selectedMode === "agent"
          ? "Si estabas en modo agente, conserva el estado, inspecciona el workspace si hace falta, termina acciones pendientes y valida cuando sea posible."
          : "Si la respuesta anterior quedo cortada, continua desde el ultimo punto coherente y cierra bloques de codigo o diff. Si dudas, termina la tarea original directamente.",
        `Motivo: ${reason}.`,
        `Tarea original:\n${originalTask}`,
        `Final ya emitido:\n${tail}`,
      ].join("\n\n");
    }
    return [
      "Continue the same task automatically in this same message.",
      "Do not greet. Do not ask for information already provided. Keep the same language as the original answer.",
      "Write only the literal continuation of the previous answer. Do not repeat already emitted text.",
      selectedMode === "agent"
        ? "If this was agent mode, preserve state, inspect the workspace when needed, finish pending actions, and validate when possible."
        : "If the previous answer was cut off, continue from the last coherent point and close code or diff blocks.",
      `Reason: ${reason}.`,
      `Original task:\n${originalTask}`,
      `Already emitted tail:\n${tail}`,
    ].join("\n\n");
  };

  const streamContinuationIntoMessage = async (
    params: {
      targetSessionId: string;
      aiMessageId: string;
      baseHistory: Message[];
      originalTask: string;
      previousText: string;
      previousFileChanges?: { path: string; diff: string }[];
      useInternet: boolean;
      selectedMode: AppMode;
      useThinking: boolean;
      projectForSend?: typeof activeProject;
      reason: string;
    },
  ) => {
    let carriedText = params.previousText;
    let carriedFileChanges = params.previousFileChanges || [];
    let reason = params.reason;

    while (reason && autoContinueDepthRef.current < 20 && !abortControllerRef.current) {
      const signature = `${params.targetSessionId}:${params.aiMessageId}:${params.selectedMode}:${reason}:${carriedText.length}`;
      if (signature === lastAutoContinueSignatureRef.current) break;
      autoContinueDepthRef.current += 1;
      lastAutoContinueSignatureRef.current = signature;
      addLog(
        "SYSTEM",
        settings.language === "es"
          ? `Auto-continuacion ${autoContinueDepthRef.current}: ${reason}`
          : `Auto-continuation ${autoContinueDepthRef.current}: ${reason}`,
      );

      setIsLoading(true);
      setIsSearching(params.useInternet);
      const continuationPrompt = buildHiddenContinuationPrompt(
        params.originalTask,
        carriedText,
        params.selectedMode,
        reason,
      );
      const continuationHistory: Message[] = [
        ...params.baseHistory,
        {
          id: params.aiMessageId,
          role: Role.AI,
          content: carriedText,
          timestamp: Date.now(),
          mode: params.selectedMode,
          fileChanges: carriedFileChanges,
        },
      ];
      const stream = vortexService.generateResponseStream(
        continuationHistory,
        continuationPrompt,
        params.useInternet,
        params.useThinking,
        params.selectedMode,
        settings.language,
        internetAllowlist,
        { accountId: currentAccountId, sessionId: params.targetSessionId },
        resolveSendPermissions(params.selectedMode, params.projectForSend || undefined),
      );

      let lastContinuationChunk: StreamChunk | null = null;
      const applyContinuationChunk = (chunk: StreamChunk): StreamChunk | null => {
        const rawAddition = chunk.text || "";
        if (!rawAddition.trim() || looksLikeBadContinuation(rawAddition)) {
          return null;
        }
        const addition = trimContinuationOverlap(carriedText, rawAddition);
        if (!addition.trim()) {
          setSessions((prev) => prev.map((session) => (
            session.id === params.targetSessionId
              ? {
                  ...session,
                  messages: session.messages.map((message) => (
                    message.id === params.aiMessageId
                      ? { ...message, finishReason: "stop" }
                      : message
                  )),
                }
              : session
          )));
          return { ...chunk, text: carriedText, fileChanges: carriedFileChanges, finishReason: "stop" };
        }
        const nextText = addition ? `${carriedText}\n\n${addition}` : carriedText;
        const nextFileChanges = mergeFileChangesLocal(carriedFileChanges, chunk.fileChanges || []);
        setSessions((prev) => prev.map((session) => (
          session.id === params.targetSessionId
            ? {
                ...session,
                updatedAt: Date.now(),
                messages: session.messages.map((message) => (
                  message.id === params.aiMessageId
                    ? {
                        ...message,
                        content: nextText,
                        thought: chunk.thought || message.thought,
                        requestId: chunk.requestId || message.requestId,
                        finishReason: chunk.finishReason ?? "stop",
                        sources: chunk.sources.length > 0 ? chunk.sources : message.sources,
                        fileChanges: nextFileChanges,
                      }
                    : message
                )),
              }
            : session
        )));
        return { ...chunk, text: nextText, fileChanges: nextFileChanges };
      };

      try {
        for await (const chunk of stream) {
          if (abortControllerRef.current) break;
          setIsSearching(false);
          lastContinuationChunk = chunk;
        }
        if (lastContinuationChunk) {
          lastContinuationChunk = applyContinuationChunk(lastContinuationChunk);
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : "continuation_failed";
        addLog("SYSTEM", repairMojibakeText(detail));
        reason = params.selectedMode === "agent" ? "agent_stream_error" : "";
        continue;
      } finally {
        setIsLoading(false);
        setIsSearching(false);
        resetInactivityTimer();
      }

      if (!lastContinuationChunk) {
        reason = reason === "bad_continuation" ? "" : "bad_continuation";
        continue;
      }
      if (lastContinuationChunk.text === carriedText) break;
      carriedText = lastContinuationChunk.text;
      carriedFileChanges = lastContinuationChunk.fileChanges || carriedFileChanges;
      reason = detectAutoContinuationReason(lastContinuationChunk, params.selectedMode);
    }
  };

  const handleSendMessageLocalFirst = async (
    content: string,
    useInternet: boolean = false,
    selectedMode: AppMode = "ask",
    useThinking: boolean = true,
    autoTrain: boolean = false,
    options?: { preserveView?: boolean; autoContinuation?: boolean },
  ) => {
    if (!options?.autoContinuation) {
      autoContinueDepthRef.current = 0;
      lastAutoContinueSignatureRef.current = "";
    }
    let agentStartupWarning: string | null = null;
    let projectForSend = activeProject;
    if (selectedMode === "agent") {
      if (!projectForSend && settings.projects.length === 1) {
        projectForSend = settings.projects[0];
        setSettings((prev) => ({
          ...prev,
          activeProjectId: projectForSend?.id || null,
          permissions: permissionsFromProject(projectForSend || null),
        }));
      }
      const validation = projectForSend
        ? await vortexService.validateWorkspaceProjects([projectForSend])
        : { ok: true, invalidIds: [] as string[] };
      if (projectForSend && validation.ok && validation.invalidIds.includes(projectForSend.id)) {
        const projectPath = projectForSend.rootPath || projectForSend.projectPath || projectForSend.name;
        agentStartupWarning = settings.language === "es"
          ? `Aviso: el backend no confirmo acceso directo a ${projectPath}. Enviare el agente igualmente; si el runtime esta en Docker, usara el repo montado o devolvera un error visible.`
          : `Warning: backend did not confirm direct access to ${projectPath}. I will still send the agent request; if the runtime is in Docker, it will use the mounted repo or return a visible error.`;
        addLog(
          "SYSTEM",
          agentStartupWarning,
        );
      } else if (projectForSend && !validation.ok) {
        agentStartupWarning = settings.language === "es"
          ? "Aviso: no pude validar el proyecto antes de enviar. El backend mostrara el error real si no puede acceder al scope."
          : "Warning: project preflight could not complete. Backend will show the real error if it cannot access the scope.";
        addLog("SYSTEM", agentStartupWarning);
      }
    }
    const targetProjectId = projectForSend?.id || null;
    let targetSessionId = currentSessionId;
    let targetSession = sessions.find((session) => session.id === targetSessionId);
    if (targetSession && (targetSession.projectId || null) !== targetProjectId) {
      targetSession = [...sessions]
        .filter((session) => (session.projectId || null) === targetProjectId)
        .sort((left, right) => right.updatedAt - left.updatedAt)[0];
      targetSessionId = targetSession?.id || null;
      if (targetSessionId) setCurrentSessionId(targetSessionId);
    }
    if (!targetSession) {
      targetSession = createEmptySession(settings.language, projectForSend);
      targetSessionId = targetSession.id;
      setSessions((prev) => [targetSession!, ...prev]);
      setCurrentSessionId(targetSessionId);
    }
    if (!targetSessionId) return;

    setMode(selectedMode);
    if (!options?.preserveView && activeView !== "chat") handleSelectView("chat");
    setHeaderVisible(true);
    setFooterVisible(true);
    shouldStickToBottomRef.current = true;
    resetInactivityTimer();
    addLog("INFO", settings.language === "es" ? `Prompt enviado (${content.length} chars) · modo=${selectedMode}` : `Prompt sent (${content.length} chars) · mode=${selectedMode}`);
    if (useInternet) {
      addLog("SEARCH", settings.language === "es" ? "Internet activado para este prompt." : "Internet enabled for this prompt.");
    }

    const userMessage: Message = { id: Date.now().toString(), role: Role.USER, content, timestamp: Date.now(), mode: selectedMode };
    const aiMessageId = (Date.now() + 1).toString();
    const initialAiMessage: Message = {
      id: aiMessageId,
      role: Role.AI,
      content: [buildInitialAssistantStatus(selectedMode, useInternet), agentStartupWarning].filter(Boolean).join("\n\n"),
      thought: "",
      requestId: undefined,
      sources: [],
      groundingSupports: [],
      timestamp: Date.now(),
      mode: selectedMode,
    };
    setSessions((prev) => prev.map((session) => (
      session.id === targetSessionId
        ? { ...session, messages: [...session.messages, userMessage, initialAiMessage], updatedAt: Date.now() }
        : session
    )));

    if ((targetSession.messages || []).length === 0) {
      void vortexService.generateChatTitle(content, settings.language).then((result) => {
        if (result.ok && result.title) {
          const normalizedTitle = repairMojibakeText(result.title);
          setSessions((prev) => prev.map((session) => (
            session.id === targetSessionId ? { ...session, title: normalizedTitle } : session
          )));
        }
      }).catch(() => {
        // Ignore title generation errors.
      });
    }

    setIsLoading(true);
    setIsSearching(useInternet);
    abortControllerRef.current = false;
    let lastStreamChunk: StreamChunk | null = null;
    let autoContinuationPrompt: string | null = null;
    let autoContinuationReason = "";
    try {
      const history = targetSession.messages || [];
      const stream = vortexService.generateResponseStream(
        history,
        content,
        useInternet,
        useThinking,
        selectedMode,
        settings.language,
        internetAllowlist,
        { accountId: currentAccountId, sessionId: targetSessionId },
        resolveSendPermissions(selectedMode, projectForSend),
      );
      let started = false;
      let aborted = false;
      let lastBrowserActions: BrowserAction[] = [];
      let pendingAnimationFrame: number | null = null;
      let pendingChunk: StreamChunk | null = null;

      const applyStreamChunk = (chunk: StreamChunk) => {
        setSessions((prev) => prev.map((session) => (
          session.id === targetSessionId
            ? {
                ...session,
                updatedAt: Date.now(),
                messages: session.messages.map((message) => (
                  message.id === aiMessageId
                    ? {
                        ...message,
                        content: chunk.text || message.content,
                        thought: chunk.thought || message.thought,
                        requestId: chunk.requestId || message.requestId,
                        finishReason: chunk.finishReason ?? message.finishReason,
                        sources: chunk.sources.length > 0 ? chunk.sources : message.sources,
                        fileChanges: chunk.fileChanges || message.fileChanges,
                      }
                    : message
                )),
              }
            : session
        )));
      };

      const scheduleStreamChunk = (chunk: StreamChunk) => {
        pendingChunk = chunk;
        if (pendingAnimationFrame !== null) return;
        pendingAnimationFrame = window.requestAnimationFrame(() => {
          pendingAnimationFrame = null;
          const next = pendingChunk;
          pendingChunk = null;
          if (next) applyStreamChunk(next);
        });
      };

      const flushStreamChunk = () => {
        if (pendingAnimationFrame !== null) {
          window.cancelAnimationFrame(pendingAnimationFrame);
          pendingAnimationFrame = null;
        }
        const next = pendingChunk;
        pendingChunk = null;
        if (next) applyStreamChunk(next);
      };

      for await (const chunk of stream) {
        if (abortControllerRef.current) {
          aborted = true;
          break;
        }
        lastStreamChunk = chunk;
        if (!started) {
          started = true;
          addLog("INFO", settings.language === "es" ? "Stream SSE conectado." : "SSE stream connected.");
        }
        setIsSearching(false);
        if (chunk.browserActions?.length) lastBrowserActions = chunk.browserActions;
        scheduleStreamChunk(chunk);
      }
      flushStreamChunk();

      if (aborted) {
        addLog("SYSTEM", settings.language === "es" ? "Ejecución abortada por el usuario." : "Run aborted by user.");
      } else {
        if (lastBrowserActions.length > 0) {
          openBrowserActions(lastBrowserActions);
        }
        const finalText = lastStreamChunk?.text || "";
        if (lastStreamChunk?.finishReason === "length") {
          autoContinuationReason = "finish_reason_length";
        } else if (isLikelyTruncatedCode(finalText)) {
          autoContinuationReason = "truncated_code";
        } else if (
          selectedMode === "agent"
          && /stopped_by_context_compaction_limit|stopped_by_wall_time_limit|stream_first_chunk_timeout|stream_connect_timeout|No se pudo completar la ejecucion del agente|Agent run could not complete/i.test(finalText)
        ) {
          autoContinuationReason = "agent_incomplete";
        }
        if (autoContinuationReason) {
          autoContinuationPrompt = settings.language === "es"
            ? "Continua la misma tarea exactamente desde el estado actual. No repitas lo ya hecho. Inspecciona el workspace, termina acciones pendientes, valida cuando sea posible y devuelve cambios completos. Si la respuesta quedo cortada, continua desde el ultimo punto y cierra bloques de codigo o diff."
            : "Continue the same task from the current state. Do not repeat completed work. Inspect the workspace, finish pending actions, validate when possible, and return complete changes. If the response was cut off, continue from the last token and close code or diff blocks.";
        }
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : (settings.language === "es" ? "Interrupción de flujo." : "Flow interrupted.");
      const cleanedDetail = repairMojibakeText(detail);
      addLog("SYSTEM", cleanedDetail);
      if (
        selectedMode === "agent"
        && /stream_first_chunk_timeout|stream_connect_timeout|network_error|Failed to fetch|No se pudo completar/i.test(cleanedDetail)
      ) {
        autoContinuationReason = "agent_stream_error";
        autoContinuationPrompt = settings.language === "es"
          ? "Continua la misma tarea de modo agente desde el estado actual. Reintenta la ejecucion, inspecciona el workspace, termina acciones pendientes y devuelve cambios completos."
          : "Continue the same agent task from the current state. Retry execution, inspect the workspace, finish pending actions, and return complete changes.";
      }
      setSessions((prev) => prev.map((session) => (
        session.id === targetSessionId
          ? {
              ...session,
              messages: session.messages.map((message) => (
                message.id === aiMessageId
                  ? {
                      ...message,
                      content: selectedMode === "agent"
                        ? (
                            settings.language === "es"
                              ? `No se pudo completar la ejecucion del agente: ${cleanedDetail}`
                              : `Agent run could not complete: ${cleanedDetail}`
                          )
                        : (
                            settings.language === "es"
                              ? `No se pudo completar la respuesta: ${cleanedDetail}`
                              : `Response could not complete: ${cleanedDetail}`
                          ),
                      finishReason: "error",
                    }
                  : message
              )),
              updatedAt: Date.now(),
            }
          : session
      )));
    } finally {
      setIsLoading(false);
      setIsSearching(false);
      resetInactivityTimer();
    }
    if (autoContinuationPrompt && !abortControllerRef.current) {
      const previousText = lastStreamChunk?.text || "";
      void streamContinuationIntoMessage({
        targetSessionId,
        aiMessageId,
        baseHistory: [...(targetSession.messages || []), userMessage],
        originalTask: content,
        previousText,
        previousFileChanges: lastStreamChunk?.fileChanges || [],
        useInternet,
        selectedMode,
        useThinking,
        projectForSend,
        reason: autoContinuationReason,
      });
    }
  };

  const handleContinueResponse = useCallback((messageId: string) => {
    const session = currentSession;
    const messageIndex = session?.messages.findIndex((item) => item.id === messageId) ?? -1;
    const message = messageIndex >= 0 ? session?.messages[messageIndex] : undefined;
    if (!message || message.role !== Role.AI) return;
    if (message.finishReason !== "length" && !isLikelyTruncatedCode(message.content)) return;
    autoContinueDepthRef.current = 0;
    lastAutoContinueSignatureRef.current = "";
    const baseHistory = session?.messages.slice(0, messageIndex) || [];
    const previousUser = [...baseHistory].reverse().find((item) => item.role === Role.USER);
    void streamContinuationIntoMessage({
      targetSessionId: session!.id,
      aiMessageId: message.id,
      baseHistory,
      originalTask: previousUser?.content || "",
      previousText: message.content,
      previousFileChanges: message.fileChanges || [],
      useInternet: false,
      selectedMode: message.mode || mode,
      useThinking: true,
      projectForSend: activeProject,
      reason: message.finishReason === "length" ? "finish_reason_length" : "truncated_code",
    });
  }, [activeProject, currentSession, mode]);

  const handleOpenModificationExplorer = useCallback((files: { path: string; diff: string }[]) => {
    setActiveModificationFiles(files);
    setHeaderVisible(false);
    setFooterVisible(false);
  }, []);

  const springConfig = { type: "spring" as const, damping: 28, stiffness: 220, mass: 0.9 };
  const direction = VIEW_INDEX[activeView] > VIEW_INDEX[prevView] ? 1 : -1;

  return (
    <div className={`relative flex h-screen w-full overflow-hidden bg-background text-foreground accelerated ask-shell`}>
      <div className="pointer-events-none absolute inset-0">
        <div className="mode-ambient absolute inset-0 transition-all duration-500" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />
      </div>

      {isCommandPaletteOpen && (
        <Suspense fallback={null}>
          <CommandPalette
            isOpen={isCommandPaletteOpen}
            onClose={() => setIsCommandPaletteOpen(false)}
            sessions={scopedSessions}
            currentSessionId={currentSessionId}
            onSelectSession={setCurrentSessionId}
            onNewChat={handleNewChat}
            onDeleteSession={handleDeleteSession}
            onClearHistory={handleClearHistory}
            onExportChat={() => {}}
            isDarkMode={isDarkMode}
            toggleDarkMode={() => setSettings({ ...settings, themeMode: isDarkMode ? "light" : "dark" })}
            isSidebarOpen={isSidebarOpen}
            onToggleSidebar={() => {
              const next = !isSidebarOpen;
              setIsSidebarOpen(next);
              if (next) setIsReasoningOpen(false);
            }}
            onOpenSettings={() => openSettings("general")}
            onOpenHelp={() => setIsHelpOpen(true)}
            categoryOrder={settings.categoryOrder}
            language={settings.language}
            onSetFontSize={(fontSize) => setSettings({ ...settings, fontSize })}
          />
        </Suspense>
      )}

      <AnimatePresence initial={false}>
        {isSidebarOpen && !activeModificationFiles && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 280, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full overflow-hidden shrink-0 z-50 flex border-r border-border/50 shadow-2xl relative">
            <Sidebar
              sessions={scopedSessions}
              currentSessionId={currentSessionId}
              activeView={activeView}
              onSelectSession={setCurrentSessionId}
              onSelectView={handleSelectView}
              onNewChat={handleNewChat}
              onDeleteSession={handleDeleteSession}
              isDarkMode={isDarkMode}
              toggleDarkMode={() => setSettings({ ...settings, themeMode: isDarkMode ? "light" : "dark" })}
              onClose={() => setIsSidebarOpen(false)}
              onOpenSettings={openSettings}
              isOpen
              language={settings.language}
              currentAccount={currentAccount}
              accounts={accounts}
              currentAccountId={currentAccountId}
              onSelectAccount={handleSelectAccount}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 flex overflow-hidden relative">
        <main className="flex-1 flex flex-col h-full bg-background relative z-0 overflow-hidden">
          {!activeModificationFiles && (
            <AppHeader
              activeView={activeView}
              headerVisible={headerVisible}
              isSidebarOpen={isSidebarOpen}
              language={settings.language}
              onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
              onSelectView={handleSelectView}
              onSetLanguage={(language) => setSettings({ ...settings, language })}
              onShowSidebar={() => {
                setIsSidebarOpen(true);
                setIsReasoningOpen(false);
              }}
              springConfig={springConfig}
            />
          )}

          <div ref={mainScrollRef} className="flex-1 overflow-y-auto custom-scrollbar flex flex-col relative h-full bg-background scroll-smooth accelerated">
            {hasMessages && !activeModificationFiles && <div className="pt-24 shrink-0" />}
            <AnimatePresence mode="popLayout" custom={direction}>
              {activeView === "chat" && (
                <motion.div
                  key="chat"
                  custom={direction}
                  variants={{
                    initial: (value: number) => ({ opacity: 0, x: value * 40, filter: "blur(10px)" }),
                    animate: { opacity: 1, x: 0, filter: "blur(0px)", transition: springConfig },
                    exit: (value: number) => ({ opacity: 0, x: -value * 40, filter: "blur(10px)", transition: { duration: 0.3 } }),
                  }}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className={`mx-auto flex min-h-full w-full flex-1 flex-col px-4 md:px-8 lg:px-10 xl:px-12 transition-all duration-500 ${!hasMessages ? "justify-center pt-20 pb-36 max-w-[1120px]" : "pt-4 max-w-[1260px]"}`}
                >
                  {!hasMessages ? (
                    <ChatHomeState
                      activeEngineLabel={activeEngineLabel}
                      activeModelLabel={activeModelLabel}
                      language={settings.language}
                      readyLabel={readyLabel}
                      sendDisabledReason={sendDisabledReason}
                      statusBody={statusBody}
                      statusHeadline={statusHeadline}
                    />
                  ) : (
                    <div className="pb-36">
                      <VirtualizedMessageList
                        messages={currentSession?.messages || []}
                        fontSize={settings.fontSize}
                        codeTheme={settings.codeTheme}
                        onShowReasoning={(messageId) => {
                          setActiveThoughtMessageId(messageId);
                          setIsReasoningOpen(true);
                          setIsSidebarOpen(false);
                        }}
                        onOpenModificationExplorer={handleOpenModificationExplorer}
                        onSuggestPatch={(messageId) => {
                          void suggestPatchFromMessage(messageId, "manual");
                        }}
                        onContinueResponse={handleContinueResponse}
                        isLoading={isLoading}
                        language={settings.language}
                        containerRef={mainScrollRef}
                        mode={mode}
                      />
                      {isSearching && (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card mt-6 flex w-fit items-center gap-3 rounded-full border border-primary/20 px-4 py-2.5 text-primary accelerated">
                          <Globe size={16} className="animate-spin-slow" />
                          <p className="text-[10px] font-black uppercase tracking-[0.14em]">
                            {settings.language === "es" ? "Internet activo en este prompt" : "Internet enabled on this prompt"}
                          </p>
                        </motion.div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}

              {activeView === "spatial" && (
                <motion.div key="spatial" custom={direction} variants={{ initial: (value: number) => ({ opacity: 0, x: value * 40, filter: "blur(10px)" }), animate: { opacity: 1, x: 0, filter: "blur(0px)", transition: springConfig }, exit: (value: number) => ({ opacity: 0, x: -value * 40, filter: "blur(10px)", transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1">
                  <Suspense fallback={lazyPanelFallback}>
                    <SpatialWorkspaceView language={settings.language} controlStatus={controlStatus} onAddLog={addLog} onSendPrompt={handleSendMessageLocalFirst} />
                  </Suspense>
                </motion.div>
              )}

            </AnimatePresence>
          </div>

          {!activeModificationFiles && activeView === "chat" && (
            <motion.div initial={false} animate={{ y: footerVisible ? 0 : 200, opacity: footerVisible ? 1 : 0 }} transition={{ type: "spring", damping: 30, stiffness: 200 }} className="absolute bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-background via-background/95 to-transparent pt-6 pb-6 pointer-events-auto accelerated">
              <div className="pointer-events-auto" data-app-mode={mode}>
                <ChatInput
                  onSend={handleSendMessageLocalFirst}
                  isLoading={isLoading}
                  isDarkMode={isDarkMode}
                  mode={mode}
                  onModeChange={setMode}
                  canUseInternet={canUseInternet}
                  sendDisabledReason={sendDisabledReason}
                  onStop={() => {
                    abortControllerRef.current = true;
                  }}
                  language={settings.language}
                  projects={settings.projects}
                  activeProjectId={settings.activeProjectId}
                  onSelectProject={handleSelectProject}
                  onOpenProjectSettings={() => openSettings("permissions")}
                  onInteraction={() => {
                    resetInactivityTimer();
                    if (!footerVisible) setFooterVisible(true);
                  }}
                  onFocusChange={setIsComposerFocused}
                  onDraftChange={setHasComposerDraft}
                />
              </div>
            </motion.div>
          )}
        </main>

        <AnimatePresence>
          {isReasoningOpen && !activeModificationFiles && (
            <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 400, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full border-l border-border/50 shrink-0 z-50 overflow-hidden bg-zinc-950/95 shadow-[-20px_0_50px_rgba(0,0,0,0.5)]">
              <Suspense fallback={null}>
                <ReasoningDrawer isOpen={isReasoningOpen} onClose={() => setIsReasoningOpen(false)} thought={activeThought} language={settings.language} isStreaming={isCurrentThoughtStreaming} />
              </Suspense>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {activeModificationFiles && (
          <Suspense fallback={null}>
            <ModificationExplorerModal
              fileChanges={activeModificationFiles}
              onClose={() => {
                setActiveModificationFiles(null);
                setHeaderVisible(true);
                setFooterVisible(true);
              }}
              language={settings.language}
            />
          </Suspense>
        )}
      </AnimatePresence>

      {isSettingsOpen && (
        <Suspense fallback={null}>
          <SettingsModal
            isOpen={isSettingsOpen}
            onClose={() => setIsSettingsOpen(false)}
            initialTab={settingsInitialTab}
            settings={settings}
            onUpdateSettings={setSettings}
            operationalStatus={operationalStatus}
            accounts={accounts}
            currentAccountId={currentAccountId}
            onSelectAccount={handleSelectAccount}
            onCreateAccount={handleCreateAccount}
          />
        </Suspense>
      )}

      {isHelpOpen && (
        <Suspense fallback={null}>
          <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} isDarkMode={isDarkMode} language={settings.language} />
        </Suspense>
      )}
    </div>
  );
};

export default App;
