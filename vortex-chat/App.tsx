import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Globe } from "lucide-react";
import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "framer-motion";
import Sidebar from "./components/Sidebar";
import ChatInput from "./components/ChatInput";
import type { SettingsTab } from "./components/SettingsModal";
import VirtualizedMessageList from "./components/VirtualizedMessageList";
import { BrowserAction, ChatSession, Message, Role, ViewType, LogEntry, AppMode, Source } from "./types";
import { isLikelyTruncatedCode, vortexService } from "./services/vortexService";
import { translations } from "./translations";
import { AppHeader } from "./app/AppHeader";
import { ChatHomeState } from "./app/ChatHomeState";
import { useSystemStatus } from "./app/useSystemStatus";
import { useWorkspaceState } from "./app/useWorkspaceState";
import { createEmptySession, repairMojibakeText, VIEW_INDEX } from "./app/shellUtils";

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
    setIsDarkMode,
    setSessions,
    setSettings,
    sessions,
    settings,
  } = useWorkspaceState({ isLoading });

  const { controlStatus, operationalStatus } = useSystemStatus({ addLog, language: settings.language });

  const inactivityTimerRef = useRef<number | null>(null);
  const isAutoScrollingRef = useRef(false);
  const lastScrollYRef = useRef(0);
  const abortControllerRef = useRef(false);
  const mainScrollRef = useRef<HTMLDivElement>(null);

  const AUTO_APPLY_SELF_EDITS = false;
  const { scrollY } = useScroll({ container: mainScrollRef });
  const t = translations[settings.language];
  const hasMessages = Boolean(currentSession?.messages?.length);
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
  const sendDisabledReason = chatReady
    ? undefined
    : rawStatusReason || (settings.language === "es" ? "Stack local no listo." : "Local stack not ready.");
  const permissionsActive = settings.permissions.level === "full";
  const permissionScope = settings.permissions.projectPath || settings.permissions.workspaceRoot;
  const permissionScopeLabel = permissionScope
    ? permissionScope.split(/[\\/]/).filter(Boolean).pop() || permissionScope
    : null;
  const activeModelLabel = operationalStatus?.active_model || (settings.language === "es" ? "Modelo base pendiente" : "Base model pending");
  const activeEngineLabel = (operationalStatus?.engine_kind || "local").toUpperCase();
  const permissionChips = permissionsActive
    ? [
        settings.language === "es" ? "Permisos: todo" : "Permissions: full",
        settings.permissions.actionMode === "full"
          ? (settings.language === "es" ? "Acciones completas" : "Full actions")
          : (settings.language === "es" ? "Solo lectura operativa" : "Read-only ops"),
        permissionScopeLabel
          ? `Scope: ${permissionScopeLabel}`
          : (settings.language === "es" ? "Scope: sin carpeta" : "Scope: no folder"),
      ]
    : [settings.language === "es" ? "Permisos: nada" : "Permissions: none"];
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
    ? (settings.language === "es" ? "Stack local listo para trabajar." : "Local stack is ready to work.")
    : degradedChatAvailable || chatMode === "fallback_degraded"
      ? (settings.language === "es" ? "Chat degradado disponible." : "Degraded chat is available.")
      : (settings.language === "es" ? "Revisa el estado antes de empezar." : "Review the stack before you start.");
  const statusBody = stackReady
    ? (settings.language === "es"
      ? "Chat y agente listos. Memoria Obsidian disponible si esta configurada."
      : "Chat and agent are ready. Obsidian memory is used when configured.")
    : degradedChatAvailable || chatMode === "fallback_degraded"
      ? (settings.language === "es"
        ? "El chat sigue disponible mientras el runtime principal se recupera."
        : "Chat stays available while the primary runtime recovers.")
      : rawStatusReason || sendDisabledReason;
  const modeThemeStyle = (mode === "agent"
    ? (isDarkMode
      ? {
          "--primary": "272 100% 72%",
          "--ring": "272 100% 72%",
          "--accent": "274 33% 18%",
          "--accent-foreground": "210 20% 98%",
          "--ambient-core": "272 100% 72%",
          "--ambient-accent": "314 100% 72%",
          "--surface-elevated": "260 29% 12%",
          "--surface-glass": "259 26% 14%",
          "--border": "270 30% 26%",
          "--input": "262 25% 16%",
          "--ambient-shadow": "254 46% 6%",
        }
      : {
          "--primary": "270 95% 68%",
          "--ring": "270 95% 68%",
          "--accent": "278 42% 95%",
          "--accent-foreground": "278 30% 18%",
          "--ambient-core": "272 100% 70%",
          "--ambient-accent": "312 96% 71%",
          "--surface-elevated": "284 60% 98%",
          "--surface-glass": "282 44% 97%",
          "--border": "278 46% 83%",
          "--input": "278 42% 89%",
          "--ambient-shadow": "283 58% 91%",
        })
    : {
        "--primary": isDarkMode ? "203 100% 58%" : "203 92% 56%",
        "--ring": isDarkMode ? "203 100% 58%" : "203 92% 56%",
        "--ambient-core": isDarkMode ? "203 100% 58%" : "203 92% 56%",
        "--ambient-accent": isDarkMode ? "189 100% 68%" : "190 94% 66%",
      }) as React.CSSProperties;

  const activeThought = useMemo(() => {
    if (!activeThoughtMessageId || !currentSessionId) return undefined;
    return currentSession?.messages.find((message) => message.id === activeThoughtMessageId)?.thought;
  }, [activeThoughtMessageId, currentSession, currentSessionId]);

  const isCurrentThoughtStreaming = useMemo(() => {
    if (!isLoading || !currentSession || !activeThoughtMessageId) return false;
    const lastMessage = currentSession.messages[currentSession.messages.length - 1];
    return lastMessage?.id === activeThoughtMessageId;
  }, [activeThoughtMessageId, currentSession, isLoading]);

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
    document.documentElement.dataset.appMode = mode;
    document.body.dataset.appMode = mode;
    return () => {
      delete document.documentElement.dataset.appMode;
      delete document.body.dataset.appMode;
    };
  }, [mode]);

  useEffect(() => {
    if (activeView === "chat" && currentSession?.messages.length) {
      const container = mainScrollRef.current;
      if (!container) return;
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 400;
      if (isAtBottom || isLoading) {
        isAutoScrollingRef.current = true;
        container.scrollTo({ top: container.scrollHeight, behavior: isLoading ? "auto" : "smooth" });
        const timer = window.setTimeout(() => {
          isAutoScrollingRef.current = false;
        }, 200);
        return () => window.clearTimeout(timer);
      }
    }
  }, [activeView, currentSession, isLoading, isSearching, sessions]);

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

  const handleNewChat = useCallback(() => {
    newChat();
    handleSelectView("chat");
    setHeaderVisible(false);
    setFooterVisible(false);
    addLog("SYSTEM", settings.language === "es" ? "Sincronización de núcleo completada." : "Kernel sync complete.");
  }, [addLog, handleSelectView, newChat, settings.language]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    if (sessions.length <= 1) {
      handleSelectView("chat");
      setHeaderVisible(false);
      setFooterVisible(false);
    }
    deleteSession(sessionId);
  }, [deleteSession, handleSelectView, sessions.length]);

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

  const handleSendMessageLocalFirst = async (
    content: string,
    useInternet: boolean = false,
    selectedMode: AppMode = "ask",
    useThinking: boolean = true,
    autoTrain: boolean = false,
    options?: { preserveView?: boolean },
  ) => {
    if (sendDisabledReason) {
      addLog("SYSTEM", sendDisabledReason);
      return;
    }

    let targetSessionId = currentSessionId;
    let targetSession = sessions.find((session) => session.id === targetSessionId);
    if (!targetSession) {
      targetSession = createEmptySession(settings.language);
      targetSessionId = targetSession.id;
      setSessions((prev) => [targetSession!, ...prev]);
      setCurrentSessionId(targetSessionId);
    }
    if (!targetSessionId) return;

    setMode(selectedMode);
    if (!options?.preserveView && activeView !== "chat") handleSelectView("chat");
    setHeaderVisible(true);
    setFooterVisible(true);
    resetInactivityTimer();
    addLog("INFO", settings.language === "es" ? `Prompt enviado (${content.length} chars) · modo=${selectedMode}` : `Prompt sent (${content.length} chars) · mode=${selectedMode}`);
    if (useInternet) {
      addLog("SEARCH", settings.language === "es" ? "Internet activado para este prompt." : "Internet enabled for this prompt.");
    }

    const userMessage: Message = { id: Date.now().toString(), role: Role.USER, content, timestamp: Date.now() };
    const aiMessageId = (Date.now() + 1).toString();
    const initialAiMessage: Message = { id: aiMessageId, role: Role.AI, content: "", thought: "", requestId: undefined, sources: [], groundingSupports: [], timestamp: Date.now() };
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
        settings.permissions,
      );
      let started = false;
      let aborted = false;
      let lastBrowserActions: BrowserAction[] = [];

      for await (const chunk of stream) {
        if (abortControllerRef.current) {
          aborted = true;
          break;
        }
        if (!started) {
          started = true;
          addLog("INFO", settings.language === "es" ? "Stream SSE conectado." : "SSE stream connected.");
        }
        setIsSearching(false);
        if (chunk.browserActions?.length) lastBrowserActions = chunk.browserActions;
        setSessions((prev) => prev.map((session) => (
          session.id === targetSessionId
            ? {
                ...session,
                messages: session.messages.map((message) => (
                  message.id === aiMessageId
                    ? {
                        ...message,
                        content: chunk.text,
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
      }

      if (aborted) {
        addLog("SYSTEM", settings.language === "es" ? "Ejecución abortada por el usuario." : "Run aborted by user.");
      } else {
        if (lastBrowserActions.length > 0) {
          openBrowserActions(lastBrowserActions);
        }
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : (settings.language === "es" ? "Interrupción de flujo." : "Flow interrupted.");
      addLog("SYSTEM", repairMojibakeText(detail));
    } finally {
      setIsLoading(false);
      setIsSearching(false);
      resetInactivityTimer();
    }
  };

  const handleContinueResponse = useCallback((messageId: string) => {
    const session = currentSession;
    const message = session?.messages.find((item) => item.id === messageId);
    if (!message || message.role !== Role.AI) return;
    if (message.finishReason !== "length" && !isLikelyTruncatedCode(message.content)) return;
    void handleSendMessageLocalFirst(
      "Continua exactamente desde donde lo dejaste. No repitas el codigo anterior. Cierra cualquier bloque de codigo abierto.",
      false,
      mode,
      true,
      false,
      { preserveView: true },
    );
  }, [currentSession, handleSendMessageLocalFirst, mode]);

  const handleOpenModificationExplorer = useCallback((files: { path: string; diff: string }[]) => {
    setActiveModificationFiles(files);
    setHeaderVisible(false);
    setFooterVisible(false);
  }, []);

  const springConfig = { type: "spring" as const, damping: 28, stiffness: 220, mass: 0.9 };
  const direction = VIEW_INDEX[activeView] > VIEW_INDEX[prevView] ? 1 : -1;

  return (
    <div style={modeThemeStyle} className={`relative flex h-screen w-full overflow-hidden bg-background text-foreground accelerated ${mode === "agent" ? "agent-shell" : "ask-shell"}`}>
      <div className="pointer-events-none absolute inset-0">
        <div className="mode-ambient absolute inset-0 transition-all duration-500" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />
      </div>

      {isCommandPaletteOpen && (
        <Suspense fallback={null}>
          <CommandPalette
            isOpen={isCommandPaletteOpen}
            onClose={() => setIsCommandPaletteOpen(false)}
            sessions={sessions}
            currentSessionId={currentSessionId}
            onSelectSession={setCurrentSessionId}
            onNewChat={handleNewChat}
            onDeleteSession={handleDeleteSession}
            onClearHistory={handleClearHistory}
            onExportChat={() => {}}
            isDarkMode={isDarkMode}
            toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
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
              sessions={sessions}
              currentSessionId={currentSessionId}
              activeView={activeView}
              onSelectSession={setCurrentSessionId}
              onSelectView={handleSelectView}
              onNewChat={handleNewChat}
              onDeleteSession={handleDeleteSession}
              isDarkMode={isDarkMode}
              toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
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
              operationalStatus={operationalStatus}
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
                      onLoadDemo={handleLoadDemo}
                      operationalStatus={operationalStatus}
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
              <div className="pointer-events-auto">
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
                  permissionChips={permissionChips}
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
