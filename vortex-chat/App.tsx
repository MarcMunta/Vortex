
import React, { Suspense, lazy, useState, useEffect, useRef, useCallback } from 'react';
import { PanelLeft, Globe, Zap, MessageSquare, BarChart3, Terminal as TerminalIcon, FileCode, FlaskConical, Layers3 } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatInput from './components/ChatInput';
import type { SettingsTab } from './components/SettingsModal';
import VortexLogo from './components/VortexLogo';
import TopBarStackStatus from './components/TopBarStackStatus';
import VirtualizedMessageList from './components/VirtualizedMessageList';
import { BrowserAction, ChatSession, Message, Role, UserSettings, ViewType, LogEntry, AppMode, Source, Language, OperationalStatus, ControlStatus, LocalAccount, WorkspacePermissions } from './types';
import { vortexService } from './services/vortexService';
import { controlService } from './services/controlService';
import { translations } from './translations';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

const CommandPalette = lazy(() => import('./components/CommandPalette'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));
const HelpModal = lazy(() => import('./components/HelpModal'));
const ReasoningDrawer = lazy(() => import('./components/ReasoningDrawer'));
const AnalysisView = lazy(() => import('./components/AnalysisView'));
const SpatialWorkspaceView = lazy(() => import('./components/spatial/SpatialWorkspaceView'));
const TrainingView = lazy(() => import('./components/TrainingView'));
const TerminalView = lazy(() => import('./components/TerminalView'));
const SelfEditsView = lazy(() => import('./components/SelfEditsView'));
const ModificationExplorerModal = lazy(() => import('./components/ModificationExplorerModal'));

const DEFAULT_PERMISSIONS: WorkspacePermissions = {
  level: 'none',
  workspaceRoot: '',
  projectPath: '',
  actionMode: 'safe',
};

const DEFAULT_SETTINGS: UserSettings = {
  categoryOrder: ['Acciones Rápidas', 'Preferencias', 'Interfaz', 'Datos', 'Chats Recientes', 'Sistema'],
  codeTheme: 'dark',
  fontSize: 'medium',
  language: 'es',
  permissions: DEFAULT_PERMISSIONS,
};

const VIEW_INDEX: Record<ViewType, number> = { 'chat': 0, 'spatial': 1, 'analysis': 2, 'training': 3, 'edits': 4, 'terminal': 5 };

const repairMojibakeText = (value: string | null | undefined): string => {
  if (!value || !/[ÃÂ]/.test(value)) return value ?? '';
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder('utf-8').decode(bytes);
    return decoded.includes('\uFFFD') ? value : decoded;
  } catch {
    return value;
  }
};

const normalizeSession = (rawSession: unknown): ChatSession | null => {
  if (!rawSession || typeof rawSession !== 'object' || !Array.isArray((rawSession as ChatSession).messages)) {
    return null;
  }

  const session = rawSession as ChatSession;
  return {
    ...session,
    title: repairMojibakeText(session.title),
    messages: session.messages.map((message) => ({
      ...message,
      content: repairMojibakeText(message.content),
      thought: typeof message.thought === 'string' ? repairMojibakeText(message.thought) : message.thought,
    })),
  };
};

const normalizeSettings = (rawSettings: unknown): UserSettings => {
  if (!rawSettings || typeof rawSettings !== 'object') return DEFAULT_SETTINGS;

  const candidate = rawSettings as Partial<UserSettings>;
  const rawPermissions = candidate.permissions && typeof candidate.permissions === 'object'
    ? candidate.permissions as Partial<WorkspacePermissions>
    : {};
  return {
    ...DEFAULT_SETTINGS,
    ...candidate,
    categoryOrder: Array.isArray(candidate.categoryOrder)
      ? candidate.categoryOrder.map((entry) => repairMojibakeText(String(entry)))
      : DEFAULT_SETTINGS.categoryOrder,
    permissions: {
      ...DEFAULT_PERMISSIONS,
      ...rawPermissions,
      workspaceRoot: repairMojibakeText(String(rawPermissions.workspaceRoot || DEFAULT_PERMISSIONS.workspaceRoot)),
      projectPath: repairMojibakeText(String(rawPermissions.projectPath || DEFAULT_PERMISSIONS.projectPath)),
    },
  };
};

const createEmptySession = (language: Language): ChatSession => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: language === 'es' ? 'Nueva Conversación' : 'New Conversation',
  messages: [],
  updatedAt: Date.now(),
});

const createLocalAccount = (name: string, email: string, handle?: string): LocalAccount => {
  const normalizedHandle = handle?.trim()
    ? (handle.trim().startsWith('@') ? handle.trim() : `@${handle.trim()}`)
    : `@${name.toLowerCase().replace(/[^a-z0-9]+/gi, '').slice(0, 12) || 'vortex'}`;
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    email,
    handle: normalizedHandle,
    avatarHue: 198 + Math.floor(Math.random() * 24),
    createdAt: Date.now(),
    lastUsedAt: Date.now(),
  };
};

const MAX_LOCAL_SESSION_CACHE_SESSIONS = 12;
const MAX_LOCAL_SESSION_CACHE_MESSAGES = 18;

const buildSessionCache = (sessions: ChatSession[]): ChatSession[] => (
  sessions.slice(0, MAX_LOCAL_SESSION_CACHE_SESSIONS).map((session) => ({
    ...session,
    messages: session.messages.slice(-MAX_LOCAL_SESSION_CACHE_MESSAGES),
  }))
);

const createDefaultAccount = (): LocalAccount => createLocalAccount('Vortex Local', 'local@vortex.dev', '@vortex');
const accountSessionsKey = (accountId: string) => `chat-sessions:${accountId}`;
const accountSettingsKey = (accountId: string) => `user-settings:${accountId}`;

const getInitialDarkMode = (): boolean => {
  const savedMode = localStorage.getItem('dark-mode');
  if (savedMode !== null) return savedMode === 'true';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

const App: React.FC = () => {
  const [accounts, setAccounts] = useState<LocalAccount[]>([]);
  const [currentAccountId, setCurrentAccountId] = useState<string | null>(null);
  const [isAccountHydrated, setIsAccountHydrated] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewType>('chat');
  const [prevView, setPrevView] = useState<ViewType>('chat');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [selfEditsPendingCount, setSelfEditsPendingCount] = useState(0);
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<SettingsTab>('general');
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(getInitialDarkMode());
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [mode, setMode] = useState<AppMode>('ask');
  const [operationalStatus, setOperationalStatus] = useState<OperationalStatus | null>(null);
  const [controlStatus, setControlStatus] = useState<ControlStatus | null>(null);
  const [analysisFocusTab, setAnalysisFocusTab] = useState<'stack' | 'learning' | 'internet'>('stack');
  const [isComposerFocused, setIsComposerFocused] = useState(false);
  const [hasComposerDraft, setHasComposerDraft] = useState(false);
  
  const [headerVisible, setHeaderVisible] = useState(true);
  const [footerVisible, setFooterVisible] = useState(true);
  const [activeModificationFiles, setActiveModificationFiles] = useState<{ path: string, diff: string }[] | null>(null);
  
  const inactivityTimerRef = useRef<number | null>(null);
  const sessionsSyncTimerRef = useRef<number | null>(null);
  const isAutoScrollingRef = useRef<boolean>(false);
  const lastScrollYRef = useRef(0);
  
  const [isReasoningOpen, setIsReasoningOpen] = useState(false);
  const [activeThoughtMessageId, setActiveThoughtMessageId] = useState<string | null>(null);
  const abortControllerRef = useRef<boolean>(false);
  const mainScrollRef = useRef<HTMLDivElement>(null);
  const AUTO_APPLY_SELF_EDITS = false;
  
  const { scrollY } = useScroll({ container: mainScrollRef });
  const t = translations[settings.language];
  const currentAccount = accounts.find((account) => account.id === currentAccountId) || accounts[0] || null;
  const currentSession = sessions.find(s => s.id === currentSessionId);
  const hasMessages = currentSession && currentSession.messages && currentSession.messages.length > 0;
  const internetAllowlist = controlStatus?.internet?.allowlist || [];
  const stackReady = Boolean(operationalStatus?.ok);
  const rawStatusReason = operationalStatus?.chat_block_reason
    || operationalStatus?.degraded_reason
    || operationalStatus?.engine_reason
    || operationalStatus?.model_reason
    || operationalStatus?.docker_reason
    || operationalStatus?.offline_reason;
  const chatReady = Boolean(operationalStatus?.chat_ready ?? operationalStatus?.ok);
  const chatMode = operationalStatus?.chat_mode || (chatReady ? 'primary' : 'unavailable');
  const canUseInternet = Boolean(controlStatus?.ok);
  const canStartTraining = Boolean(controlStatus?.ok);
  const degradedChatAvailable = !stackReady && chatReady;
  const sendDisabledReason = chatReady
    ? undefined
    : rawStatusReason
      || (settings.language === 'es' ? 'Stack local no listo.' : 'Local stack not ready.');
  const permissionsActive = settings.permissions.level === 'full';
  const permissionScope = settings.permissions.projectPath || settings.permissions.workspaceRoot;
  const permissionScopeLabel = permissionScope
    ? permissionScope.split(/[\\/]/).filter(Boolean).pop() || permissionScope
    : null;
  const activeModelLabel = operationalStatus?.active_model || (settings.language === 'es' ? 'Modelo base pendiente' : 'Base model pending');
  const activeEngineLabel = (operationalStatus?.engine_kind || 'local').toUpperCase();
  const permissionChips = permissionsActive
    ? [
        settings.language === 'es' ? 'Permisos: todo' : 'Permissions: full',
        settings.permissions.actionMode === 'full'
          ? (settings.language === 'es' ? 'Acciones completas' : 'Full actions')
          : (settings.language === 'es' ? 'Solo lectura operativa' : 'Read-only ops'),
        permissionScopeLabel
          ? `${settings.language === 'es' ? 'Scope' : 'Scope'}: ${permissionScopeLabel}`
          : (settings.language === 'es' ? 'Scope: sin carpeta' : 'Scope: no folder'),
      ]
    : [
        settings.language === 'es' ? 'Permisos: nada' : 'Permissions: none',
      ];
  const readyLabel = stackReady
    ? (settings.language === 'es' ? 'Listo' : 'Ready')
    : degradedChatAvailable || chatMode === 'fallback_degraded'
      ? (settings.language === 'es' ? 'Degradado' : 'Degraded')
      : (settings.language === 'es' ? 'Pendiente' : 'Pending');
  const lazyPanelFallback = (
    <div className="flex h-full items-center justify-center px-6 text-xs font-black uppercase tracking-[0.14em] text-muted-foreground">
      {settings.language === 'es' ? 'Cargando vista...' : 'Loading view...'}
    </div>
  );
  const statusHeadline = stackReady
    ? (settings.language === 'es' ? 'Stack local listo para trabajar.' : 'Local stack is ready to work.')
    : degradedChatAvailable || chatMode === 'fallback_degraded'
      ? (settings.language === 'es' ? 'Chat degradado disponible.' : 'Degraded chat is available.')
      : (settings.language === 'es' ? 'Revisa el estado antes de empezar.' : 'Review the stack before you start.');
  const statusBody = stackReady
    ? (settings.language === 'es'
      ? 'Consulta, agente, navegación puntual y entrenamiento siguen visibles desde la misma interfaz.'
      : 'Query, agent mode, prompt browsing, and training stay visible from the same interface.')
    : degradedChatAvailable || chatMode === 'fallback_degraded'
      ? (settings.language === 'es'
        ? 'El chat sigue disponible mientras el runtime principal termina de recuperarse o mientras el entrenamiento usa el fallback.'
        : 'Chat stays available while the primary runtime recovers or training uses the fallback.')
      : rawStatusReason || sendDisabledReason;
  const heroCards = settings.language === 'es'
    ? [
        { label: 'Modo', value: 'Consulta y agente' },
        { label: 'Internet', value: 'Solo al activarlo' },
        { label: 'Entreno', value: 'Visible y manual' },
      ]
    : [
        { label: 'Mode', value: 'Query and agent' },
        { label: 'Internet', value: 'Only when enabled' },
        { label: 'Training', value: 'Visible and manual' },
      ];
  const modeThemeStyle = (mode === 'agent'
    ? (isDarkMode
      ? {
          '--primary': '272 100% 72%',
          '--ring': '272 100% 72%',
          '--accent': '274 33% 18%',
          '--accent-foreground': '210 20% 98%',
          '--ambient-core': '272 100% 72%',
          '--ambient-accent': '314 100% 72%',
          '--surface-elevated': '260 29% 12%',
          '--surface-glass': '259 26% 14%',
          '--border': '270 30% 26%',
          '--input': '262 25% 16%',
          '--ambient-shadow': '254 46% 6%',
        }
      : {
          '--primary': '270 95% 68%',
          '--ring': '270 95% 68%',
          '--accent': '278 42% 95%',
          '--accent-foreground': '278 30% 18%',
          '--ambient-core': '272 100% 70%',
          '--ambient-accent': '312 96% 71%',
          '--surface-elevated': '284 60% 98%',
          '--surface-glass': '282 44% 97%',
          '--border': '278 46% 83%',
          '--input': '278 42% 89%',
          '--ambient-shadow': '283 58% 91%',
        })
    : {
        '--primary': isDarkMode ? '203 100% 58%' : '203 92% 56%',
        '--ring': isDarkMode ? '203 100% 58%' : '203 92% 56%',
        '--ambient-core': isDarkMode ? '203 100% 58%' : '203 92% 56%',
        '--ambient-accent': isDarkMode ? '189 100% 68%' : '190 94% 66%',
      }) as React.CSSProperties;

  const openSettings = useCallback((tab: SettingsTab = 'general') => {
    setSettingsInitialTab(tab);
    setIsSettingsOpen(true);
  }, []);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const newLog: LogEntry = { id: Math.random().toString(36).substr(2, 9), timestamp: Date.now(), level, message };
    setLogs(prev => [...prev.slice(-149), newLog]);
  }, []);

  const activeThought = React.useMemo(() => {
    if (!activeThoughtMessageId || !currentSessionId) return undefined;
    return currentSession?.messages.find(m => m.id === activeThoughtMessageId)?.thought;
  }, [activeThoughtMessageId, currentSessionId, currentSession]);

  const isCurrentThoughtStreaming = React.useMemo(() => {
    if (!isLoading || !currentSession || !activeThoughtMessageId) return false;
    const lastMsg = currentSession.messages[currentSession.messages.length - 1];
    return lastMsg.id === activeThoughtMessageId;
  }, [isLoading, currentSession, activeThoughtMessageId]);

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
    const resp = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/accept`, { method: 'POST' });
    if (!resp.ok) return { ok: false, stage: 'accept' };
    const apply = await fetch(`/v1/self-edits/proposals/${encodeURIComponent(proposalId)}/apply`, { method: 'POST' });
    if (!apply.ok) return { ok: false, stage: 'apply' };
    return { ok: true };
  }, []);

  const suggestPatchFromMessage = useCallback(async (messageId: string, reason: string) => {
    const session = sessions.find(s => s.id === currentSessionId);
    const msg = session?.messages.find(m => m.id === messageId);
    if (!msg || !msg.content) {
      addLog('SYSTEM', settings.language === 'es' ? 'No hay contenido para sugerir parche.' : 'No content available to suggest a patch.');
      return;
    }
    const diffText = extractDiffBlocks(msg.content);
    if (!diffText) {
      addLog('SYSTEM', settings.language === 'es' ? 'No se detectó un bloque diff en la respuesta.' : 'No diff block detected in the response.');
      return;
    }
    const title = settings.language === 'es' ? 'Parche sugerido (frontend)' : 'Suggested patch (frontend)';
    const summary = settings.language === 'es' ? `Generado desde chat · ${reason}` : `Generated from chat · ${reason}`;
    const proposal = await vortexService.proposeSelfEditFromDiff(diffText, title, summary);
    if (!proposal.ok || !proposal.id) {
      addLog('SYSTEM', settings.language === 'es' ? `No se pudo crear propuesta: ${proposal.error || 'error'}` : `Failed to create proposal: ${proposal.error || 'error'}`);
      return;
    }
    addLog('LEARN', settings.language === 'es' ? `Propuesta creada: ${proposal.id}` : `Proposal created: ${proposal.id}`);
    if (AUTO_APPLY_SELF_EDITS) {
      const applyRes = await acceptAndApplyProposal(proposal.id);
      if (applyRes.ok) {
        addLog('LEARN', settings.language === 'es' ? `Parche aplicado: ${proposal.id}` : `Patch applied: ${proposal.id}`);
      } else {
        addLog('SYSTEM', settings.language === 'es' ? `No se pudo aplicar: ${proposal.id}` : `Failed to apply: ${proposal.id}`);
      }
    }
  }, [acceptAndApplyProposal, addLog, currentSessionId, extractDiffBlocks, sessions, settings.language]);

  const openBrowserActions = useCallback((actions: BrowserAction[]) => {
    if (settings.permissions.level !== 'full' || settings.permissions.actionMode !== 'full') return;
    const openedTargets = new Set<string>();
    for (const action of actions) {
      const target = repairMojibakeText(String(action?.target || '').trim());
      if (!target || openedTargets.has(target) || !/^https?:\/\//i.test(target)) continue;
      openedTargets.add(target);
      try {
        const handle = window.open(target, '_blank', 'noopener,noreferrer');
        if (handle) {
          addLog('SYSTEM', settings.language === 'es' ? `Navegador abierto: ${target}` : `Browser opened: ${target}`);
        } else {
          addLog('SYSTEM', settings.language === 'es' ? `Apertura bloqueada por el navegador: ${target}` : `Browser blocked opening: ${target}`);
        }
      } catch {
        addLog('SYSTEM', settings.language === 'es' ? `No se pudo abrir el navegador: ${target}` : `Could not open browser: ${target}`);
      }
    }
  }, [addLog, settings.language, settings.permissions.actionMode, settings.permissions.level]);

  const resetInactivityTimer = useCallback(() => {
    if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current);
    if (activeModificationFiles) return;
    if (isLoading || isSearching) { setFooterVisible(true); return; }
    if (!hasMessages && activeView === 'chat') { setFooterVisible(true); return; }
    if (isComposerFocused || hasComposerDraft) { setFooterVisible(true); return; }
    if (hasMessages) setFooterVisible(true);
    inactivityTimerRef.current = window.setTimeout(() => {
      if (activeView === 'chat') {
        setFooterVisible(false);
      }
    }, 6000);
  }, [isLoading, isSearching, activeView, hasMessages, activeModificationFiles, isComposerFocused, hasComposerDraft]);

  useEffect(() => { resetInactivityTimer(); return () => { if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current); }; }, [resetInactivityTimer]);

  useEffect(() => {
    let disposed = false;

    const pollStatus = async () => {
      const [runtimeStatus, nextControlStatus] = await Promise.all([
        vortexService.fetchOperationalStatus(),
        controlService.fetchStatus(),
      ]);
      if (disposed) return;
      setOperationalStatus(runtimeStatus);
      setControlStatus(nextControlStatus);
    };

    pollStatus();
    const timer = window.setInterval(pollStatus, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  // --- Poll backend /v1/status for REAL kernel activity ---
  useEffect(() => {
    let disposed = false;
    let prevEpisodes = -1;
    let prevRequests = -1;
    let prevKnowledge = -1;
    let prevBackendKey = '';
    let prevWebChunks = -1;
    let prevCodeChunks = -1;
    let prevAnalyses = -1;
    let prevProposals = -1;
    let prevDiscoveredUrls = -1;

    const poll = async () => {
      try {
        const resp = await fetch('/v1/status');
        if (!resp.ok || disposed) return;
        const data = await resp.json().catch(() => null);
        if (!data || disposed) return;

        const lang = settings.language;
        const backends = data.backends || [];
        const adaptersLoaded = data.adapters
          ? Object.values(data.adapters).filter(Boolean).length
          : 0;
        const metrics = data.metrics || {};
        const episodes = data.episodes || 0;
        const knowledge = data.knowledge_chunks || 0;
        const al = data.autolearn || {};

        // -- Backend & adapter status (only on CHANGE) --
        const backendKey = `${backends.join(',')}|${adaptersLoaded}`;
        if (backendKey !== prevBackendKey) {
          prevBackendKey = backendKey;
          if (adaptersLoaded > 0) {
            addLog('LEARN', lang === 'es'
              ? `Adaptadores LoRA activos: ${adaptersLoaded} en ${backends.join(', ')}.`
              : `Active LoRA adapters: ${adaptersLoaded} on ${backends.join(', ')}.`);
          } else if (backends.length > 0) {
            addLog('INFO', lang === 'es'
              ? `Backend: ${backends.join(', ')} — modelo cargado y listo.`
              : `Backend: ${backends.join(', ')} — model loaded and ready.`);
          }
        }

        // -- Episode growth --
        if (prevEpisodes >= 0 && episodes > prevEpisodes) {
          const diff = episodes - prevEpisodes;
          addLog('LEARN', lang === 'es'
            ? `+${diff} episodio${diff > 1 ? 's' : ''} registrado${diff > 1 ? 's' : ''} (total: ${episodes}).`
            : `+${diff} new episode${diff > 1 ? 's' : ''} logged (total: ${episodes}).`);
        } else if (prevEpisodes < 0 && episodes > 0) {
          addLog('INFO', lang === 'es'
            ? `Episodios almacenados: ${episodes}.`
            : `Stored episodes: ${episodes}.`);
        }
        prevEpisodes = episodes;

        // -- Knowledge chunks growth --
        if (prevKnowledge >= 0 && knowledge > prevKnowledge) {
          const diff = knowledge - prevKnowledge;
          addLog('LEARN', lang === 'es'
            ? `+${diff} chunk${diff > 1 ? 's' : ''} de conocimiento indexado${diff > 1 ? 's' : ''} (total: ${knowledge}).`
            : `+${diff} knowledge chunk${diff > 1 ? 's' : ''} indexed (total: ${knowledge}).`);
        } else if (prevKnowledge < 0 && knowledge > 0) {
          addLog('INFO', lang === 'es'
            ? `Base de conocimiento: ${knowledge} chunks indexados.`
            : `Knowledge base: ${knowledge} chunks indexed.`);
        }
        prevKnowledge = knowledge;

        // -- Autolearn: web ingest activity --
        const webChunks = al.total_web_chunks || 0;
        if (prevWebChunks >= 0 && webChunks > prevWebChunks) {
          const diff = webChunks - prevWebChunks;
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: +${diff} fragmentos web ingestados (total: ${webChunks}).`
            : `Autolearn: +${diff} web chunks ingested (total: ${webChunks}).`);
        } else if (prevWebChunks < 0 && webChunks > 0) {
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: ${webChunks} fragmentos web en base de conocimiento.`
            : `Autolearn: ${webChunks} web chunks in knowledge base.`);
        }
        prevWebChunks = webChunks;

        // -- Autolearn: code index activity --
        const codeChunks = al.total_code_chunks || 0;
        if (prevCodeChunks >= 0 && codeChunks > prevCodeChunks) {
          const diff = codeChunks - prevCodeChunks;
          addLog('LEARN', lang === 'es'
            ? `Autolearn: +${diff} fragmentos de código propio indexados.`
            : `Autolearn: +${diff} self-code chunks indexed.`);
        } else if (prevCodeChunks < 0 && codeChunks > 0) {
          addLog('LEARN', lang === 'es'
            ? `Autolearn: ${codeChunks} fragmentos de código propio indexados.`
            : `Autolearn: ${codeChunks} self-code chunks indexed.`);
        }
        prevCodeChunks = codeChunks;

        // -- Autolearn: analysis & proposals --
        const analyses = al.total_analyses || 0;
        const proposals = al.total_proposals || 0;
        if (prevAnalyses >= 0 && analyses > prevAnalyses) {
          const diff = analyses - prevAnalyses;
          addLog('LEARN', lang === 'es'
            ? `Autolearn: ${diff} archivo${diff > 1 ? 's' : ''} analizado${diff > 1 ? 's' : ''} — ${proposals} propuestas generadas.`
            : `Autolearn: ${diff} file${diff > 1 ? 's' : ''} analyzed — ${proposals} proposals generated.`);
        }
        prevAnalyses = analyses;
        if (prevProposals >= 0 && proposals > prevProposals) {
          const diff = proposals - prevProposals;
          addLog('SYSTEM', lang === 'es'
            ? `Autolearn: +${diff} propuesta${diff > 1 ? 's' : ''} de auto-mejora generada${diff > 1 ? 's' : ''}.`
            : `Autolearn: +${diff} self-improvement proposal${diff > 1 ? 's' : ''} generated.`);
        }
        prevProposals = proposals;

        // -- Autolearn: URL discovery --
        const discoveredUrls = (al.discovered_urls || []).length;
        if (prevDiscoveredUrls >= 0 && discoveredUrls > prevDiscoveredUrls) {
          const diff = discoveredUrls - prevDiscoveredUrls;
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: +${diff} URL${diff > 1 ? 's' : ''} descubierta${diff > 1 ? 's' : ''} por el modelo (total: ${discoveredUrls}).`
            : `Autolearn: +${diff} URL${diff > 1 ? 's' : ''} discovered by model (total: ${discoveredUrls}).`);
        } else if (prevDiscoveredUrls < 0 && discoveredUrls > 0) {
          addLog('SEARCH', lang === 'es'
            ? `Autolearn: ${discoveredUrls} URLs descubiertas para aprendizaje autónomo.`
            : `Autolearn: ${discoveredUrls} URLs discovered for autonomous learning.`);
        }
        prevDiscoveredUrls = discoveredUrls;

        // -- Request metrics (only show when new requests arrived) --
        if (prevRequests >= 0 && metrics.chat_requests > prevRequests) {
          const diff = metrics.chat_requests - prevRequests;
          const lat = metrics.avg_latency_ms || 0;
          const tokens = metrics.completion_tokens_est || 0;
          addLog('INFO', lang === 'es'
            ? `+${diff} petición${diff > 1 ? 'es' : ''} procesada${diff > 1 ? 's' : ''} — latencia media: ${lat}ms, tokens generados: ${tokens}.`
            : `+${diff} request${diff > 1 ? 's' : ''} processed — avg latency: ${lat}ms, tokens generated: ${tokens}.`);
        }
        prevRequests = metrics.chat_requests || 0;

      } catch { /* backend offline */ }
    };

    // Initial poll after short delay
    const t1 = setTimeout(poll, 2000);
    // Then every 20s
    const interval = setInterval(poll, 20000);
    return () => { disposed = true; clearTimeout(t1); clearInterval(interval); };
  }, [addLog, settings.language]);

  useEffect(() => {
    let disposed = false;
    const fetchPending = async () => {
      try {
        const resp = await fetch("/v1/self-edits/proposals?status=pending");
        if (!resp.ok) return;
        const payload = await resp.json().catch(() => ({}));
        const nextCount = Array.isArray(payload?.data) ? payload.data.length : 0;
        if (!disposed) setSelfEditsPendingCount(nextCount);
      } catch {
        // ignore
      }
    };

    void fetchPending();
    const interval = window.setInterval(fetchPending, 8000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, []);

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (activeModificationFiles) return;
    const container = mainScrollRef.current;
    if (!container || isAutoScrollingRef.current) return;
    const diff = latest - lastScrollYRef.current;
    lastScrollYRef.current = latest;
    if (latest < 10) { if (hasMessages) setHeaderVisible(true); return; }
    if (Math.abs(diff) < 10) return;
    if (diff > 15) setHeaderVisible(false);
    else if (diff < -20) setHeaderVisible(true);
  });

  useEffect(() => {
    const handleGlobalActivity = (e: MouseEvent) => {
      if (activeModificationFiles) return;
      if (e.clientY < 80 && lastScrollYRef.current < 10) {
        if (!headerVisible) setHeaderVisible(true);
      }
      if (e.clientY > window.innerHeight - 120) { if (!footerVisible) setFooterVisible(true); resetInactivityTimer(); }
    };
    window.addEventListener('mousemove', handleGlobalActivity);
    return () => window.removeEventListener('mousemove', handleGlobalActivity);
  }, [footerVisible, headerVisible, resetInactivityTimer, activeModificationFiles]);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      resetInactivityTimer();
      if (!footerVisible) setFooterVisible(true);
      if (e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setIsCommandPaletteOpen(prev => !prev); return; }
      if (e.key === 'Escape') {
        if (activeModificationFiles) setActiveModificationFiles(null);
        else if (isSettingsOpen) setIsSettingsOpen(false);
        else if (isCommandPaletteOpen) setIsCommandPaletteOpen(false);
        else if (isHelpOpen) setIsHelpOpen(false);
        else if (isReasoningOpen) setIsReasoningOpen(false);
        else openSettings('general');
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [resetInactivityTimer, footerVisible, isCommandPaletteOpen, isHelpOpen, isReasoningOpen, isSettingsOpen, activeModificationFiles, openSettings]);

  useEffect(() => {
    const savedAccounts = localStorage.getItem('vortex-accounts');
    const savedCurrentAccountId = localStorage.getItem('vortex-current-account-id');
    let nextAccounts: LocalAccount[] = [];

    if (savedAccounts) {
      try {
        const parsed = JSON.parse(savedAccounts);
        if (Array.isArray(parsed)) {
          nextAccounts = parsed as LocalAccount[];
        }
      } catch {
        nextAccounts = [];
      }
    }

    if (nextAccounts.length === 0) {
      nextAccounts = [createDefaultAccount()];
    }

    const safeCurrentAccountId = nextAccounts.some((account) => account.id === savedCurrentAccountId)
      ? savedCurrentAccountId
      : nextAccounts[0].id;

    setAccounts(nextAccounts);
    setCurrentAccountId(safeCurrentAccountId);
    setIsAccountHydrated(false);
  }, []);

  useEffect(() => {
    if (!currentAccountId) return;

    const savedSessions = localStorage.getItem(accountSessionsKey(currentAccountId));
    const savedSettings = localStorage.getItem(accountSettingsKey(currentAccountId));
    let disposed = false;

    const nextSettings = savedSettings
      ? (() => {
          try {
            return normalizeSettings(JSON.parse(savedSettings));
          } catch {
            return DEFAULT_SETTINGS;
          }
        })()
      : DEFAULT_SETTINGS;

    const applyCandidateSessions = (candidate: unknown, language: Language): boolean => {
      const normalizedSessions = Array.isArray(candidate)
        ? candidate
            .map((session) => normalizeSession(session))
            .filter((session): session is ChatSession => session !== null)
            .filter((session, index, all) => {
              const isEmptyDraft = session.messages.length === 0;
              if (!isEmptyDraft) return true;
              return all.findIndex((item) => item && item.title === session.title && item.messages.length === 0) === index;
            })
        : [];

      if (normalizedSessions.length > 0) {
        setSessions(normalizedSessions);
        setCurrentSessionId(normalizedSessions[0].id);
        return true;
      }

      const freshSession = createEmptySession(language);
      setSessions([freshSession]);
      setCurrentSessionId(freshSession.id);
      return false;
    };

    setSettings(nextSettings);

    let usedCache = false;
    if (savedSessions) {
      try {
        usedCache = applyCandidateSessions(JSON.parse(savedSessions), nextSettings.language);
      } catch {
        usedCache = applyCandidateSessions([], nextSettings.language);
      }
    } else {
      applyCandidateSessions([], nextSettings.language);
    }

    const finalizeHydration = () => {
      if (disposed) return;
      setAccounts((prev) => prev.map((account) => (
        account.id === currentAccountId
          ? { ...account, lastUsedAt: Date.now() }
          : account
      )));
      setIsAccountHydrated(true);
    };

    void vortexService.fetchChatSessions(currentAccountId).then((result) => {
      if (disposed) return;
      if (result.ok && Array.isArray(result.sessions) && result.sessions.length > 0) {
        applyCandidateSessions(result.sessions, nextSettings.language);
        try {
          localStorage.setItem(accountSessionsKey(currentAccountId), JSON.stringify(buildSessionCache(result.sessions)));
        } catch {
          // keep remote state authoritative
        }
      } else if (!usedCache) {
        applyCandidateSessions([], nextSettings.language);
      }
      finalizeHydration();
    }).catch(() => {
      finalizeHydration();
    });

    return () => {
      disposed = true;
    };
  }, [currentAccountId]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.body.classList.toggle('dark', isDarkMode);
    document.documentElement.style.colorScheme = isDarkMode ? 'dark' : 'light';
    document.body.style.colorScheme = isDarkMode ? 'dark' : 'light';
    localStorage.setItem('dark-mode', String(isDarkMode));
  }, [isDarkMode]);
  useEffect(() => {
    document.documentElement.dataset.appMode = mode;
    document.body.dataset.appMode = mode;
    return () => {
      delete document.documentElement.dataset.appMode;
      delete document.body.dataset.appMode;
    };
  }, [mode]);
  useEffect(() => { localStorage.setItem('vortex-accounts', JSON.stringify(accounts)); }, [accounts]);
  useEffect(() => {
    if (currentAccountId) localStorage.setItem('vortex-current-account-id', currentAccountId);
  }, [currentAccountId]);
  useEffect(() => {
    if (isAccountHydrated && currentAccountId) {
      try {
        localStorage.setItem(accountSessionsKey(currentAccountId), JSON.stringify(buildSessionCache(sessions)));
      } catch {
        // Backend persistence remains the source of truth when cache is full.
      }
    }
  }, [currentAccountId, isAccountHydrated, sessions]);
  useEffect(() => {
    if (isAccountHydrated && currentAccountId) {
      localStorage.setItem(accountSettingsKey(currentAccountId), JSON.stringify(settings));
    }
  }, [currentAccountId, isAccountHydrated, settings]);
  useEffect(() => {
    if (!isAccountHydrated || !currentAccountId) return;
    if (sessionsSyncTimerRef.current) {
      window.clearTimeout(sessionsSyncTimerRef.current);
      sessionsSyncTimerRef.current = null;
    }
    if (isLoading) return;
    sessionsSyncTimerRef.current = window.setTimeout(() => {
      void vortexService.syncChatSessions(currentAccountId, sessions);
    }, 600);
    return () => {
      if (sessionsSyncTimerRef.current) {
        window.clearTimeout(sessionsSyncTimerRef.current);
        sessionsSyncTimerRef.current = null;
      }
    };
  }, [currentAccountId, isAccountHydrated, isLoading, sessions]);
  useEffect(() => {
    if (sessions.length === 0) {
      setCurrentSessionId(null);
      return;
    }
    if (!currentSessionId || !sessions.some(session => session.id === currentSessionId)) {
      setCurrentSessionId(sessions[0].id);
    }
  }, [sessions, currentSessionId]);

  useEffect(() => {
    if (activeView === 'chat' && currentSession?.messages.length) {
      const container = mainScrollRef.current;
      if (!container) return;
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 400;
      if (isAtBottom || isLoading) {
        isAutoScrollingRef.current = true;
        container.scrollTo({ top: container.scrollHeight, behavior: isLoading ? 'auto' : 'smooth' });
        const timer = setTimeout(() => { isAutoScrollingRef.current = false; }, 200);
        return () => clearTimeout(timer);
      }
    }
  }, [sessions, isLoading, isSearching, activeView, currentSessionId, currentSession]);

  const handleSelectAccount = useCallback((accountId: string) => {
    if (accountId === currentAccountId) return;
    setIsAccountHydrated(false);
    setCurrentAccountId(accountId);
    setHeaderVisible(true);
    setFooterVisible(true);
  }, [currentAccountId]);

  const handleCreateAccount = useCallback((draft: { name: string; email: string; handle?: string }) => {
    const nextAccount = createLocalAccount(draft.name.trim(), draft.email.trim(), draft.handle?.trim());
    const freshSettings = { ...DEFAULT_SETTINGS, language: settings.language };
    const freshSession = createEmptySession(freshSettings.language);

    localStorage.setItem(accountSettingsKey(nextAccount.id), JSON.stringify(freshSettings));
    localStorage.setItem(accountSessionsKey(nextAccount.id), JSON.stringify(buildSessionCache([freshSession])));

    setAccounts((prev) => [nextAccount, ...prev]);
    setIsAccountHydrated(false);
    setCurrentAccountId(nextAccount.id);
    setSessions([freshSession]);
    setCurrentSessionId(freshSession.id);
    setHeaderVisible(true);
    setFooterVisible(true);
  }, [settings.language]);

  const handleNewChat = useCallback(() => {
    const newSession = createEmptySession(settings.language);
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    handleSelectView('chat');
    setHeaderVisible(false); setFooterVisible(false);
    addLog('SYSTEM', settings.language === 'es' ? 'Sincronización de núcleo completada.' : 'Kernel sync complete.');
  }, [addLog, settings.language]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    const remainingSessions = sessions.filter(session => session.id !== sessionId);
    if (remainingSessions.length === 0) {
      const replacementSession = createEmptySession(settings.language);
      setSessions([replacementSession]);
      setCurrentSessionId(replacementSession.id);
      setActiveView(prev => { setPrevView(prev); return 'chat'; });
      setHeaderVisible(false);
      setFooterVisible(false);
      return;
    }
    setSessions(remainingSessions);
    if (!remainingSessions.some(session => session.id === currentSessionId)) {
      setCurrentSessionId(remainingSessions[0].id);
    }
  }, [sessions, currentSessionId, settings.language]);

  const handleClearHistory = useCallback(() => {
    const freshSession = createEmptySession(settings.language);
    setSessions([freshSession]);
    setCurrentSessionId(freshSession.id);
    setActiveView(prev => { setPrevView(prev); return 'chat'; });
    setHeaderVisible(false);
    setFooterVisible(false);
    addLog('SYSTEM', settings.language === 'es' ? 'Historial purgado. Conversación reiniciada.' : 'History cleared. Conversation reset.');
  }, [addLog, settings.language]);

  const handleSelectView = useCallback((newView: ViewType) => {
    if (newView === 'analysis') setAnalysisFocusTab('stack');
    setActiveView(prev => { setPrevView(prev); return newView; });
    setHeaderVisible(true);
  }, []);

  const openAnalysisTab = useCallback((tab: 'stack' | 'learning' | 'internet') => {
    setAnalysisFocusTab(tab);
    setActiveView(prev => { setPrevView(prev); return 'analysis'; });
    setHeaderVisible(true);
  }, []);

  const openTrainingView = useCallback(() => {
    setActiveView(prev => { setPrevView(prev); return 'training'; });
    setHeaderVisible(true);
  }, []);

  const handleNavigateToChat = useCallback((sessionId: string, messageId?: string) => {
    setCurrentSessionId(sessionId); setActiveView('chat');
    if (messageId) { setTimeout(() => { document.getElementById(messageId)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 300); }
  }, []);

  const handleLoadDemo = useCallback(() => {
    if (!currentSessionId) return;
    const mockSources: Source[] = [
      { url: 'https://react.dev', title: t.analysis_library + ': React Hooks', domain: 'react.dev', kind: 'web', index: 0 },
      { url: 'https://fastapi.tiangolo.com', title: t.analysis_library + ': FastAPI', domain: 'fastapi.tiangolo.com', kind: 'web', index: 1 },
      { url: 'https://developer.mozilla.org/es/docs/Web/API/Element/scrollIntoView', title: 'MDN: Element.scrollIntoView()', domain: 'developer.mozilla.org', kind: 'web', index: 2 }
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
      { id: 'demo-1', role: Role.USER, content: settings.language === 'es' ? "Activar protocolo de demostración." : "Activate demo protocol.", timestamp: Date.now() - 60000 },
      { 
        id: 'demo-2', 
        role: Role.AI, 
        content: settings.language === 'es' ? demoContentEs : demoContentEn, 
        thought: settings.language === 'es' ? "Análisis completado. Se han identificado cuellos de botella en la renderización y se han ajustado las físicas del sidebar para una respuesta táctil superior." : "Analysis complete. Rendering bottlenecks identified and sidebar physics adjusted for superior tactile response.", 
        sources: mockSources, 
        fileChanges: [{ path: 'App.tsx', diff: `- const timer = 100;\n+ const timer = 60;` }, { path: 'components/Sidebar.tsx', diff: `- stiffness: 400;\n+ stiffness: 500;` }], 
        timestamp: Date.now() - 30000 
      }
    ];
    setSessions(prev => prev.map(s => s.id === currentSessionId ? { ...s, messages: demoMessages, updatedAt: Date.now() } : s));
    setHeaderVisible(true); setFooterVisible(true);
    addLog('SYSTEM', settings.language === 'es' ? 'Carga de demostración completada.' : 'Demo load complete.');
  }, [currentSessionId, addLog, settings.language, t]);

  const handleSendMessageLocalFirst = async (
    content: string,
    useInternet: boolean = false,
    selectedMode: AppMode = 'ask',
    useThinking: boolean = true,
    autoTrain: boolean = true,
    options?: { preserveView?: boolean },
  ) => {
    if (sendDisabledReason) {
      addLog('SYSTEM', sendDisabledReason);
      return;
    }
    let targetSessionId = currentSessionId;
    let targetSession = sessions.find(session => session.id === targetSessionId);
    if (!targetSession) {
      targetSession = createEmptySession(settings.language);
      targetSessionId = targetSession.id;
      setSessions(prev => [targetSession!, ...prev]);
      setCurrentSessionId(targetSessionId);
    }
    if (!targetSessionId) return;

    setMode(selectedMode);
    if (!options?.preserveView && activeView !== 'chat') handleSelectView('chat');
    setHeaderVisible(true);
    setFooterVisible(true);
    resetInactivityTimer();
    addLog('INFO', settings.language === 'es' ? `Prompt enviado (${content.length} chars) · modo=${selectedMode}` : `Prompt sent (${content.length} chars) · mode=${selectedMode}`);
    if (useInternet) {
      addLog('SEARCH', settings.language === 'es' ? 'Internet activado para este prompt.' : 'Internet enabled for this prompt.');
    }

    const userMessage: Message = { id: Date.now().toString(), role: Role.USER, content, timestamp: Date.now() };
    const aiMessageId = (Date.now() + 1).toString();
    const initialAiMessage: Message = { id: aiMessageId, role: Role.AI, content: "", thought: "", requestId: undefined, sources: [], groundingSupports: [], timestamp: Date.now() };
    setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: [...s.messages, userMessage, initialAiMessage], updatedAt: Date.now() } : s));

    const currentMessages = targetSession.messages || [];
    if (currentMessages.length === 0) {
      vortexService.generateChatTitle(content, settings.language).then(result => {
        if (result.ok && result.title) {
          const normalizedTitle = repairMojibakeText(result.title);
          setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, title: normalizedTitle } : s));
        }
      }).catch(() => {});
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
        settings.permissions
      );
      let started = false;
      let aborted = false;
      let lastText = '';
      let lastRequestId: string | undefined;
      let lastBrowserActions: BrowserAction[] = [];
      for await (const chunk of stream) {
        if (abortControllerRef.current) {
          aborted = true;
          break;
        }
        if (!started) {
          started = true;
          addLog('INFO', settings.language === 'es' ? 'Stream SSE conectado.' : 'SSE stream connected.');
        }
        setIsSearching(false);
        lastText = chunk.text || lastText;
        if (chunk.requestId) lastRequestId = chunk.requestId;
        if (chunk.browserActions?.length) lastBrowserActions = chunk.browserActions;
        setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: s.messages.map(m => m.id === aiMessageId ? { ...m, content: chunk.text, thought: chunk.thought || m.thought, requestId: chunk.requestId || m.requestId, sources: chunk.sources.length > 0 ? chunk.sources : m.sources, fileChanges: chunk.fileChanges || m.fileChanges } : m) } : s));
      }
      if (aborted) {
        addLog('SYSTEM', settings.language === 'es' ? 'Ejecución abortada por el usuario.' : 'Run aborted by user.');
      } else {
        if (lastBrowserActions.length > 0) {
          openBrowserActions(lastBrowserActions);
        }
        if (autoTrain && lastRequestId && lastText) {
          addLog('LEARN', settings.language === 'es' ? 'Aprendizaje: encolando respuesta para curaciÃ³n...' : 'Learning: queueing answer for curation...');
          const feedback = await vortexService.submitFeedback(lastRequestId, lastText);
          if (feedback.ok && feedback.trainingEvent) {
            const learningStatus = feedback.quickTrainScheduled
              ? 'scheduled'
              : (feedback.learningQueueItem?.status || 'queued');
            setSessions(prev => prev.map(s => s.id === targetSessionId ? {
              ...s,
              messages: s.messages.map(m => m.id === aiMessageId ? {
                ...m,
                trainingEvent: true,
                learningStatus,
                learningQueueId: feedback.learningQueueItem?.id,
                learningRunId: feedback.scheduledRunId || undefined,
              } : m),
            } : s));
            addLog(
              'LEARN',
              settings.language === 'es'
                ? `Respuesta en cola (${feedback.learningQueueDepth || 0}). Motivo: ${feedback.queueReason || 'queued'}.`
                : `Answer queued (${feedback.learningQueueDepth || 0}). Reason: ${feedback.queueReason || 'queued'}.`,
            );
            if (feedback.quickTrainScheduled && feedback.scheduledRunId) {
              addLog('LEARN', settings.language === 'es' ? `Quick learning programado: ${feedback.scheduledRunId}` : `Quick learning scheduled: ${feedback.scheduledRunId}`);
            }
            await suggestPatchFromMessage(aiMessageId, 'auto-train');
            return;
          }
          if (feedback.ok) {
            addLog('SYSTEM', settings.language === 'es' ? 'Aprendizaje no encolado: falta muestra curable para training.' : 'Learning not queued: no curatable sample available for training.');
            return;
          }
          addLog('SYSTEM', settings.language === 'es' ? `Aprendizaje fallÃ³: ${feedback.error || 'error'}` : `Learning failed: ${feedback.error || 'error'}`);
          return;
        } else if (autoTrain) {
          addLog('SYSTEM', settings.language === 'es' ? 'Aprendizaje omitido: request_id ausente.' : 'Learning skipped: missing request_id.');
        }
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : (settings.language === 'es' ? 'Interrupción de flujo.' : 'Flow interrupted.');
      addLog('SYSTEM', repairMojibakeText(detail));
    } finally {
      setIsLoading(false);
      setIsSearching(false);
      resetInactivityTimer();
    }
  };

  const handleOpenModificationExplorer = (files: { path: string, diff: string }[]) => {
    setActiveModificationFiles(files);
    setHeaderVisible(false);
    setFooterVisible(false);
  };

  const springConfig = { type: 'spring' as const, damping: 28, stiffness: 220, mass: 0.9 };
  const direction = VIEW_INDEX[activeView] > VIEW_INDEX[prevView] ? 1 : -1;

  return (
    <div style={modeThemeStyle} className={`relative flex h-screen w-full overflow-hidden bg-background text-foreground accelerated ${mode === 'agent' ? 'agent-shell' : 'ask-shell'}`}>
      <div className="pointer-events-none absolute inset-0">
        <div className="mode-ambient absolute inset-0 transition-all duration-500" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />
      </div>
      {isCommandPaletteOpen && (
        <Suspense fallback={null}>
          <CommandPalette isOpen={isCommandPaletteOpen} onClose={() => setIsCommandPaletteOpen(false)} sessions={sessions} currentSessionId={currentSessionId} onSelectSession={setCurrentSessionId} onNewChat={handleNewChat} onDeleteSession={handleDeleteSession} onClearHistory={handleClearHistory} onExportChat={() => {}} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} isSidebarOpen={isSidebarOpen} onToggleSidebar={() => { const next = !isSidebarOpen; setIsSidebarOpen(next); if (next) setIsReasoningOpen(false); }} onOpenSettings={() => openSettings('general')} onOpenHelp={() => setIsHelpOpen(true)} categoryOrder={settings.categoryOrder} language={settings.language} onSetFontSize={(size) => setSettings({ ...settings, fontSize: size })} />
        </Suspense>
      )}
      <AnimatePresence initial={false}>{isSidebarOpen && !activeModificationFiles && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 280, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full overflow-hidden shrink-0 z-50 flex border-r border-border/50 shadow-2xl relative"><Sidebar sessions={sessions} currentSessionId={currentSessionId} activeView={activeView} onSelectSession={setCurrentSessionId} onSelectView={handleSelectView} onNewChat={handleNewChat} onDeleteSession={handleDeleteSession} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} onClose={() => setIsSidebarOpen(false)} onOpenSettings={openSettings} isOpen={true} language={settings.language} selfEditsPendingCount={selfEditsPendingCount} currentAccount={currentAccount} accounts={accounts} currentAccountId={currentAccountId} onSelectAccount={handleSelectAccount} /></motion.div>
      )}</AnimatePresence>
      <div className="flex-1 flex overflow-hidden relative">
        <main className="flex-1 flex flex-col h-full bg-background relative z-0 overflow-hidden">
          {!activeModificationFiles && (
            <motion.header initial={false} animate={{ y: headerVisible ? 0 : -100, opacity: headerVisible ? 1 : 0 }} transition={springConfig} className="absolute top-0 left-0 right-0 z-40 flex h-[72px] items-center justify-between border-b border-border/60 bg-background/88 px-5 backdrop-blur-xl pointer-events-auto accelerated lg:px-8">
              <div className="flex items-center gap-8">
                <AnimatePresence mode="wait">{!isSidebarOpen && (<motion.button initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.8, opacity: 0 }} whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--muted-foreground) / 0.1)' }} whileTap={{ scale: 0.9 }} onClick={() => { setIsSidebarOpen(true); setIsReasoningOpen(false); }} className="p-3.5 rounded-2xl transition-all"><PanelLeft size={24} /></motion.button>)}</AnimatePresence>
                <div className="flex items-center gap-4">
                  <motion.div whileHover={{ rotate: -8, scale: 1.04 }} transition={{ type: 'spring', stiffness: 320, damping: 18 }}>
                    <VortexLogo size={40} alt="Vortex" />
                  </motion.div>
                  <div className="flex flex-col">
                    <div className="flex items-center gap-3">
                      <h1 className="text-[18px] font-black tracking-tight leading-none">Vortex</h1>
                      <span className="rounded-full border border-border/60 bg-muted/25 px-3 py-1 text-[9px] font-black uppercase tracking-[0.14em] text-primary">
                        {activeEngineLabel}
                      </span>
                    </div>
                    <span className="mt-1 text-[9px] font-black uppercase tracking-[0.14em] text-muted-foreground">{t.system_kernel}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <motion.button whileHover={{ scale: 1.06, backgroundColor: 'hsla(var(--muted) / 0.8)' }} whileTap={{ scale: 0.94 }} onClick={() => setSettings({ ...settings, language: settings.language === 'es' ? 'en' : 'es' })} className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-full border border-border/60 bg-muted/20 transition-all shadow-sm"><img src={settings.language === 'es' ? 'https://flagcdn.com/w80/es.png' : 'https://flagcdn.com/w80/us.png'} alt={settings.language} className="h-6 w-6 rounded-full object-cover select-none" /></motion.button>
                <div className="flex items-center gap-1 rounded-[1rem] border border-border/60 bg-muted/20 p-1 relative">{(['chat', 'spatial', 'analysis', 'training', 'edits', 'terminal'] as ViewType[]).map(v => (<button key={v} onClick={() => handleSelectView(v)} className={`relative rounded-[0.85rem] p-2.5 transition-all z-10 ${activeView === v ? 'text-primary-foreground' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>{v === 'chat' ? <MessageSquare size={16} /> : v === 'spatial' ? <Layers3 size={16} /> : v === 'analysis' ? <BarChart3 size={16} /> : v === 'training' ? <FlaskConical size={16} /> : v === 'edits' ? <FileCode size={16} /> : <TerminalIcon size={16} />}{activeView === v && <motion.div layoutId="header-nav-indicator" className="absolute inset-0 bg-primary rounded-[0.85rem] -z-10" transition={springConfig} />}</button>))}</div>
                <TopBarStackStatus
                  status={operationalStatus}
                  controlStatus={controlStatus}
                  language={settings.language}
                  onBootstrap={() => controlService.bootstrap(false)}
                  onModelInit={() => controlService.initModel()}
                  onRestartRuntime={() => controlService.restartRuntime()}
                  onStartTraining={() => controlService.startTraining('quick')}
                  onOpenTraining={openTrainingView}
                  onStartAutonomy={() => controlService.startAutonomy()}
                  onStopAutonomy={() => controlService.stopAutonomy()}
                />
                <motion.button whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.95 }} onClick={() => setIsCommandPaletteOpen(true)} className="flex items-center gap-3 rounded-[1rem] border border-border/60 bg-muted/20 px-4 py-2.5 transition-all hover:bg-background"><Zap size={16} className={'text-primary'} /><kbd className="hidden lg:inline-block rounded-lg border bg-background px-2 py-0.5 text-[8px] font-black opacity-40">ALT+K</kbd></motion.button>
              </div>
            </motion.header>
          )}

          <div ref={mainScrollRef} className="flex-1 overflow-y-auto custom-scrollbar flex flex-col relative h-full bg-background scroll-smooth accelerated">
            {hasMessages && !activeModificationFiles && <div className="pt-24 shrink-0" />}
            <AnimatePresence mode="popLayout" custom={direction}>
              {activeView === 'chat' && (
                <motion.div key="chat" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className={`mx-auto flex min-h-full w-full flex-1 flex-col px-4 md:px-8 lg:px-10 xl:px-12 transition-all duration-500 ${!hasMessages ? 'justify-center pt-20 pb-36 max-w-[1120px]' : 'pt-4 max-w-[1260px]'}`}>
                  {!hasMessages ? (
                    <div className="mx-auto flex w-full max-w-[980px] flex-col items-center justify-center gap-8 text-center">
                      <div className="flex flex-col items-center gap-5">
                        <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground shadow-sm">
                          <VortexLogo size={20} alt="Vortex" />
                          <span>{settings.language === 'es' ? 'Vortex local core' : 'Vortex local core'}</span>
                          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                          <span className="text-primary">{activeEngineLabel}</span>
                        </div>

                        <div className="space-y-4">
                          <h2 className="max-w-3xl text-4xl font-extrabold tracking-[-0.045em] leading-[1.02] text-foreground lg:text-5xl">
                            {settings.language === 'es'
                              ? 'Una sola consola para chatear, controlar y mejorar Vortex.'
                              : 'One console to chat, control, and improve Vortex.'}
                          </h2>
                          <p className="mx-auto max-w-2xl text-[15px] leading-7 text-muted-foreground lg:text-base">
                            {settings.language === 'es'
                              ? 'La interfaz principal se comporta como una app de trabajo real: limpia, local y centrada en conversación, control del runtime y entrenamiento visible.'
                              : 'The main interface behaves like a real work app: clean, local, and focused on conversation, runtime control, and visible training.'}
                          </p>
                        </div>

                        <div className="grid w-full gap-3 md:grid-cols-3">
                          {heroCards.map((card) => (
                            <div key={card.label} className="surface-panel rounded-[1.2rem] px-5 py-5 text-left">
                              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">{card.label}</p>
                              <p className="mt-2 text-[15px] font-bold tracking-tight text-foreground">{card.value}</p>
                            </div>
                          ))}
                        </div>

                        <div className="flex flex-wrap items-center justify-center gap-4">
                          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={handleLoadDemo} className="rounded-full bg-foreground px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-background transition-all dark:bg-primary dark:text-primary-foreground">
                            {t.initialize_vortex}
                          </motion.button>
                          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={() => handleSelectView('analysis')} className="rounded-full border border-border/70 bg-background px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-foreground shadow-sm">
                            {settings.language === 'es' ? 'Abrir control' : 'Open control'}
                          </motion.button>
                          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={openTrainingView} className="rounded-full border border-primary/25 bg-primary/[0.10] px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-primary shadow-sm">
                            {settings.language === 'es' ? 'Entrenamiento' : 'Training'}
                          </motion.button>
                        </div>

                        <p className="max-w-2xl text-sm font-medium text-muted-foreground">{statusBody}</p>
                      </div>

                      <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }} className="surface-panel relative isolate w-full max-w-[820px] overflow-hidden rounded-[1.6rem] p-5 text-foreground">
                        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.10),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />

                        <div className="relative z-10 flex flex-col gap-5">
                          <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">
                            <span>{settings.language === 'es' ? 'Estado base' : 'Base status'}</span>
                            <span>{readyLabel}</span>
                          </div>

                          <div className="flex items-center gap-4 rounded-[1.2rem] border border-border/60 bg-muted/20 px-4 py-4">
                            <motion.div animate={{ rotate: [0, 2, -2, 0], scale: [1, 1.01, 1] }} transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}>
                              <VortexLogo size={52} alt="Vortex mark" className="max-w-full" />
                            </motion.div>
                            <div className="text-left">
                              <p className="text-sm font-bold tracking-tight text-foreground">
                                {statusHeadline}
                              </p>
                              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                                {operationalStatus?.ok
                                  ? (settings.language === 'es'
                                    ? 'Consulta, agente, navegación puntual y entrenamiento siguen visibles desde la misma interfaz.'
                                    : 'Query, agent mode, prompt browsing, and training stay visible from the same interface.')
                                  : sendDisabledReason}
                              </p>
                            </div>
                          </div>

                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-[1rem] border border-border/60 bg-muted/20 p-4">
                              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                                {settings.language === 'es' ? 'Runtime' : 'Runtime'}
                              </p>
                              <p className="mt-2 text-sm font-bold tracking-tight text-foreground">{activeEngineLabel}</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {operationalStatus?.engine_base_url || '127.0.0.1'}
                              </p>
                            </div>
                            <div className="rounded-[1rem] border border-border/60 bg-muted/20 p-4">
                              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                                {settings.language === 'es' ? 'Modelo activo' : 'Active model'}
                              </p>
                              <p className="mt-2 text-sm font-bold tracking-tight text-foreground break-words">{activeModelLabel}</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {operationalStatus?.web_disabled
                                  ? (settings.language === 'es' ? 'Internet solo al activarlo' : 'Internet only when enabled')
                                  : (settings.language === 'es' ? 'Política web editable' : 'Editable web policy')}
                              </p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </div>
                  ) : (
                    <div className="pb-36">
                      <VirtualizedMessageList 
                        messages={currentSession.messages} 
                        fontSize={settings.fontSize} 
                        codeTheme={settings.codeTheme}
                        onShowReasoning={messageId => { setActiveThoughtMessageId(messageId); setIsReasoningOpen(true); setIsSidebarOpen(false); }} 
                        onOpenModificationExplorer={handleOpenModificationExplorer} 
                        onSuggestPatch={messageId => { void suggestPatchFromMessage(messageId, 'manual'); }}
                        isLoading={isLoading} 
                        language={settings.language} 
                        containerRef={mainScrollRef} 
                      />
                      {isSearching && (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card mt-6 flex w-fit items-center gap-3 rounded-full border border-primary/20 px-4 py-2.5 text-primary accelerated"><Globe size={16} className="animate-spin-slow" /><p className="text-[10px] font-black uppercase tracking-[0.14em]">{settings.language === 'es' ? 'Internet activo en este prompt' : 'Internet enabled on this prompt'}</p></motion.div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
              {activeView === 'spatial' && <motion.div key="spatial" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><Suspense fallback={lazyPanelFallback}><SpatialWorkspaceView language={settings.language} controlStatus={controlStatus} onAddLog={addLog} onSendPrompt={handleSendMessageLocalFirst} /></Suspense></motion.div>}
              {activeView === 'analysis' && <motion.div key="analysis" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><Suspense fallback={lazyPanelFallback}><AnalysisView sessions={sessions} onNavigateToChat={handleNavigateToChat} onAddLog={addLog} language={settings.language} controlStatus={controlStatus} operationalStatus={operationalStatus} onBootstrap={() => controlService.bootstrap(false)} onModelInit={() => controlService.initModel()} onRestartRuntime={() => controlService.restartRuntime()} onReloadInstructions={() => controlService.reloadInstructions()} onStartTraining={(trainingMode) => controlService.startTraining(trainingMode)} onSaveAllowlist={(domains) => controlService.saveAllowlist(domains)} focusTab={analysisFocusTab} /></Suspense></motion.div>}
              {activeView === 'training' && <motion.div key="training" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><Suspense fallback={lazyPanelFallback}><TrainingView sessions={sessions} language={settings.language} controlStatus={controlStatus} onAddLog={addLog} onStartTraining={(trainingMode) => controlService.startTraining(trainingMode)} onStartAutonomy={() => controlService.startAutonomy()} onStopAutonomy={() => controlService.stopAutonomy()} onConfigureAutonomy={(config) => controlService.configureAutonomy(config)} /></Suspense></motion.div>}
              {activeView === 'edits' && <motion.div key="edits" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><Suspense fallback={lazyPanelFallback}><SelfEditsView language={settings.language} onAddLog={addLog} onPendingCountChange={setSelfEditsPendingCount} /></Suspense></motion.div>}
              {activeView === 'terminal' && <motion.div key="terminal" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><Suspense fallback={lazyPanelFallback}><TerminalView logs={logs} onClear={() => setLogs([])} language={settings.language} /></Suspense></motion.div>}
            </AnimatePresence>
          </div>

          {!activeModificationFiles && activeView === 'chat' && (
            <motion.div initial={false} animate={{ y: footerVisible ? 0 : 200, opacity: footerVisible ? 1 : 0 }} transition={{ type: 'spring', damping: 30, stiffness: 200 }} className="absolute bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-background via-background/95 to-transparent pt-6 pb-6 pointer-events-auto accelerated">
              <div className="pointer-events-auto">
                <ChatInput
                  onSend={handleSendMessageLocalFirst}
                  isLoading={isLoading}
                  isDarkMode={isDarkMode}
                  mode={mode}
                  onModeChange={setMode}
                  canUseInternet={canUseInternet}
                  allowAutoTrain={canStartTraining}
                  sendDisabledReason={sendDisabledReason}
                  onStop={() => { abortControllerRef.current = true; }}
                  language={settings.language}
                  permissionChips={permissionChips}
                  onInteraction={() => { resetInactivityTimer(); if (!footerVisible) setFooterVisible(true); }}
                  onFocusChange={setIsComposerFocused}
                  onDraftChange={setHasComposerDraft}
                />
              </div>
            </motion.div>
          )}
        </main>
        
        <AnimatePresence>{isReasoningOpen && !activeModificationFiles && (
            <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 400, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full border-l border-border/50 shrink-0 z-50 overflow-hidden bg-zinc-950/95 shadow-[-20px_0_50px_rgba(0,0,0,0.5)]"><Suspense fallback={null}><ReasoningDrawer isOpen={isReasoningOpen} onClose={() => setIsReasoningOpen(false)} thought={activeThought} language={settings.language} isStreaming={isCurrentThoughtStreaming} /></Suspense></motion.div>
        )}</AnimatePresence>
      </div>

      <AnimatePresence>
        {activeModificationFiles && (
          <Suspense fallback={null}>
            <ModificationExplorerModal 
              fileChanges={activeModificationFiles} 
              onClose={() => { setActiveModificationFiles(null); setHeaderVisible(true); setFooterVisible(true); }} 
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
