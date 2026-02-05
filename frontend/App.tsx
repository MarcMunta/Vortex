
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { PanelLeft, Bot, Sparkles, Globe, Zap, MessageSquare, BarChart3, Terminal as TerminalIcon } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatInput from './components/ChatInput';
import CommandPalette from './components/CommandPalette';
import SettingsModal from './components/SettingsModal';
import HelpModal from './components/HelpModal';
import ReasoningDrawer from './components/ReasoningDrawer';
import AnalysisView from './components/AnalysisView';
import TerminalView from './components/TerminalView';
import VirtualizedMessageList from './components/VirtualizedMessageList';
import ModificationExplorerModal from './components/ModificationExplorerModal';
import { ChatSession, Message, Role, UserSettings, ViewType, LogEntry, AppMode, Source, Language } from './types';
import { klimeaiService } from './services/klimeaiService';
import { translations } from './translations';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

const _readNumberEnv = (key: string, fallback: number) => {
  const raw = (import.meta as any).env?.[key];
  const val = typeof raw === 'string' ? Number(raw) : Number(raw);
  return Number.isFinite(val) ? val : fallback;
};

const DEFAULT_SETTINGS: UserSettings = {
  categoryOrder: ['Acciones Rápidas', 'Preferencias', 'Interfaz', 'Datos', 'Chats Recientes', 'Sistema'],
  codeTheme: 'dark',
  fontSize: 'medium',
  language: 'es',
  llm: {
    baseUrl: ((import.meta as any).env?.VITE_API_BASE_URL as string | undefined) ?? '',
    token: ((import.meta as any).env?.VITE_API_TOKEN as string | undefined) ?? '',
    model: ((import.meta as any).env?.VITE_DEFAULT_MODEL as string | undefined) ?? 'core',
    temperature: _readNumberEnv('VITE_DEFAULT_TEMPERATURE', 0.7),
    topP: _readNumberEnv('VITE_DEFAULT_TOP_P', 1.0),
    maxTokens: Math.floor(_readNumberEnv('VITE_DEFAULT_MAX_TOKENS', 2048)),
  }
};

const VIEW_INDEX: Record<ViewType, number> = { 'chat': 0, 'analysis': 1, 'terminal': 2 };

const getInitialDarkMode = (): boolean => {
  const savedMode = localStorage.getItem('dark-mode');
  if (savedMode !== null) return savedMode === 'true';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

const App: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewType>('chat');
  const [prevView, setPrevView] = useState<ViewType>('chat');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(getInitialDarkMode());
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [mode, setMode] = useState<AppMode>('ask');
  
  const [headerVisible, setHeaderVisible] = useState(false);
  const [footerVisible, setFooterVisible] = useState(false);
  const [activeModificationFiles, setActiveModificationFiles] = useState<{ path: string, diff: string }[] | null>(null);
  
  const inactivityTimerRef = useRef<number | null>(null);
  const isAutoScrollingRef = useRef<boolean>(false);
  const lastScrollYRef = useRef(0);
  
  const [isReasoningOpen, setIsReasoningOpen] = useState(false);
  const [activeThoughtMessageId, setActiveThoughtMessageId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const mainScrollRef = useRef<HTMLDivElement>(null);
  
  const { scrollY } = useScroll({ container: mainScrollRef });
  const t = translations[settings.language];
  const currentSession = sessions.find(s => s.id === currentSessionId);
  const hasMessages = currentSession && currentSession.messages && currentSession.messages.length > 0;

  const activeThought = React.useMemo(() => {
    if (!activeThoughtMessageId || !currentSessionId) return undefined;
    return currentSession?.messages.find(m => m.id === activeThoughtMessageId)?.thought;
  }, [activeThoughtMessageId, currentSessionId, currentSession]);

  const isCurrentThoughtStreaming = React.useMemo(() => {
    if (!isLoading || !currentSession || !activeThoughtMessageId) return false;
    const lastMsg = currentSession.messages[currentSession.messages.length - 1];
    return lastMsg.id === activeThoughtMessageId;
  }, [isLoading, currentSession, activeThoughtMessageId]);

  const resetInactivityTimer = useCallback(() => {
    if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current);
    if (activeModificationFiles) return;
    if (isLoading || isSearching) { setFooterVisible(true); setHeaderVisible(true); return; }
    if (hasMessages) setFooterVisible(true);
    inactivityTimerRef.current = window.setTimeout(() => { if (activeView === 'chat') { setFooterVisible(false); setHeaderVisible(false); } }, 6000);
  }, [isLoading, isSearching, activeView, hasMessages, activeModificationFiles]);

  useEffect(() => { resetInactivityTimer(); return () => { if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current); }; }, [resetInactivityTimer]);

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (activeModificationFiles) return;
    const container = mainScrollRef.current;
    if (!container || isAutoScrollingRef.current) return;
    const diff = latest - lastScrollYRef.current;
    lastScrollYRef.current = latest;
    if (latest < 10) { if (hasMessages) setHeaderVisible(true); return; }
    if (Math.abs(diff) < 10) return;
    if (diff > 15) setHeaderVisible(false);
    else if (diff < -20) { setHeaderVisible(true); resetInactivityTimer(); }
  });

  useEffect(() => {
    const handleGlobalActivity = (e: MouseEvent) => {
      if (activeModificationFiles) return;
      if (e.clientY < 80) { if (!headerVisible) setHeaderVisible(true); resetInactivityTimer(); }
      if (e.clientY > window.innerHeight - 120) { if (!footerVisible) setFooterVisible(true); resetInactivityTimer(); }
    };
    window.addEventListener('mousemove', handleGlobalActivity);
    return () => window.removeEventListener('mousemove', handleGlobalActivity);
  }, [footerVisible, headerVisible, resetInactivityTimer, activeModificationFiles]);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      resetInactivityTimer();
      if (!headerVisible) setHeaderVisible(true);
      if (!footerVisible) setFooterVisible(true);
      if (e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setIsCommandPaletteOpen(prev => !prev); return; }
      if (e.key === 'Escape') {
        if (activeModificationFiles) setActiveModificationFiles(null);
        else if (isSettingsOpen) setIsSettingsOpen(false);
        else if (isCommandPaletteOpen) setIsCommandPaletteOpen(false);
        else if (isHelpOpen) setIsHelpOpen(false);
        else if (isReasoningOpen) setIsReasoningOpen(false);
        else setIsSettingsOpen(true);
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [resetInactivityTimer, headerVisible, footerVisible, isCommandPaletteOpen, isHelpOpen, isReasoningOpen, isSettingsOpen, activeModificationFiles]);

  useEffect(() => {
    const savedSessions = localStorage.getItem('chat-sessions');
    const savedSettings = localStorage.getItem('user-settings');
    if (savedSessions) {
      const parsedSessions = JSON.parse(savedSessions);
      setSessions(parsedSessions);
      if (parsedSessions.length > 0) setCurrentSessionId(parsedSessions[0].id);
    } else handleNewChat();
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings);
      setSettings({
        ...DEFAULT_SETTINGS,
        ...parsed,
        llm: { ...DEFAULT_SETTINGS.llm, ...(parsed?.llm || {}) },
      });
    }
  }, []);

  useEffect(() => { document.documentElement.classList.toggle('dark', isDarkMode); localStorage.setItem('dark-mode', String(isDarkMode)); }, [isDarkMode]);
  useEffect(() => { if (sessions.length > 0) localStorage.setItem('chat-sessions', JSON.stringify(sessions)); }, [sessions]);
  useEffect(() => { localStorage.setItem('user-settings', JSON.stringify(settings)); }, [settings]);

  useEffect(() => {
    if (activeView === 'chat' && currentSession?.messages.length) {
      const container = mainScrollRef.current;
      if (!container) return;
      const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 100;
      if (isAtBottom) {
        isAutoScrollingRef.current = true;
        container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
        const timer = setTimeout(() => { isAutoScrollingRef.current = false; }, 200);
        return () => clearTimeout(timer);
      }
    }
  }, [sessions, isLoading, isSearching, activeView, currentSessionId, currentSession]);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const newLog: LogEntry = { id: Math.random().toString(36).substr(2, 9), timestamp: Date.now(), level, message };
    setLogs(prev => [...prev.slice(-149), newLog]);
  }, []);

  const handleNewChat = useCallback(() => {
    const newSession: ChatSession = { id: Date.now().toString(), title: settings.language === 'es' ? 'Nueva Conversación' : 'New Conversation', messages: [], updatedAt: Date.now() };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    handleSelectView('chat');
    setHeaderVisible(false); setFooterVisible(false);
    addLog('SYSTEM', settings.language === 'es' ? 'Sincronización de núcleo completada.' : 'Kernel sync complete.');
  }, [addLog, settings.language]);

  const handleSelectView = useCallback((newView: ViewType) => {
    setActiveView(prev => { setPrevView(prev); return newView; });
    setHeaderVisible(true);
  }, []);

  const handleNavigateToChat = useCallback((sessionId: string, messageId?: string) => {
    setCurrentSessionId(sessionId); setActiveView('chat');
    if (messageId) { setTimeout(() => { document.getElementById(messageId)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 300); }
  }, []);

  const handleLoadDemo = useCallback(() => {
    if (!currentSessionId) return;
    const mockSources: Source[] = [
      { url: 'https://react.dev', title: t.analysis_library + ': React Hooks', domain: 'react.dev', index: 0 },
      { url: 'https://github.com/google/genai', title: 'GitHub - google/genai: SDK JS', domain: 'github.com', index: 1 },
      { url: 'https://developer.mozilla.org/es/docs/Web/API/Element/scrollIntoView', title: 'MDN: Element.scrollIntoView()', domain: 'developer.mozilla.org', index: 2 }
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

  const handleSendMessage = async (content: string, useInternet: boolean = false, selectedMode: AppMode = 'ask', useThinking: boolean = true) => {
    let targetSessionId = currentSessionId;
    let isNewSession = false;
    
    // Check if session actually exists
    const sessionExists = sessions.some(s => s.id === targetSessionId);

    if (!targetSessionId || !sessionExists) {
      targetSessionId = Date.now().toString();
      isNewSession = true;
      setCurrentSessionId(targetSessionId);
      handleSelectView('chat');
    }

    setMode(selectedMode);
    if (activeView !== 'chat') handleSelectView('chat');
    setHeaderVisible(true); setFooterVisible(true); resetInactivityTimer();
    
    const userMessage: Message = { id: Date.now().toString(), role: Role.USER, content, timestamp: Date.now() };
    const aiMessageId = (Date.now() + 1).toString();
    const initialAiMessage: Message = { id: aiMessageId, role: Role.AI, content: "", thought: "", sources: [], groundingSupports: [], timestamp: Date.now() };
    
    setSessions(prev => {
      if (isNewSession) {
         const newSession: ChatSession = { 
             id: targetSessionId!, 
             title: settings.language === 'es' ? 'Nueva Conversación' : 'New Conversation', 
             messages: [userMessage, initialAiMessage], 
             updatedAt: Date.now() 
         };
         return [newSession, ...prev];
      }
      return prev.map(s => s.id === targetSessionId ? { ...s, messages: [...s.messages, userMessage, initialAiMessage], updatedAt: Date.now() } : s);
    });

    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    setIsLoading(true); setIsSearching(useInternet);
    try {
      const history = isNewSession ? [] : (sessions.find(s => s.id === targetSessionId)?.messages || []);
      const stream = klimeaiService.generateResponseStream({
        history,
        prompt: content,
        api: settings.llm,
        mode: selectedMode,
        useInternet,
        useThinking,
        signal: abortControllerRef.current!.signal,
      });
      for await (const chunk of stream) {
        setIsSearching(false);
        setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: s.messages.map(m => m.id === aiMessageId ? { ...m, content: chunk.text, thought: chunk.thought || m.thought, sources: chunk.sources.length > 0 ? chunk.sources : m.sources, fileChanges: chunk.fileChanges || m.fileChanges } : m) } : s));
      }
    } catch (error: any) {
      if (error?.name !== 'AbortError') addLog('SYSTEM', 'Interrupción de flujo.');
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false); setIsSearching(false); resetInactivityTimer();
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
    <div className={`flex h-screen w-full bg-background transition-colors duration-1000 overflow-hidden text-foreground accelerated ${mode === 'agent' ? 'ring-[6px] ring-primary/10' : ''}`}>
      <CommandPalette isOpen={isCommandPaletteOpen} onClose={() => setIsCommandPaletteOpen(false)} sessions={sessions} currentSessionId={currentSessionId} onSelectSession={setCurrentSessionId} onNewChat={handleNewChat} onDeleteSession={id => setSessions(p => p.filter(x => x.id !== id))} onClearHistory={() => setSessions([])} onExportChat={() => {}} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} isSidebarOpen={isSidebarOpen} onToggleSidebar={() => { const next = !isSidebarOpen; setIsSidebarOpen(next); if (next) setIsReasoningOpen(false); }} onOpenSettings={() => setIsSettingsOpen(true)} onOpenHelp={() => setIsHelpOpen(true)} categoryOrder={settings.categoryOrder} language={settings.language} onSetFontSize={(size) => setSettings({ ...settings, fontSize: size })} />
      <AnimatePresence initial={false}>{isSidebarOpen && !activeModificationFiles && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 280, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full overflow-hidden shrink-0 z-50 flex border-r border-border/50 shadow-2xl relative"><Sidebar sessions={sessions} currentSessionId={currentSessionId} activeView={activeView} onSelectSession={setCurrentSessionId} onSelectView={handleSelectView} onNewChat={handleNewChat} onDeleteSession={id => setSessions(p => p.filter(x => x.id !== id))} isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} onClose={() => setIsSidebarOpen(false)} onOpenSettings={() => setIsSettingsOpen(true)} isOpen={true} language={settings.language} /></motion.div>
      )}</AnimatePresence>
      <div className="flex-1 flex overflow-hidden relative">
        <main className="flex-1 flex flex-col h-full bg-background relative z-0 overflow-hidden">
          {!activeModificationFiles && (
            <motion.header initial={false} animate={{ y: headerVisible ? 0 : -100, opacity: headerVisible ? 1 : 0 }} transition={springConfig} className={`absolute top-0 left-0 right-0 h-24 border-b border-border/40 flex items-center justify-between px-10 bg-background/80 dark:bg-zinc-950/80 backdrop-blur-3xl z-40 shrink-0 shadow-sm pointer-events-auto accelerated ${mode === 'agent' ? 'bg-primary/5 border-primary/20' : ''}`}>
              <div className="flex items-center gap-8">
                <AnimatePresence mode="wait">{!isSidebarOpen && (<motion.button initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.8, opacity: 0 }} whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--muted-foreground) / 0.1)' }} whileTap={{ scale: 0.9 }} onClick={() => { setIsSidebarOpen(true); setIsReasoningOpen(false); }} className="p-3.5 rounded-2xl transition-all"><PanelLeft size={24} /></motion.button>)}</AnimatePresence>
                <div className="flex items-center gap-5"><motion.div whileHover={{ rotate: -10, scale: 1.1 }} className="w-12 h-12 bg-primary rounded-[1.5rem] flex items-center justify-center text-primary-foreground shadow-xl transition-all duration-700"><Bot size={28} strokeWidth={2.5} /></motion.div><div className="flex flex-col"><h1 className="text-[17px] font-black tracking-tight leading-none">Vortex</h1><span className="text-[9px] font-black uppercase tracking-[0.3em] mt-2 transition-colors text-primary">{t.system_kernel}</span></div></div>
              </div>
              <div className="flex items-center gap-4">
                <motion.button whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--primary) / 0.1)' }} whileTap={{ scale: 0.9 }} onClick={() => setSettings({ ...settings, language: settings.language === 'es' ? 'en' : 'es' })} className="w-12 h-12 flex items-center justify-center bg-muted/40 dark:bg-zinc-900/40 border border-border/50 rounded-2xl hover:border-primary/40 transition-all shadow-sm overflow-hidden"><img src={settings.language === 'es' ? 'https://flagcdn.com/w80/es.png' : 'https://flagcdn.com/w80/us.png'} alt={settings.language} className="w-7 h-auto object-contain rounded-sm select-none" /></motion.button>
                <div className="flex items-center gap-1 bg-muted/40 dark:bg-zinc-900/40 p-1 rounded-2xl border border-border/50 relative">{['chat', 'analysis', 'terminal'].map(v => (<button key={v} onClick={() => handleSelectView(v as ViewType)} className={`relative p-2.5 rounded-xl transition-all z-10 ${activeView === v ? 'text-primary-foreground' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>{v === 'chat' ? <MessageSquare size={16} /> : v === 'analysis' ? <BarChart3 size={16} /> : <TerminalIcon size={16} />}{activeView === v && <motion.div layoutId="header-nav-indicator" className="absolute inset-0 bg-primary rounded-xl shadow-lg -z-10" transition={springConfig} />}</button>))}</div>
                <motion.button whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }} onClick={() => setIsCommandPaletteOpen(true)} className="flex items-center gap-3 px-5 py-2.5 bg-muted/50 dark:bg-zinc-900/50 hover:bg-primary/10 rounded-2xl border border-border/50 transition-all shadow-sm"><Zap size={16} className={'text-primary'} /><kbd className="hidden lg:inline-block px-2 py-0.5 bg-background border rounded-lg text-[8px] font-black opacity-40">ALT+K</kbd></motion.button>
              </div>
            </motion.header>
          )}

          <div ref={mainScrollRef} className="flex-1 overflow-y-auto custom-scrollbar flex flex-col relative h-full bg-background scroll-smooth accelerated">
            {hasMessages && !activeModificationFiles && <div className="pt-32 shrink-0" />}
            <AnimatePresence mode="popLayout" custom={direction}>
              {activeView === 'chat' && (
                <motion.div key="chat" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className={`mx-auto w-full flex-1 flex flex-col px-6 lg:px-16 min-h-full transition-all duration-500 ${!hasMessages ? 'justify-center max-w-[1200px]' : 'pt-6 max-w-full'}`}>
                  {!hasMessages ? (
                    <div className="flex flex-col items-center justify-center text-center space-y-12">
                      <motion.div initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1, rotate: [0, 4, -4, 0] }} transition={{ duration: 8, repeat: Infinity }} whileHover={{ scale: 1.1, rotate: 10 }} className="relative w-40 h-40 bg-primary/10 rounded-[3.5rem] flex items-center justify-center text-primary border border-primary/20 shadow-2xl cursor-pointer"><Sparkles size={70} strokeWidth={1} /></motion.div>
                      <h2 className="text-5xl font-black tracking-tighter leading-tight whitespace-pre-line">{t.welcome_title}</h2>
                      <motion.button whileHover={{ scale: 1.05, boxShadow: '0 20px 40px -10px rgba(0,0,0,0.2)' }} whileTap={{ scale: 0.95 }} onClick={handleLoadDemo} className="flex items-center gap-5 px-12 py-6 bg-foreground text-background rounded-[2.5rem] font-black uppercase tracking-[0.3em] text-[10px] transition-all">{t.initialize_vortex}</motion.button>
                    </div>
                  ) : (
                    <div className="pb-40">
                      <VirtualizedMessageList 
                        messages={currentSession.messages} 
                        fontSize={settings.fontSize} 
                        codeTheme={settings.codeTheme}
                        onShowReasoning={messageId => { setActiveThoughtMessageId(messageId); setIsReasoningOpen(true); setIsSidebarOpen(false); }} 
                        onOpenModificationExplorer={handleOpenModificationExplorer} 
                        isLoading={isLoading} 
                        language={settings.language} 
                        containerRef={mainScrollRef} 
                      />
                      {isSearching && (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-5 px-8 py-5 mt-6 bg-primary/5 border border-primary/20 rounded-[2.5rem] text-primary shadow-xl w-fit glass-card accelerated"><Globe size={22} className="animate-spin-slow" /><p className="text-[12px] font-black uppercase tracking-widest">{settings.language === 'es' ? 'Capas de conocimiento activas...' : 'Active knowledge layers...'}</p></motion.div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
              {activeView === 'analysis' && <motion.div key="analysis" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><AnalysisView sessions={sessions} onNavigateToChat={handleNavigateToChat} onAddLog={addLog} language={settings.language}/></motion.div>}
              {activeView === 'terminal' && <motion.div key="terminal" custom={direction} variants={{ initial: (d: number) => ({ opacity: 0, x: d * 40, filter: 'blur(10px)' }), animate: { opacity: 1, x: 0, filter: 'blur(0px)', transition: springConfig }, exit: (d: number) => ({ opacity: 0, x: -d * 40, filter: 'blur(10px)', transition: { duration: 0.3 } }) }} initial="initial" animate="animate" exit="exit" className="flex-1"><TerminalView logs={logs} onClear={() => setLogs([])} language={settings.language} /></motion.div>}
            </AnimatePresence>
          </div>

          {!activeModificationFiles && (
            <motion.div initial={false} animate={{ y: footerVisible ? 0 : 200, opacity: footerVisible ? 1 : 0 }} transition={{ type: 'spring', damping: 30, stiffness: 200 }} className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background via-background/95 to-transparent pt-12 pb-8 z-30 pointer-events-auto accelerated ${mode === 'agent' ? 'from-primary/5' : ''}`}>
              <div className="pointer-events-auto"><ChatInput onSend={handleSendMessage} isLoading={isLoading} isDarkMode={isDarkMode} onStop={() => { abortControllerRef.current?.abort(); }} language={settings.language} onInteraction={() => { resetInactivityTimer(); if (!footerVisible) setFooterVisible(true); }} /></div>
            </motion.div>
          )}
        </main>
        
        <AnimatePresence>{isReasoningOpen && !activeModificationFiles && (
            <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 400, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={springConfig} className="h-full border-l border-border/50 shrink-0 z-50 overflow-hidden bg-zinc-950/95 shadow-[-20px_0_50px_rgba(0,0,0,0.5)]"><ReasoningDrawer isOpen={isReasoningOpen} onClose={() => setIsReasoningOpen(false)} thought={activeThought} language={settings.language} isStreaming={isCurrentThoughtStreaming} /></motion.div>
        )}</AnimatePresence>
      </div>

      <AnimatePresence>
        {activeModificationFiles && (
          <ModificationExplorerModal 
            fileChanges={activeModificationFiles} 
            onClose={() => { setActiveModificationFiles(null); setHeaderVisible(true); setFooterVisible(true); }} 
            language={settings.language} 
          />
        )}
      </AnimatePresence>

      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onUpdateSettings={setSettings} />
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} isDarkMode={isDarkMode} language={settings.language} />
    </div>
  );
};

export default App;
