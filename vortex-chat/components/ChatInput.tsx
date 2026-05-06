
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Square, ArrowUp, Globe, Timer, FolderOpen, ChevronDown, ChevronUp, ShieldCheck, SquareTerminal, Cpu } from 'lucide-react';
import { motion } from 'framer-motion';
import { AppMode, Language, WorkspaceProject } from '../types';
import { translations } from '../translations';

interface ChatInputProps {
  onSend: (message: string, useInternet: boolean, mode: AppMode, useThinking: boolean, autoTrain: boolean) => void;
  isLoading: boolean;
  isDarkMode: boolean;
  language: Language;
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
  canUseInternet?: boolean;
  sendDisabledReason?: string;
  isThinking?: boolean;
  onStop?: () => void;
  onInteraction?: () => void;
  onFocusChange?: (focused: boolean) => void;
  onDraftChange?: (hasDraft: boolean) => void;
  projects?: WorkspaceProject[];
  activeProjectId?: string | null;
  onSelectProject?: (projectId: string | null) => void;
  onOpenProjectSettings?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  isLoading,
  isDarkMode,
  language,
  mode,
  onModeChange,
  canUseInternet = true,
  sendDisabledReason,
  isThinking,
  onStop,
  onInteraction,
  onFocusChange,
  onDraftChange,
  projects = [],
  activeProjectId = null,
  onSelectProject,
  onOpenProjectSettings,
}) => {
  const [input, setInput] = useState('');
  const [isInternetEnabled, setIsInternetEnabled] = useState(false);
  const [useThinking, setUseThinking] = useState(true);
  const [isProjectPickerOpen, setIsProjectPickerOpen] = useState(false);
  const [isProjectDetailsOpen, setIsProjectDetailsOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const t = translations[language];
  const MAX_HEIGHT = 360; 

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const newHeight = Math.max(54, Math.min(scrollHeight, MAX_HEIGHT));
      textarea.style.height = `${newHeight}px`;
    }
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [input, adjustHeight]);

  useEffect(() => {
    onDraftChange?.(input.trim().length > 0);
  }, [input, onDraftChange]);

  useEffect(() => {
    if (!canUseInternet) setIsInternetEnabled(false);
  }, [canUseInternet]);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onInteraction?.();
      onSend(input.trim(), isInternetEnabled, mode, useThinking, false);
      setInput('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const footerHint = sendDisabledReason
    || (language === 'es'
      ? 'Enter para enviar. Shift+Enter para nueva línea.'
      : 'Enter to send. Shift+Enter for a new line.');
  const activeProject = projects.find((project) => project.id === activeProjectId) || null;
  const projectLabel = activeProject?.name || (language === 'es' ? 'Añadir workspace' : 'Add workspace');
  const permissionLabel = activeProject
    ? activeProject.permissionLevel === 'full'
      ? (language === 'es' ? 'total' : 'full')
      : activeProject.permissionLevel === 'edit'
        ? (language === 'es' ? 'edicion' : 'edit')
        : activeProject.permissionLevel === 'read'
          ? (language === 'es' ? 'lectura' : 'read')
          : (language === 'es' ? 'sin acceso' : 'none')
    : (language === 'es' ? 'sin proyecto' : 'no project');
  const actionLabel = activeProject?.actionMode === 'full'
    ? (language === 'es' ? 'acciones completas' : 'full actions')
    : (language === 'es' ? 'modo seguro' : 'safe mode');
  const scopeLabel = activeProject?.projectPath || activeProject?.rootPath || (language === 'es' ? 'sin scope' : 'no scope');
  const projectDetails = [
    { label: language === 'es' ? 'Permisos' : 'Permissions', value: permissionLabel },
    { label: language === 'es' ? 'Acciones' : 'Actions', value: actionLabel },
    { label: 'Scope', value: scopeLabel },
    { label: language === 'es' ? 'Modo' : 'Mode', value: mode === 'agent' ? t.input_agent_mode : t.input_ask_mode },
    { label: 'Thinking', value: useThinking ? 'on' : 'off' },
  ];

  return (
    <div className="mx-auto w-full max-w-[1160px] relative px-6 accelerated">
      <motion.div 
        layout
        className={`surface-panel relative flex w-full items-end rounded-[1.5rem] p-1.5 transition-all duration-500 group accelerated ${
          isLoading 
            ? 'border-border/70 opacity-90' 
            : mode === 'agent'
              ? 'focus-within:border-primary/40 focus-within:ring-[4px] focus-within:ring-primary/10 shadow-[0_20px_70px_-48px_hsla(var(--primary)/0.55)]'
              : 'focus-within:border-primary/35 focus-within:ring-[4px] focus-within:ring-primary/5'
        }`}
      >
        <div className="flex items-center mb-1 ml-1 relative z-10 shrink-0">
          <button
            onClick={() => { setUseThinking(!useThinking); onInteraction?.(); }}
            aria-label={useThinking ? (language === 'es' ? 'Modo Thinking activo' : 'Thinking mode active') : (language === 'es' ? 'Modo Fast activo' : 'Fast mode active')}
            title={useThinking ? (language === 'es' ? 'Modo Thinking Activo' : 'Thinking Mode Active') : (language === 'es' ? 'Modo Fast Activo' : 'Fast Mode Active')}
            className={`flex h-10 w-10 items-center justify-center rounded-full border border-transparent transition-all duration-300 ${
              useThinking 
                ? 'text-primary bg-primary/10 border-primary/10' 
                : 'text-muted-foreground/40 dark:text-zinc-600 hover:text-foreground/60'
            }`}
          >
            <Timer size={20} strokeWidth={useThinking ? 2.4 : 1.5} className="transition-all duration-500" />
          </button>
        </div>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => { setInput(e.target.value); onInteraction?.(); }}
          onKeyDown={handleKeyDown}
          onFocus={() => { onInteraction?.(); onFocusChange?.(true); }}
          onBlur={() => onFocusChange?.(false)}
          placeholder={mode === 'agent' ? t.input_placeholder_agent : t.input_placeholder_ask}
          rows={1}
          className="relative z-10 flex-1 resize-none border-none bg-transparent px-3 py-3.5 text-[15px] font-medium leading-7 text-foreground outline-none custom-scrollbar placeholder:text-muted-foreground/45 focus:ring-0 dark:placeholder:text-zinc-500"
        />
        
        <div className="flex items-center gap-2 mb-1 mr-1 relative z-10 shrink-0">
          <div className="relative flex h-[42px] min-w-[120px] overflow-hidden rounded-2xl border border-border/50 bg-muted/35 p-0.5">
            <motion.div
              animate={{ x: mode === 'ask' ? 0 : 58, backgroundColor: 'hsl(var(--primary))' }}
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              className="mode-pill-active absolute top-0.5 bottom-0.5 w-[58px] rounded-xl shadow-lg z-0"
            />
            <button onClick={() => { onModeChange('ask'); onInteraction?.(); }} className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 text-[8px] font-black uppercase tracking-widest transition-all duration-300 ${mode === 'ask' ? 'text-white' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>
               {t.input_ask_mode}
            </button>
            <button onClick={() => { onModeChange('agent'); onInteraction?.(); }} className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 text-[8px] font-black uppercase tracking-widest transition-all duration-300 ${mode === 'agent' ? 'text-white' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>
               {t.input_agent_mode}
            </button>
          </div>

          <button
            onClick={() => {
              if (!canUseInternet) return;
              setIsInternetEnabled(!isInternetEnabled);
              onInteraction?.();
            }}
            aria-label={isInternetEnabled ? (language === 'es' ? 'Desactivar Internet' : 'Disable internet') : (language === 'es' ? 'Activar Internet' : 'Enable internet')}
            title={isInternetEnabled ? (language === 'es' ? 'Internet activado' : 'Internet enabled') : (language === 'es' ? 'Internet desactivado' : 'Internet disabled')}
            className={`flex items-center justify-center rounded-full border border-transparent p-2.5 transition-all duration-300 ${
              !canUseInternet
                ? 'text-muted-foreground/20 dark:text-zinc-700 cursor-not-allowed'
                : isInternetEnabled 
                ? 'bg-primary/12 text-primary border-primary/20' 
                : 'text-muted-foreground dark:text-zinc-400 hover:bg-muted hover:text-foreground'
            }`}
            disabled={!canUseInternet}
          >
            <Globe size={18} className={isInternetEnabled ? 'animate-pulse' : ''} />
          </button>

          {isLoading ? (
            <button
              onClick={() => onStop?.()}
              aria-label={language === 'es' ? 'Detener' : 'Stop'}
              title={language === 'es' ? 'Detener' : 'Stop'}
              className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500 text-white shadow-lg transition-all hover:scale-105 active:scale-90"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              aria-label={language === 'es' ? 'Enviar mensaje' : 'Send message'}
              title={sendDisabledReason || (language === 'es' ? 'Enviar mensaje' : 'Send message')}
              className={`group/send flex h-10 w-10 items-center justify-center rounded-full transition-all duration-300 ${
                input.trim()
                  ? mode === 'agent' 
                    ? 'bg-primary text-white shadow-lg shadow-primary/10 hover:scale-105 active:scale-90'
                    : 'bg-primary text-white shadow-lg shadow-primary/10 hover:scale-105 active:scale-90'
                  : 'cursor-not-allowed bg-muted/40 text-muted-foreground/10 shadow-none dark:bg-zinc-800/40'
              }`}
            >
              <ArrowUp size={20} strokeWidth={3} className={`transition-transform duration-300 ${input.trim() ? 'group-hover/send:-translate-y-0.5' : ''}`} />
            </button>
          )}
        </div>
      </motion.div>

      <div className="mt-3 flex flex-col gap-3 px-2 md:flex-row md:items-start md:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setIsProjectPickerOpen((value) => !value);
                setIsProjectDetailsOpen(false);
                onInteraction?.();
              }}
              className="inline-flex max-w-[280px] items-center gap-2 rounded-full border border-border/60 bg-background/80 px-3 py-1.5 text-[11px] font-bold text-foreground shadow-sm transition-colors hover:border-primary/30"
              title={activeProject?.rootPath || projectLabel}
            >
              <FolderOpen size={14} className="text-primary" />
              <span className="truncate">{projectLabel}</span>
              <ChevronDown size={13} className="text-muted-foreground" />
            </button>
            {isProjectPickerOpen && (
              <div className="absolute bottom-full left-0 z-50 mb-2 w-[340px] overflow-hidden rounded-2xl border border-border/70 bg-background shadow-2xl">
                <button
                  type="button"
                  onClick={() => {
                    onSelectProject?.(null);
                    setIsProjectPickerOpen(false);
                  }}
                  className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm hover:bg-muted/40"
                >
                  <span>{language === 'es' ? 'Sin workspace' : 'No workspace'}</span>
                  {!activeProject && <ShieldCheck size={15} className="text-primary" />}
                </button>
                {projects.map((project) => (
                  <button
                    key={project.id}
                    type="button"
                    onClick={() => {
                      onSelectProject?.(project.id);
                      setIsProjectPickerOpen(false);
                    }}
                    className="flex w-full items-center justify-between gap-3 border-t border-border/50 px-4 py-3 text-left hover:bg-muted/40"
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold text-foreground">{project.name}</span>
                      <span className="block truncate text-xs text-muted-foreground">{project.projectPath || project.rootPath}</span>
                    </span>
                    <span className="shrink-0 rounded-full bg-muted px-2 py-1 text-[9px] font-black uppercase tracking-[0.1em] text-muted-foreground">{project.permissionLevel}</span>
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    setIsProjectPickerOpen(false);
                    onOpenProjectSettings?.();
                  }}
                  className="flex w-full items-center gap-2 border-t border-border/60 px-4 py-3 text-left text-sm font-bold text-primary hover:bg-primary/10"
                >
                  <FolderOpen size={15} />
                  {language === 'es' ? 'Gestionar proyectos' : 'Manage projects'}
                </button>
              </div>
            )}
          </div>
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setIsProjectDetailsOpen((value) => !value);
                setIsProjectPickerOpen(false);
                onInteraction?.();
              }}
              aria-label={language === 'es' ? 'Ver detalles del chat' : 'Show chat details'}
              title={language === 'es' ? 'Detalles del chat' : 'Chat details'}
              className="flex h-8 w-8 items-center justify-center rounded-full border border-border/60 bg-background/80 text-muted-foreground shadow-sm transition-colors hover:border-primary/30 hover:text-foreground"
            >
              <ChevronUp size={15} className={`transition-transform duration-200 ${isProjectDetailsOpen ? 'rotate-180' : ''}`} />
            </button>
            {isProjectDetailsOpen && (
              <div className="absolute bottom-full left-0 z-50 mb-2 w-[360px] overflow-hidden rounded-2xl border border-border/70 bg-background shadow-2xl">
                <div className="border-b border-border/60 bg-muted/20 px-4 py-3">
                  <div className="flex items-center gap-3">
                    <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
                      <Cpu size={17} />
                    </span>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-extrabold text-foreground">{activeProject?.name || (language === 'es' ? 'Chat sin proyecto' : 'Chat without project')}</p>
                      <p className="truncate text-xs text-muted-foreground">{scopeLabel}</p>
                    </div>
                  </div>
                </div>
                <div className="grid gap-2 p-3">
                  {projectDetails.map((item) => (
                    <div key={item.label} className="flex items-center justify-between gap-4 rounded-xl bg-muted/20 px-3 py-2">
                      <span className="text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">{item.label}</span>
                      <span className="min-w-0 truncate text-right text-xs font-bold text-foreground">{item.value}</span>
                    </div>
                  ))}
                  <div className="flex items-center justify-between gap-4 rounded-xl bg-muted/20 px-3 py-2">
                    <span className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">
                      <SquareTerminal size={13} />
                      {language === 'es' ? 'Estado' : 'Status'}
                    </span>
                    <span className="min-w-0 truncate text-right text-xs font-bold text-foreground">
                      {isLoading ? (language === 'es' ? 'ejecutando' : 'running') : (language === 'es' ? 'listo' : 'ready')}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        <p className={`text-xs ${sendDisabledReason ? 'text-primary' : 'text-muted-foreground'}`}>
          {footerHint}
        </p>
      </div>
    </div>
  );
};

export default ChatInput;
