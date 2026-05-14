import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronUp,
  ChevronsUpDown,
  CircleUserRound,
  MessageSquare,
  Moon,
  PanelLeftClose,
  Plus,
  Settings,
  Sun,
  Trash2,
  AlertTriangle,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChatSession, Language, LocalAccount, ViewType } from '../types';
import { translations } from '../translations';
import VortexLogo from './VortexLogo';
import type { SettingsTab } from './SettingsModal';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  activeView: ViewType;
  onSelectSession: (id: string) => void;
  onSelectView: (view: ViewType) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  onClose: () => void;
  onOpenSettings: (tab?: SettingsTab) => void;
  isOpen: boolean;
  language: Language;
  currentAccount?: LocalAccount | null;
  accounts: LocalAccount[];
  currentAccountId: string | null;
  onSelectAccount: (id: string) => void;
}

const springTransition = { type: 'spring' as const, stiffness: 420, damping: 34 };

const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  currentSessionId,
  activeView,
  onSelectSession,
  onSelectView,
  onNewChat,
  onDeleteSession,
  isDarkMode,
  toggleDarkMode,
  onClose,
  onOpenSettings,
  language,
  currentAccount,
  accounts,
  currentAccountId,
  onSelectAccount,
}) => {
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);
  const t = translations[language];

  const themeLabel = language === 'es'
    ? (isDarkMode ? 'Cambiar a claro' : 'Cambiar a oscuro')
    : (isDarkMode ? 'Switch to light' : 'Switch to dark');

  const navigation = useMemo(
    () => [
      { id: 'chat' as ViewType, label: t.nav_chat, icon: MessageSquare },
    ],
    [t.nav_chat],
  );

  const orderedSessions = useMemo(
    () => [...sessions].sort((left, right) => right.updatedAt - left.updatedAt),
    [sessions],
  );
  const sortedAccounts = useMemo(
    () => [...accounts].sort((left, right) => right.lastUsedAt - left.lastUsedAt),
    [accounts],
  );

  const formatSessionTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString(language === 'es' ? 'es-ES' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const handleDeleteConfirm = () => {
    if (!sessionToDelete) return;
    onDeleteSession(sessionToDelete);
    setSessionToDelete(null);
  };

  useEffect(() => {
    if (!isProfileMenuOpen) return;
    const handleOutsideClick = (event: MouseEvent) => {
      if (!profileMenuRef.current?.contains(event.target as Node)) {
        setIsProfileMenuOpen(false);
      }
    };
    window.addEventListener('mousedown', handleOutsideClick);
    return () => window.removeEventListener('mousedown', handleOutsideClick);
  }, [isProfileMenuOpen]);

  return (
    <>
      <div className="relative flex h-full w-full flex-col overflow-hidden border-r border-border/70 bg-background/95 transition-colors duration-500">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.18),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />
        </div>

        <div className="relative z-10 border-b border-border/50 px-4 pb-4 pt-4">
          <div className="flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => onSelectView('chat')}
              className="group flex min-w-0 items-center gap-3 rounded-xl px-2 py-2 text-left transition-all hover:bg-muted/30"
            >
              <VortexLogo size={28} alt="Vortex" />
              <div className="min-w-0">
                <p className="truncate text-sm font-extrabold tracking-tight">Vortex</p>
              </div>
            </button>

            <motion.button
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              onClick={onClose}
              className="rounded-xl p-2.5 text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
              aria-label={language === 'es' ? 'Cerrar lateral' : 'Close sidebar'}
            >
              <PanelLeftClose size={20} />
            </motion.button>
          </div>

          <button
            type="button"
            onClick={onNewChat}
            className="mt-4 flex w-full items-center justify-between rounded-xl border border-border/70 bg-muted/15 px-4 py-3 text-left shadow-sm transition-all hover:border-primary/20 hover:bg-background"
          >
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">
                {language === 'es' ? 'Nueva sesión' : 'New session'}
              </p>
              <p className="mt-1 text-sm font-semibold text-foreground">
                {language === 'es' ? 'Abrir chat limpio' : 'Start a fresh chat'}
              </p>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Plus size={18} />
            </div>
          </button>
        </div>

        <div className="relative z-10 px-4 pt-4">
          <div className="space-y-1.5">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectView(item.id)}
                  className={`relative flex w-full items-center gap-3 rounded-[1rem] px-4 py-3 text-left transition-all ${
                    isActive
                      ? 'bg-primary/[0.12] text-foreground'
                      : 'text-foreground/75 hover:bg-muted/30 hover:text-foreground'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active-pill"
                      transition={springTransition}
                      className="absolute inset-0 rounded-[1rem] bg-primary/[0.12] border border-primary/20"
                    />
                  )}
                  <span className="relative z-10 flex h-9 w-9 items-center justify-center rounded-xl bg-background/70 border border-border/40">
                    <Icon size={18} />
                  </span>
                  <span className="relative z-10 flex-1 text-sm font-semibold tracking-tight">
                    {item.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="relative z-10 mt-5 flex-1 overflow-y-auto px-4 pb-5 custom-scrollbar">
          <div className="mb-3 flex items-center justify-between px-2">
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">
                {t.smart_sessions}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {language === 'es'
                  ? `${orderedSessions.length} activas`
                  : `${orderedSessions.length} active`}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            {orderedSessions.map((session) => {
              const isCurrent = activeView === 'chat' && currentSessionId === session.id;
              return (
                <div
                  key={session.id}
                  className={`group relative rounded-[1.1rem] border px-4 py-3 transition-all ${
                    isCurrent
                      ? 'border-primary/20 bg-primary/[0.08] shadow-sm'
                      : 'border-transparent bg-transparent hover:border-border/60 hover:bg-muted/20'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelectSession(session.id);
                      onSelectView('chat');
                    }}
                    className="flex w-full items-start gap-3 text-left"
                  >
                    <div className={`mt-0.5 h-2.5 w-2.5 rounded-full ${isCurrent ? 'bg-primary' : 'bg-muted-foreground/30'}`} />
                    <div className="min-w-0 flex-1">
                      <p className={`truncate text-sm font-semibold tracking-tight ${isCurrent ? 'text-foreground' : 'text-foreground/80'}`}>
                        {session.title}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {session.messages.length} {language === 'es' ? 'mensajes' : 'messages'} · {formatSessionTime(session.updatedAt)}
                      </p>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      setSessionToDelete(session.id);
                    }}
                    className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-xl text-muted-foreground opacity-0 transition-all hover:bg-red-500/10 hover:text-red-500 group-hover:opacity-100"
                    aria-label={language === 'es' ? 'Eliminar sesión' : 'Delete session'}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div className="relative z-10 border-t border-border/50 px-4 py-4">
          <div className="mb-3 flex items-center gap-2">
            <button
              type="button"
              onClick={toggleDarkMode}
              className="flex flex-1 items-center justify-center gap-2 rounded-[1rem] border border-border/60 bg-muted/20 px-3 py-2.5 text-xs font-semibold text-foreground transition-all hover:border-primary/20 hover:bg-background"
            >
              {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
              <span>{isDarkMode ? t.interface_light : t.interface_dark}</span>
            </button>
          <button
            type="button"
            onClick={() => onOpenSettings('general')}
            className="flex h-10 w-10 items-center justify-center rounded-[1rem] border border-border/60 bg-muted/20 text-foreground transition-all hover:border-primary/20 hover:bg-background"
            aria-label={t.configuration}
            title={t.configuration}
            >
              <Settings size={18} />
            </button>
          </div>

          <div ref={profileMenuRef} className="relative">
            <AnimatePresence>
              {isProfileMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: 'easeOut' }}
                  className="absolute bottom-[calc(100%+12px)] left-0 right-0 rounded-[1.25rem] border border-border/70 bg-background/98 p-2 shadow-[0_32px_80px_-48px_rgba(15,23,42,0.35)] backdrop-blur-xl"
                >
                  <div className="rounded-[1rem] border border-border/60 bg-muted/15 px-3 py-3">
                    <div className="flex items-center gap-3">
                      <div
                        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-sm font-black text-white"
                        style={{ background: `linear-gradient(135deg, hsl(${currentAccount?.avatarHue ?? 198} 100% 58%), hsl(${(currentAccount?.avatarHue ?? 198) + 18} 100% 46%))` }}
                      >
                        {currentAccount?.name?.slice(0, 2).toUpperCase() || <CircleUserRound size={16} />}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold tracking-tight text-foreground">
                          {currentAccount?.name || 'Vortex Local'}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-muted-foreground">
                          {currentAccount?.email || 'local@vortex.dev'}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-2 space-y-1">
                    {sortedAccounts.map((account) => {
                      const isCurrentAccount = account.id === currentAccountId;
                      return (
                        <button
                          key={account.id}
                          type="button"
                          onClick={() => {
                            onSelectAccount(account.id);
                            setIsProfileMenuOpen(false);
                          }}
                          className={`flex w-full items-center gap-3 rounded-[0.95rem] px-3 py-2.5 text-left transition-all ${
                            isCurrentAccount
                              ? 'bg-primary/[0.10] text-foreground'
                              : 'hover:bg-muted/35'
                          }`}
                        >
                          <div
                            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[11px] font-black text-white"
                            style={{ background: `linear-gradient(135deg, hsl(${account.avatarHue} 100% 58%), hsl(${account.avatarHue + 18} 100% 46%))` }}
                          >
                            {account.name.slice(0, 2).toUpperCase()}
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-semibold text-foreground">{account.name}</p>
                            <p className="truncate text-xs text-muted-foreground">{account.handle}</p>
                          </div>
                          {isCurrentAccount && (
                            <span className="rounded-full bg-primary px-2 py-1 text-[9px] font-black uppercase tracking-[0.12em] text-primary-foreground">
                              {t.account_current}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  <div className="mt-2 border-t border-border/60 pt-2">
                    <button
                      type="button"
                      onClick={() => {
                        setIsProfileMenuOpen(false);
                        onOpenSettings('profiles');
                      }}
                      className="flex w-full items-center gap-3 rounded-[0.95rem] px-3 py-2.5 text-left text-sm font-semibold text-foreground transition-all hover:bg-muted/35"
                    >
                      <Settings size={16} />
                      <span>{language === 'es' ? 'Gestionar perfiles' : 'Manage profiles'}</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="button"
              onClick={() => setIsProfileMenuOpen((prev) => !prev)}
              className="flex w-full items-center gap-3 rounded-[1.15rem] border border-border/60 bg-muted/20 px-3 py-3 text-left transition-all hover:border-primary/20 hover:bg-background"
            >
              <div
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-sm font-black text-white"
                style={{ background: `linear-gradient(135deg, hsl(${currentAccount?.avatarHue ?? 198} 100% 58%), hsl(${(currentAccount?.avatarHue ?? 198) + 18} 100% 46%))` }}
              >
                {currentAccount?.name?.slice(0, 2).toUpperCase() || <CircleUserRound size={16} />}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold tracking-tight text-foreground">
                  {currentAccount?.name || 'Vortex Local'}
                </p>
                <p className="mt-0.5 truncate text-xs text-muted-foreground">
                  {currentAccount?.email || 'local@vortex.dev'}
                </p>
              </div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-background/80 text-muted-foreground">
                {isProfileMenuOpen ? <ChevronUp size={15} /> : <ChevronsUpDown size={15} />}
              </div>
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {sessionToDelete && (
          <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/45 px-6 backdrop-blur-md">
            <motion.button
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0"
              onClick={() => setSessionToDelete(null)}
              aria-label={language === 'es' ? 'Cerrar diálogo' : 'Close dialog'}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.94, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: 20 }}
              transition={springTransition}
              className="relative w-full max-w-md rounded-[1.8rem] border border-border bg-background px-8 py-8 shadow-[0_28px_80px_-48px_rgba(15,23,42,0.35)]"
            >
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[1.8rem] bg-red-500/10 text-red-500">
                <AlertTriangle size={28} />
              </div>
              <h3 className="mt-6 text-center text-2xl font-black tracking-tight">
                {language === 'es' ? 'Eliminar sesión' : 'Delete session'}
              </h3>
              <p className="mt-3 text-center text-sm leading-7 text-muted-foreground">
                {language === 'es'
                  ? 'Se borrará esta conversación del historial local. No se eliminarán los datos del runtime ni del modelo.'
                  : 'This will remove the conversation from local history. Runtime and model data will not be deleted.'}
              </p>

              <div className="mt-8 flex flex-col gap-3">
                <button
                  type="button"
                  onClick={handleDeleteConfirm}
                  className="rounded-[1.2rem] bg-red-500 px-5 py-3 text-sm font-black text-white transition-all hover:bg-red-600"
                >
                  {language === 'es' ? 'Eliminar sesión' : 'Delete session'}
                </button>
                <button
                  type="button"
                  onClick={() => setSessionToDelete(null)}
                  className="rounded-[1.2rem] border border-border px-5 py-3 text-sm font-semibold text-foreground transition-all hover:bg-muted/50"
                >
                  {language === 'es' ? 'Cancelar' : 'Cancel'}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
};

export default Sidebar;
