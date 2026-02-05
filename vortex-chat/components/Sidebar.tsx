
import React, { useState } from 'react';
import { 
  Plus, 
  MessageSquare, 
  Trash2, 
  Settings, 
  Moon, 
  Sun,
  PanelLeftClose,
  Zap,
  BarChart3,
  Terminal as TerminalIcon,
  Layers,
  AlertTriangle,
  X,
  ChevronRight,
  Sparkles
} from 'lucide-react';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { ChatSession, ViewType, Language } from '../types';
import { translations } from '../translations';

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
  onOpenSettings: () => void;
  isOpen: boolean;
  language: Language;
  selfEditsPendingCount?: number;
}

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
  isOpen,
  language,
  selfEditsPendingCount
}) => {
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const t = translations[language];

  const handleDeleteConfirm = () => {
    if (sessionToDelete) {
      onDeleteSession(sessionToDelete);
      setSessionToDelete(null);
    }
  };

  const mainNav = [
    { id: 'chat', label: t.nav_chat, icon: <MessageSquare size={18} /> },
    { id: 'analysis', label: t.nav_analysis, icon: <BarChart3 size={18} /> },
    { id: 'edits', label: t.nav_edits, icon: <Layers size={18} />, badge: selfEditsPendingCount || 0 },
    { id: 'terminal', label: t.nav_terminal, icon: <TerminalIcon size={18} /> },
  ];

  const springTransition = { type: "spring" as const, stiffness: 400, damping: 30 };

  return (
    <>
      <div className="h-full flex flex-col bg-muted/95 dark:bg-zinc-950 backdrop-blur-3xl w-full transition-colors duration-500 overflow-hidden border-r border-border/60 relative">
        <div className="p-6 flex flex-col gap-6 shrink-0">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
                <motion.div 
                  whileHover={{ rotate: 15, scale: 1.15 }}
                  whileTap={{ scale: 0.9 }}
                  className="w-10 h-10 bg-primary rounded-2xl flex items-center justify-center text-primary-foreground shadow-lg shadow-primary/30 cursor-pointer"
                >
                  <Zap size={22} fill="currentColor" strokeWidth={0} />
                </motion.div>
                <div>
                  <h1 className="text-sm font-black tracking-tighter text-foreground">Vortex</h1>
                  <p className="text-[9px] font-black uppercase tracking-[0.2em] text-primary">{language === 'es' ? 'Operativo' : 'Operational'}</p>
                </div>
            </div>
            <motion.button 
              whileHover={{ scale: 1.1, backgroundColor: 'hsla(var(--muted-foreground) / 0.1)' }}
              whileTap={{ scale: 0.9 }}
              onClick={onClose} 
              className="p-2.5 rounded-xl text-muted-foreground transition-all hover:text-primary active:scale-90"
            >
              <PanelLeftClose size={22} />
            </motion.button>
          </div>

          <div className="space-y-2">
            {mainNav.map((item) => {
              const isActive = activeView === item.id;
              return (
                <motion.button
                  key={item.id}
                  whileHover={{ x: 6 }}
                  onClick={() => onSelectView(item.id as ViewType)}
                  className={`relative w-full flex items-center gap-4 px-6 py-4 rounded-2xl text-[12px] font-black uppercase tracking-widest transition-all group overflow-hidden ${
                    isActive 
                      ? 'bg-primary text-primary-foreground shadow-xl shadow-primary/20' 
                      : 'text-muted-foreground dark:text-zinc-400 hover:bg-background hover:text-foreground border border-transparent hover:border-border/40 hover:shadow-sm'
                  }`}
                >
                  <motion.div 
                    animate={isActive ? { scale: 1.1, rotate: 0 } : { scale: 1, rotate: 0 }}
                    className={`${isActive ? 'text-primary-foreground' : 'text-primary/60 group-hover:text-primary group-hover:scale-110'} transition-all duration-300 relative z-10`}
                  >
                    {item.icon}
                  </motion.div>
                  <span className="relative z-10 flex-1 text-left">{item.label}</span>
                  {typeof item.badge === 'number' && item.badge > 0 && (
                    <span className="relative z-10 px-2 py-0.5 rounded-full bg-red-500 text-white text-[9px] font-black tabular-nums">
                      {item.badge > 99 ? "99+" : item.badge}
                    </span>
                  )}
                  
                  <AnimatePresence>
                    {!isActive && (
                      <motion.div 
                        initial={{ opacity: 0, x: -10 }}
                        whileHover={{ opacity: 1, x: 0 }}
                        className="absolute left-0 w-1 h-5 bg-primary/40 rounded-r-full"
                      />
                    )}
                  </AnimatePresence>

                  {isActive && (
                    <motion.div 
                      layoutId="sidebar-active-pill" 
                      className="absolute inset-0 bg-primary -z-10"
                      transition={springTransition}
                    />
                  )}
                </motion.button>
              );
            })}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 custom-scrollbar">
          <div className="flex items-center justify-between px-3 mb-4 mt-6">
            <p className="text-[10px] font-black text-muted-foreground dark:text-zinc-500 uppercase tracking-[0.25em] opacity-80">{t.smart_sessions}</p>
            <motion.button 
              whileHover={{ scale: 1.1, rotate: 90, backgroundColor: 'hsla(var(--primary) / 0.2)' }}
              whileTap={{ scale: 0.9 }}
              onClick={onNewChat}
              className="w-8 h-8 bg-primary/10 text-primary rounded-xl flex items-center justify-center transition-all shadow-sm"
            >
              <Plus size={16} strokeWidth={3} />
            </motion.button>
          </div>
          
          <div className="space-y-1.5 pb-8">
            {sessions.map((session) => {
              const isActive = currentSessionId === session.id && activeView === 'chat';
              const isPendingDelete = sessionToDelete === session.id;
              
              return (
                <motion.div
                  key={session.id}
                  layout
                  whileHover={{ x: 4, scale: 1.01 }}
                  className={`group relative flex items-center justify-between w-full px-5 py-3.5 rounded-xl text-[13px] transition-all cursor-pointer border accelerated ${
                    isPendingDelete
                      ? 'border-red-500 ring-4 ring-red-500/10 shadow-lg z-10 bg-red-500/5'
                      : isActive
                      ? 'bg-primary/[0.08] border-primary/30 text-foreground font-bold shadow-sm'
                      : 'bg-transparent hover:bg-background text-muted-foreground dark:text-zinc-400 border-transparent hover:border-border/40 hover:shadow-sm'
                  }`}
                  onClick={() => {
                    if (!isPendingDelete) {
                      onSelectSession(session.id);
                      onSelectView('chat');
                    }
                  }}
                >
                  <div className="flex items-center gap-3.5 overflow-hidden relative z-10">
                    <Layers size={16} className={`flex-shrink-0 transition-all duration-500 ${isPendingDelete ? 'text-red-500' : isActive ? 'text-primary' : 'opacity-40 group-hover:opacity-100 group-hover:scale-110'}`} />
                    <span className={`truncate tracking-tight transition-all duration-500 ${isPendingDelete ? 'text-red-500' : isActive ? 'text-foreground font-black' : ''}`}>{session.title}</span>
                  </div>
                  
                  {!isPendingDelete && (
                    <motion.button
                      initial={{ opacity: 0.4, scale: 0.9 }}
                      whileHover={{ 
                        opacity: 1, 
                        scale: 1.15, 
                        backgroundColor: 'rgba(239, 68, 68, 0.15)' 
                      }}
                      className="p-1.5 rounded-lg transition-all z-20 group/trash"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSessionToDelete(session.id);
                      }}
                    >
                      <Trash2 
                        size={15} 
                        className="text-muted-foreground transition-colors group-hover/trash:text-red-500" 
                      />
                    </motion.button>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="p-6 border-t border-border/40 bg-muted/20 dark:bg-zinc-900/40 space-y-3 mt-auto shrink-0">
          <motion.button
            whileHover={{ y: -3, backgroundColor: 'hsla(var(--background) / 0.8)', boxShadow: '0 10px 25px -10px rgba(0,0,0,0.1)' }}
            whileTap={{ scale: 0.98 }}
            onClick={toggleDarkMode}
            className="flex items-center gap-4 w-full px-5 py-3.5 rounded-2xl text-[12px] font-bold transition-all text-muted-foreground dark:text-zinc-300 hover:text-foreground border border-transparent hover:border-border/60 group glass-card"
          >
            <div className="w-8 h-8 rounded-xl bg-muted/50 dark:bg-zinc-800 flex items-center justify-center group-hover:bg-primary/10 group-hover:text-primary transition-all group-hover:rotate-12">
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </div>
            <span className="flex-1 text-left uppercase tracking-widest">{isDarkMode ? t.interface_light : t.interface_dark}</span>
          </motion.button>
          
          <motion.button 
            whileHover={{ y: -3, backgroundColor: 'hsla(var(--background) / 0.8)', boxShadow: '0 10px 25px -10px rgba(0,0,0,0.1)' }}
            whileTap={{ scale: 0.98 }}
            onClick={onOpenSettings}
            className="flex items-center gap-4 w-full px-5 py-3.5 rounded-2xl text-[12px] font-bold transition-all text-muted-foreground dark:text-zinc-300 hover:text-foreground border border-transparent hover:border-border/60 group glass-card"
          >
            <div className="w-8 h-8 rounded-xl bg-muted/50 dark:bg-zinc-800 flex items-center justify-center group-hover:bg-primary/10 group-hover:text-primary transition-all group-hover:rotate-[30deg]">
              <Settings size={18} />
            </div>
            <span className="flex-1 text-left uppercase tracking-widest">{t.configuration}</span>
          </motion.button>
        </div>
      </div>

      <AnimatePresence>
        {sessionToDelete && (
          <div className="fixed inset-0 z-[10000] flex items-center justify-center p-6 bg-black/40 backdrop-blur-md pointer-events-auto">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 cursor-default"
              onClick={() => setSessionToDelete(null)}
            />

            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              transition={springTransition}
              className="relative w-full max-w-[400px] bg-background rounded-[3rem] p-10 shadow-2xl border border-border overflow-hidden text-center"
            >
              <div className="flex justify-center mb-6">
                <div className="w-20 h-20 rounded-[2rem] bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-500">
                  <AlertTriangle size={36} />
                </div>
              </div>
              <h3 className="text-2xl font-black tracking-tight mb-2">{language === 'es' ? '¿Purgar Sesión?' : 'Purge Session?'}</h3>
              <p className="text-muted-foreground text-sm font-medium mb-8">
                {language === 'es' 
                  ? 'Esta acción eliminará todos los rastros neuronales de esta conversación de forma permanente.' 
                  : 'This action will permanently remove all neural traces of this conversation.'}
              </p>
              
              <div className="flex flex-col gap-3">
                <motion.button
                  whileHover={{ scale: 1.02, backgroundColor: 'hsla(0, 72%, 51%, 1)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleDeleteConfirm}
                  className="w-full py-4 bg-red-500 text-white rounded-2xl font-black uppercase tracking-widest text-[10px] shadow-lg shadow-red-500/20"
                >
                  {language === 'es' ? 'Confirmar Eliminación' : 'Confirm Deletion'}
                </motion.button>
                <button
                  onClick={() => setSessionToDelete(null)}
                  className="w-full py-4 text-muted-foreground hover:text-foreground font-bold text-sm transition-colors"
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
