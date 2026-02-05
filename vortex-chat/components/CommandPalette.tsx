
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Search, 
  MessageSquare, 
  Moon, 
  Sun, 
  Plus, 
  Settings, 
  Command,
  X,
  Layout,
  Download,
  Eraser,
  HelpCircle,
  Sparkles,
  ChevronRight,
  Type
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatSession, FontSize, Language } from '../types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onClearHistory: () => void;
  onExportChat: () => void;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
  categoryOrder: string[];
  onSetFontSize: (size: FontSize) => void;
  language: Language;
}

const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearHistory,
  onExportChat,
  isDarkMode,
  toggleDarkMode,
  isSidebarOpen,
  onToggleSidebar,
  onOpenSettings,
  onOpenHelp,
  categoryOrder,
  onSetFontSize,
  language
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const actions = useMemo(() => [
    { id: 'new-chat', title: 'Nuevo Chat', icon: <Plus size={18} />, action: onNewChat, category: 'Acciones Rápidas' },
    { id: 'toggle-dark', title: isDarkMode ? 'Modo Claro' : 'Modo Oscuro', icon: isDarkMode ? <Sun size={18} /> : <Moon size={18} />, action: toggleDarkMode, category: 'Preferencias' },
    { id: 'toggle-sidebar', title: isSidebarOpen ? 'Ocultar Lateral' : 'Mostrar Lateral', icon: <Layout size={18} />, action: onToggleSidebar, category: 'Interfaz' },
    { id: 'font-small', title: 'Fuente: Pequeña', icon: <Type size={14} />, action: () => onSetFontSize('small'), category: 'Interfaz' },
    { id: 'font-medium', title: 'Fuente: Normal', icon: <Type size={18} />, action: () => onSetFontSize('medium'), category: 'Interfaz' },
    { id: 'font-large', title: 'Fuente: Grande', icon: <Type size={22} />, action: () => onSetFontSize('large'), category: 'Interfaz' },
    { 
        id: 'export-chat', 
        title: 'Exportar Chat (Markdown)', 
        icon: <Download size={18} />, 
        action: onExportChat, 
        category: 'Datos',
        disabled: !currentSessionId
    },
    { 
        id: 'clear-history', 
        title: 'Purgar Historial', 
        icon: <Eraser size={18} />, 
        action: onClearHistory, 
        category: 'Datos',
        disabled: sessions.length === 0
    },
    { id: 'settings', title: 'Configuración Avanzada', icon: <Settings size={18} />, action: onOpenSettings, category: 'Sistema' },
    { id: 'help', title: 'Ayuda y Atajos', icon: <HelpCircle size={18} />, action: onOpenHelp, category: 'Sistema' },
  ], [isDarkMode, isSidebarOpen, currentSessionId, sessions.length, onNewChat, toggleDarkMode, onToggleSidebar, onExportChat, onClearHistory, onOpenSettings, onOpenHelp, onSetFontSize]);

  const filteredItems = useMemo(() => {
    const search = query.toLowerCase();
    
    const matchingActions = actions.filter(a => !a.disabled && a.title.toLowerCase().includes(search));
    const matchingSessions = sessions
      .filter(s => s.title.toLowerCase().includes(search))
      .map(s => ({
        id: s.id,
        title: s.title,
        icon: <MessageSquare size={18} />,
        action: () => onSelectSession(s.id),
        category: 'Chats Recientes'
      }));

    const allItems = [...matchingActions, ...matchingSessions];

    return allItems.sort((a, b) => {
      const indexA = categoryOrder.indexOf(a.category);
      const indexB = categoryOrder.indexOf(b.category);
      return indexA - indexB;
    });
  }, [query, actions, sessions, onSelectSession, categoryOrder]);

  const orderedCategories = useMemo(() => {
    const activeCategories = new Set(filteredItems.map(i => i.category));
    return categoryOrder.filter(cat => activeCategories.has(cat));
  }, [filteredItems, categoryOrder]);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % Math.max(1, filteredItems.length));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredItems.length) % Math.max(1, filteredItems.length));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredItems[selectedIndex]) {
          filteredItems[selectedIndex].action();
          onClose();
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredItems, selectedIndex, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] px-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-zinc-950/40 backdrop-blur-xl"
            onClick={onClose}
          />
          
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 40, filter: 'blur(10px)' }}
            animate={{ opacity: 1, scale: 1, y: 0, filter: 'blur(0px)' }}
            exit={{ opacity: 0, scale: 0.95, y: 20, filter: 'blur(10px)' }}
            className="relative w-full max-w-2xl bg-white/95 dark:bg-zinc-900/95 border border-border shadow-2xl rounded-[2.5rem] overflow-hidden flex flex-col"
          >
            <div className="relative flex items-center px-10 border-b border-border/50 h-24 bg-muted/20 dark:bg-zinc-900/40">
              <Search size={22} className="text-primary mr-6 shrink-0 opacity-60" />
              <input
                ref={inputRef}
                type="text"
                className="flex-1 bg-transparent border-none outline-none text-foreground placeholder:text-muted-foreground/30 dark:placeholder:text-zinc-600 text-xl font-bold tracking-tight"
                placeholder="Busca un comando o sesión..."
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setSelectedIndex(0);
                }}
              />
              <button
                onClick={onClose}
                aria-label={language === 'es' ? 'Cerrar' : 'Close'}
                title={language === 'es' ? 'Cerrar' : 'Close'}
                className="p-3 hover:bg-muted dark:hover:bg-zinc-800 rounded-full transition-all"
              >
                <X size={20} className="text-muted-foreground" />
              </button>
            </div>

            <div className="max-h-[50vh] overflow-y-auto p-5 custom-scrollbar">
              <AnimatePresence mode="popLayout">
                {filteredItems.length === 0 ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="py-16 flex flex-col items-center justify-center text-muted-foreground gap-4"
                  >
                    <Sparkles size={32} className="opacity-20" />
                    <p className="text-xs font-black uppercase tracking-[0.3em] opacity-40">No se encontraron protocolos</p>
                  </motion.div>
                ) : (
                  <div className="space-y-8 py-4">
                    {orderedCategories.map(category => (
                      <div key={category} className="space-y-3">
                        <h3 className="px-6 text-[10px] font-black text-primary/60 uppercase tracking-[0.4em] mb-4">
                          {category}
                        </h3>
                        <div className="space-y-1.5">
                          {filteredItems
                            .map((item, index) => ({ item, index }))
                            .filter(({ item }) => item.category === category)
                            .map(({ item, index }) => {
                              const isSelected = index === selectedIndex;
                              return (
                                <motion.button
                                  key={item.id}
                                  layout
                                  onMouseEnter={() => setSelectedIndex(index)}
                                  onClick={() => {
                                    item.action();
                                    onClose();
                                  }}
                                  className={`w-full flex items-center gap-5 px-6 py-4.5 rounded-2xl text-left transition-all relative group overflow-hidden ${
                                    isSelected 
                                      ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20 z-10' 
                                      : 'hover:bg-muted/60 dark:hover:bg-zinc-800/60 text-foreground/70 dark:text-zinc-400'
                                  }`}
                                >
                                  {isSelected && (
                                    <motion.div 
                                      layoutId="command-highlight"
                                      className="absolute inset-0 bg-primary -z-10"
                                      transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                                    />
                                  )}
                                  
                                  <div className={`shrink-0 transition-transform duration-500 ${isSelected ? 'scale-110 rotate-3' : 'opacity-40'}`}>
                                    {item.icon}
                                  </div>
                                  
                                  <div className="flex-1 min-w-0 flex items-center justify-between">
                                    <p className={`text-[14.5px] font-bold tracking-tight ${isSelected ? 'text-white' : ''}`}>
                                      {item.title}
                                    </p>
                                    {isSelected && (
                                      <motion.div 
                                        initial={{ x: -5, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        className="shrink-0"
                                      >
                                        <ChevronRight size={16} />
                                      </motion.div>
                                    )}
                                  </div>
                                </motion.button>
                              );
                            })}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </AnimatePresence>
            </div>

            <div className="px-10 py-5 bg-muted/10 dark:bg-zinc-900/60 border-t border-border/40 flex items-center justify-between text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/40">
              <div className="flex gap-6">
                <span className="flex items-center gap-2">
                  <kbd className="px-2 py-0.5 bg-background dark:bg-zinc-800 border border-border rounded-lg">↑↓</kbd>
                  Navegar
                </span>
                <span className="flex items-center gap-2">
                  <kbd className="px-2 py-0.5 bg-background dark:bg-zinc-800 border border-border rounded-lg">↵</kbd>
                  Ejecutar
                </span>
              </div>
              <div className="flex items-center gap-2">
                 <Command size={14} />
                 <span>Consola de Inteligencia</span>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default CommandPalette;
