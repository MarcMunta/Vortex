
import React, { useMemo } from 'react';
import { X, Keyboard, MessageSquare, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Language } from '../types';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
  isDarkMode: boolean;
  language: Language;
}

const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose, isDarkMode, language }) => {
  const keyboardShortcuts = [
    { key: 'Alt + K', desc: 'Abrir paleta de comandos' },
    { key: 'Esc', desc: 'Cerrar ventanas emergentes' },
    { key: 'Enter', desc: 'Enviar mensaje' },
    { key: 'Shift + ↵', desc: 'Salto de línea' },
  ];

  const chatCommands = useMemo(() => [
    { cmd: '/new', desc: 'Iniciar un nuevo chat' },
    { cmd: '/clear', desc: 'Borrar todo el historial' },
    { cmd: isDarkMode ? '/light' : '/dark', desc: isDarkMode ? 'Modo claro' : 'Modo oscuro' },
    { cmd: '/export', desc: 'Exportar chat actual' },
    { cmd: '/settings', desc: 'Abrir configuración' },
    { cmd: '/help', desc: 'Mostrar esta ayuda' },
  ], [isDarkMode]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[120] flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm" 
            onClick={onClose} 
          />
          <motion.div 
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            className="relative w-full max-w-lg bg-background border border-border shadow-2xl rounded-3xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-muted/20">
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-primary/10 rounded-lg text-primary">
                  <Sparkles size={18} />
                </div>
                <h2 className="text-lg font-bold">Ayuda y Atajos</h2>
              </div>
              <button
                onClick={onClose}
                aria-label={language === 'es' ? 'Cerrar' : 'Close'}
                title={language === 'es' ? 'Cerrar' : 'Close'}
                className="p-2 hover:bg-muted rounded-full text-muted-foreground transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            
            <div className="p-6 space-y-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
              <section>
                <div className="flex items-center gap-2 mb-4 text-primary">
                  <Keyboard size={18} />
                  <h3 className="text-sm font-bold uppercase tracking-widest">Atajos de Teclado</h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {keyboardShortcuts.map((item) => (
                    <div key={item.key} className="flex items-center justify-between p-3 bg-muted/30 border border-border rounded-xl">
                      <span className="text-xs font-medium text-muted-foreground">{item.desc}</span>
                      <kbd className="px-2 py-1 bg-background border border-border rounded text-[10px] font-bold shadow-sm whitespace-nowrap">
                        {item.key}
                      </kbd>
                    </div>
                  ))}
                </div>
              </section>

              <section>
                <div className="flex items-center gap-2 mb-4 text-primary">
                  <MessageSquare size={18} />
                  <h3 className="text-sm font-bold uppercase tracking-widest">Comandos de Chat</h3>
                </div>
                <div className="space-y-2">
                  {chatCommands.map((item) => (
                    <div key={item.cmd} className="flex items-center justify-between p-3 bg-muted/30 border border-border rounded-xl group hover:border-primary/30 transition-colors">
                      <code className="text-sm font-bold text-primary">{item.cmd}</code>
                      <span className="text-xs font-medium text-muted-foreground">{item.desc}</span>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <div className="p-6 bg-muted/20 border-t border-border flex justify-center">
              <p className="text-[11px] font-medium text-muted-foreground italic text-center">
                Escribe <span className="font-bold text-primary">/</span> en el chat para ver sugerencias en tiempo real.
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default HelpModal;
