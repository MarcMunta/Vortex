
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Square, ArrowUp, Globe, Timer } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { AppMode, Language } from '../types';
import { translations } from '../translations';

interface ChatInputProps {
  onSend: (message: string, useInternet: boolean, mode: AppMode, useThinking: boolean) => void;
  isLoading: boolean;
  isDarkMode: boolean;
  language: Language;
  isThinking?: boolean;
  onStop?: () => void;
  onInteraction?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, isLoading, isDarkMode, language, isThinking, onStop, onInteraction }) => {
  const [input, setInput] = useState('');
  const [isInternetEnabled, setIsInternetEnabled] = useState(false);
  const [useThinking, setUseThinking] = useState(true);
  const [mode, setMode] = useState<AppMode>('ask');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const t = translations[language];
  const MAX_HEIGHT = 200; 

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

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onInteraction?.();
      onSend(input.trim(), isInternetEnabled, mode, useThinking);
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

  return (
    <div className="max-w-[840px] mx-auto w-full relative px-6 accelerated">
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            className="absolute -top-12 left-10 flex items-center gap-3 px-5 py-2 bg-background dark:bg-zinc-900 border border-border/50 shadow-2xl rounded-2xl glass-card z-20"
          >
            <div className="flex gap-1.5">
              {[0, 0.2, 0.4].map((d) => (
                <motion.span
                  key={d}
                  animate={{ opacity: [0.3, 1, 0.3], scale: [0.9, 1.1, 0.9] }}
                  transition={{ duration: 1.2, repeat: Infinity, delay: d }}
                  className={`w-1.5 h-1.5 rounded-full ${mode === 'agent' ? 'bg-purple-500' : 'bg-primary'}`}
                />
              ))}
            </div>
            <span className="text-[9px] font-black uppercase tracking-[0.2em] opacity-70">
              {mode === 'agent' 
                ? (language === 'es' ? 'Motor de Agente Activo' : 'Agent Engine Active') 
                : (language === 'es' ? 'Procesando Flujo' : 'Processing Flow')}
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div 
        layout
        className={`relative flex items-end w-full bg-white/95 dark:bg-zinc-900/95 backdrop-blur-3xl border rounded-[2.2rem] transition-all duration-500 p-2 group shadow-2xl accelerated ${
          isLoading 
            ? 'border-border/30 opacity-90' 
            : mode === 'agent'
              ? 'border-purple-500/30 focus-within:border-purple-500/60 focus-within:ring-[8px] focus-within:ring-purple-500/5'
              : 'border-border/50 focus-within:border-primary/60 focus-within:ring-[8px] focus-within:ring-primary/5'
        }`}
      >
        <div className="flex items-center mb-1 ml-1 relative z-10 shrink-0">
          <button
            onClick={() => { setUseThinking(!useThinking); onInteraction?.(); }}
            aria-label={useThinking ? (language === 'es' ? 'Modo Thinking activo' : 'Thinking mode active') : (language === 'es' ? 'Modo Fast activo' : 'Fast mode active')}
            title={useThinking ? (language === 'es' ? 'Modo Thinking Activo' : 'Thinking Mode Active') : (language === 'es' ? 'Modo Fast Activo' : 'Fast Mode Active')}
            className={`w-11 h-11 rounded-full transition-all duration-300 flex items-center justify-center border border-transparent ${
              useThinking 
                ? 'text-primary bg-primary/5 border-primary/10' 
                : 'text-muted-foreground/40 dark:text-zinc-600 hover:text-foreground/60'
            }`}
          >
            <Timer size={22} strokeWidth={useThinking ? 2.5 : 1.5} className="transition-all duration-500" />
          </button>
        </div>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => { setInput(e.target.value); onInteraction?.(); }}
          onKeyDown={handleKeyDown}
          onFocus={onInteraction}
          placeholder={mode === 'agent' ? t.input_placeholder_agent : t.input_placeholder_ask}
          rows={1}
          className="flex-1 bg-transparent border-none focus:ring-0 outline-none resize-none py-4 px-3 text-[15px] leading-relaxed placeholder:text-muted-foreground/30 dark:placeholder:text-zinc-600 font-medium text-foreground relative z-10 custom-scrollbar"
        />
        
        <div className="flex items-center gap-2 mb-1 mr-1 relative z-10 shrink-0">
          <div className="relative flex bg-muted/40 dark:bg-zinc-800/60 p-0.5 rounded-2xl border border-border/30 h-[44px] min-w-[120px] shadow-inner overflow-hidden">
            <motion.div
              animate={{ x: mode === 'ask' ? 0 : 58, backgroundColor: mode === 'ask' ? 'hsl(var(--primary))' : '#8b5cf6' }}
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              className="absolute top-0.5 bottom-0.5 w-[58px] rounded-xl shadow-lg z-0"
            />
            <button onClick={() => { setMode('ask'); onInteraction?.(); }} className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 text-[8px] font-black uppercase tracking-widest transition-all duration-300 ${mode === 'ask' ? 'text-white' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>
               {t.input_ask_mode}
            </button>
            <button onClick={() => { setMode('agent'); onInteraction?.(); }} className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 text-[8px] font-black uppercase tracking-widest transition-all duration-300 ${mode === 'agent' ? 'text-white' : 'text-muted-foreground dark:text-zinc-400 hover:text-foreground'}`}>
               {t.input_agent_mode}
            </button>
          </div>

          <button
            onClick={() => { setIsInternetEnabled(!isInternetEnabled); onInteraction?.(); }}
            aria-label={isInternetEnabled ? (language === 'es' ? 'Desactivar Internet' : 'Disable internet') : (language === 'es' ? 'Activar Internet' : 'Enable internet')}
            title={isInternetEnabled ? (language === 'es' ? 'Internet activado' : 'Internet enabled') : (language === 'es' ? 'Internet desactivado' : 'Internet disabled')}
            className={`p-2.5 rounded-full transition-all duration-300 flex items-center justify-center border border-transparent ${
              isInternetEnabled 
                ? 'bg-primary/15 text-primary border-primary/20 shadow-inner' 
                : 'text-muted-foreground dark:text-zinc-400 hover:bg-muted dark:hover:bg-zinc-800 hover:text-foreground'
            }`}
          >
            <Globe size={18} className={isInternetEnabled ? 'animate-pulse' : ''} />
          </button>

          {isLoading ? (
            <button
              onClick={() => onStop?.()}
              aria-label={language === 'es' ? 'Detener' : 'Stop'}
              title={language === 'es' ? 'Detener' : 'Stop'}
              className="w-11 h-11 bg-red-500 text-white rounded-full transition-all hover:scale-110 active:scale-90 shadow-xl flex items-center justify-center"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              aria-label={language === 'es' ? 'Enviar mensaje' : 'Send message'}
              title={language === 'es' ? 'Enviar mensaje' : 'Send message'}
              className={`w-11 h-11 rounded-full transition-all duration-300 shadow-xl flex items-center justify-center group/send ${
                input.trim()
                  ? mode === 'agent' 
                    ? 'bg-purple-600 text-white hover:scale-110 active:scale-90 shadow-purple-500/20'
                    : 'bg-primary text-white hover:scale-110 active:scale-90 shadow-primary/20'
                  : 'bg-muted/40 dark:bg-zinc-800/40 text-muted-foreground/10 cursor-not-allowed shadow-none'
              }`}
            >
              <ArrowUp size={22} strokeWidth={3} className={`transition-transform duration-300 ${input.trim() ? 'group-hover/send:-translate-y-0.5' : ''}`} />
            </button>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default ChatInput;
