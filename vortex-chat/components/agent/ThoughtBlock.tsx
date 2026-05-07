import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Loader2 } from 'lucide-react';
import type { Language } from '../../types';

interface ThoughtBlockProps {
  text: string;
  isStreaming: boolean;
  language: Language;
}

const ThoughtBlock: React.FC<ThoughtBlockProps> = ({ text, isStreaming, language }) => {
  if (!text) return null;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="overflow-hidden"
    >
      <div className="flex items-start gap-3 rounded-xl border border-primary/15 bg-primary/5 px-4 py-3">
        <div className="mt-0.5 shrink-0">
          {isStreaming ? (
            <div className="relative">
              <Brain size={14} className="text-primary animate-pulse" />
              <motion.div
                animate={{ opacity: [0, 1, 0] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
                className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full blur-[2px]"
              />
            </div>
          ) : (
            <Brain size={14} className="text-primary/60" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[9px] font-black uppercase tracking-[0.2em] text-primary/50 mb-1.5">
            {isStreaming
              ? (language === 'es' ? 'Razonando...' : 'Thinking...')
              : (language === 'es' ? 'Razonamiento' : 'Reasoning')
            }
          </div>
          <p className="text-[12px] leading-relaxed text-foreground/70 font-medium">
            {text}
            {isStreaming && (
              <motion.span
                animate={{ opacity: [1, 0] }}
                transition={{ repeat: Infinity, duration: 0.8 }}
                className="inline-block w-[2px] h-[14px] bg-primary ml-0.5 align-text-bottom"
              />
            )}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default ThoughtBlock;
