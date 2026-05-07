import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, ChevronDown, ChevronRight, Copy, CheckCircle2 } from 'lucide-react';
import type { Language } from '../../types';

interface TerminalBlockProps {
  command: string;
  output?: string;
  isStreaming?: boolean;
  language: Language;
}

const TerminalBlock: React.FC<TerminalBlockProps> = ({ command, output, isStreaming, language }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [copied, setCopied] = useState(false);
  const outputRef = useRef<HTMLPreElement>(null);

  // Auto-scroll to bottom when streaming
  useEffect(() => {
    if (isStreaming && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output, isStreaming]);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    const text = command + (output ? '\n' + output : '');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-zinc-700/60 bg-zinc-900/90 overflow-hidden shadow-lg"
    >
      {/* Terminal Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between px-4 py-2.5 bg-zinc-800/60 border-b border-zinc-700/40 cursor-pointer hover:bg-zinc-800/80 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <Terminal size={13} className="text-emerald-400 shrink-0" />
          <code className="text-[12px] font-mono font-bold text-emerald-300/90 truncate">
            $ {command}
          </code>
          {isStreaming && (
            <motion.div
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0"
            />
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-white/5 transition-all active:scale-95"
          >
            {copied ? <CheckCircle2 size={12} className="text-emerald-400" /> : <Copy size={12} />}
          </button>
          <div className="text-zinc-600">
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>
        </div>
      </div>

      {/* Terminal Output */}
      <AnimatePresence>
        {isExpanded && output && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <pre
              ref={outputRef}
              className="px-4 py-3 text-[11px] font-mono leading-[1.6] text-zinc-400 overflow-x-auto custom-scrollbar max-h-[300px] overflow-y-auto"
            >
              {output}
              {isStreaming && (
                <motion.span
                  animate={{ opacity: [1, 0] }}
                  transition={{ repeat: Infinity, duration: 0.6 }}
                  className="inline-block w-[7px] h-[14px] bg-emerald-400/80 ml-0.5 align-text-bottom"
                />
              )}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default TerminalBlock;
