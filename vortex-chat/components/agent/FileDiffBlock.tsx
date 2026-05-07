import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileCode, Plus, Minus, ChevronDown, ChevronRight } from 'lucide-react';
import type { Language } from '../../types';

interface FileDiffBlockProps {
  path: string;
  diff: string;
  language: Language;
  defaultExpanded?: boolean;
}

const FileDiffBlock: React.FC<FileDiffBlockProps> = ({ path, diff, language, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const { lines, stats } = useMemo(() => {
    const rawLines = diff.split('\n');
    let added = 0;
    let removed = 0;
    let lineNum = 1;
    const processed = rawLines.map((line) => {
      const isAdded = line.startsWith('+');
      const isRemoved = line.startsWith('-');
      if (isAdded) added++;
      if (isRemoved) removed++;
      return {
        type: isAdded ? 'added' as const : isRemoved ? 'removed' as const : 'neutral' as const,
        content: line.replace(/^[+-]/, ''),
        lineNum: lineNum++,
      };
    });
    return { lines: processed, stats: { added, removed, total: processed.length } };
  }, [diff]);

  const fileName = path.split(/[/\\]/).pop() || path;
  const dirPath = path.includes('/') || path.includes('\\')
    ? path.slice(0, path.lastIndexOf(path.includes('/') ? '/' : '\\'))
    : '';

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-zinc-700/50 bg-zinc-900/80 overflow-hidden shadow-md"
    >
      {/* File Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between px-4 py-2.5 bg-zinc-800/50 border-b border-zinc-700/30 cursor-pointer hover:bg-zinc-800/70 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <FileCode size={14} className="text-primary/70 shrink-0" />
          <div className="flex items-baseline gap-1.5 min-w-0 truncate">
            {dirPath && (
              <span className="text-[10px] font-mono text-zinc-600 truncate">{dirPath}/</span>
            )}
            <span className="text-[12px] font-mono font-bold text-foreground/90">{fileName}</span>
          </div>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 text-[10px] font-black text-emerald-400">
              <Plus size={9} strokeWidth={4} /> {stats.added}
            </span>
            <span className="flex items-center gap-1 text-[10px] font-black text-red-400">
              <Minus size={9} strokeWidth={4} /> {stats.removed}
            </span>
          </div>
          <div className="text-zinc-600">
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>
        </div>
      </div>

      {/* Diff Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="overflow-x-auto custom-scrollbar max-h-[400px] overflow-y-auto">
              {lines.map((line, i) => (
                <div
                  key={i}
                  className={`flex items-stretch min-h-[24px] border-l-[3px] ${
                    line.type === 'added'
                      ? 'bg-emerald-500/8 border-emerald-500/60'
                      : line.type === 'removed'
                        ? 'bg-red-500/8 border-red-500/60'
                        : 'bg-transparent border-transparent'
                  }`}
                >
                  {/* Line Number */}
                  <div className="w-12 shrink-0 flex items-center justify-end pr-3 text-[9px] font-mono text-zinc-600 select-none tabular-nums bg-black/20 border-r border-zinc-800/50">
                    {line.lineNum}
                  </div>
                  {/* Sign */}
                  <div className="w-6 shrink-0 flex items-center justify-center">
                    <span className={`font-mono text-[12px] font-black ${
                      line.type === 'added' ? 'text-emerald-400' : line.type === 'removed' ? 'text-red-400' : 'text-transparent'
                    }`}>
                      {line.type === 'added' ? '+' : line.type === 'removed' ? '−' : ' '}
                    </span>
                  </div>
                  {/* Content */}
                  <div className="flex-1 flex items-center px-3 min-w-0">
                    <span className={`whitespace-pre font-mono text-[11px] truncate ${
                      line.type === 'added' ? 'text-emerald-300/90' : line.type === 'removed' ? 'text-red-300/90' : 'text-zinc-400'
                    }`}>
                      {line.content || ' '}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default FileDiffBlock;
