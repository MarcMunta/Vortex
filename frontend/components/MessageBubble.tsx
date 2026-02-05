
import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { 
  User, Bot, Copy, CheckCircle2, Brain, Globe, 
  FileCode, Plus, Minus, Equal, Maximize2, Minimize2, 
  Code2, ChevronRight, Loader2, Fingerprint, 
  GitBranch, FileStack, ChevronDown, ChevronUp, ExternalLink, ShieldCheck
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { Message, Role, Source, FontSize, Language } from '../types';

interface MessageBubbleProps {
  message: Message;
  fontSize?: FontSize;
  codeTheme?: 'dark' | 'light' | 'match-app';
  onShowReasoning?: (messageId: string) => void;
  onOpenModificationExplorer: (fileChanges: { path: string, diff: string }[]) => void;
  isStreaming?: boolean;
  language: Language;
}

const GroundingPill: React.FC<{ source: Source; index: number }> = ({ source }) => {
  const [isHovered, setIsHovered] = useState(false);
  const timeoutRef = useRef<number | null>(null);
  const safeTitle = source.title || source.domain || source.url;

  const handleMouseEnter = () => {
    if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = window.setTimeout(() => setIsHovered(false), 150);
  };

  return (
    <div className="relative inline-block" onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      <motion.a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2 px-3 py-1.5 bg-background border border-border/60 hover:border-primary/60 rounded-xl transition-all shadow-sm group glass-card relative z-10"
      >
        <div className="w-4 h-4 rounded-md bg-muted/40 flex items-center justify-center overflow-hidden shrink-0 text-muted-foreground/70">
          <ExternalLink size={12} />
        </div>
        <span className="text-[10px] font-bold text-foreground/70 group-hover:text-primary transition-colors truncate max-w-[120px]">
          {safeTitle}
        </span>
      </motion.a>

      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 15, scale: 0.95 }}
            animate={{ opacity: 1, y: -12, scale: 1 }}
            exit={{ opacity: 0, y: 15, scale: 0.95 }}
            className="absolute bottom-full left-0 mb-4 w-72 z-[2000] pointer-events-none"
          >
            <div className="bg-[#050505] border-[2px] border-black rounded-[2rem] shadow-[0_40px_80px_-15px_rgba(0,0,0,0.8)] overflow-hidden relative ring-1 ring-white/10">
              <div className="aspect-video w-full bg-zinc-950 overflow-hidden relative flex items-center justify-center">
                <div className="absolute inset-0 bg-gradient-to-b from-primary/10 to-transparent" />
                <div className="relative z-10 flex items-center gap-3 text-white/60">
                  <ExternalLink size={18} />
                  <span className="text-[10px] font-black uppercase tracking-[0.3em]">Source</span>
                </div>
              </div>
              <div className="px-4 py-3.5 bg-zinc-950 flex items-center gap-3">
                <div className="w-8 h-8 bg-white/5 border border-white/10 rounded-lg shrink-0 shadow-sm flex items-center justify-center text-white/70">
                  <ExternalLink size={16} />
                </div>
                <div className="flex flex-col min-w-0">
                  <div className="text-[11px] font-black text-white truncate leading-tight tracking-tight">{safeTitle}</div>
                  <div className="text-[9px] font-mono text-zinc-500 truncate mt-0.5">{String(source.url || '').replace(/^https?:\/\/(www\.)?/, '')}</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const DiffLine = ({ line, isDarkMode }: { line: any, isDarkMode: boolean }) => (
  <div 
    className={`grid grid-cols-[56px_28px_1fr] items-stretch w-full h-[24px] border-l-4 transition-colors relative group/line ${
      line.type === 'added' 
        ? 'bg-emerald-500/10 border-emerald-500' 
        : line.type === 'removed' 
          ? 'bg-red-500/10 border-red-500' 
          : 'bg-transparent border-transparent hover:bg-white/[0.02]'
    }`}
  >
    {/* Gutter: Número de línea con alineación perfecta */}
    <div className={`flex items-center justify-end pr-3 border-r select-none tabular-nums shrink-0 ${
      isDarkMode 
        ? 'text-zinc-600 bg-black/20 border-white/5' 
        : 'text-zinc-400 bg-black/5 border-zinc-200'
    }`}>
      <span className="text-[9px] font-mono leading-none tracking-tighter opacity-40">
        {line.type === 'added' ? line.newLine : line.oldLine || line.newLine}
      </span>
    </div>

    {/* Signo: Posicionado simétricamente en su columna */}
    <div className="flex items-center justify-center shrink-0">
      <span className={`font-mono text-[13px] font-black leading-none ${
        line.type === 'added' ? 'text-emerald-500' : line.type === 'removed' ? 'text-red-500' : 'text-zinc-800'
      }`}>
        {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
      </span>
    </div>

    {/* Código: Alineado exactamente al inicio de su área */}
    <div className="flex items-center px-2 min-w-0 overflow-hidden">
      <span className={`whitespace-pre font-mono text-[13px] font-medium tracking-tight truncate leading-none ${
        line.type === 'added' ? 'text-emerald-300' : line.type === 'removed' ? 'text-red-300' : isDarkMode ? 'text-zinc-300' : 'text-zinc-700'
      }`}>
        {line.content || ' '}
      </span>
    </div>
  </div>
);

const CodeBlock = React.memo(({ children, className, isCollapsed, onToggle, codeTheme }: any) => {
  const isFilePath = !!className?.includes('file:');
  const path = isFilePath ? className.split('file:')[1] : null;
  const [filter, setFilter] = useState<'all' | 'added' | 'removed'>('all');
  const [copied, setCopied] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const hoverTimeoutRef = useRef<number | null>(null);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const rotateX = useSpring(useTransform(mouseY, [-100, 100], [5, -5]), { stiffness: 200, damping: 30 });
  const rotateY = useSpring(useTransform(mouseX, [-100, 100], [-5, 5]), { stiffness: 200, damping: 30 });

  const isGlobalDark = document.documentElement.classList.contains('dark');
  const effectiveTheme = codeTheme === 'match-app' ? (isGlobalDark ? 'dark' : 'light') : codeTheme;
  const isCodeDark = effectiveTheme !== 'light';

  const { lines, stats } = useMemo(() => {
    const rawContent = String(children).trim();
    if (!rawContent) return { lines: [], stats: { added: 0, removed: 0, total: 0 } };
    const rawLines = rawContent.split('\n');
    let oldNum = 1;
    let newNum = 1;
    let added = 0;
    let removed = 0;
    const processed = rawLines.map(line => {
      const isAdded = line.startsWith('+');
      const isRemoved = line.startsWith('-');
      if (isAdded) added++;
      if (isRemoved) removed++;
      return {
        type: isAdded ? 'added' : isRemoved ? 'removed' : 'neutral',
        content: line.replace(/^[+-]/, ''),
        oldLine: isAdded ? null : oldNum++,
        newLine: isRemoved ? null : newNum++,
      };
    });
    return { lines: processed, stats: { added, removed, total: processed.length } };
  }, [children]);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(String(children));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    mouseX.set(x);
    mouseY.set(y);
  };

  const handleMouseEnter = () => {
    if (!isCollapsed) return;
    if (hoverTimeoutRef.current) window.clearTimeout(hoverTimeoutRef.current);
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) window.clearTimeout(hoverTimeoutRef.current);
    hoverTimeoutRef.current = window.setTimeout(() => setIsHovered(false), 150);
    mouseX.set(0);
    mouseY.set(0);
  };

  const springTransition = { type: 'spring' as const, stiffness: 350, damping: 40, mass: 1 };

  if (!isFilePath) {
    const lang = className?.replace('language-', '') || 'snippet';
    return (
      <div className={`mt-[14px] mb-6 rounded-3xl border border-border/40 overflow-hidden shadow-2xl group/code accelerated ring-1 ring-white/5 ${isCodeDark ? 'bg-[#0a0a0a]' : 'bg-zinc-50 border-zinc-200'}`}>
        <div className={`px-5 py-3 border-b border-white/5 flex items-center justify-between ${isCodeDark ? 'bg-white/[0.03]' : 'bg-zinc-100/50'}`}>
          <div className="flex items-center gap-3">
            <div className={`w-7 h-7 rounded-lg flex items-center justify-center border ${isCodeDark ? 'bg-primary/20 text-primary border-primary/20' : 'bg-primary/10 text-primary border-primary/10'}`}>
              <Code2 size={14} />
            </div>
            <span className={`text-[10px] font-black uppercase tracking-[0.2em] ${isCodeDark ? 'text-zinc-500' : 'text-zinc-400'}`}>{lang}</span>
          </div>
          <button onClick={handleCopy} className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[10px] font-black transition-all border active:scale-95 ${isCodeDark ? 'bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white border-white/5' : 'bg-white hover:bg-zinc-200 text-zinc-500 border-zinc-200'}`}>
            {copied ? <CheckCircle2 size={12} className="text-emerald-500" /> : <Copy size={12} />} {copied ? 'Copiado' : 'Copiar'}
          </button>
        </div>
        <pre className={`p-6 overflow-x-auto text-[13px] font-mono leading-relaxed custom-scrollbar ${isCodeDark ? 'text-zinc-300 bg-black/40' : 'text-zinc-800 bg-white'}`}><code>{children}</code></pre>
      </div>
    );
  }

  const filteredLines = lines.filter(l => filter === 'all' || l.type === filter);

  return (
    <div id={`file-${path}`} className="mt-[14px] mb-6 w-full max-w-full relative group/file overflow-visible scroll-mt-24">
      <AnimatePresence mode="popLayout" initial={false}>
        {isCollapsed ? (
          <motion.div
            key="collapsed"
            layoutId={`file-block-${path}`}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={springTransition}
            onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} onMouseMove={handleMouseMove}
            className={`inline-flex items-center gap-4 p-2.5 border rounded-2xl shadow-xl cursor-pointer transition-all group/pill active:scale-95 z-40 relative overflow-visible ${isCodeDark ? 'bg-zinc-900 border-white/10 hover:bg-zinc-800' : 'bg-zinc-100 border-zinc-200 hover:bg-zinc-200 text-zinc-900'}`}
          >
            <div className="w-10 h-10 bg-primary/20 text-primary rounded-xl flex items-center justify-center border border-primary/20 shrink-0 pointer-events-none"><FileCode size={20} /></div>
            <div className="flex flex-col mr-4 pointer-events-none min-w-0">
              <span className={`text-[11px] font-black uppercase tracking-tight whitespace-nowrap ${isCodeDark ? 'text-white' : 'text-zinc-900'}`}>{path}</span>
              <div className="flex items-center gap-2 mt-0.5">
                <div className="flex items-center gap-1 text-[9px] font-black text-emerald-500"><Plus size={8} strokeWidth={4} /> {stats.added}</div>
                <div className="flex items-center gap-1 text-[9px] font-black text-red-500"><Minus size={8} strokeWidth={4} /> {stats.removed}</div>
              </div>
            </div>
            <Maximize2 size={14} className="text-zinc-600 group-hover/pill:text-primary transition-colors mr-2 shrink-0 pointer-events-none" />
            
            <AnimatePresence>
              {isHovered && (
                <motion.div 
                  initial={{ opacity: 0, y: 10, scale: 0.92, filter: 'blur(15px)' }} 
                  animate={{ opacity: 1, y: -20, scale: 1, filter: 'blur(0px)' }} 
                  exit={{ opacity: 0, y: 10, scale: 0.92, filter: 'blur(15px)' }} 
                  style={{ rotateX, rotateY, perspective: 1500 }} 
                  className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 w-[540px] z-[3000] pointer-events-none"
                >
                  <div className={`border-2 rounded-[2rem] shadow-[0_60px_120px_-20px_rgba(0,0,0,1)] overflow-hidden ring-1 ring-primary/20 relative ${isCodeDark ? 'bg-[#020202] border-black' : 'bg-white border-zinc-200 shadow-xl'}`}>
                    <div className="absolute inset-0 pointer-events-none overflow-hidden">
                       <motion.div 
                         animate={{ y: ['-100%', '200%'] }} 
                         transition={{ duration: 4, repeat: Infinity, ease: 'linear' }} 
                         className="absolute top-0 left-0 right-0 h-[80px] bg-gradient-to-b from-transparent via-primary/5 to-transparent z-0" 
                       />
                       <motion.div 
                         animate={{ y: ['0%', '100%', '0%'], opacity: [0, 0.4, 0] }} 
                         transition={{ duration: 3, repeat: Infinity, ease: 'linear' }} 
                         className="absolute top-0 left-0 right-0 h-[1px] bg-primary shadow-[0_0_25px_rgba(var(--primary),0.8)] z-50" 
                       />
                    </div>
                    
                    <div className={`px-8 py-5 border-b flex items-center justify-between relative z-10 backdrop-blur-xl ${isCodeDark ? 'border-white/10 bg-zinc-900/90' : 'border-zinc-200 bg-zinc-50/90'}`}>
                      <div className="flex items-center gap-4">
                         <div className="flex gap-1.5 mr-2">
                           <div className="w-2 h-2 rounded-full bg-red-500/80" />
                           <div className="w-2 h-2 rounded-full bg-yellow-500/80" />
                           <div className="w-2 h-2 rounded-full bg-green-500/80" />
                         </div>
                         <div className={`h-4 w-[1px] mx-1 ${isCodeDark ? 'bg-white/10' : 'bg-zinc-300'}`} />
                         <div className="flex flex-col">
                           <div className="flex items-center gap-2 mb-0.5">
                             <span className="text-[9px] font-black text-primary uppercase tracking-[0.4em] leading-none">PATCH_INTEGRITY: VERIFIED</span>
                           </div>
                           <span className={`text-[15px] font-black uppercase tracking-tight ${isCodeDark ? 'text-white' : 'text-zinc-900'}`}>{path}</span>
                         </div>
                      </div>
                      <div className={`p-2.5 rounded-xl border shadow-inner ${isCodeDark ? 'bg-white/5 border-white/5' : 'bg-white border-zinc-200'}`}>
                        <Fingerprint size={16} className="text-primary/60" />
                      </div>
                    </div>
                    
                    <div className={`p-6 space-y-px relative z-10 ${isCodeDark ? 'bg-[#010101]' : 'bg-white'}`}>
                      {lines.slice(0, 8).map((l, i) => (
                        <div key={i} className={`flex gap-0 overflow-hidden first:rounded-t-xl last:rounded-b-xl border-x border-y-[0.5px] transition-all ${l.type === 'added' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : l.type === 'removed' ? 'bg-red-500/10 border-red-500/30 text-red-400' : isCodeDark ? 'bg-white/[0.03] border-white/5 text-zinc-400' : 'bg-zinc-50 border-zinc-200 text-zinc-600'}`}>
                          <div className={`w-12 shrink-0 font-mono font-black text-center text-[10px] py-2.5 border-r border-inherit tabular-nums ${isCodeDark ? 'bg-black/40' : 'bg-zinc-100'} ${l.type === 'added' ? 'text-emerald-500' : l.type === 'removed' ? 'text-red-500' : 'opacity-20'}`}>
                            {l.type === 'added' ? `+${l.newLine}` : l.type === 'removed' ? `-${l.oldLine}` : (l.newLine || l.oldLine)}
                          </div>
                          <span className="truncate font-mono text-[12px] font-bold tracking-tight py-2.5 px-4 flex-1">{l.content || ' '}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <button onClick={(e) => { e.stopPropagation(); onToggle(path); }} className="absolute inset-0 z-[45]" />
          </motion.div>
        ) : (
          <motion.div
            key="expanded"
            layoutId={`file-block-${path}`}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={springTransition}
            className={`border overflow-hidden shadow-2xl ring-1 ring-white/5 w-full max-w-full rounded-[2.8rem] relative z-40 flex flex-col ${isCodeDark ? 'bg-[#050505] border-white/10' : 'bg-white border-zinc-200'}`}
          >
            <div className={`px-8 py-5 border-b flex flex-col md:flex-row md:items-center justify-between gap-4 ${isCodeDark ? 'border-white/5 bg-white/[0.02]' : 'border-zinc-200 bg-zinc-50'}`}>
              <div className="flex items-center gap-5 min-w-0">
                <div className="w-10 h-10 bg-primary/10 text-primary rounded-xl flex items-center justify-center border border-primary/10 shrink-0"><FileCode size={20} /></div>
                <span className={`text-[14px] font-black uppercase tracking-tight truncate max-w-[200px] md:max-w-[400px] ${isCodeDark ? 'text-white' : 'text-zinc-900'}`}>{path}</span>
                <div className={`flex items-center gap-1.5 ml-3 p-1.5 rounded-2xl border shrink-0 ${isCodeDark ? 'bg-black/40 border-white/5' : 'bg-white border-zinc-200'}`}>
                  <button onClick={(e) => { e.stopPropagation(); setFilter('added'); }} className={`flex items-center gap-2 px-3.5 py-2 rounded-xl transition-all ${filter === 'added' ? 'bg-emerald-500 text-white shadow-lg' : 'text-emerald-500/40 hover:text-emerald-500'}`}><Plus size={10} strokeWidth={4} /><span className="text-[10px] font-black">{stats.added}</span></button>
                  <button onClick={(e) => { e.stopPropagation(); setFilter('removed'); }} className={`flex items-center gap-2 px-3.5 py-2 rounded-xl transition-all ${filter === 'removed' ? 'bg-red-500 text-white shadow-lg' : 'text-red-500/40 hover:text-red-500'}`}><Minus size={10} strokeWidth={4} /><span className="text-[10px] font-black">{stats.removed}</span></button>
                  <button onClick={(e) => { e.stopPropagation(); setFilter('all'); }} className={`flex items-center gap-2 px-3.5 py-2 rounded-xl transition-all ${filter === 'all' ? 'bg-blue-600 text-white shadow-lg' : 'text-zinc-600 hover:text-zinc-400'}`}><Equal size={10} strokeWidth={4} /><span className="text-[10px] font-black">{stats.total}</span></button>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button onClick={handleCopy} className={`flex items-center gap-2 px-5 py-2.5 rounded-xl transition-all text-[10px] font-black uppercase tracking-widest border active:scale-95 ${isCodeDark ? 'bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white border-white/5' : 'bg-white hover:bg-zinc-100 text-zinc-600 hover:text-zinc-900 border-zinc-200'}`}>
                  {copied ? <CheckCircle2 size={12} className="text-emerald-500" /> : <Copy size={12} />} {copied ? 'Copiado' : 'Copiar'}
                </button>
                <button onClick={() => onToggle(path)} className={`p-2.5 rounded-xl transition-all border active:scale-95 ${isCodeDark ? 'bg-white/5 hover:bg-white/10 text-zinc-500 hover:text-white border-white/5' : 'bg-white hover:bg-zinc-100 text-zinc-500 border-zinc-200'}`}><Minimize2 size={18} /></button>
              </div>
            </div>
            <div className={`overflow-x-auto text-[13px] font-mono custom-scrollbar py-8 ${isCodeDark ? 'bg-[#050505]' : 'bg-white'}`}>
              <div className="min-w-full inline-block">
                {filteredLines.map((line, i) => <DiffLine key={i} line={line} isDarkMode={isCodeDark} />)}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

const PatchOverview: React.FC<{ 
  fileChanges: { path: string, diff: string }[], 
  onToggleAll: (collapsed: boolean) => void,
  onToggleSingle: (path: string) => void,
  onOpenExplorer: () => void,
  collapsedPaths: Record<string, boolean>,
  language: Language
}> = ({ fileChanges, onToggleAll, onToggleSingle, onOpenExplorer, collapsedPaths, language }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [localLimit, setLocalLimit] = useState(4);
  const INITIAL_LIMIT = 4;
  
  const stats = useMemo(() => {
    let added = 0;
    let removed = 0;
    fileChanges.forEach(f => {
      const lines = f.diff.split('\n');
      lines.forEach(l => {
        if (l.startsWith('+')) added++;
        if (l.startsWith('-')) removed++;
      });
    });
    return { added, removed };
  }, [fileChanges]);

  const visibleFiles = fileChanges.slice(0, localLimit);
  const isAllShownLocally = localLimit >= fileChanges.length;

  return (
    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mb-8 border border-primary/20 bg-primary/5 rounded-[2rem] overflow-hidden glass-card shadow-2xl">
      <div onClick={() => setIsOpen(!isOpen)} className="px-6 py-4 flex items-center justify-between cursor-pointer hover:bg-primary/10 transition-colors">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-primary/20 text-primary rounded-xl flex items-center justify-center border border-primary/20 shadow-inner"><GitBranch size={20} /></div>
          <div>
            <h4 className="text-[13px] font-black tracking-tight leading-none text-foreground">{language === 'es' ? 'Gestor de Modificaciones Vortex' : 'Vortex Manager'}</h4>
            <div className="flex items-center gap-4 mt-1.5">
              <span className="text-[9px] font-black uppercase tracking-widest text-primary/60">{fileChanges.length} {language === 'es' ? 'Archivos' : 'Files'}</span>
              <div className="flex items-center gap-2 text-[9px] font-black text-emerald-500"><Plus size={10} strokeWidth={4}/> {stats.added}</div>
              <div className="flex items-center gap-2 text-[9px] font-black text-red-500"><Minus size={10} strokeWidth={4}/> {stats.removed}</div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex bg-black/30 p-1 rounded-xl border border-white/5">
            <button onClick={(e) => { e.stopPropagation(); onToggleAll(false); }} className="px-3 py-1.5 text-[9px] font-black uppercase tracking-widest text-white/40 hover:text-white transition-colors">Expandir</button>
            <button onClick={(e) => { e.stopPropagation(); onToggleAll(true); }} className="px-3 py-1.5 text-[9px] font-black uppercase tracking-widest text-white/40 hover:text-white border-l border-white/5 transition-colors">Colapsar</button>
          </div>
          <div className={`transition-transform duration-500 ${isOpen ? 'rotate-180' : ''}`}>
             <ChevronDown size={18} className="text-muted-foreground" />
          </div>
        </div>
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden bg-black/20">
            <div className="p-4 flex flex-col gap-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {visibleFiles.map((file) => (
                  <motion.button key={file.path} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
                    onClick={() => { const el = document.getElementById(`file-${file.path}`); el?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }}
                    className="flex items-center justify-between p-3 bg-white/[0.04] border border-white/5 rounded-xl hover:bg-white/10 hover:border-primary/40 transition-all text-left group"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <FileCode size={18} className="text-primary/60 group-hover:text-primary shrink-0 transition-transform group-hover:scale-110" />
                      <span className="text-[11px] font-bold text-foreground truncate">{file.path}</span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <div className="flex gap-2">
                        <span className="text-[9px] font-black text-emerald-500/60">+{file.diff.split('\n').filter(l => l.startsWith('+')).length}</span>
                        <span className="text-[9px] font-black text-red-500/60">-{file.diff.split('\n').filter(l => l.startsWith('-')).length}</span>
                      </div>
                      <button onClick={(e) => { e.stopPropagation(); onToggleSingle(file.path); }} className="p-1.5 bg-white/5 rounded-lg text-zinc-500 hover:text-white transition-all active:scale-90">
                        {collapsedPaths[file.path] !== false ? <Maximize2 size={12} /> : <Minimize2 size={12} />}
                      </button>
                    </div>
                  </motion.button>
                ))}
              </div>
              <div className="flex flex-col gap-2 mt-2">
                 {fileChanges.length > INITIAL_LIMIT && (
                    <button onClick={(e) => { e.stopPropagation(); setLocalLimit(isAllShownLocally ? INITIAL_LIMIT : fileChanges.length); }} className="w-full p-2.5 bg-white/5 border border-white/5 rounded-xl hover:bg-white/10 flex items-center justify-center gap-3 text-zinc-400 group/more transition-all">
                      {isAllShownLocally ? <><ChevronUp size={14} /><span className="text-[9px] font-black uppercase tracking-widest">{language === 'es' ? 'Mostrar menos' : 'Show less'}</span></> : <><ChevronDown size={14} /><span className="text-[9px] font-black uppercase tracking-widest">{language === 'es' ? `Ver todos (${fileChanges.length} archivos)` : `Show all (${fileChanges.length} files)`}</span></>}
                    </button>
                 )}
                <button onClick={(e) => { e.stopPropagation(); onOpenExplorer(); }} className="w-full p-3.5 bg-primary/10 border border-primary/20 rounded-2xl hover:bg-primary/20 flex items-center justify-center gap-4 group/expand transition-all shadow-[0_15px_35px_-10px_rgba(var(--primary),0.3)]">
                  <FileStack size={18} className="text-primary group-hover/expand:scale-125 transition-transform" />
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">{language === 'es' ? 'Abrir Explorador de Modificaciones' : 'Open Modification Explorer'}</span>
                  <ExternalLink size={14} className="text-primary opacity-40 group-hover/expand:opacity-100 transition-opacity" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, fontSize = 'medium', codeTheme = 'dark', onShowReasoning, onOpenModificationExplorer, isStreaming = false, language }) => {
  const isUser = message.role === Role.USER;
  const [collapsedPaths, setCollapsedPaths] = useState<Record<string, boolean>>({});
  const [msgCopied, setMsgCopied] = useState(false);

  const togglePath = useCallback((path: string) => setCollapsedPaths(prev => ({ ...prev, [path]: !prev[path] })), []);
  const toggleAll = useCallback((collapsed: boolean) => {
    if (message.fileChanges) {
      const newState: Record<string, boolean> = {};
      message.fileChanges.forEach(f => { newState[f.path] = collapsed; });
      setCollapsedPaths(newState);
    }
  }, [message.fileChanges]);

  const handleCopyMessage = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setMsgCopied(true);
    setTimeout(() => setMsgCopied(false), 2000);
  }, [message.content]);

  const fontSizeClass = { small: 'text-[12px]', medium: 'text-[15px]', large: 'text-[17px]' }[fontSize];

  const markdownComponents = useMemo(() => ({
    ul: ({ children }: any) => <ul className="space-y-4 mb-6 list-none">{children}</ul>,
    ol: ({ children }: any) => <ol className="space-y-4 mb-6 list-none counter-reset-pulse">{children}</ol>,
    li: ({ children, ordered, index }: any) => (
      <motion.li initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} className="flex items-start gap-5 group/li">
        <div className="shrink-0 mt-2">{ordered ? <div className="w-6 h-6 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-[10px] font-black text-primary shadow-inner">{index + 1}</div> : <div className="w-2 h-2 rounded-full bg-primary mt-1.5 shadow-[0_0_10px_rgba(var(--primary),0.8)] group-hover/li:scale-150 transition-transform duration-500" />}</div>
        <span className="text-foreground/80 leading-relaxed font-medium group-hover/li:text-foreground transition-colors">{children}</span>
      </motion.li>
    ),
    code({ node, inline, className, children, ...props }: any) {
      if (inline) return <code className="px-2 py-0.5 bg-muted rounded text-primary font-bold" {...props}>{children}</code>;
      const path = className?.includes('file:') ? className.split('file:')[1] : null;
      return <CodeBlock key={path || 'code'} className={className} isCollapsed={path ? (collapsedPaths[path] ?? true) : false} onToggle={togglePath} codeTheme={codeTheme}>{children}</CodeBlock>;
    }
  }), [collapsedPaths, togglePath, codeTheme]);

  const isThinking = isStreaming && !message.content && !!message.thought;

  return (
    <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className={`group flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-10 accelerated`}>
      <div className={`flex max-w-[96%] md:max-w-[88%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start gap-5`}>
        <div className={`w-11 h-11 rounded-2xl flex items-center justify-center shrink-0 border transition-all duration-700 group-hover:scale-110 group-hover:rotate-6 ${isUser ? 'bg-primary text-white border-primary/20 shadow-xl' : 'bg-background border-border glass-card shadow-lg'}`}>{isUser ? <User size={20} /> : <Bot size={20} />}</div>
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} min-w-0 flex-1`}>
          <div className={`px-8 py-6 rounded-[2.5rem] ${isUser ? 'bg-primary text-white rounded-tr-none shadow-2xl' : 'bg-muted/10 border border-border/40 rounded-tl-none glass-card'} ${fontSizeClass} leading-relaxed w-full relative`}>
            {!isUser && message.fileChanges && message.fileChanges.length >= 2 && (
              <PatchOverview fileChanges={message.fileChanges} onToggleAll={toggleAll} onToggleSingle={togglePath} onOpenExplorer={() => onOpenModificationExplorer(message.fileChanges!)} collapsedPaths={collapsedPaths} language={language} />
            )}
            <div className="markdown-content relative overflow-visible z-10">
              {message.content ? <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{message.content}</ReactMarkdown> : isThinking ? <div className="flex items-center gap-4 py-3 text-primary"><Loader2 size={20} className="animate-spin" /><span className="text-[13px] font-black tracking-[0.2em] opacity-70 uppercase">{language === 'es' ? 'Vortex está procesando...' : 'Vortex is processing...'}</span></div> : null}
              {isStreaming && message.content && <span className="typing-cursor" />}
            </div>
            {!isUser && message.sources && message.sources.length > 0 && <div className="mt-10 pt-8 border-t border-border/20 flex flex-wrap gap-3 mb-2 relative z-[20]">{message.sources.map((src, i) => <GroundingPill key={i} source={src} index={i} />)}</div>}
          </div>
          <div className="flex gap-8 mt-4 px-6 opacity-0 group-hover:opacity-100 transition-all duration-500 items-center w-full">
            {!isUser && (
              <div className="flex gap-6">
                <button onClick={handleCopyMessage} className="text-[11px] font-black uppercase tracking-widest text-muted-foreground hover:text-primary transition-all flex items-center gap-2.5 active:scale-95 py-1.5 px-3 rounded-xl hover:bg-muted">{msgCopied ? <CheckCircle2 size={14} className="text-emerald-500" /> : <Copy size={14} />} {msgCopied ? 'Copiado' : 'Copiar'}</button>
                {(message.thought || (isStreaming && message.role === Role.AI)) && (
                  <button onClick={() => onShowReasoning?.(message.id)} className="group/btn text-[11px] font-black uppercase tracking-widest text-primary flex items-center gap-2.5 py-2 px-4 bg-primary/10 rounded-2xl hover:bg-primary/20 transition-all active:scale-95 border border-primary/10">
                    <div className="relative"><Brain size={14} className={`${isStreaming ? 'animate-pulse' : 'group-hover/btn:rotate-12'} transition-transform`} />{isStreaming && <motion.div animate={{ opacity: [0, 1, 0] }} transition={{ repeat: Infinity, duration: 1.5 }} className="absolute -top-1.5 -right-1.5 w-2.5 h-2.5 bg-primary rounded-full blur-[3px]" />}</div>
                    <span>{isStreaming && !message.content ? (language === 'es' ? 'Razonando...' : 'Thinking...') : (language === 'es' ? 'Razonamiento' : 'Reasoning')}</span>
                    <ChevronRight size={14} className="opacity-40 group-hover/btn:translate-x-1.5 transition-transform" />
                  </button>
                )}
              </div>
            )}
            <span className="text-[10px] font-black text-muted-foreground/30 uppercase tracking-[0.3em] ml-auto">{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default MessageBubble;
