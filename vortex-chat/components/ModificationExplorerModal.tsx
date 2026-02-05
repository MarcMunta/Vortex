
import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, FileCode, Plus, Minus, Search as SearchIcon, 
  FileText, Fingerprint, ShieldCheck, Copy, FileStack, 
  ChevronRight, Terminal, Binary, CheckCircle2, Zap
} from 'lucide-react';
import { Language } from '../types';

// Interface de props para ModificationExplorerModal
interface ModificationExplorerModalProps {
  fileChanges: { path: string, diff: string }[];
  onClose: () => void;
  language: Language;
}

const DiffLine = ({ line }: { line: any }) => (
  <div 
    className={`grid grid-cols-[64px_32px_1fr] items-stretch w-full h-[26px] border-l-4 transition-all relative ${
      line.type === 'added' 
        ? 'bg-emerald-500/15 border-emerald-500' 
        : line.type === 'removed' 
          ? 'bg-red-500/15 border-red-500' 
          : 'bg-transparent border-transparent hover:bg-white/[0.04]'
    }`}
  >
    {/* Gutter: Simetría absoluta para números de línea */}
    <div className="flex items-center justify-end pr-4 border-r border-white/5 select-none tabular-nums shrink-0 bg-black/40">
      <span className="text-[10px] font-mono leading-none tracking-tighter text-zinc-600">
        {line.type === 'added' ? line.newLine : line.oldLine || line.newLine}
      </span>
    </div>

    {/* Signo: Centrado en columna fija */}
    <div className="flex items-center justify-center shrink-0">
      <span className={`font-mono text-[14px] font-black leading-none ${
        line.type === 'added' ? 'text-emerald-500' : line.type === 'removed' ? 'text-red-500' : 'text-zinc-800'
      }`}>
        {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
      </span>
    </div>

    {/* Código: Alineación monospaciada de alta precisión */}
    <div className="flex items-center px-4 min-w-0 overflow-hidden">
      <span className={`whitespace-pre font-mono text-[13px] font-bold tracking-tight truncate leading-none ${
        line.type === 'added' ? 'text-emerald-300' : line.type === 'removed' ? 'text-red-300' : 'text-zinc-300'
      }`}>
        {line.content || ' '}
      </span>
    </div>
  </div>
);

const ModificationExplorerModal: React.FC<ModificationExplorerModalProps> = ({ fileChanges, onClose, language }) => {
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const [search, setSearch] = useState('');
  const [isCopying, setIsCopying] = useState(false);
  
  const filteredFiles = useMemo(() => {
    return fileChanges.filter(f => f.path.toLowerCase().includes(search.toLowerCase()));
  }, [fileChanges, search]);

  const selectedFile = filteredFiles[selectedFileIdx] || filteredFiles[0] || fileChanges[0];

  const processedLines = useMemo(() => {
    if (!selectedFile) return [];
    const rawLines = selectedFile.diff.split('\n');
    let oldNum = 1;
    let newNum = 1;
    return rawLines.map(line => {
      const isAdded = line.startsWith('+');
      const isRemoved = line.startsWith('-');
      return {
        type: isAdded ? 'added' : isRemoved ? 'removed' : 'neutral',
        content: line.replace(/^[+-]/, ''),
        oldLine: isAdded ? null : oldNum++,
        newLine: isRemoved ? null : newNum++,
      };
    });
  }, [selectedFile]);

  const handleCopy = () => {
    if (!selectedFile) return;
    navigator.clipboard.writeText(selectedFile.diff);
    setIsCopying(true);
    setTimeout(() => setIsCopying(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-[100000] flex items-center justify-center bg-zinc-950 p-4 md:p-10 overflow-hidden selection:bg-primary/30">
      <div className="absolute inset-0 opacity-5 pointer-events-none vortex-grid-bg" />
      
      <motion.div 
        initial={{ opacity: 0, scale: 0.95, y: 40 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 40 }}
        className="relative w-full h-full bg-[#030303] border border-white/10 rounded-[2.5rem] shadow-[0_80px_160px_-40px_rgba(0,0,0,1)] overflow-hidden flex flex-col ring-1 ring-white/5"
      >
        <header className="px-8 py-5 border-b border-white/5 bg-white/[0.01] flex items-center justify-between shrink-0">
          <div className="flex items-center gap-6">
            <div className="w-12 h-12 bg-primary rounded-2xl flex items-center justify-center text-white shadow-2xl relative group">
              <div className="absolute inset-[-4px] bg-primary/20 blur-lg group-hover:blur-xl transition-all" />
              <FileStack size={22} className="relative z-10" />
            </div>
            <div className="space-y-0.5">
              <h2 className="text-xl font-black text-white tracking-tighter leading-none">
                {language === 'es' ? 'Explorador Vortex' : 'Vortex Explorer'}
              </h2>
              <div className="flex items-center gap-3">
                <span className="text-[8px] font-black uppercase tracking-[0.4em] text-primary">Kernel_Patch_Explorer_v3</span>
                <div className="w-1 h-1 rounded-full bg-white/10" />
                <span className="text-[8px] font-black uppercase tracking-[0.4em] text-white/30">{fileChanges.length} Entidades Mapeadas</span>
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            aria-label={language === 'es' ? 'Cerrar' : 'Close'}
            title={language === 'es' ? 'Cerrar' : 'Close'}
            className="w-11 h-11 bg-white/5 hover:bg-red-500/10 text-white/20 hover:text-red-500 rounded-xl border border-white/5 hover:border-red-500/20 transition-all active:scale-90 flex items-center justify-center group"
          >
            <X size={20} className="group-hover:rotate-90 transition-transform" />
          </button>
        </header>

        <div className="flex-1 flex overflow-hidden">
          <aside className="w-[280px] border-r border-white/5 flex flex-col bg-black/40 shrink-0">
            <div className="p-5 border-b border-white/5">
              <div className="relative group">
                <SearchIcon size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/10 group-focus-within:text-primary transition-colors" />
                <input 
                  type="text" 
                  placeholder={language === 'es' ? 'Localizar...' : 'Locate...'}
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setSelectedFileIdx(0); 
                  }}
                  className="w-full h-10 bg-white/5 border border-white/10 rounded-xl pl-11 pr-4 text-[13px] font-bold text-white outline-none focus:border-primary/40 focus:ring-4 focus:ring-primary/5 transition-all placeholder:text-white/10"
                />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-1.5 overflow-x-hidden">
              <AnimatePresence mode="popLayout">
                {filteredFiles.map((file, i) => {
                  const isSelected = selectedFile === file;
                  return (
                    <motion.button
                      key={file.path}
                      layout
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20, scale: 0.95 }}
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      onClick={() => setSelectedFileIdx(i)}
                      className={`w-full p-4 rounded-xl text-left transition-all border group relative overflow-hidden ${
                        isSelected 
                          ? 'bg-primary/10 border-primary/30 shadow-lg' 
                          : 'border-transparent hover:bg-white/5 hover:border-white/10'
                      }`}
                    >
                      {isSelected && <motion.div layoutId="active-patch-file" className="absolute left-0 top-3 bottom-3 w-1 bg-primary rounded-full" />}
                      <div className="flex items-center gap-3 mb-2">
                        <FileCode size={16} className={isSelected ? 'text-primary' : 'text-white/20'} />
                        <span className={`text-[13px] font-black truncate tracking-tight transition-colors ${isSelected ? 'text-white' : 'text-white/40 group-hover:text-white/70'}`}>
                          {file.path.split('/').pop()}
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[8px] font-mono font-black text-white/10 truncate uppercase tracking-widest leading-none">{file.path}</span>
                        <div className="flex gap-1.5 shrink-0">
                           <div className="px-1.5 py-0.5 bg-emerald-500/10 rounded-md text-[8px] font-black text-emerald-500">+{file.diff.split('\n').filter(l => l.startsWith('+')).length}</div>
                           <div className="px-1.5 py-0.5 bg-red-500/10 rounded-md text-[8px] font-black text-red-500">-{file.diff.split('\n').filter(l => l.startsWith('-')).length}</div>
                        </div>
                      </div>
                    </motion.button>
                  );
                })}
              </AnimatePresence>
            </div>
          </aside>

          <main className="flex-1 flex flex-col bg-black/80 relative overflow-hidden">
            <div className="px-8 py-5 bg-white/[0.01] border-b border-white/5 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-5">
                <div className="w-10 h-10 bg-primary/10 text-primary rounded-xl flex items-center justify-center border border-primary/20 shadow-inner">
                  <Terminal size={20} />
                </div>
                <div className="flex flex-col">
                  <span className="text-[8px] font-black text-primary uppercase tracking-[0.4em] mb-0.5">Vortex_Kernel_Mapper</span>
                  <span className="text-[16px] font-black text-white/90 uppercase tracking-tight">{selectedFile?.path}</span>
                </div>
              </div>
              <button 
                onClick={handleCopy}
                className={`flex items-center gap-3 px-6 py-3 rounded-xl text-[9px] font-black uppercase tracking-[0.2em] transition-all active:scale-95 ${
                  isCopying ? 'bg-emerald-500 text-white' : 'bg-primary text-white shadow-xl hover:scale-105'
                }`}
              >
                {isCopying ? <CheckCircle2 size={14} /> : <Copy size={14} />}
                {isCopying ? 'Copiado' : 'Copiar Parche'}
              </button>
            </div>
            
            <div className="flex-1 overflow-auto custom-scrollbar p-6 bg-[#010101] relative">
               <div className="w-full max-w-full inline-block bg-[#050505] rounded-2xl border border-white/5 py-4 overflow-hidden shadow-2xl relative">
                 <div className="overflow-x-auto min-w-full">
                   {processedLines.map((line, i) => (
                     <DiffLine key={i} line={line} />
                   ))}
                 </div>
               </div>
            </div>
          </main>
        </div>

        <footer className="px-8 py-5 border-t border-white/5 bg-white/[0.01] flex items-center justify-between shrink-0">
           <div className="flex items-center gap-5">
              <div className="flex -space-x-2.5">
                {[0, 1].map(i => (
                  <motion.div 
                    key={i} 
                    animate={{ opacity: [0.4, 1, 0.4] }} 
                    transition={{ duration: 2, repeat: Infinity, delay: i * 0.5 }}
                    className="w-9 h-9 rounded-lg border-[2px] border-zinc-950 bg-zinc-900 flex items-center justify-center shadow-lg"
                  >
                    <ShieldCheck size={14} className="text-primary/60" />
                  </motion.div>
                ))}
              </div>
              <div className="flex flex-col">
                <span className="text-[8px] font-black text-white/20 uppercase tracking-[0.3em] mb-0.5">Integridad de Núcleo</span>
                <div className="flex items-center gap-2">
                   <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                   <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">En Línea</span>
                </div>
              </div>
           </div>
           <button 
             onClick={onClose}
             className="px-8 py-4 bg-white text-black rounded-xl text-[10px] font-black uppercase tracking-[0.3em] shadow-xl hover:scale-105 active:scale-95 transition-all"
           >
             Cerrar Entorno de Ingenería
           </button>
        </footer>
      </motion.div>
    </div>
  );
};

export default ModificationExplorerModal;
