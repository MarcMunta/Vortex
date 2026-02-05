
import React, { useMemo, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, Brain, Target, Zap, Layers, Activity, Sparkles, 
  CheckCircle2, Cpu, BarChart, ShieldCheck, Clock,
  ArrowDownCircle, Fingerprint, Network, Gauge, Loader2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Language } from '../types';
import { translations } from '../translations';

interface ReasoningDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  thought: string | undefined;
  language: Language;
  isStreaming?: boolean;
}

const ReasoningDrawer: React.FC<ReasoningDrawerProps> = ({ isOpen, onClose, thought, language, isStreaming }) => {
  const t = translations[language];
  const scrollRef = useRef<HTMLDivElement>(null);

  const steps = useMemo(() => {
    if (!thought) return [];
    
    return thought
      .split(/\n\n|(?=\bStep\s\d:)|(?=\d\.\s)|(?=\n[*-]\s)/)
      .filter(step => step.trim().length > 5)
      .map(step => {
        return step
          .replace(/^Step\s\d:\s*/i, '')
          .replace(/^\d\.\s*/, '')
          .replace(/^[*-]\s*/, '')
          .trim();
      });
  }, [thought]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thought, steps.length]);

  const complexityScore = useMemo(() => {
    if (!thought) return 0;
    const wordCount = thought.split(/\s+/).length;
    return Math.min(100, Math.floor((wordCount / 500) * 100));
  }, [thought]);

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20, filter: 'blur(10px)' },
    show: { 
      opacity: 1, 
      y: 0, 
      filter: 'blur(0px)',
      transition: { type: 'spring' as const, damping: 20, stiffness: 100 } 
    }
  };

  return (
    <div className="h-full w-[400px] flex flex-col relative bg-zinc-950 overflow-hidden">
      {/* Background HUD Decorator */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-0 right-0 w-full h-full bg-[radial-gradient(circle_at_100%_0%,rgba(var(--primary),0.15),transparent_50%)]" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-primary/5 blur-[100px] rounded-full" />
      </div>

      <header className="relative px-6 pt-10 pb-6 border-b border-white/5 bg-white/[0.02] shrink-0 z-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative group">
              <div className="absolute inset-0 bg-primary/40 blur-xl rounded-full animate-pulse" />
              <div className="w-12 h-12 bg-primary rounded-2xl flex items-center justify-center text-white shadow-2xl relative z-10">
                <Brain size={24} fill="currentColor" />
              </div>
            </div>
            <div>
              <h2 className="text-[10px] font-black tracking-[0.3em] text-primary uppercase leading-none mb-1.5">{t.reasoning_title}</h2>
              <div className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${isStreaming ? 'bg-primary' : 'bg-emerald-500'}`} />
                <span className="text-sm font-black text-white tracking-tight">
                  {isStreaming ? (language === 'es' ? 'Analizando...' : 'Analyzing...') : 'Kernel Active'}
                </span>
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            aria-label={language === 'es' ? 'Cerrar panel de razonamiento' : 'Close reasoning panel'}
            title={language === 'es' ? 'Cerrar' : 'Close'}
            className="p-2.5 bg-white/5 hover:bg-white/10 rounded-xl text-zinc-400 transition-all border border-white/5"
          >
            <X size={18} />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="p-4 bg-white/[0.03] border border-white/5 rounded-2xl space-y-2">
            <div className="flex items-center justify-between text-[8px] font-black uppercase tracking-widest text-zinc-500">
              <div className="flex items-center gap-1.5"><Gauge size={10} /> Complexity</div>
              <span className="text-primary">{complexityScore}%</span>
            </div>
            <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${complexityScore}%` }}
                className="h-full bg-primary"
              />
            </div>
          </div>
          <div className="p-4 bg-white/[0.03] border border-white/5 rounded-2xl space-y-2">
            <div className="flex items-center justify-between text-[8px] font-black uppercase tracking-widest text-zinc-500">
              <div className="flex items-center gap-1.5"><Network size={10} /> Neural Path</div>
              <span className="text-emerald-500">{isStreaming ? 'Flow' : 'Sync'}</span>
            </div>
            <div className="flex gap-1">
              {[0, 1, 2, 3].map(i => (
                <motion.div 
                  key={i}
                  animate={isStreaming ? { opacity: [0.2, 1, 0.2] } : { opacity: 1 }}
                  transition={isStreaming ? { duration: 1, repeat: Infinity, delay: i * 0.2 } : {}}
                  className="h-1 flex-1 bg-emerald-500/40 rounded-full"
                />
              ))}
            </div>
          </div>
        </div>
      </header>

      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar px-6 py-8 space-y-10 bg-black/20 relative z-10"
      >
        {steps.length > 0 ? (
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="show"
            className="space-y-10"
          >
            {steps.map((step, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="relative pl-12 group"
              >
                {(index !== steps.length - 1 || isStreaming) && (
                  <div className="absolute left-[20px] top-10 bottom-[-40px] w-[1px] bg-white/5">
                    <motion.div 
                      animate={{ top: ['0%', '100%'], opacity: [0, 1, 0] }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="absolute left-0 right-0 h-12 bg-gradient-to-b from-transparent via-primary to-transparent"
                    />
                  </div>
                )}
                
                <div className="absolute left-0 top-0 w-10 h-10 rounded-[1rem] bg-zinc-900 border border-white/10 flex items-center justify-center z-10 shadow-lg overflow-hidden">
                  <div className="absolute inset-0 bg-primary/5 group-hover:bg-primary/10 transition-colors" />
                  {index === steps.length - 1 && isStreaming ? (
                    <Loader2 size={16} className="text-primary animate-spin" />
                  ) : index === steps.length - 1 && !isStreaming ? (
                    <Zap size={16} className="text-primary" />
                  ) : (
                    <CheckCircle2 size={16} className="text-emerald-500/60" />
                  )}
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                     <span className="text-[9px] font-black text-white/20 uppercase tracking-[0.3em]">Node_0{index + 1}</span>
                     <div className="h-[1px] flex-1 bg-white/5" />
                  </div>
                  <div className="bg-white/[0.015] border border-white/5 rounded-2xl p-5 glass-card">
                    <div className="markdown-content text-[13px] leading-relaxed text-zinc-400 font-medium">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{step}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}

            {isStreaming && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="relative pl-12"
              >
                <div className="absolute left-0 top-0 w-10 h-10 rounded-[1rem] bg-zinc-900 border border-primary/20 flex items-center justify-center z-10 shadow-lg">
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="w-2 h-2 bg-primary rounded-full"
                  />
                </div>
                <div className="space-y-3">
                   <div className="flex items-center gap-3">
                      <span className="text-[9px] font-black text-primary/40 uppercase tracking-[0.3em]">{language === 'es' ? 'Inferencia Activa' : 'Active Inference'}</span>
                      <div className="h-[1px] flex-1 bg-primary/10" />
                   </div>
                   <div className="py-2 flex gap-1">
                      <motion.span animate={{ opacity: [0, 1, 0] }} transition={{ repeat: Infinity, duration: 1.5 }} className="w-1 h-1 bg-primary rounded-full" />
                      <motion.span animate={{ opacity: [0, 1, 0] }} transition={{ repeat: Infinity, duration: 1.5, delay: 0.2 }} className="w-1 h-1 bg-primary rounded-full" />
                      <motion.span animate={{ opacity: [0, 1, 0] }} transition={{ repeat: Infinity, duration: 1.5, delay: 0.4 }} className="w-1 h-1 bg-primary rounded-full" />
                   </div>
                </div>
              </motion.div>
            )}
            
            {!isStreaming && (
              <motion.div
                variants={itemVariants}
                className="flex flex-col items-center justify-center py-8 gap-4"
              >
                <div className="w-12 h-[1px] bg-white/10" />
                <div className="flex items-center gap-3 px-6 py-3 bg-emerald-500/10 rounded-full border border-emerald-500/20 text-emerald-400">
                  <Sparkles size={14} />
                  <span className="text-[9px] font-black uppercase tracking-[0.3em]">{t.reasoning_complete}</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 py-10">
            {isStreaming ? (
              <div className="flex flex-col items-center gap-6">
                 <div className="relative">
                   <div className="absolute inset-0 bg-primary/20 blur-2xl rounded-full animate-pulse" />
                   <Loader2 size={48} className="text-primary animate-spin" />
                 </div>
                 <p className="text-[11px] font-black uppercase tracking-[0.4em] text-primary">{language === 'es' ? 'Construyendo Traza Neural...' : 'Building Neural Trace...'}</p>
              </div>
            ) : (
              <>
                <div className="w-20 h-20 bg-white/5 rounded-[2.5rem] flex items-center justify-center border border-white/10 text-zinc-600">
                  <Fingerprint size={36} strokeWidth={1} />
                </div>
                <div className="space-y-2 max-w-[220px]">
                  <p className="text-[11px] font-black uppercase tracking-[0.3em] text-white leading-tight">{t.reasoning_empty}</p>
                  <p className="text-[10px] leading-relaxed font-medium text-zinc-500">{t.reasoning_empty_desc}</p>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <footer className="relative p-6 border-t border-white/5 bg-zinc-950 shrink-0 z-10">
        <div className="p-4 bg-white/[0.02] border border-white/5 rounded-2xl flex items-center gap-4">
          <div className="w-10 h-10 bg-emerald-500/10 text-emerald-500 rounded-xl flex items-center justify-center shrink-0 border border-emerald-500/20">
            <ShieldCheck size={20} />
          </div>
          <div className="space-y-0.5 min-w-0">
            <p className="text-[10px] font-black uppercase text-white truncate">Integrity Verified</p>
            <p className="text-[9px] text-zinc-500 truncate opacity-70">Secure logic sequence mapping.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ReasoningDrawer;
