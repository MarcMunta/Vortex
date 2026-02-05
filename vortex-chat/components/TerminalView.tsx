
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Terminal as TerminalIcon, ShieldCheck, Cpu, Trash2, Power, 
  Zap, Activity, Database, Network, ChevronRight, Filter, 
  Search, ShieldAlert, CpuIcon, Layers, Radio, Share2, Binary,
  ArrowRight, TerminalSquare
} from 'lucide-react';
import { LogEntry, Language } from '../types';

interface TerminalViewProps {
  logs: LogEntry[];
  onClear: () => void;
  language: Language;
}

const TerminalView: React.FC<TerminalViewProps> = ({ logs, onClear, language }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [stats, setStats] = useState({ cpu: 12, mem: 4.2, net: 0 });
  const [activeFilter, setActiveFilter] = useState<LogEntry['level'] | 'ALL'>('ALL');
  const [command, setCommand] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setStats({
        cpu: Math.floor(8 + Math.random() * 25),
        mem: parseFloat((4.0 + Math.random() * 0.8).toFixed(1)),
        net: Math.floor(Math.random() * 1200)
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const getLevelStyles = (level: LogEntry['level']) => {
    switch (level) {
      case 'LEARN': return { text: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/20' };
      case 'SEARCH': return { text: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' };
      case 'SYSTEM': return { text: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' };
      default: return { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' };
    }
  };

  const filteredLogs = useMemo(() => {
    return activeFilter === 'ALL' ? logs : logs.filter(l => l.level === activeFilter);
  }, [logs, activeFilter]);

  const systemMetrics = [
    { label: 'CPU Usage', val: stats.cpu, color: 'bg-emerald-500' },
    { label: 'Neural Buffer', val: (stats.mem / 8) * 100, color: 'bg-blue-500' },
    { label: 'Grounding Link', val: Math.min(100, (stats.net / 1200) * 100), color: 'bg-amber-500' }
  ];

  return (
    <div className="h-full flex flex-row bg-[#020202] text-[#e0e0e0]/90 font-mono text-[13px] overflow-hidden selection:bg-emerald-500/30 pt-24">
      
      <aside className="hidden lg:flex w-72 border-r border-white/5 bg-black/60 flex-col shrink-0">
        <div className="p-8 border-b border-white/5 bg-emerald-500/[0.02]">
           <div className="flex items-center gap-4 mb-6">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 border border-emerald-500/20 shadow-lg shadow-emerald-500/10">
                <Radio size={20} className="animate-pulse" />
              </div>
              <div>
                <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-emerald-500">Telemetry</h3>
                <p className="text-[9px] text-white/30 uppercase font-black">Secure Kernel Link</p>
              </div>
           </div>
           
           <div className="space-y-8">
              {systemMetrics.map((m, i) => (
                <div key={i} className="space-y-3">
                  <div className="flex justify-between text-[9px] font-black uppercase tracking-widest text-white/40">
                    <span>{m.label}</span>
                    <span className="text-white/80 tabular-nums">{Math.round(m.val)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden p-[1px]">
                     <motion.div animate={{ width: `${m.val}%` }} className={`h-full ${m.color} rounded-full`} />
                  </div>
                </div>
              ))}
           </div>
        </div>

        <div className="flex-1 p-8 space-y-10 overflow-y-auto custom-scrollbar">
           <div>
              <h4 className="text-[9px] font-black uppercase tracking-[0.4em] text-white/20 mb-6">Filtros de Nivel</h4>
              <div className="space-y-2">
                 {['ALL', 'INFO', 'LEARN', 'SEARCH', 'SYSTEM'].map(f => (
                   <button 
                     key={f}
                     onClick={() => setActiveFilter(f as any)}
                     className={`w-full flex items-center justify-between px-4 py-3 rounded-xl border transition-all text-[10px] font-black uppercase tracking-widest ${
                       activeFilter === f 
                        ? 'bg-emerald-500 text-black border-emerald-500 shadow-xl' 
                        : 'bg-white/5 border-white/5 text-white/40 hover:bg-white/10 hover:text-white'
                     }`}
                   >
                     {f}
                     {activeFilter === f && <ChevronRight size={12} />}
                   </button>
                 ))}
              </div>
           </div>

           <div>
              <h4 className="text-[9px] font-black uppercase tracking-[0.4em] text-white/20 mb-6">Estado de Servicios</h4>
              <div className="space-y-4">
                 {[
                   { name: 'Grounding_Service', status: 'Active' },
                   { name: 'Neural_Buffer', status: 'Synced' },
                   { name: 'Secure_Tunnel', status: 'Locked' }
                 ].map((s, i) => (
                   <div key={i} className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      <span className="text-[10px] font-bold text-white/60 tracking-tight">{s.name}</span>
                   </div>
                 ))}
              </div>
           </div>
        </div>

        <div className="p-8 border-t border-white/5 flex flex-col gap-4">
           <button 
             onClick={onClear}
             className="w-full flex items-center justify-center gap-3 px-5 py-4 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 rounded-2xl transition-all text-[10px] font-black uppercase tracking-widest group"
           >
             <Trash2 size={16} className="group-hover:rotate-12 transition-transform" />
             Purgar Registros
           </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col h-full bg-[#020202] relative">
        <header className="px-10 py-8 border-b border-white/5 flex items-center justify-between bg-white/[0.01]">
          <div className="flex items-center gap-6">
             <div className="w-12 h-12 bg-white/5 rounded-2xl flex items-center justify-center text-emerald-400 border border-white/10 group-hover:border-emerald-500/40 transition-colors">
                <TerminalIcon size={24} />
             </div>
             <div>
                <h1 className="text-[11px] font-black uppercase tracking-[0.4em] text-emerald-500">Vortex_Kernel_Console</h1>
                <p className="text-[10px] text-white/20 font-black uppercase tracking-widest mt-1">Secure Session Verified</p>
             </div>
          </div>
          <div className="flex gap-1.5">
             {['bg-red-500', 'bg-amber-500', 'bg-emerald-500'].map((bg, i) => (
               <div key={i} className={`w-3 h-3 rounded-full border border-white/10 ${bg}`} />
             ))}
          </div>
        </header>

        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-12 space-y-4 custom-scrollbar scroll-smooth"
        >
          <div className="mb-12 space-y-3">
             <p className="text-[10px] font-black uppercase tracking-[0.5em] text-emerald-500/40">Iniciando Inferencia de Kernel...</p>
             <div className="p-6 bg-emerald-500/5 border border-emerald-500/10 rounded-2xl flex items-center gap-5">
                <ShieldCheck size={20} className="text-emerald-500" />
                <p className="text-[11px] font-bold text-emerald-500 opacity-80 uppercase tracking-widest">Protocolo de seguridad habilitado. Sesión local cifrada.</p>
             </div>
          </div>

          <AnimatePresence initial={false}>
            {filteredLogs.map((log) => {
              const styles = getLevelStyles(log.level);
              return (
                <motion.div 
                  key={log.id}
                  initial={{ opacity: 0, x: -10, filter: 'blur(5px)' }}
                  animate={{ opacity: 1, x: 0, filter: 'blur(0)' }}
                  className="flex items-start gap-8 group py-2 hover:bg-white/[0.02] rounded-xl px-4 -mx-4 transition-colors relative"
                >
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-500/0 group-hover:bg-emerald-500/20 transition-all rounded-full" />
                  
                  <div className="shrink-0 flex flex-col items-center gap-1.5 pt-1 tabular-nums">
                    <span className="text-[10px] font-black text-white/10 group-hover:text-white/30 transition-colors">
                      {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span className="text-[8px] font-black text-white/5 uppercase">0x{log.id.slice(0, 4)}</span>
                  </div>

                  <div className={`shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[9px] font-black uppercase tracking-widest ${styles.text} ${styles.bg} ${styles.border} shadow-sm`}>
                    <Binary size={10} />
                    {log.level}
                  </div>

                  <span className="flex-1 text-[13.5px] leading-relaxed text-white/70 group-hover:text-white transition-colors font-medium tracking-tight">
                    {log.message}
                  </span>
                </motion.div>
              );
            })}
          </AnimatePresence>
          
          <div className="flex items-center gap-4 mt-12 py-4">
            <div className="w-2.5 h-6 bg-emerald-500 animate-pulse shadow-[0_0_15px_#10b981]" />
            <span className="text-[10px] font-black text-emerald-500/30 uppercase tracking-[0.4em]">Awaiting Instruction...</span>
          </div>
        </div>

        <footer className="px-10 py-8 bg-black/40 border-t border-white/5 flex items-center gap-6">
           <div className="w-10 h-10 bg-emerald-500/5 text-emerald-500 rounded-xl flex items-center justify-center border border-emerald-500/20 shadow-inner">
             <ChevronRight size={20} />
           </div>
           <input 
             type="text" 
             value={command}
             onChange={(e) => setCommand(e.target.value)}
             placeholder="Ejecutar comando de sistema (/help para ver lista)..."
             className="flex-1 bg-transparent border-none outline-none text-[14px] font-bold placeholder:text-white/10 text-emerald-400 font-mono tracking-tight"
           />
           <div className="flex items-center gap-6 text-[9px] font-black uppercase tracking-[0.2em] text-white/20">
              <span className="flex items-center gap-2"><kbd className="px-2 py-1 bg-white/5 border border-white/10 rounded-lg text-white/40">ENTER</kbd> EXEC</span>
              <div className="h-4 w-[1px] bg-white/10" />
              <div className="flex items-center gap-2 text-emerald-500/60">
                 <Power size={14} />
                 <span>Operational</span>
              </div>
           </div>
        </footer>
      </main>
    </div>
  );
};

export default TerminalView;
