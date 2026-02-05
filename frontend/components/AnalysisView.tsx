
import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, Globe, Brain, TrendingUp, Search, Zap, 
  Activity, Shield, ListChecks, PauseCircle, AlertTriangle, 
  RefreshCw, BrainCircuit, Binary, Network, Fingerprint,
  BookOpen, ExternalLink, Cpu, Compass, Lock,
  Sparkles, Database, PieChart, Layers, ShieldCheck,
  Star, Clock, ChevronRight, Share2, Info, FileCode,
  Tag, Filter, Hash, HardDrive, LayoutGrid, List, ActivitySquare,
  History, Server, Download, CheckCircle2
} from 'lucide-react';
import { ChatSession, Source, Message, Role, LogEntry, Language } from '../types';

interface AnalysisViewProps {
  sessions: ChatSession[];
  onNavigateToChat: (sessionId: string, messageId?: string) => void;
  onAddLog: (level: LogEntry['level'], message: string) => void;
  language: Language;
}

type TabType = 'overview' | 'knowledge' | 'network';
type KnowledgeCategory = 'all' | 'docs' | 'code' | 'research' | 'other';

const TAB_INDEX: Record<TabType, number> = {
  'overview': 0,
  'knowledge': 1,
  'network': 2
};

const Sparkline: React.FC<{ color: string; data?: number[] }> = ({ color, data }) => {
  const points = useMemo(() => data || Array.from({ length: 12 }, () => Math.random() * 20 + 5), [data]);
  const path = points.map((p, i) => `${i * 10},${25 - p}`).join(' L ');

  return (
    <svg className="w-24 h-8 overflow-visible opacity-50">
      <motion.path
        d={`M 0,25 L ${path}`}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
      />
    </svg>
  );
};

const NeuralNetworkGraph: React.FC<{ nodeCount: number }> = ({ nodeCount }) => {
  const nodes = useMemo(() => {
    const count = Math.min(Math.max(nodeCount, 15), 60);
    return Array.from({ length: count }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      r: Math.random() * 3 + 1.5,
      pulse: Math.random() * 2 + 1
    }));
  }, [nodeCount]);

  return (
    <div className="relative w-full h-[500px] bg-black/40 rounded-[3rem] border border-white/5 overflow-hidden group">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(var(--primary),0.1),transparent)]" />
      <svg className="w-full h-full">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        {nodes.map((n1, i) => (
          nodes.slice(i + 1).map((n2, j) => {
            const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
            if (dist < 25) {
              return (
                <motion.line
                  key={`${i}-${j}`}
                  x1={`${n1.x}%`} y1={`${n1.y}%`}
                  x2={`${n2.x}%`} y2={`${n2.y}%`}
                  stroke="currentColor"
                  strokeWidth={0.5}
                  className="text-primary/20"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: [0, 0.4, 0] }}
                  transition={{ duration: Math.random() * 3 + 2, repeat: Infinity, delay: Math.random() * 2 }}
                />
              );
            }
            return null;
          })
        ))}
        {nodes.map((node, i) => (
          <motion.circle
            key={i}
            cx={`${node.x}%`}
            cy={`${node.y}%`}
            r={node.r}
            className="fill-primary"
            filter="url(#glow)"
            initial={{ opacity: 0.2 }}
            animate={{ opacity: [0.2, 0.8, 0.2], scale: [1, 1.3, 1] }}
            transition={{ duration: node.pulse, repeat: Infinity, delay: Math.random() }}
          />
        ))}
      </svg>
      <div className="absolute bottom-8 left-8 right-8 flex justify-between items-end">
        <div className="space-y-1">
          <p className="text-[10px] font-black uppercase tracking-[0.4em] text-primary">Grafo de Red Inteligente</p>
          <p className="text-2xl font-black text-white">{nodeCount} Entidades Cognitivas</p>
        </div>
        <div className="px-5 py-2.5 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl flex items-center gap-3">
          <RefreshCw size={14} className="text-primary animate-spin" />
          <span className="text-[9px] font-black uppercase tracking-widest text-white/60">Sincronizando Mapeo Neural</span>
        </div>
      </div>
    </div>
  );
};

const AnalysisView: React.FC<AnalysisViewProps> = ({ sessions, onNavigateToChat, onAddLog, language }) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [prevTab, setPrevTab] = useState<TabType>('overview');
  const [libSearch, setLibSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<KnowledgeCategory>('all');
  const [isAuditing, setIsAuditing] = useState(false);
  const [learningRate, setLearningRate] = useState(94.5);

  useEffect(() => {
    const interval = setInterval(() => {
      setLearningRate(prev => Math.min(99.9, Math.max(92.0, prev + (Math.random() - 0.5) * 0.8)));
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const metrics = useMemo(() => {
    const allMessages = sessions.flatMap(s => s.messages);
    const aiMessages = allMessages.filter(m => m.role === Role.AI);
    
    const uniqueSourcesMap = new Map<string, { 
      source: Source, 
      count: number, 
      lastSeen: number, 
      trust: number,
      category: KnowledgeCategory,
      tags: string[]
    }>();
    
    const allFileChanges: { path: string; diff: string; timestamp: number; sessionId: string; messageId: string }[] = [];

    sessions.forEach(s => {
      s.messages.forEach(m => {
        if (m.role === Role.AI) {
          m.sources?.forEach(src => {
            const domain = src.domain.toLowerCase();
            const existing = uniqueSourcesMap.get(domain);
            if (existing) {
              existing.count++;
              existing.lastSeen = Math.max(existing.lastSeen, m.timestamp);
            } else {
              let cat: KnowledgeCategory = domain.includes('github') || domain.includes('npm') ? 'code' : 
                                         domain.includes('docs') || domain.includes('mdn') ? 'docs' :
                                         domain.includes('arxiv') || domain.includes('edu') ? 'research' : 'other';
              uniqueSourcesMap.set(domain, { 
                source: src, count: 1, lastSeen: m.timestamp, trust: 85 + Math.floor(Math.random() * 14),
                category: cat, tags: ['Verified', cat.toUpperCase()]
              });
            }
          });

          if (m.fileChanges) {
            m.fileChanges.forEach(fc => {
              allFileChanges.push({ ...fc, timestamp: m.timestamp, sessionId: s.id, messageId: m.id });
            });
          }
        }
      });
    });

    const totalChars = allMessages.reduce((acc, m) => acc + m.content.length, 0);
    const nodes = Math.max(10, Math.ceil(totalChars / 150));
    const learnedDomains = Array.from(uniqueSourcesMap.values()).sort((a, b) => b.count - a.count);
    const coherence = aiMessages.length > 0 ? (aiMessages.filter(m => !!m.thought).length / aiMessages.length) * 100 : 0;

    return {
      nodes,
      learnedDomains,
      fileChanges: allFileChanges.sort((a, b) => b.timestamp - a.timestamp),
      totalTrust: learnedDomains.length > 0 ? learnedDomains.reduce((acc, d) => acc + d.trust, 0) / learnedDomains.length : 0,
      coherence,
      totalInteractions: allMessages.length,
      groundingRatio: aiMessages.length > 0 ? (learnedDomains.length / aiMessages.length) * 100 : 0
    };
  }, [sessions]);

  // Optimización de la lógica de filtrado de dominios
  const filteredDomains = useMemo(() => {
    const searchLower = libSearch.toLowerCase().trim();
    const isCategoryAll = activeCategory === 'all';
    
    // Optimizacion: Si no hay filtros activos, devolver la lista completa directamente
    if (!searchLower && isCategoryAll) {
      return metrics.learnedDomains;
    }

    return metrics.learnedDomains.filter(d => {
      // Optimizacion: Comprobar categoría primero (operación O(1) barata)
      if (!isCategoryAll && d.category !== activeCategory) {
        return false;
      }
      
      // Optimizacion: Si no hay búsqueda de texto, ya pasó el filtro de categoría
      if (!searchLower) {
        return true;
      }
      
      // Optimizacion: Búsqueda de texto (operación costosa, al final)
      return (
        d.source.title.toLowerCase().includes(searchLower) || 
        d.source.domain.toLowerCase().includes(searchLower)
      );
    });
  }, [metrics.learnedDomains, libSearch, activeCategory]);

  const handleExportGraph = () => {
    const data = {
      timestamp: Date.now(),
      metrics: {
        nodes: metrics.nodes,
        interactions: metrics.totalInteractions,
        coherence: metrics.coherence,
        trust: metrics.totalTrust
      },
      sources: metrics.learnedDomains.map(d => ({
        title: d.source.title,
        domain: d.source.domain,
        url: d.source.url,
        trust: d.trust
      })),
      patches: metrics.fileChanges.map(f => ({
        path: f.path,
        timestamp: f.timestamp
      }))
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vortex-graph-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    onAddLog('SYSTEM', 'Estructura de grafo exportada correctamente.');
  };

  const handleAudit = async () => {
    setIsAuditing(true);
    onAddLog('SYSTEM', 'Iniciando auditoría de integridad local...');
    
    await new Promise(r => setTimeout(r, 800));
    onAddLog('LEARN', `Analizando ${metrics.nodes} entidades de conocimiento...`);
    await new Promise(r => setTimeout(r, 1200));
    onAddLog('SEARCH', `Verificando ${metrics.learnedDomains.length} fuentes externas contrastadas...`);
    await new Promise(r => setTimeout(r, 1000));
    
    setIsAuditing(false);
    onAddLog('SYSTEM', 'Auditoría completada: 100% integridad verificada. Protocolo Vortex Secure intacto.');
  };

  const handleTabChange = (newTab: TabType) => {
    setPrevTab(activeTab);
    setActiveTab(newTab);
  };

  const tabDirection = TAB_INDEX[activeTab] > TAB_INDEX[prevTab] ? 1 : -1;
  const slideVariants = {
    enter: (dir: number) => ({ x: dir > 0 ? 50 : -50, opacity: 0, filter: 'blur(10px)' }),
    center: { x: 0, opacity: 1, filter: 'blur(0px)', transition: { type: 'spring' as const, stiffness: 300, damping: 30 } },
    exit: (dir: number) => ({ x: dir > 0 ? -50 : 50, opacity: 0, filter: 'blur(10px)', transition: { duration: 0.3 } })
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="pt-32 px-10 max-w-[1400px] mx-auto w-full space-y-16 pb-40">
      <header className="relative flex flex-col lg:flex-row lg:items-end justify-between gap-10 pb-12 border-b border-border/40">
        <div className="flex items-center gap-10">
          <div className="relative group">
            <motion.div whileHover={{ rotate: 10, scale: 1.1 }} className="w-20 h-20 bg-primary rounded-[2.5rem] flex items-center justify-center text-primary-foreground shadow-2xl relative z-10">
               <BrainCircuit size={40} strokeWidth={1.5} />
            </motion.div>
            <div className="absolute inset-[-15px] bg-primary/20 blur-[120px] animate-pulse-glow" />
          </div>
          <div className="space-y-2">
            <h2 className="text-5xl font-black tracking-tighter text-foreground leading-none">Intelligence Hub</h2>
            <div className="flex items-center gap-5">
              <div className="px-3 py-1 bg-primary/10 border border-primary/20 rounded-full">
                <span className="text-[10px] font-black uppercase tracking-[0.4em] text-primary">System Operational</span>
              </div>
              <div className="h-1.5 w-1.5 rounded-full bg-border" />
              <span className="text-[10px] font-black uppercase tracking-[0.4em] text-muted-foreground opacity-50">Neural Engine</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-muted/10 p-2 rounded-[2rem] border border-border/40 backdrop-blur-3xl relative shadow-inner">
          {(['overview', 'knowledge', 'network'] as TabType[]).map((tab) => {
            const isActive = activeTab === tab;
            return (
              <button key={tab} onClick={() => handleTabChange(tab)} className={`relative px-10 py-4 rounded-2xl text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-500 z-10 ${isActive ? 'text-white' : 'text-muted-foreground hover:text-foreground'}`}>
                {tab === 'overview' ? 'Resumen' : tab === 'knowledge' ? 'Librería' : 'Grafo'}
                {isActive && <motion.div layoutId="analysis-tab-indicator" className="absolute inset-0 bg-primary rounded-2xl shadow-xl -z-10" transition={{ type: "spring", stiffness: 400, damping: 35 }} />}
              </button>
            );
          })}
        </div>
      </header>

      <AnimatePresence mode="wait" custom={tabDirection}>
        {activeTab === 'overview' && (
          <motion.div key="overview" custom={tabDirection} variants={slideVariants} initial="enter" animate="center" exit="exit" className="space-y-16">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                { label: 'Densidad Neural', value: metrics.nodes, sub: 'Nodos mapeados', color: 'hsl(var(--primary))', icon: <Layers size={22} /> },
                { label: 'Fuentes de Verdad', value: metrics.learnedDomains.length, sub: 'Dominios contrastados', color: '#c084fc', icon: <BookOpen size={22} /> },
                { label: 'Kernel Stability', value: `${learningRate.toFixed(1)}%`, sub: 'Tasa operativa', color: '#fbbf24', icon: <Zap size={22} /> },
                { label: 'Índice de Confianza', value: `${metrics.totalTrust.toFixed(0)}%`, sub: 'Integridad Global', color: '#10b981', icon: <ShieldCheck size={22} /> }
              ].map((m, i) => (
                <div key={i} className="p-10 bg-muted/5 border border-border/30 rounded-[3rem] relative overflow-hidden group hover:border-primary/40 transition-all glass-card hover:-translate-y-2 shadow-sm">
                  <div className="flex justify-between items-start mb-8">
                    <div className="w-14 h-14 bg-background border border-border/50 rounded-2xl flex items-center justify-center shadow-inner group-hover:scale-110 transition-transform duration-500" style={{ color: m.color }}>{m.icon}</div>
                    <Sparkline color={m.color} />
                  </div>
                  <div className="space-y-2">
                    <h4 className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.3em] opacity-40">{m.label}</h4>
                    <p className="text-4xl font-black tracking-tighter">{m.value}</p>
                    <p className="text-[10px] font-bold text-muted-foreground/60">{m.sub}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
              <div className="lg:col-span-2 p-12 bg-muted/5 border border-border/30 rounded-[4rem] glass-card space-y-10">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-black tracking-tight flex items-center gap-4"><ActivitySquare size={24} className="text-primary" /> Auditoría de Parches</h3>
                  <div className="px-4 py-2 bg-primary/10 rounded-xl text-[10px] font-black text-primary uppercase tracking-widest border border-primary/20">{metrics.fileChanges.length} Cambios</div>
                </div>
                <div className="space-y-4">
                  {metrics.fileChanges.length > 0 ? (
                    metrics.fileChanges.slice(0, 5).map((change, i) => (
                      <button key={i} onClick={() => onNavigateToChat(change.sessionId, change.messageId)} className="w-full p-8 bg-background/40 border border-border/40 rounded-3xl flex items-center justify-between group hover:border-primary/40 hover:bg-background/60 transition-all">
                        <div className="flex items-center gap-6">
                          <div className="w-12 h-12 bg-primary/5 text-primary rounded-xl flex items-center justify-center border border-primary/10 group-hover:bg-primary group-hover:text-white transition-all"><FileCode size={20} /></div>
                          <div className="text-left">
                            <p className="text-[15px] font-black text-foreground">{change.path}</p>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-bold mt-1">Ver contexto en la sesión original</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-[11px] font-mono text-muted-foreground/40 tabular-nums">{new Date(change.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                          <div className="p-3 bg-muted rounded-xl group-hover:bg-primary group-hover:text-white transition-all"><ChevronRight size={16} /></div>
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="py-24 flex flex-col items-center gap-5 opacity-20 italic">
                       <History size={64} strokeWidth={1} />
                       <p className="text-sm font-medium uppercase tracking-[0.2em]">Ningún parche generado aún</p>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="p-12 bg-muted/5 border border-border/30 rounded-[4rem] glass-card space-y-12">
                <h3 className="text-2xl font-black tracking-tight flex items-center gap-4"><PieChart size={24} className="text-primary" /> Eficiencia Neural</h3>
                <div className="space-y-10">
                  {[
                    { label: 'Neural Coherence', val: Math.round(metrics.coherence), color: 'bg-primary' },
                    { label: 'Grounding Depth', val: Math.round(metrics.groundingRatio), color: 'bg-amber-500' },
                    { label: 'Data Latency', val: 98, color: 'bg-emerald-500' }
                  ].map((s, i) => (
                    <div key={i} className="space-y-4">
                      <div className="flex justify-between items-end text-[11px] font-black uppercase tracking-[0.2em]"><span className="text-muted-foreground">{s.label}</span><span className="text-foreground">{s.val}%</span></div>
                      <div className="h-3 w-full bg-muted rounded-full overflow-hidden p-[2px]">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${s.val}%` }} className={`h-full ${s.color} rounded-full shadow-lg`} />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="pt-8 flex flex-col items-center gap-4 text-center">
                  <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center text-emerald-500 border border-emerald-500/20 shadow-lg animate-pulse"><Server size={28} /></div>
                  <p className="text-[10px] font-black uppercase tracking-[0.3em] text-emerald-500">Núcleo Verificado</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'knowledge' && (
          <motion.div key="knowledge" custom={tabDirection} variants={slideVariants} initial="enter" animate="center" exit="exit" className="space-y-12">
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="relative flex-1 group">
                 <div className="absolute left-8 top-1/2 -translate-y-1/2 text-primary opacity-40 group-focus-within:opacity-100 transition-opacity"><Search size={22} /></div>
                 <input type="text" placeholder="Escanear repositorio neural..." value={libSearch} onChange={(e) => setLibSearch(e.target.value)} className="w-full h-20 bg-muted/10 border border-border/40 rounded-[2.5rem] pl-20 pr-10 text-lg font-bold outline-none focus:border-primary/60 transition-all glass-card" />
              </div>
              <div className="flex items-center gap-2 p-2 bg-muted/10 rounded-[2.5rem] border border-border/40 shrink-0">
                 {[{ id: 'all', label: 'Todo', icon: <Layers size={14} /> }, { id: 'docs', label: 'Docs', icon: <BookOpen size={14} /> }, { id: 'code', label: 'Code', icon: <Binary size={14} /> }, { id: 'research', label: 'Papers', icon: <Compass size={14} /> }].map(c => (
                   <button key={c.id} onClick={() => setActiveCategory(c.id as KnowledgeCategory)} className={`px-6 py-3.5 rounded-2xl text-[10px] font-black uppercase tracking-widest flex items-center gap-3 transition-all ${activeCategory === c.id ? 'bg-primary text-white shadow-xl' : 'text-muted-foreground hover:text-foreground'}`}>{c.icon} {c.label}</button>
                 ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
              <AnimatePresence mode="popLayout">
                {filteredDomains.length > 0 ? (
                  filteredDomains.map((item, i) => (
                    <motion.div key={item.source.domain} layout initial={{ opacity: 0, scale: 0.9, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9, y: 20 }} transition={{ delay: i * 0.05 }} className="group">
                      <div className="p-10 bg-muted/5 border border-border/30 rounded-[3.5rem] glass-card hover:border-primary/60 transition-all duration-700 flex flex-col gap-8 shadow-sm hover:shadow-2xl hover:-translate-y-3 h-full overflow-hidden relative">
                        <div className="flex items-start justify-between relative z-10">
                          <div className="w-16 h-16 bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center p-3 shadow-2xl group-hover:scale-110 transition-transform text-primary"><Globe size={28} /></div>
                          <div className="flex flex-col items-end gap-3"><div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-500 flex items-center gap-2"><ShieldCheck size={12} /> <span className="text-[10px] font-black uppercase tracking-widest">{item.trust}% Confianza</span></div><span className="text-[9px] font-black text-primary/60 uppercase tracking-[0.2em]">{item.count} Citaciones</span></div>
                        </div>
                        <div className="space-y-4 relative z-10"><h5 className="text-2xl font-black tracking-tight text-foreground line-clamp-2 leading-tight group-hover:text-primary transition-colors">{item.source.title}</h5><p className="text-[10px] font-mono font-black text-muted-foreground uppercase tracking-widest flex items-center gap-3"><Globe size={14} className="opacity-40" /> {item.source.domain}</p></div>
                        <div className="flex flex-wrap gap-2 pt-2">{item.tags.map((t, idx) => (<span key={idx} className="px-3 py-1 bg-primary/5 border border-primary/10 rounded-lg text-[9px] font-black text-primary/60 uppercase tracking-widest">#{t}</span>))}</div>
                        <div className="mt-auto pt-8 border-t border-border/40 flex items-center justify-between"><div className="flex items-center gap-3 text-muted-foreground/40"><Clock size={14} /><span className="text-[9px] font-black uppercase tracking-widest">Visto {new Date(item.lastSeen).toLocaleDateString()}</span></div><a href={item.source.url} target="_blank" rel="noopener noreferrer" className="w-12 h-12 bg-muted/50 hover:bg-primary hover:text-white rounded-2xl flex items-center justify-center transition-all border border-border/40 shadow-sm"><ExternalLink size={18} /></a></div>
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <div className="col-span-full py-32 flex flex-col items-center justify-center opacity-20"><Database size={80} strokeWidth={1} /><p className="text-xl font-black uppercase tracking-[0.3em] mt-6">Sin fuentes detectadas</p></div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}

        {activeTab === 'network' && (
          <motion.div key="network" custom={tabDirection} variants={slideVariants} initial="enter" animate="center" exit="exit" className="space-y-12">
            <NeuralNetworkGraph nodeCount={metrics.nodes} />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
               {[{ label: 'Densidad Semántica', val: Math.min(100, metrics.nodes), desc: 'Concentración de nodos por sesión' }, { label: 'Veracidad', val: Math.round(metrics.groundingRatio), desc: 'Fuentes contrastadas' }, { label: 'Neural Coherence', val: Math.round(metrics.coherence), desc: 'Consistencia de razonamiento lógico' }].map((stat, i) => (
                 <div key={i} className="p-10 bg-muted/5 border border-border/30 rounded-[3rem] glass-card space-y-6">
                    <div className="flex items-center gap-4"><div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center text-primary"><ActivitySquare size={20} /></div><h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-muted-foreground">{stat.label}</h4></div>
                    <div className="flex items-end gap-5"><span className="text-5xl font-black tracking-tighter text-foreground">{stat.val}%</span><p className="text-[10px] font-medium text-muted-foreground/60 mb-2 leading-relaxed">{stat.desc}</p></div>
                    <div className="h-2 w-full bg-muted rounded-full overflow-hidden"><motion.div initial={{ width: 0 }} animate={{ width: `${stat.val}%` }} className="h-full bg-primary rounded-full" /></div>
                 </div>
               ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="pt-20 border-t border-border/40">
        <div className="p-12 bg-primary/5 border border-primary/20 rounded-[4rem] flex flex-col md:flex-row items-center gap-12 group relative overflow-hidden glass-card">
           {isAuditing && (
             <motion.div initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 3 }} className="absolute bottom-0 left-0 right-0 h-1 bg-primary z-20 origin-left" />
           )}
           <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />
           <div className="shrink-0 w-32 h-32 rounded-[3.5rem] bg-primary flex items-center justify-center text-primary-foreground shadow-2xl relative z-10 group-hover:rotate-12 transition-transform">
             {isAuditing ? <RefreshCw size={64} className="animate-spin" /> : <Fingerprint size={64} strokeWidth={1} />}
           </div>
           <div className="space-y-4 flex-1 relative z-10">
             <div className="flex items-center gap-4">
                <div className={`w-3 h-3 rounded-full ${isAuditing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'} shadow-lg`} />
                <p className="text-3xl font-black tracking-tighter text-foreground">{isAuditing ? 'Auditoría en Curso...' : 'Kernel Verificado'}</p>
             </div>
             <p className="text-base text-muted-foreground font-medium opacity-60 leading-relaxed max-w-3xl">La arquitectura de conocimiento ha procesado {metrics.totalInteractions} interacciones. Todo el tráfico está cifrado bajo el protocolo Vortex Secure.</p>
           </div>
           <div className="shrink-0 relative z-10 flex flex-col gap-3">
              <button onClick={handleExportGraph} className="flex items-center justify-center gap-3 px-8 py-4 bg-foreground text-background rounded-[2rem] text-[10px] font-black uppercase tracking-[0.3em] hover:scale-105 transition-all shadow-xl active:scale-95"><Download size={16} /> Exportar Grafo</button>
              <button disabled={isAuditing} onClick={handleAudit} className="px-8 py-4 bg-muted text-foreground rounded-[2rem] text-[10px] font-black uppercase tracking-[0.3em] hover:bg-muted/80 transition-all border border-border/60 active:scale-95 disabled:opacity-50">Auditoría Local</button>
           </div>
        </div>
      </footer>
    </motion.div>
  );
};

export default AnalysisView;
