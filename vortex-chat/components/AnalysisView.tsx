import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  ArrowUpRight,
  BookOpenText,
  Bot,
  Clock3,
  FileCode2,
  FlaskConical,
  Globe,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';
import { ChatSession, ControlStatus, Language, LogEntry, OperationalStatus, Role, Source, TrainingRunSummary, TrainingStreamPayload } from '../types';
import { controlService } from '../services/controlService';
import { translations } from '../translations';

interface AnalysisViewProps {
  sessions: ChatSession[];
  onNavigateToChat: (sessionId: string, messageId?: string) => void;
  onAddLog: (level: LogEntry['level'], message: string) => void;
  language: Language;
  controlStatus: ControlStatus | null;
  operationalStatus: OperationalStatus | null;
  onBootstrap: () => Promise<unknown> | unknown;
  onModelInit: () => Promise<unknown> | unknown;
  onRestartRuntime: () => Promise<unknown> | unknown;
  onReloadInstructions: () => Promise<unknown> | unknown;
  onStartTraining: (mode: 'quick' | 'full') => Promise<unknown> | unknown;
  onSaveAllowlist: (domains: string[]) => Promise<string[]>;
  focusTab?: ControlTab;
}

type ControlTab = 'stack' | 'learning' | 'internet';

type SourceDomainSummary = {
  domain: string;
  count: number;
  lastSeen: number;
  title: string;
  url: string;
  kind: Source['kind'];
};

type ActivityItem = {
  id: string;
  label: string;
  detail: string;
  timestamp: number;
  sessionId: string;
  messageId?: string;
};

type PromotedResponseItem = {
  id: string;
  content: string;
  timestamp: number;
  sessionId: string;
  messageId: string;
};

const TAB_INDEX: Record<ControlTab, number> = {
  stack: 0,
  learning: 1,
  internet: 2,
};

const StatusPill: React.FC<{ ok: boolean; label: string }> = ({ ok, label }) => (
  <span
    className={`inline-flex items-center rounded-full px-3 py-1 text-[10px] font-black uppercase tracking-[0.24em] ${
      ok ? 'bg-primary/[0.10] text-primary' : 'bg-primary/[0.08] text-primary/70'
    }`}
  >
    {label}
  </span>
);

const Panel: React.FC<{ title: string; eyebrow?: string; children: React.ReactNode; className?: string }> = ({
  title,
  eyebrow,
  children,
  className = '',
}) => (
  <section className={`surface-panel rounded-[1.5rem] p-6 ${className}`}>
    {(eyebrow || title) && (
      <header className="mb-5">
        {eyebrow && <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{eyebrow}</p>}
        <h3 className="mt-2 text-2xl font-extrabold tracking-tight">{title}</h3>
      </header>
    )}
    {children}
  </section>
);

const AnalysisView: React.FC<AnalysisViewProps> = ({
  sessions,
  onNavigateToChat,
  onAddLog,
  language,
  controlStatus,
  operationalStatus,
  onBootstrap,
  onModelInit,
  onRestartRuntime,
  onReloadInstructions,
  onStartTraining,
  onSaveAllowlist,
  focusTab,
}) => {
  const t = translations[language];
  const [activeTab, setActiveTab] = useState<ControlTab>('stack');
  const [previousTab, setPreviousTab] = useState<ControlTab>('stack');
  const [allowlistDraft, setAllowlistDraft] = useState('');
  const [isSavingAllowlist, setIsSavingAllowlist] = useState(false);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [trainingStream, setTrainingStream] = useState<TrainingStreamPayload | null>(null);

  useEffect(() => {
    setAllowlistDraft((controlStatus?.internet?.allowlist || []).join(', '));
  }, [controlStatus?.internet?.allowlist]);

  useEffect(() => {
    if (!focusTab || focusTab === activeTab) return;
    setPreviousTab(activeTab);
    setActiveTab(focusTab);
  }, [activeTab, focusTab]);

  useEffect(() => {
    const unsubscribe = controlService.subscribeTrainingStream(
      (payload) => setTrainingStream(payload),
      () => {},
    );
    return unsubscribe;
  }, []);

  const metrics = useMemo(() => {
    const sourceMap = new Map<string, SourceDomainSummary>();
    const activities: ActivityItem[] = [];
    const fileChanges: Array<{ id: string; path: string; diff: string; timestamp: number; sessionId: string; messageId: string }> = [];
    const promotedResponses: PromotedResponseItem[] = [];
    let totalMessages = 0;
    let sourcedResponses = 0;
    let trainingEvents = 0;

    for (const session of sessions) {
      for (const message of session.messages) {
        totalMessages += 1;

        if (message.role === Role.AI) {
          if (message.sources?.length) sourcedResponses += 1;
          if (message.trainingEvent) {
            trainingEvents += 1;
            promotedResponses.push({
              id: `${session.id}:${message.id}`,
              content: message.content.slice(0, 220).trim(),
              timestamp: message.timestamp,
              sessionId: session.id,
              messageId: message.id,
            });
          }
          if (message.fileChanges?.length) {
            for (const change of message.fileChanges) {
              fileChanges.push({
                id: `${session.id}:${message.id}:${change.path}`,
                path: change.path,
                diff: change.diff,
                timestamp: message.timestamp,
                sessionId: session.id,
                messageId: message.id,
              });
            }
          }
        }

        if (message.sources?.length) {
          for (const source of message.sources) {
            const domain = source.domain || 'local';
            const existing = sourceMap.get(domain);
            if (existing) {
              existing.count += 1;
              existing.lastSeen = Math.max(existing.lastSeen, message.timestamp);
            } else {
              sourceMap.set(domain, {
                domain,
                count: 1,
                lastSeen: message.timestamp,
                title: source.title,
                url: source.url,
                kind: source.kind,
              });
            }
          }
        }

        activities.push({
          id: `${session.id}:${message.id}`,
          label:
            message.role === Role.USER
              ? (language === 'es' ? 'Prompt' : 'Prompt')
              : message.trainingEvent
                ? (language === 'es' ? 'Respuesta en cola' : 'Queued answer')
                : (language === 'es' ? 'Respuesta' : 'Response'),
          detail: message.content.slice(0, 140).trim() || (language === 'es' ? 'Sin contenido.' : 'No content.'),
          timestamp: message.timestamp,
          sessionId: session.id,
          messageId: message.id,
        });
      }
    }

    const topDomains = Array.from(sourceMap.values()).sort((left, right) => {
      if (right.count !== left.count) return right.count - left.count;
      return right.lastSeen - left.lastSeen;
    });

    return {
      totalMessages,
      sourcedResponses,
      trainingEvents,
      promotedResponses: promotedResponses.sort((left, right) => right.timestamp - left.timestamp).slice(0, 6),
      fileChanges: fileChanges.sort((left, right) => right.timestamp - left.timestamp),
      topDomains,
      recentActivity: activities.sort((left, right) => right.timestamp - left.timestamp).slice(0, 8),
      recentSessions: [...sessions].sort((left, right) => right.updatedAt - left.updatedAt).slice(0, 6),
    };
  }, [language, sessions]);

  const displayRuns = (trainingStream?.runs && trainingStream.runs.length > 0)
    ? trainingStream.runs
    : (controlStatus?.runs || []);
  const activeRunId = trainingStream?.active_run_id || controlStatus?.active_run_id || null;
  const activeRun = (activeRunId
    ? displayRuns.find((run) => run.run_id === activeRunId)
    : null) || displayRuns[0] || null;
  const latestRun = activeRun;
  const tabDirection = TAB_INDEX[activeTab] > TAB_INDEX[previousTab] ? 1 : -1;
  const modelReady = Boolean(
    controlStatus?.model?.cached
    || operationalStatus?.model_ready
    || controlStatus?.runtime?.runtime_ready
  );

  const summaryCards = [
    {
      title: language === 'es' ? 'Runtime' : 'Runtime',
      value: operationalStatus?.engine_ready ? (language === 'es' ? 'Activo' : 'Live') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: (operationalStatus?.engine_kind || 'sglang').toUpperCase(),
      icon: <Activity size={18} />,
      accent: 'text-primary',
    },
    {
      title: language === 'es' ? 'Modelo' : 'Model',
      value: modelReady ? (language === 'es' ? 'Listo' : 'Ready') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: operationalStatus?.active_model || controlStatus?.model?.model_id || 'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ',
      icon: <Bot size={18} />,
      accent: 'text-foreground',
    },
    {
      title: language === 'es' ? 'Aprendizaje' : 'Learning',
      value: `${metrics.trainingEvents}`,
      caption: language === 'es' ? 'respuestas en cola para aprendizaje' : 'answers queued for learning',
      icon: <FlaskConical size={18} />,
      accent: 'text-primary',
    },
    {
      title: language === 'es' ? 'Fuentes' : 'Sources',
      value: `${metrics.topDomains.length}`,
      caption: language === 'es' ? 'dominios visibles en sesiones' : 'domains visible in sessions',
      icon: <Globe size={18} />,
      accent: 'text-primary',
    },
  ];

  const runAction = async (label: string, action: () => Promise<unknown> | unknown, successMessage: string) => {
    setPendingAction(label);
    try {
      await action();
      onAddLog('SYSTEM', successMessage);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      onAddLog('SYSTEM', detail);
    } finally {
      setPendingAction(null);
    }
  };

  const handleSaveAllowlist = async () => {
    const domains = Array.from(new Set(allowlistDraft.split(/[\n,]/).map((item) => item.trim()).filter(Boolean)));
    setIsSavingAllowlist(true);
    try {
      const saved = await onSaveAllowlist(domains);
      setAllowlistDraft(saved.join(', '));
      onAddLog(
        'SEARCH',
        language === 'es' ? `Allowlist actualizada: ${saved.length} dominio(s).` : `Allowlist updated: ${saved.length} domain(s).`,
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      onAddLog('SYSTEM', detail);
    } finally {
      setIsSavingAllowlist(false);
    }
  };

  const renderRunCard = (run: TrainingRunSummary) => (
    <div key={run.run_id} className="rounded-[1.4rem] border border-border/40 bg-muted/20 px-4 py-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-black tracking-tight">
            {run.mode === 'quick'
              ? (language === 'es' ? 'Aprendizaje rápido' : 'Quick learning')
              : (language === 'es' ? 'Entreno completo' : 'Full training')}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">{run.run_id}</p>
        </div>
        <StatusPill ok={run.status === 'completed'} label={run.status} />
      </div>
      <div className="mt-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
        <p>{run.adapter_dir || (language === 'es' ? 'Sin adapter todavía' : 'No adapter yet')}</p>
        <p>{run.promotion?.decision || (language === 'es' ? 'Promoción manual' : 'Manual promotion')}</p>
      </div>
    </div>
  );

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto w-full max-w-[1280px] space-y-8 px-6 pb-32 pt-24 lg:px-8">
      <header className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-muted/20 px-4 py-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary">
            <Sparkles size={14} />
            <span>{t.analysis_hub_title}</span>
          </div>
          <div className="space-y-3">
            <h2 className="max-w-3xl text-4xl font-extrabold tracking-[-0.05em] text-foreground lg:text-5xl">
              {language === 'es' ? 'Un solo panel para arrancar, entrenar y controlar Vortex.' : 'One panel to boot, train, and control Vortex.'}
            </h2>
            <p className="max-w-2xl text-base leading-8 text-muted-foreground lg:text-lg">
              {language === 'es'
                ? 'Aquí ves el estado real del runtime, el modelo, los runs de entrenamiento y las fuentes que entran cuando activas internet por prompt.'
                : 'This view shows the real runtime, model, training runs, and sources that enter when internet is enabled for a prompt.'}
            </p>
          </div>
        </div>

        <Panel title={language === 'es' ? 'Resumen operativo' : 'Operational summary'} eyebrow={language === 'es' ? 'Estado actual' : 'Current status'}>
              <div className="grid gap-3 sm:grid-cols-2">
            {summaryCards.map((card) => (
              <div key={card.title} className="rounded-[1.1rem] border border-border/60 bg-muted/20 px-4 py-4">
                <div className={`flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.24em] ${card.accent}`}>
                  {card.icon}
                  <span>{card.title}</span>
                </div>
                <p className="mt-3 text-lg font-black tracking-tight">{card.value}</p>
                <p className="mt-1 break-words text-xs text-muted-foreground">{card.caption}</p>
              </div>
            ))}
          </div>
        </Panel>
      </header>

      <div className="flex flex-wrap items-center gap-3 rounded-[1.2rem] border border-border/60 bg-muted/20 p-2 backdrop-blur-xl">
        {([
          { id: 'stack', label: t.analysis_overview },
          { id: 'learning', label: t.analysis_library },
          { id: 'internet', label: t.analysis_graph },
        ] as Array<{ id: ControlTab; label: string }>).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => {
              setPreviousTab(activeTab);
              setActiveTab(tab.id);
            }}
            className={`relative rounded-[0.95rem] px-5 py-3 text-[11px] font-black uppercase tracking-[0.14em] transition-all ${
              activeTab === tab.id ? 'text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {activeTab === tab.id && (
              <motion.div
                layoutId="analysis-active-tab"
                className="absolute inset-0 rounded-[0.95rem] bg-primary"
                transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              />
            )}
            <span className="relative z-10">{tab.label}</span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait" custom={tabDirection}>
        {activeTab === 'stack' && (
          <motion.div
            key="stack"
            custom={tabDirection}
            initial={{ opacity: 0, x: tabDirection > 0 ? 40 : -40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: tabDirection > 0 ? -40 : 40 }}
            transition={{ type: 'spring', stiffness: 260, damping: 26 }}
            className="grid gap-6 lg:grid-cols-[1.08fr_0.92fr]"
          >
            <Panel title={language === 'es' ? 'Stack local' : 'Local stack'} eyebrow={language === 'es' ? 'Arranque y salud' : 'Boot and health'}>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-[1.1rem] border border-border/60 bg-muted/20 p-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.24em] text-primary"><Activity size={14} /><span>{language === 'es' ? 'Runtime' : 'Runtime'}</span></div>
                  <p className="mt-3 text-lg font-black tracking-tight">{operationalStatus?.engine_ready ? (language === 'es' ? 'Respondiendo' : 'Responding') : (language === 'es' ? 'Sin levantar' : 'Not started')}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{operationalStatus?.engine_base_url || 'http://127.0.0.1:30000'}</p>
                </div>
                <div className="rounded-[1.1rem] border border-border/60 bg-muted/20 p-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.24em] text-primary"><Bot size={14} /><span>{language === 'es' ? 'Modelo servido' : 'Served model'}</span></div>
                  <p className="mt-3 text-lg font-black tracking-tight break-words">{operationalStatus?.active_model || controlStatus?.model?.model_id || 'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ'}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{controlStatus?.model?.cached ? (language === 'es' ? 'Cache local detectada.' : 'Local cache detected.') : (language === 'es' ? 'Modelo pendiente de descarga.' : 'Model still needs download.')}</p>
                </div>
                <div className="rounded-[1.1rem] border border-border/60 bg-muted/20 p-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.24em] text-primary"><ShieldCheck size={14} /><span>Docker</span></div>
                  <p className="mt-3 text-lg font-black tracking-tight">{controlStatus?.docker?.ready ? (language === 'es' ? 'Listo' : 'Ready') : (language === 'es' ? 'Pendiente' : 'Pending')}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{controlStatus?.docker?.reason || 'docker_ready'}</p>
                </div>
                <div className="rounded-[1.1rem] border border-border/60 bg-muted/20 p-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.24em] text-primary"><Workflow size={14} /><span>{language === 'es' ? 'Bootstrap' : 'Bootstrap'}</span></div>
                  <p className="mt-3 text-lg font-black tracking-tight">{controlStatus?.bootstrap?.stage || (language === 'es' ? 'Sin actividad' : 'Idle')}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{controlStatus?.bootstrap?.message || (language === 'es' ? 'No hay tareas en curso.' : 'No tasks in progress.')}</p>
                </div>
              </div>

              <div className="mt-6 flex flex-wrap gap-3">
                <button type="button" onClick={() => { void runAction('bootstrap', () => onBootstrap(), language === 'es' ? 'Arranque solicitado.' : 'Bootstrap requested.'); }} className="rounded-full border border-primary/25 bg-primary/[0.10] px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] text-primary transition-all hover:bg-primary/[0.16]">{pendingAction === 'bootstrap' ? (language === 'es' ? 'Arrancando...' : 'Starting...') : (language === 'es' ? 'Iniciar stack' : 'Start stack')}</button>
                <button type="button" onClick={() => { void runAction('model', () => onModelInit(), language === 'es' ? 'Descarga del modelo iniciada.' : 'Model download started.'); }} className="rounded-full border border-border/50 bg-background px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] transition-all hover:border-primary/25">{pendingAction === 'model' ? (language === 'es' ? 'Descargando...' : 'Downloading...') : (language === 'es' ? 'Descargar modelo' : 'Download model')}</button>
                <button type="button" onClick={() => { void runAction('restart', () => onRestartRuntime(), language === 'es' ? 'Reinicio solicitado.' : 'Restart requested.'); }} className="rounded-full border border-border/50 bg-background px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] transition-all hover:border-primary/25">{pendingAction === 'restart' ? (language === 'es' ? 'Reiniciando...' : 'Restarting...') : (language === 'es' ? 'Reiniciar runtime' : 'Restart runtime')}</button>
                <button type="button" onClick={() => { void runAction('instructions', () => onReloadInstructions(), language === 'es' ? 'Instrucciones recargadas.' : 'Instructions reloaded.'); }} className="rounded-full border border-border/50 bg-background px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] transition-all hover:border-primary/25">{pendingAction === 'instructions' ? (language === 'es' ? 'Recargando...' : 'Reloading...') : (language === 'es' ? 'Recargar instrucciones' : 'Reload instructions')}</button>
              </div>
            </Panel>

            <div className="space-y-6">
              <Panel title={language === 'es' ? 'Contrato operativo' : 'Operational contract'} eyebrow={language === 'es' ? 'Lo que consume el frontend' : 'What the frontend reads'}>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><span className="font-semibold text-foreground">readyz</span><span className="text-right text-muted-foreground">{operationalStatus?.ok ? 'ok' : operationalStatus?.chat_ready ? (language === 'es' ? 'chat degradado' : 'degraded chat') : operationalStatus?.chat_block_reason || operationalStatus?.degraded_reason || 'pending'}</span></div>
                  <div className="flex items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><span className="font-semibold text-foreground">instructions.digest</span><span className="break-words text-right text-muted-foreground">{operationalStatus?.instructions?.digest || 'n/a'}</span></div>
                  <div className="flex items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><span className="font-semibold text-foreground">{language === 'es' ? 'Fuentes de instrucciones' : 'Instruction sources'}</span><span className="text-right text-muted-foreground">{(operationalStatus?.instructions?.sources || controlStatus?.instructions?.sources || []).length || 0}</span></div>
                </div>
              </Panel>

              <Panel title={language === 'es' ? 'Actividad reciente' : 'Recent activity'} eyebrow={language === 'es' ? 'Sesiones y prompts' : 'Sessions and prompts'}>
                <div className="space-y-3">
                  {metrics.recentActivity.length > 0 ? metrics.recentActivity.map((item) => (
                    <button key={item.id} type="button" onClick={() => onNavigateToChat(item.sessionId, item.messageId)} className="flex w-full items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3 text-left transition-all hover:border-primary/30">
                      <div className="min-w-0">
                        <p className="text-sm font-black tracking-tight">{item.label}</p>
                        <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{item.detail}</p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground"><Clock3 size={14} /><span>{new Date(item.timestamp).toLocaleTimeString(language === 'es' ? 'es-ES' : 'en-US', { hour: '2-digit', minute: '2-digit' })}</span></div>
                    </button>
                  )) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'Todavía no hay actividad registrada.' : 'No activity recorded yet.'}</p>}
                </div>
              </Panel>
            </div>
          </motion.div>
        )}
        {activeTab === 'learning' && (
          <motion.div
            key="learning"
            custom={tabDirection}
            initial={{ opacity: 0, x: tabDirection > 0 ? 40 : -40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: tabDirection > 0 ? -40 : 40 }}
            transition={{ type: 'spring', stiffness: 260, damping: 26 }}
            className="grid gap-6 lg:grid-cols-[1.02fr_0.98fr]"
          >
            <div className="space-y-6">
              <Panel title={language === 'es' ? 'Run activo' : 'Active run'} eyebrow={language === 'es' ? 'Tiempo real' : 'Real-time'}>
                {activeRun ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-4 rounded-[1.4rem] border border-border/40 bg-muted/15 p-4">
                      <div>
                        <p className="text-[10px] font-black uppercase tracking-[0.24em] text-primary">
                          {activeRun.mode === 'quick'
                            ? (language === 'es' ? 'Aprendizaje rápido' : 'Quick learning')
                            : (language === 'es' ? 'Entreno completo' : 'Full training')}
                        </p>
                        <h3 className="mt-2 text-xl font-black tracking-tight">{activeRun.run_id}</h3>
                        <p className="mt-2 text-sm text-muted-foreground">
                          {language === 'es'
                            ? `Estado ${activeRun.status} · etapa ${activeRun.stage || 'queued'}`
                            : `Status ${activeRun.status} · stage ${activeRun.stage || 'queued'}`}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 rounded-full border border-primary/20 bg-primary/[0.08] px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.24em] text-primary">
                        <span className={`h-2 w-2 rounded-full ${activeRunId ? 'animate-pulse bg-primary' : 'bg-primary'}`} />
                        <span>{activeRunId ? (language === 'es' ? 'Ejecutando' : 'Running') : (language === 'es' ? 'Último run' : 'Latest run')}</span>
                      </div>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-3">
                      <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3">
                        <p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">Stage</p>
                        <p className="mt-3 text-lg font-black tracking-tight">{activeRun.stage || 'queued'}</p>
                      </div>
                      <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3">
                        <p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">Dataset</p>
                        <p className="mt-3 break-all text-sm font-black tracking-tight">{activeRun.dataset_hash || 'pending'}</p>
                      </div>
                      <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3">
                        <p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">{language === 'es' ? 'Adapter' : 'Adapter'}</p>
                        <p className="mt-3 break-all text-sm font-black tracking-tight">{activeRun.adapter_dir || 'pending'}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {language === 'es'
                      ? 'Todavía no hay runs activos. Desde aquí puedes lanzar aprendizaje rápido o un entreno completo.'
                      : 'No active runs yet. Launch quick learning or a full training run from here.'}
                  </p>
                )}
              </Panel>

            <Panel title={language === 'es' ? 'Entrenamiento manual' : 'Manual training'} eyebrow={language === 'es' ? 'Quick y full' : 'Quick and full'}>
              <div className="grid gap-4 md:grid-cols-2">
                <button type="button" onClick={() => { void runAction('quick-train', () => onStartTraining('quick'), language === 'es' ? 'Aprendizaje rápido lanzado.' : 'Quick learning launched.'); }} className="rounded-[1.5rem] border border-primary/25 bg-primary/[0.10] p-5 text-left transition-all hover:bg-primary/[0.16]">
                  <p className="text-[10px] font-black uppercase tracking-[0.24em] text-primary">{language === 'es' ? 'Aprender rápido' : 'Quick learning'}</p>
                  <p className="mt-3 text-lg font-black tracking-tight">{language === 'es' ? 'Feedback reciente + dataset incremental' : 'Recent feedback + incremental dataset'}</p>
                  <p className="mt-2 text-sm leading-7 text-muted-foreground">{language === 'es' ? 'Lanza un run conservador para aprovechar feedback positivo y episodios recientes.' : 'Runs a conservative job that reuses positive feedback and recent episodes.'}</p>
                </button>
                <button type="button" onClick={() => { void runAction('full-train', () => onStartTraining('full'), language === 'es' ? 'Entreno completo lanzado.' : 'Full training launched.'); }} className="rounded-[1.5rem] border border-border/45 bg-background p-5 text-left transition-all hover:border-primary/25">
                  <p className="text-[10px] font-black uppercase tracking-[0.24em] text-primary">{language === 'es' ? 'Entreno completo' : 'Full training'}</p>
                  <p className="mt-3 text-lg font-black tracking-tight">{language === 'es' ? 'Eval + bench + cuarentena' : 'Eval + bench + quarantine'}</p>
                  <p className="mt-2 text-sm leading-7 text-muted-foreground">{language === 'es' ? 'Genera adapter, manifiesto y métricas, pero mantiene la promoción en manual.' : 'Produces adapter, manifest, and metrics while keeping promotion manual.'}</p>
                </button>
              </div>

              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">{language === 'es' ? 'Runs detectados' : 'Detected runs'}</p><p className="mt-3 text-2xl font-black tracking-tight">{displayRuns.length}</p></div>
                <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">{language === 'es' ? 'Respuestas en cola' : 'Queued answers'}</p><p className="mt-3 text-2xl font-black tracking-tight">{metrics.trainingEvents}</p></div>
                <div className="rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3"><p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">{language === 'es' ? 'Cambios sugeridos' : 'Suggested changes'}</p><p className="mt-3 text-2xl font-black tracking-tight">{metrics.fileChanges.length}</p></div>
              </div>
            </Panel>
            </div>

            <div className="space-y-6">
              <Panel title={language === 'es' ? 'Últimos runs' : 'Latest runs'} eyebrow={language === 'es' ? 'Artefactos y estado' : 'Artifacts and state'}>
                <div className="space-y-3">
                  {(controlStatus?.runs || []).length > 0 ? (controlStatus?.runs || []).slice(0, 4).map(renderRunCard) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'Todavía no hay runs guardados.' : 'No saved runs yet.'}</p>}
                </div>
              </Panel>

              <Panel title={language === 'es' ? 'Lo que aprende desde el chat' : 'What it learns from chat'} eyebrow={language === 'es' ? 'Dataset incremental' : 'Incremental dataset'}>
                <div className="space-y-3">
                  {metrics.promotedResponses.length > 0 ? metrics.promotedResponses.map((item) => (
                    <button key={item.id} type="button" onClick={() => onNavigateToChat(item.sessionId, item.messageId)} className="flex w-full items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3 text-left transition-all hover:border-primary/30">
                      <div className="min-w-0">
                        <p className="text-sm font-black tracking-tight">{language === 'es' ? 'Respuesta promovida' : 'Promoted answer'}</p>
                        <p className="mt-1 line-clamp-3 text-sm text-muted-foreground">{item.content || (language === 'es' ? 'Sin contenido visible.' : 'No visible content.')}</p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground"><Clock3 size={14} /><span>{new Date(item.timestamp).toLocaleTimeString(language === 'es' ? 'es-ES' : 'en-US', { hour: '2-digit', minute: '2-digit' })}</span></div>
                    </button>
                  )) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'Todavía no hay respuestas promovidas para auto-mejora.' : 'No promoted answers for self-improvement yet.'}</p>}
                </div>
              </Panel>

              <Panel title={language === 'es' ? 'Cambios detectados en respuestas' : 'Changes detected in answers'} eyebrow={language === 'es' ? 'Parches y diffs' : 'Patches and diffs'}>
                <div className="space-y-3">
                  {metrics.fileChanges.length > 0 ? metrics.fileChanges.slice(0, 5).map((change) => (
                    <button key={change.id} type="button" onClick={() => onNavigateToChat(change.sessionId, change.messageId)} className="flex w-full items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3 text-left transition-all hover:border-primary/30">
                      <div className="min-w-0">
                        <p className="text-sm font-black tracking-tight">{change.path}</p>
                        <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{change.diff}</p>
                      </div>
                      <span className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground"><FileCode2 size={14} /><ArrowUpRight size={14} /></span>
                    </button>
                  )) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'Aún no hay diffs sugeridos desde el chat.' : 'No diffs have been suggested from chat yet.'}</p>}
                </div>
              </Panel>
            </div>
          </motion.div>
        )}

        {activeTab === 'internet' && (
          <motion.div
            key="internet"
            custom={tabDirection}
            initial={{ opacity: 0, x: tabDirection > 0 ? 40 : -40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: tabDirection > 0 ? -40 : 40 }}
            transition={{ type: 'spring', stiffness: 260, damping: 26 }}
            className="grid gap-6 lg:grid-cols-[0.96fr_1.04fr]"
          >
            <div className="space-y-6">
              <Panel title={language === 'es' ? 'Modo internet por prompt' : 'Per-prompt internet mode'} eyebrow={language === 'es' ? 'Control de dominios' : 'Domain control'}>
                <div className="rounded-[1.35rem] border border-border/40 bg-muted/15 px-4 py-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-black tracking-tight">{language === 'es' ? 'Allowlist local' : 'Local allowlist'}</p>
                      <p className="mt-1 text-sm leading-7 text-muted-foreground">{language === 'es' ? 'Se aplica solo cuando activas el globo en el prompt. No enciende rastreo web continuo.' : 'Applied only when you enable the globe on a prompt. It does not enable continuous web crawling.'}</p>
                    </div>
                    <StatusPill ok={!operationalStatus?.web_disabled || (controlStatus?.internet?.allowlist || []).length > 0} label={operationalStatus?.web_disabled ? (language === 'es' ? 'Manual' : 'Manual') : (language === 'es' ? 'Revisar' : 'Review')} />
                  </div>
                  <textarea value={allowlistDraft} onChange={(event) => setAllowlistDraft(event.target.value)} className="mt-4 min-h-[130px] w-full rounded-[1.2rem] border border-border/50 bg-background px-4 py-3 text-sm outline-none transition-all focus:border-primary/30" placeholder="docs.python.org, react.dev, fastapi.tiangolo.com" />
                  <div className="mt-4 flex flex-wrap gap-3">
                    <button type="button" onClick={() => { void handleSaveAllowlist(); }} className="rounded-full border border-primary/25 bg-primary/[0.10] px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] text-primary transition-all hover:bg-primary/[0.16]">{isSavingAllowlist ? (language === 'es' ? 'Guardando...' : 'Saving...') : (language === 'es' ? 'Guardar dominios' : 'Save domains')}</button>
                  </div>
                </div>
              </Panel>

              <Panel title={language === 'es' ? 'Fuentes detectadas' : 'Detected sources'} eyebrow={language === 'es' ? 'Grounding visible' : 'Visible grounding'}>
                <div className="space-y-3">
                  {metrics.topDomains.length > 0 ? metrics.topDomains.slice(0, 6).map((domain) => (
                    <a key={domain.domain} href={domain.url} target="_blank" rel="noreferrer" className="flex items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3 transition-all hover:border-primary/30">
                      <div className="min-w-0">
                        <p className="text-sm font-black tracking-tight">{domain.domain}</p>
                        <p className="mt-1 line-clamp-1 text-sm text-muted-foreground">{domain.title}</p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="text-sm font-black">{domain.count}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{new Date(domain.lastSeen).toLocaleDateString(language === 'es' ? 'es-ES' : 'en-US')}</p>
                      </div>
                    </a>
                  )) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'Aún no hay fuentes registradas en respuestas.' : 'No sources have been recorded in responses yet.'}</p>}
                </div>
              </Panel>
            </div>

            <div className="space-y-6">
              <Panel title={language === 'es' ? 'Cómo se usa internet' : 'How internet is used'} eyebrow={language === 'es' ? 'Política actual' : 'Current policy'}>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[1.25rem] border border-border/40 bg-muted/15 px-4 py-4"><div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.22em] text-primary"><ShieldCheck size={14} /><span>{language === 'es' ? 'Modo por defecto' : 'Default mode'}</span></div><p className="mt-3 text-lg font-black tracking-tight">{operationalStatus?.web_disabled ? 'Local-only' : (language === 'es' ? 'Mixto' : 'Mixed')}</p><p className="mt-1 text-sm text-muted-foreground">{language === 'es' ? 'Solo se consulta internet cuando lo marcas en el prompt.' : 'Internet is only used when you enable it on the prompt.'}</p></div>
                  <div className="rounded-[1.25rem] border border-border/40 bg-muted/15 px-4 py-4"><div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.22em] text-primary"><BookOpenText size={14} /><span>{language === 'es' ? 'Trazabilidad' : 'Traceability'}</span></div><p className="mt-3 text-lg font-black tracking-tight">{metrics.sourcedResponses}</p><p className="mt-1 text-sm text-muted-foreground">{language === 'es' ? 'respuestas con fuentes visibles en el chat' : 'responses with visible sources in chat'}</p></div>
                </div>

                <div className="mt-6 rounded-[1.35rem] border border-border/40 bg-muted/15 px-4 py-4">
                  <p className="text-[10px] font-black uppercase tracking-[0.22em] text-primary">{language === 'es' ? 'Último run visible' : 'Latest visible run'}</p>
                  {latestRun ? (
                    <div className="mt-3 flex items-start justify-between gap-4">
                      <div>
                        <p className="text-sm font-black tracking-tight">{latestRun.run_id}</p>
                        <p className="mt-1 text-sm text-muted-foreground">{latestRun.mode === 'quick' ? (language === 'es' ? 'Aprendizaje rápido' : 'Quick learning') : (language === 'es' ? 'Entreno completo' : 'Full training')}</p>
                      </div>
                      <StatusPill ok={latestRun.status === 'completed'} label={latestRun.status} />
                    </div>
                  ) : <p className="mt-3 text-sm text-muted-foreground">{language === 'es' ? 'No hay runs todavía.' : 'No runs yet.'}</p>}
                </div>
              </Panel>

              <Panel title={language === 'es' ? 'Sesiones recientes' : 'Recent sessions'} eyebrow={language === 'es' ? 'Entrada manual' : 'Manual input'}>
                <div className="space-y-3">
                  {metrics.recentSessions.length > 0 ? metrics.recentSessions.map((session) => (
                    <button key={session.id} type="button" onClick={() => onNavigateToChat(session.id)} className="flex w-full items-start justify-between gap-4 rounded-[1.2rem] border border-border/40 bg-muted/15 px-4 py-3 text-left transition-all hover:border-primary/30">
                      <div className="min-w-0">
                        <p className="text-sm font-black tracking-tight">{session.title}</p>
                        <p className="mt-1 text-sm text-muted-foreground">{session.messages.length} {language === 'es' ? 'mensajes' : 'messages'}</p>
                      </div>
                      <span className="text-xs text-muted-foreground">{new Date(session.updatedAt).toLocaleTimeString(language === 'es' ? 'es-ES' : 'en-US', { hour: '2-digit', minute: '2-digit' })}</span>
                    </button>
                  )) : <p className="text-sm text-muted-foreground">{language === 'es' ? 'No hay sesiones guardadas.' : 'No saved sessions yet.'}</p>}
                </div>
              </Panel>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default AnalysisView;
