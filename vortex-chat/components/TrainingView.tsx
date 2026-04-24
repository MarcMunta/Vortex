import React, { useDeferredValue, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Bot, BrainCircuit, Clock3, FileCode2, FlaskConical, GitBranch, PauseCircle, PlayCircle, RefreshCw, Search, ShieldCheck, Sparkles } from 'lucide-react';
import { AutonomyEvent, AutonomyStatus, ChatSession, ControlStatus, Language, LogEntry, TrainingCampaignSummary, TrainingRunSummary, TrainingStreamPayload } from '../types';
import { controlService } from '../services/controlService';
import TrainingReviewModal from './TrainingReviewModal';

interface TrainingViewProps {
  sessions: ChatSession[];
  language: Language;
  controlStatus: ControlStatus | null;
  onAddLog: (level: LogEntry['level'], message: string) => void;
  onStartTraining: (mode: 'quick' | 'full') => Promise<unknown> | unknown;
  onStartAutonomy: () => Promise<unknown> | unknown;
  onStopAutonomy: () => Promise<unknown> | unknown;
  onConfigureAutonomy: (config: {
    enabled?: boolean;
    reflection_enabled?: boolean;
    training_enabled?: boolean;
    autoedit_enabled?: boolean;
    multi_agent_dialogue_enabled?: boolean;
    descriptive_reports_enabled?: boolean;
    live_autoedit_enabled?: boolean;
  }) => Promise<unknown> | unknown;
}

const Panel: React.FC<{ title: string; eyebrow?: string; className?: string; children: React.ReactNode }> = ({ title, eyebrow, className = '', children }) => (
  <section className={`surface-panel rounded-[1.6rem] p-6 ${className}`}>
    <header className="mb-5">
      {eyebrow && <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{eyebrow}</p>}
      <h3 className="mt-2 text-2xl font-black tracking-tight">{title}</h3>
    </header>
    {children}
  </section>
);

const formatTime = (ts: number | null | undefined, language: Language): string => {
  if (!ts) return language === 'es' ? 'Sin fecha' : 'No timestamp';
  const normalized = ts > 1_000_000_000_000 ? ts : ts * 1000;
  return new Date(normalized).toLocaleString(language === 'es' ? 'es-ES' : 'en-US', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
};

const normalizeProgress = (value: number | null | undefined): number => {
  const raw = Number(value || 0);
  if (!Number.isFinite(raw)) return 0;
  if (raw > 1) return Math.max(0, Math.min(100, Math.round(raw)));
  return Math.max(0, Math.min(100, Math.round(raw * 100)));
};

const chipTone = (value: string | null | undefined): string => {
  switch (String(value || '').toLowerCase()) {
    case 'completed': return 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300';
    case 'rolled_back': return 'border-amber-500/25 bg-amber-500/10 text-amber-300';
    case 'degraded': return 'border-rose-500/25 bg-rose-500/10 text-rose-300';
    case 'blocked': return 'border-orange-500/25 bg-orange-500/10 text-orange-300';
    case 'training':
    case 'evaluating':
    case 'applying':
    case 'verifying': return 'border-sky-500/25 bg-sky-500/10 text-sky-300';
    default: return 'border-border/60 bg-muted/20 text-foreground/80';
  }
};

const TrainingView: React.FC<TrainingViewProps> = ({ sessions, language, controlStatus, onAddLog, onStartTraining, onStartAutonomy, onStopAutonomy, onConfigureAutonomy }) => {
  const [trainingStream, setTrainingStream] = useState<TrainingStreamPayload | null>(null);
  const [autonomyStream, setAutonomyStream] = useState<{ status: AutonomyStatus; events: AutonomyEvent[] } | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<TrainingRunSummary | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState('all');
  const [resultFilter, setResultFilter] = useState('all');
  const [campaignFilter, setCampaignFilter] = useState('all');
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    const closeTraining = controlService.subscribeTrainingStream((payload) => setTrainingStream(payload));
    const closeAutonomy = controlService.subscribeAutonomyStream((payload) => setAutonomyStream({ status: payload.status, events: payload.events || [] }));
    return () => { closeTraining(); closeAutonomy(); };
  }, []);

  const autonomy = autonomyStream?.status || controlStatus?.autonomy || null;
  const runs = useMemo(() => ((trainingStream?.runs && trainingStream.runs.length > 0) ? trainingStream.runs : (controlStatus?.runs || [])), [trainingStream?.runs, controlStatus?.runs]);
  const campaign = (trainingStream?.campaign || autonomy?.campaign || null) as TrainingCampaignSummary | null;
  const activeRunId = trainingStream?.active_run_id || controlStatus?.active_run_id || null;
  const activeRun = trainingStream?.active_run || runs.find((run) => run.run_id === activeRunId) || runs[0] || null;
  const pipelineRuns = (trainingStream?.pipeline_runs || runs.filter((run) => ['planned', 'blocked', 'training', 'evaluating', 'applying', 'verifying'].includes(String(run.lifecycle_state || '').toLowerCase()))).slice(0, 8);
  const blockedRuns = (trainingStream?.blocked_runs || autonomy?.blocked_runs || []).slice(0, 6);
  const nextRunAt = trainingStream?.next_run_scheduled_at || autonomy?.next_run_scheduled_at || null;
  const campaignOptions = useMemo(() => Array.from(new Set(runs.map((run) => String(run.campaign_id || '')).filter(Boolean))), [runs]);
  const filteredRuns = useMemo(() => {
    const text = deferredQuery.trim().toLowerCase();
    return runs.filter((run) => {
      const lifecycle = String(run.lifecycle_state || '').toLowerCase();
      const haystack = [run.display_name, run.display_description, run.objective, run.run_id, run.campaign_id, ...(run.learning_focus || [])].join(' ').toLowerCase();
      const resultOk = resultFilter === 'all'
        || (resultFilter === 'terminal' && ['completed', 'rolled_back', 'degraded'].includes(lifecycle))
        || lifecycle === resultFilter;
      return (statusFilter === 'all' || lifecycle === statusFilter) && resultOk && (campaignFilter === 'all' || String(run.campaign_id || '') === campaignFilter) && (!text || haystack.includes(text));
    });
  }, [runs, deferredQuery, statusFilter, resultFilter, campaignFilter]);

  useEffect(() => {
    if (!selectedRunId) return;
    const streamedRun = runs.find((run) => run.run_id === selectedRunId);
    if (streamedRun) setSelectedRun((prev) => (prev && prev.run_id === selectedRunId ? { ...prev, ...streamedRun } : streamedRun));
  }, [runs, selectedRunId]);

  const runAction = async (label: string, action: () => Promise<unknown> | unknown, successMessage: string) => {
    setBusyAction(label);
    try { await action(); onAddLog('LEARN', successMessage); } catch (error) { onAddLog('SYSTEM', error instanceof Error ? error.message : String(error)); } finally { setBusyAction(null); }
  };

  const openTrainingReview = async (run: TrainingRunSummary) => {
    setSelectedRunId(run.run_id);
    setSelectedRun(run);
    setReviewLoading(true);
    try { setSelectedRun((await controlService.getTrainingRun(run.run_id)) || run); } finally { setReviewLoading(false); }
  };

  const resetTrainingState = async () => {
    setSelectedRunId(null); setSelectedRun(null); setReviewLoading(false); setTrainingStream(null);
    await controlService.resetTrainingState({ clear_runs: true, clear_learning_queue: true });
  };

  const latestNotebook = (activeRun?.notebook_sections || []).slice(-4).reverse();
  const latestDialogue = (activeRun?.agent_dialogue || []).slice(-4).reverse();
  const gateEntries = Object.entries(activeRun?.gate_results || {}).slice(0, 8);
  const executionProgress = normalizeProgress(trainingStream?.execution_progress_pct ?? activeRun?.execution_progress_pct ?? activeRun?.progress_pct);
  const pipelineProgress = normalizeProgress(trainingStream?.pipeline_progress_pct ?? activeRun?.pipeline_progress_pct);
  const sessionsWithMessages = sessions.filter((session) => session.messages.length > 0).length;

  return (
    <>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto w-full max-w-[1380px] space-y-8 px-6 pb-32 pt-24 lg:px-8">
        <header className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="space-y-5">
            <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-muted/15 px-4 py-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary"><Sparkles size={14} /><span>{language === 'es' ? 'Centro de control de aprendizaje' : 'Learning control center'}</span></div>
            <div className="space-y-3">
              <h2 className="max-w-3xl text-4xl font-extrabold tracking-[-0.05em] text-foreground lg:text-5xl">{language === 'es' ? 'Entrenamiento 24/7 con estados, gates y campanas visibles.' : '24/7 training with visible states, gates, and campaigns.'}</h2>
              <p className="max-w-3xl text-base leading-8 text-muted-foreground lg:text-lg">{language === 'es' ? 'La vista principal ya muestra que pasa ahora, que viene despues, como rinde la campana y como buscar runs por estado o foco.' : 'The main view now shows what is happening now, what comes next, how the campaign performs, and how to search runs by state or focus.'}</p>
            </div>
            <div className="flex flex-wrap gap-3">
              <button type="button" onClick={() => { void runAction('enable', async () => { await onConfigureAutonomy({ enabled: true, reflection_enabled: true, training_enabled: true, autoedit_enabled: true, multi_agent_dialogue_enabled: true, descriptive_reports_enabled: true, live_autoedit_enabled: true }); await onStartAutonomy(); }, language === 'es' ? 'Autonomia 24/7 activada.' : '24/7 autonomy enabled.'); }} className="rounded-full border border-primary/25 bg-primary/[0.10] px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-primary transition-all hover:bg-primary/[0.16]"><span className="inline-flex items-center gap-2"><PlayCircle size={14} />24/7</span></button>
              <button type="button" onClick={() => { void runAction('pause', () => onStopAutonomy(), language === 'es' ? 'Autonomia pausada.' : 'Autonomy paused.'); }} className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"><span className="inline-flex items-center gap-2"><PauseCircle size={14} />{language === 'es' ? 'Pausar' : 'Pause'}</span></button>
              <button type="button" onClick={() => { void runAction('quick', () => onStartTraining('quick'), language === 'es' ? 'Run quick lanzado.' : 'Quick run started.'); }} className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"><span className="inline-flex items-center gap-2"><RefreshCw size={14} />Quick</span></button>
              <button type="button" onClick={() => { void runAction('full', () => onStartTraining('full'), language === 'es' ? 'Run full lanzado.' : 'Full run started.'); }} className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"><span className="inline-flex items-center gap-2"><FlaskConical size={14} />Full</span></button>
              <button type="button" onClick={() => { void runAction('reset', () => resetTrainingState(), language === 'es' ? 'Estado de entrenamiento reiniciado.' : 'Training state reset.'); }} className="rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80 transition-all hover:border-primary/20 hover:text-foreground"><span className="inline-flex items-center gap-2"><BrainCircuit size={14} />{language === 'es' ? 'Reiniciar estado' : 'Reset state'}</span></button>
            </div>
            {busyAction && <p className="text-xs text-primary">{language === 'es' ? `Aplicando: ${busyAction}` : `Applying: ${busyAction}`}</p>}
          </div>
          <Panel title={language === 'es' ? 'Resumen operativo' : 'Operational summary'} eyebrow="Live">
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                { label: language === 'es' ? 'Campana' : 'Campaign', value: campaign?.campaign_id || 'n/a', icon: <GitBranch size={16} />, caption: campaign?.objective || '-' },
                { label: language === 'es' ? 'Autonomia' : 'Autonomy', value: autonomy?.enabled ? 'active' : 'paused', icon: <Activity size={16} />, caption: autonomy?.state || '-' },
                { label: language === 'es' ? 'Siguiente run' : 'Next run', value: nextRunAt ? formatTime(nextRunAt, language) : '-', icon: <Clock3 size={16} />, caption: trainingStream?.scheduled_followup_reason || autonomy?.scheduled_followup_reason || '-' },
                { label: language === 'es' ? 'Autoedicion' : 'Self-edit', value: autonomy?.config?.live_autoedit_enabled ? 'live' : 'safe', icon: <FileCode2 size={16} />, caption: language === 'es' ? 'Repo limpio requerido.' : 'Requires clean repo.' },
              ].map((item) => <div key={item.label} className="rounded-[1.2rem] border border-border/50 bg-muted/15 p-4"><div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary">{item.icon}<span>{item.label}</span></div><p className="mt-3 text-sm font-black break-all">{item.value}</p><p className="mt-1 text-xs text-muted-foreground">{item.caption}</p></div>)}
            </div>
          </Panel>
        </header>

        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title={language === 'es' ? 'Ahora' : 'Now'} eyebrow={language === 'es' ? 'Run activo' : 'Active run'}>
            {activeRun ? (
              <div className="space-y-5">
                <button type="button" onClick={() => { void openTrainingReview(activeRun); }} className="w-full rounded-[1.3rem] border border-primary/25 bg-primary/[0.08] p-5 text-left transition-all hover:bg-primary/[0.12]">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-lg font-black tracking-tight">{activeRun.display_name || activeRun.run_id}</p>
                      <p className="mt-2 text-sm leading-7 text-muted-foreground">{activeRun.display_description || activeRun.objective || activeRun.blocked_reason || activeRun.terminal_reason || '-'}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <span className={`rounded-full border px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] ${chipTone(activeRun.lifecycle_state || activeRun.status)}`}>{activeRun.lifecycle_state || activeRun.status}</span>
                      <span className="rounded-full border border-border/60 bg-background px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80">{activeRun.mode}</span>
                    </div>
                  </div>
                  <div className="mt-5 grid gap-3 md:grid-cols-2">
                    <div>
                      <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground"><span>Execution</span><span>{executionProgress}%</span></div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted/30"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.max(3, executionProgress)}%` }} /></div>
                    </div>
                    <div>
                      <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground"><span>Pipeline</span><span>{pipelineProgress}%</span></div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted/30"><div className="h-full rounded-full bg-sky-400" style={{ width: `${Math.max(3, pipelineProgress)}%` }} /></div>
                    </div>
                  </div>
                </button>

                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded-[1.15rem] border border-border/60 bg-muted/15 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Fase' : 'Phase'}</p><p className="mt-3 text-sm font-black">{activeRun.stage || '-'}</p></div>
                  <div className="rounded-[1.15rem] border border-border/60 bg-muted/15 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">ETA</p><p className="mt-3 text-sm font-black">{activeRun.retry_in_s ? `${Math.round(activeRun.retry_in_s)}s` : (nextRunAt ? formatTime(nextRunAt, language) : '-')}</p></div>
                  <div className="rounded-[1.15rem] border border-border/60 bg-muted/15 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Motivo' : 'Reason'}</p><p className="mt-3 text-sm font-black break-words">{activeRun.blocked_reason || activeRun.queue_reason || activeRun.terminal_reason || '-'}</p></div>
                </div>

                <div className="grid gap-4 lg:grid-cols-2">
                  <div className="rounded-[1.2rem] border border-border/60 bg-muted/15 p-4">
                    <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary"><ShieldCheck size={14} /><span>Gates</span></div>
                    <div className="mt-4 grid gap-2">
                      {gateEntries.length > 0 ? gateEntries.map(([key, value]) => <div key={key} className="flex items-center justify-between rounded-[0.95rem] border border-border/50 bg-background/70 px-3 py-2 text-sm"><span className="text-muted-foreground">{key.replaceAll('_', ' ')}</span><span className={typeof value === 'boolean' ? (value ? 'text-emerald-300' : 'text-rose-300') : 'text-foreground'}>{String(value)}</span></div>) : <div className="rounded-[0.95rem] border border-dashed border-border/60 bg-background/50 px-3 py-4 text-sm text-muted-foreground">{language === 'es' ? 'Sin gates todavia.' : 'No gates yet.'}</div>}
                    </div>
                  </div>
                  <div className="rounded-[1.2rem] border border-border/60 bg-muted/15 p-4">
                    <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary"><Bot size={14} /><span>{language === 'es' ? 'Dialogo vivo' : 'Live dialogue'}</span></div>
                    <div className="mt-4 space-y-2">
                      {latestDialogue.length > 0 ? latestDialogue.map((turn) => <div key={turn.id} className={`rounded-[0.95rem] border px-3 py-3 ${turn.speaker === 'builder' ? 'border-primary/25 bg-primary/[0.08]' : 'border-border/60 bg-background/70'}`}><div className="flex items-center justify-between gap-2 text-[10px] font-black uppercase tracking-[0.14em] text-primary"><span>{turn.speaker_label || turn.speaker}</span><span className="text-muted-foreground">{formatTime(turn.ts, language)}</span></div><p className="mt-2 text-sm leading-6 text-foreground">{turn.message}</p></div>) : <div className="rounded-[0.95rem] border border-dashed border-border/60 bg-background/50 px-3 py-4 text-sm text-muted-foreground">{language === 'es' ? 'Sin dialogo todavia.' : 'No dialogue yet.'}</div>}
                    </div>
                  </div>
                </div>

                <div className="rounded-[1.2rem] border border-border/60 bg-muted/15 p-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary"><BrainCircuit size={14} /><span>{language === 'es' ? 'Libreta por fases' : 'Notebook by phase'}</span></div>
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {latestNotebook.length > 0 ? latestNotebook.map((entry) => <div key={entry.id} className="rounded-[0.95rem] border border-border/50 bg-background/70 p-3"><div className="flex items-center justify-between gap-2"><p className="text-sm font-black">{entry.title}</p><span className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">{entry.phase}</span></div><p className="mt-2 text-sm leading-6 text-muted-foreground">{entry.content}</p></div>) : <div className="rounded-[0.95rem] border border-dashed border-border/60 bg-background/50 px-3 py-4 text-sm text-muted-foreground md:col-span-2">{language === 'es' ? 'La libreta se llenara por fases.' : 'The notebook will fill phase by phase.'}</div>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-[1.2rem] border border-dashed border-border/60 bg-muted/10 px-4 py-10 text-sm text-muted-foreground">{language === 'es' ? 'No hay run activo ahora mismo.' : 'There is no active run right now.'}</div>
            )}
          </Panel>
          <Panel title={language === 'es' ? 'Campana' : 'Campaign'} eyebrow="24/7">
            <div className="space-y-4">
              <div className="rounded-[1.2rem] border border-border/60 bg-muted/15 p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Objetivo' : 'Objective'}</p>
                <p className="mt-3 text-sm leading-7 text-foreground">{campaign?.objective || (language === 'es' ? 'Sin campana activa.' : 'No active campaign.')}</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {[
                  { label: 'Runs', value: String(campaign?.run_count || 0) },
                  { label: language === 'es' ? 'Completados' : 'Completed', value: String(campaign?.completed_count || 0) },
                  { label: 'Rollback', value: String(campaign?.rolled_back_count || 0) },
                  { label: language === 'es' ? 'Degradados' : 'Degraded', value: String(campaign?.degraded_count || 0) },
                  { label: language === 'es' ? 'Racha OK' : 'Success streak', value: String(campaign?.success_streak || 0) },
                  { label: 'Throughput/h', value: String(campaign?.throughput_per_hour || 0) },
                ].map((item) => <div key={item.label} className="rounded-[1rem] border border-border/50 bg-background/70 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{item.label}</p><p className="mt-3 text-2xl font-black tracking-tight">{item.value}</p></div>)}
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-[1rem] border border-border/50 bg-muted/15 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Ultimo apply' : 'Last apply'}</p><p className="mt-3 text-sm break-words">{String((campaign?.last_apply as { decision?: string } | null)?.decision || '-')}</p></div>
                <div className="rounded-[1rem] border border-border/50 bg-muted/15 p-4"><p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Ultimo rollback' : 'Last rollback'}</p><p className="mt-3 text-sm break-words">{String((campaign?.last_rollback as { reason?: string } | null)?.reason || '-')}</p></div>
              </div>
              <div className="rounded-[1rem] border border-border/50 bg-muted/15 p-4">
                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === 'es' ? 'Contexto operativo' : 'Operational context'}</p>
                <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                  <p>{language === 'es' ? 'Siguiente run' : 'Next run'}: {nextRunAt ? formatTime(nextRunAt, language) : '-'}</p>
                  <p>{language === 'es' ? 'Bloqueados' : 'Blocked'}: {blockedRuns.length}</p>
                  <p>{language === 'es' ? 'Sesiones con mensajes' : 'Sessions with messages'}: {sessionsWithMessages}</p>
                  <p>{language === 'es' ? 'Autonomia' : 'Autonomy'}: {autonomy?.state || '-'}</p>
                </div>
              </div>
            </div>
          </Panel>
          <Panel title={language === 'es' ? 'Pipeline' : 'Pipeline'} eyebrow={language === 'es' ? 'Cola visible' : 'Visible queue'}>
            <div className="space-y-4">
              {nextRunAt && <div className="rounded-[1.2rem] border border-primary/25 bg-primary/[0.08] p-4"><div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.16em] text-primary"><Clock3 size={14} /><span>{language === 'es' ? 'Siguiente lanzamiento' : 'Next launch'}</span></div><p className="mt-3 text-base font-black">{formatTime(nextRunAt, language)}</p><p className="mt-1 text-sm text-muted-foreground">{trainingStream?.scheduled_followup_reason || autonomy?.scheduled_followup_reason || '-'}</p></div>}
              {pipelineRuns.length > 0 ? pipelineRuns.map((run) => <button key={run.run_id} type="button" onClick={() => { void openTrainingReview(run); }} className="w-full rounded-[1.15rem] border border-border/60 bg-muted/15 p-4 text-left transition-all hover:border-primary/20 hover:bg-background"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-sm font-black">{run.display_name || run.run_id}</p><p className="mt-2 text-sm leading-6 text-muted-foreground">{run.blocked_reason || run.display_description || run.objective || '-'}</p></div><div className="flex flex-wrap gap-2"><span className={`rounded-full border px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] ${chipTone(run.lifecycle_state || run.status)}`}>{run.lifecycle_state || run.status}</span>{run.retry_in_s ? <span className="rounded-full border border-border/60 bg-background px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80">{Math.round(run.retry_in_s)}s</span> : null}</div></div><div className="mt-4 h-2 overflow-hidden rounded-full bg-muted/30"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.max(3, normalizeProgress(run.pipeline_progress_pct ?? run.progress_pct))}%` }} /></div></button>) : <div className="rounded-[1.15rem] border border-dashed border-border/60 bg-muted/10 px-4 py-10 text-sm text-muted-foreground">{language === 'es' ? 'No hay pipeline visible ahora mismo.' : 'There is no visible pipeline right now.'}</div>}
            </div>
          </Panel>
          <Panel title={language === 'es' ? 'Biblioteca' : 'Library'} eyebrow={language === 'es' ? 'Historial navegable' : 'Browsable history'} className="xl:row-span-2">
            <div className="space-y-4">
              <div className="grid gap-3 lg:grid-cols-[1.2fr_0.7fr_0.7fr_0.7fr]">
                <label className="relative">
                  <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={language === 'es' ? 'Buscar run, foco o motivo' : 'Search run, focus, or reason'} className="w-full rounded-[1rem] border border-border/60 bg-background py-3 pl-10 pr-3 text-sm outline-none transition-all focus:border-primary/30" />
                </label>
                <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} className="rounded-[1rem] border border-border/60 bg-background px-3 py-3 text-sm outline-none focus:border-primary/30">
                  <option value="all">{language === 'es' ? 'Todos los estados' : 'All states'}</option>
                  {['planned', 'blocked', 'training', 'evaluating', 'applying', 'verifying', 'completed', 'rolled_back', 'degraded'].map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
                <select value={resultFilter} onChange={(event) => setResultFilter(event.target.value)} className="rounded-[1rem] border border-border/60 bg-background px-3 py-3 text-sm outline-none focus:border-primary/30">
                  <option value="all">{language === 'es' ? 'Todos los resultados' : 'All results'}</option>
                  <option value="terminal">{language === 'es' ? 'Terminales' : 'Terminal'}</option>
                  <option value="completed">completed</option>
                  <option value="rolled_back">rolled_back</option>
                  <option value="degraded">degraded</option>
                </select>
                <select value={campaignFilter} onChange={(event) => setCampaignFilter(event.target.value)} className="rounded-[1rem] border border-border/60 bg-background px-3 py-3 text-sm outline-none focus:border-primary/30">
                  <option value="all">{language === 'es' ? 'Todas las campanas' : 'All campaigns'}</option>
                  {campaignOptions.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </div>

              <div className="grid max-h-[980px] gap-3 overflow-auto pr-1">
                {filteredRuns.length > 0 ? filteredRuns.map((run) => {
                  const execution = normalizeProgress(run.execution_progress_pct ?? run.progress_pct);
                  const pipeline = normalizeProgress(run.pipeline_progress_pct ?? run.progress_pct);
                  return (
                    <button key={run.run_id} type="button" onClick={() => { void openTrainingReview(run); }} className="rounded-[1.25rem] border border-border/60 bg-muted/15 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:bg-background">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <p className="text-sm font-black tracking-tight">{run.display_name || run.run_id}</p>
                          <p className="mt-2 text-sm leading-6 text-muted-foreground">{run.display_description || run.blocked_reason || run.terminal_reason || run.objective || '-'}</p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <span className={`rounded-full border px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] ${chipTone(run.lifecycle_state || run.status)}`}>{run.lifecycle_state || run.status}</span>
                          <span className="rounded-full border border-border/60 bg-background px-3 py-1 text-[10px] font-black uppercase tracking-[0.14em] text-foreground/80">{run.mode}</span>
                        </div>
                      </div>
                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        <div><div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground"><span>Execution</span><span>{execution}%</span></div><div className="h-2 overflow-hidden rounded-full bg-muted/30"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.max(3, execution)}%` }} /></div></div>
                        <div><div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground"><span>Pipeline</span><span>{pipeline}%</span></div><div className="h-2 overflow-hidden rounded-full bg-muted/30"><div className="h-full rounded-full bg-sky-400" style={{ width: `${Math.max(3, pipeline)}%` }} /></div></div>
                      </div>
                      <div className="mt-4 grid gap-2 text-xs text-muted-foreground md:grid-cols-4">
                        <p>{language === 'es' ? 'Fase' : 'Phase'}: {run.stage || '-'}</p>
                        <p>{language === 'es' ? 'Campana' : 'Campaign'}: {run.campaign_id || '-'}</p>
                        <p>{language === 'es' ? 'Intento' : 'Attempt'}: {run.attempt || 1}</p>
                        <p>{language === 'es' ? 'Actualizado' : 'Updated'}: {formatTime(run.updated_at, language)}</p>
                      </div>
                    </button>
                  );
                }) : <div className="rounded-[1.25rem] border border-dashed border-border/60 bg-muted/10 px-4 py-10 text-sm text-muted-foreground">{language === 'es' ? 'No hay runs que coincidan con los filtros.' : 'No runs match the filters.'}</div>}
              </div>
            </div>
          </Panel>
        </div>
      </motion.div>
      <TrainingReviewModal run={selectedRun} loading={reviewLoading} language={language} onClose={() => { setSelectedRunId(null); setSelectedRun(null); setReviewLoading(false); }} />
    </>
  );
};

export default TrainingView;
