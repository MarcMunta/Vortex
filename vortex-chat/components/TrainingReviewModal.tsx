import React, { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Activity, Bot, FileCode2, FlaskConical, TerminalSquare, Workflow, X } from 'lucide-react';
import { Language, TrainingDialogueTurn, TrainingNotebookSection, TrainingReviewSection, TrainingRunSummary } from '../types';

interface TrainingReviewModalProps {
  run: TrainingRunSummary | null;
  language: Language;
  loading?: boolean;
  onClose: () => void;
}

const formatTime = (ts: number | null | undefined, language: Language): string => {
  if (!ts) return language === 'es' ? 'Sin fecha' : 'No timestamp';
  const normalized = ts > 1_000_000_000_000 ? ts : ts * 1000;
  return new Date(normalized).toLocaleString(language === 'es' ? 'es-ES' : 'en-US', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const speakerTone = (speaker: TrainingDialogueTurn['speaker']): string => {
  if (speaker === 'builder') return 'border-[#d97706]/25 bg-[#fff3d6]';
  if (speaker === 'analyst') return 'border-[#0f766e]/25 bg-[#e7f8f2]';
  return 'border-slate-300/70 bg-white';
};

const TrainingReviewModal: React.FC<TrainingReviewModalProps> = ({ run, language, loading = false, onClose }) => {
  const reviewSections = (run?.review_sections || []) as TrainingReviewSection[];
  const notebook = (run?.notebook_sections || []) as TrainingNotebookSection[];
  const dialogue = (run?.agent_dialogue || []) as TrainingDialogueTurn[];
  const liveSeries = run?.live_metrics_series || [];
  const latestMetrics = Object.entries(run?.latest_metrics || {});
  const artifacts = Object.entries(run?.artifacts || {});
  const gateEntries = Object.entries(run?.gate_results || {});
  const applyResult = run?.apply_result || null;
  const rollbackResult = run?.rollback_result || null;
  const comparison = run?.comparison || null;
  const events = [...(run?.events || [])].reverse();
  const logs = Object.entries(run?.logs || {});
  const rawJson = useMemo(() => (run ? JSON.stringify(run, null, 2) : ''), [run]);

  return (
    <AnimatePresence>
      {run && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[120] overflow-hidden bg-black/70 backdrop-blur-md"
          onClick={onClose}
        >
          <div className="flex h-full w-full items-center justify-center overflow-hidden p-4 lg:p-8">
            <motion.div
              initial={{ y: 28, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 28, opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={(event) => event.stopPropagation()}
              className="flex h-full max-h-[92vh] w-full max-w-[1120px] flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-[#0b1118] shadow-[0_30px_120px_rgba(0,0,0,0.55)]"
            >
              <div className="flex items-center justify-between border-b border-white/10 px-5 py-4 text-white">
                <div>
                  <p className="text-[10px] font-black uppercase tracking-[0.18em] text-sky-300">
                    {language === 'es' ? 'Informe completo' : 'Full review'}
                  </p>
                  <h3 className="mt-2 text-xl font-black tracking-tight">
                    {run.display_name || run.run_id}
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white transition hover:bg-white/10"
                  aria-label={language === 'es' ? 'Cerrar informe' : 'Close review'}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="flex-1 overflow-hidden px-4 py-4 lg:px-6 lg:py-6">
                <article className="mx-auto flex h-full w-full max-w-[920px] flex-col overflow-auto rounded-[2rem] border border-[#d5cdbd] bg-[#f6efe3] text-[#171717] shadow-[0_28px_70px_rgba(0,0,0,0.18)]">
                  <div className="border-b border-[#d7cfbf] px-6 py-6 lg:px-10 lg:py-8">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="max-w-2xl">
                        <p className="text-[11px] font-black uppercase tracking-[0.24em] text-[#6c5d48]">
                          {language === 'es' ? 'Entrenamiento descriptivo' : 'Descriptive training'}
                        </p>
                        <h4 className="mt-3 text-3xl font-black tracking-[-0.04em] text-[#111827]">
                          {run.display_name || run.run_id}
                        </h4>
                        <p className="mt-3 max-w-2xl text-sm leading-7 text-[#50473a]">
                          {run.display_description || run.objective || (language === 'es' ? 'Sin descripcion disponible.' : 'No description available.')}
                        </p>
                      </div>
                      <div className="min-w-[220px] rounded-[1.3rem] border border-[#d6ccb8] bg-white/70 p-4">
                        <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          {language === 'es' ? 'Estado del ciclo' : 'Cycle state'}
                        </p>
                        <p className="mt-3 text-lg font-black tracking-tight text-[#111827]">{run.status}</p>
                        <p className="mt-1 text-xs text-[#61584b]">{run.stage || run.mode}</p>
                        <p className="mt-4 text-xs text-[#61584b]">
                          {language === 'es' ? 'Creado' : 'Created'}: {formatTime(run.created_at, language)}
                        </p>
                        <p className="mt-1 text-xs text-[#61584b]">
                          {language === 'es' ? 'Actualizado' : 'Updated'}: {formatTime(run.updated_at, language)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-4 border-b border-[#d7cfbf] px-6 py-6 lg:grid-cols-4 lg:px-10">
                    <div className="rounded-[1.2rem] border border-[#d6ccb8] bg-white/65 p-4">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        {language === 'es' ? 'Modo' : 'Mode'}
                      </p>
                      <p className="mt-3 text-base font-black text-[#111827]">{run.mode}</p>
                    </div>
                    <div className="rounded-[1.2rem] border border-[#d6ccb8] bg-white/65 p-4">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        {language === 'es' ? 'Foco' : 'Focus'}
                      </p>
                      <p className="mt-3 text-sm font-semibold leading-6 text-[#111827]">
                        {(run.learning_focus || []).join(', ') || (language === 'es' ? 'Sin foco declarado' : 'No declared focus')}
                      </p>
                    </div>
                    <div className="rounded-[1.2rem] border border-[#d6ccb8] bg-white/65 p-4">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        {language === 'es' ? 'Metricas visibles' : 'Visible metrics'}
                      </p>
                      <p className="mt-3 text-base font-black text-[#111827]">{liveSeries.length || latestMetrics.length}</p>
                    </div>
                    <div className="rounded-[1.2rem] border border-[#d6ccb8] bg-white/65 p-4">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        {language === 'es' ? 'Decision final' : 'Final decision'}
                      </p>
                      <p className="mt-3 text-base font-black text-[#111827]">{run.lifecycle_state || run.status}</p>
                    </div>
                  </div>

                  <div className="space-y-6 px-6 py-6 lg:px-10 lg:py-8">
                    <section>
                      <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        <FlaskConical size={14} />
                        <span>{language === 'es' ? 'Resumen ejecutivo' : 'Executive summary'}</span>
                      </div>
                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        {reviewSections.length > 0 ? reviewSections.map((section) => (
                          <div key={section.key} className="rounded-[1.25rem] border border-[#d6ccb8] bg-white/70 p-4">
                            <p className="text-sm font-black tracking-tight text-[#111827]">{section.title}</p>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[#4b5563]">{section.content}</p>
                          </div>
                        )) : (
                          <div className="rounded-[1.25rem] border border-dashed border-[#cdbfa6] bg-white/60 p-5 text-sm text-[#61584b] md:col-span-2">
                            {language === 'es' ? 'Aun no hay secciones descriptivas disponibles para este run.' : 'No descriptive sections are available for this run yet.'}
                          </div>
                        )}
                      </div>
                    </section>

                    <section>
                      <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        <Activity size={14} />
                        <span>{language === 'es' ? 'Libreta por fases' : 'Notebook by phase'}</span>
                      </div>
                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        {notebook.length > 0 ? notebook.map((entry) => (
                          <div key={entry.id} className="rounded-[1.25rem] border border-[#d6ccb8] bg-white/70 p-4">
                            <div className="flex items-center justify-between gap-3">
                              <p className="text-sm font-black tracking-tight text-[#111827]">{entry.title}</p>
                              <span className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">{entry.phase}</span>
                            </div>
                            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[#4b5563]">{entry.content}</p>
                          </div>
                        )) : (
                          <div className="rounded-[1.25rem] border border-dashed border-[#cdbfa6] bg-white/60 p-5 text-sm text-[#61584b] md:col-span-2">
                            {language === 'es' ? 'Este run aun no tiene libreta de aprendizaje.' : 'This run does not have a learning notebook yet.'}
                          </div>
                        )}
                      </div>
                    </section>

                    <section>
                      <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                        <Bot size={14} />
                        <span>{language === 'es' ? 'Dialogo multiagente' : 'Multi-agent dialogue'}</span>
                      </div>
                      <div className="mt-4 space-y-3">
                        {dialogue.length > 0 ? dialogue.map((turn) => (
                          <div key={turn.id} className={`rounded-[1.25rem] border p-4 ${speakerTone(turn.speaker)}`}>
                            <div className="flex items-center justify-between gap-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.18em] text-[#6b7280]">
                                {turn.speaker_label || turn.speaker}
                              </p>
                              <span className="text-[11px] text-[#6b7280]">{formatTime(turn.ts, language)}</span>
                            </div>
                            <p className="mt-3 text-sm leading-7 text-[#111827]">{turn.message}</p>
                          </div>
                        )) : (
                          <div className="rounded-[1.25rem] border border-dashed border-[#cdbfa6] bg-white/60 p-5 text-sm text-[#61584b]">
                            {language === 'es' ? 'No hay dialogo guardado para este entrenamiento.' : 'There is no stored dialogue for this training run.'}
                          </div>
                        )}
                      </div>
                    </section>

                    <section className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <Activity size={14} />
                          <span>{language === 'es' ? 'Metricas y senales' : 'Metrics and signals'}</span>
                        </div>
                        <div className="mt-4 grid gap-3 sm:grid-cols-2">
                          {latestMetrics.length > 0 ? latestMetrics.map(([label, value]) => (
                            <div key={label} className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">{label}</p>
                              <p className="mt-2 break-words text-sm font-semibold text-[#111827]">{String(value)}</p>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b] sm:col-span-2">
                              {language === 'es' ? 'Sin metricas en vivo para este run.' : 'No live metrics for this run.'}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <FileCode2 size={14} />
                          <span>{language === 'es' ? 'Artefactos del entrenamiento' : 'Training artifacts'}</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {artifacts.length > 0 ? artifacts.map(([label, value]) => (
                            <div key={label} className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">{label}</p>
                              <p className="mt-2 break-all text-sm text-[#4b5563]">{value}</p>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b]">
                              {language === 'es' ? 'Este run aun no ha publicado artefactos.' : 'This run has not published artifacts yet.'}
                            </div>
                          )}
                        </div>
                      </div>
                    </section>

                    <section className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <Workflow size={14} />
                          <span>{language === 'es' ? 'Gates y apply' : 'Gates and apply'}</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {gateEntries.length > 0 ? gateEntries.map(([label, value]) => (
                            <div key={label} className="flex items-center justify-between rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] px-3 py-3 text-sm">
                              <span className="text-[#61584b]">{label}</span>
                              <span className="font-semibold text-[#111827]">{String(value)}</span>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b]">
                              {language === 'es' ? 'No hay gates registrados.' : 'No gates recorded.'}
                            </div>
                          )}
                          <div className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3 text-sm text-[#374151]">
                            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">apply_result</p>
                            <p className="mt-2 break-words">{applyResult ? JSON.stringify(applyResult) : '-'}</p>
                          </div>
                          <div className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3 text-sm text-[#374151]">
                            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">rollback_result</p>
                            <p className="mt-2 break-words">{rollbackResult ? JSON.stringify(rollbackResult) : '-'}</p>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <FileCode2 size={14} />
                          <span>{language === 'es' ? 'Comparacion con run padre' : 'Parent-run comparison'}</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {comparison ? Object.entries(comparison).map(([label, value]) => (
                            <div key={label} className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">{label}</p>
                              <p className="mt-2 break-words text-sm text-[#374151]">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</p>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b]">
                              {language === 'es' ? 'No hay comparacion disponible para este ciclo.' : 'No comparison is available for this cycle.'}
                            </div>
                          )}
                        </div>
                      </div>
                    </section>

                    <section className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <Workflow size={14} />
                          <span>{language === 'es' ? 'Eventos del ciclo' : 'Cycle events'}</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {events.length > 0 ? events.map((event) => (
                            <div key={event.id} className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3">
                              <div className="flex items-center justify-between gap-3">
                                <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">
                                  {event.phase} / {event.kind}
                                </p>
                                <span className="text-[11px] text-[#6b7280]">{formatTime(event.ts, language)}</span>
                              </div>
                              <p className="mt-2 text-sm leading-6 text-[#111827]">{event.message}</p>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b]">
                              {language === 'es' ? 'Sin eventos registrados.' : 'No recorded events.'}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                        <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                          <TerminalSquare size={14} />
                          <span>{language === 'es' ? 'Logs del entrenamiento' : 'Training logs'}</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {logs.length > 0 ? logs.map(([label, lines]) => (
                            <div key={label} className="rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-3">
                              <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">{label}</p>
                              <pre className="mt-3 whitespace-pre-wrap break-words text-[11px] leading-6 text-[#374151]">
                                {Array.isArray(lines) && lines.length > 0
                                  ? lines.join('\n')
                                  : (language === 'es' ? 'Sin salida.' : 'No output.')}
                              </pre>
                            </div>
                          )) : (
                            <div className="rounded-[1rem] border border-dashed border-[#d6ccb8] bg-[#fffdfa] p-4 text-sm text-[#61584b]">
                              {language === 'es' ? 'No hay logs visibles para este run.' : 'No visible logs for this run.'}
                            </div>
                          )}
                        </div>
                      </div>
                    </section>

                    <section className="rounded-[1.35rem] border border-[#d6ccb8] bg-white/70 p-5">
                      <details>
                        <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] font-black uppercase tracking-[0.18em] text-[#8b6f47]">
                              {language === 'es' ? 'Estado bruto completo' : 'Complete raw state'}
                            </p>
                            <p className="mt-2 text-sm text-[#61584b]">
                              {language === 'es'
                                ? 'Abre este bloque solo si necesitas inspeccionar el payload entero del backend.'
                                : 'Open this only when you need to inspect the full backend payload.'}
                            </p>
                          </div>
                          {loading && (
                            <span className="rounded-full border border-[#d6ccb8] bg-[#fffdfa] px-3 py-1 text-[10px] font-black uppercase tracking-[0.16em] text-[#8b6f47]">
                              {language === 'es' ? 'Sincronizando' : 'Syncing'}
                            </span>
                          )}
                        </summary>
                        <pre className="mt-4 whitespace-pre-wrap break-words rounded-[1rem] border border-[#ddd2bf] bg-[#fffdfa] p-4 text-[11px] leading-6 text-[#1f2937]">
                          {rawJson || '{}'}
                        </pre>
                      </details>
                    </section>
                  </div>
                </article>
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default TrainingReviewModal;
