import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, Clock, Zap, FileCode, Terminal } from 'lucide-react';
import type { AgentEvent, AgentRun, Language } from '../../types';
import StatusBadge from './StatusBadge';
import AgentStepCard from './AgentStepCard';
import ThoughtBlock from './ThoughtBlock';
import TerminalBlock from './TerminalBlock';
import FileDiffBlock from './FileDiffBlock';

interface AgentTimelineProps {
  run: AgentRun;
  isStreaming: boolean;
  language: Language;
}

/**
 * Groups events into steps. Each step starts with a 'step' event
 * and includes all events until the next step or the end.
 */
function groupEventsIntoSteps(events: AgentEvent[]): { title: string; index: number; events: AgentEvent[] }[] {
  const steps: { title: string; index: number; events: AgentEvent[] }[] = [];
  let currentStep: { title: string; index: number; events: AgentEvent[] } | null = null;

  // Collect events that appear before any step
  const preStepEvents: AgentEvent[] = [];

  for (const event of events) {
    if (event.type === 'step') {
      if (currentStep) steps.push(currentStep);
      currentStep = { title: event.title, index: event.index, events: [] };
    } else if (event.type === 'status' || event.type === 'done') {
      // Skip status/done events from step grouping
      continue;
    } else {
      if (currentStep) {
        currentStep.events.push(event);
      } else {
        preStepEvents.push(event);
      }
    }
  }

  if (currentStep) steps.push(currentStep);

  // If there are pre-step events (like initial thoughts), create a synthetic step
  if (preStepEvents.length > 0 && steps.length === 0) {
    steps.unshift({ title: 'Processing', index: 0, events: preStepEvents });
  } else if (preStepEvents.length > 0 && steps.length > 0) {
    steps[0].events = [...preStepEvents, ...steps[0].events];
  }

  return steps;
}

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
};

const AgentTimeline: React.FC<AgentTimelineProps> = ({ run, isStreaming, language }) => {
  const steps = useMemo(() => groupEventsIntoSteps(run.events), [run.events]);

  const elapsedMs = run.completedAt
    ? run.completedAt - run.startedAt
    : isStreaming
      ? Date.now() - run.startedAt
      : 0;

  const totalTokens = run.totalInputTokens + run.totalOutputTokens;

  // Collect standalone events (thoughts before steps, orphan diffs, etc.)
  const orphanDiffs = useMemo(() => {
    const stepPaths = new Set<string>();
    for (const step of steps) {
      for (const event of step.events) {
        if (event.type === 'file_change') stepPaths.add(event.path);
      }
    }
    return run.events
      .filter((e): e is Extract<AgentEvent, { type: 'file_change' }> =>
        e.type === 'file_change' && !stepPaths.has(e.path)
      );
  }, [run.events, steps]);

  return (
    <div className="w-full">
      {/* Run Header */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-4"
      >
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-primary/15 text-primary flex items-center justify-center border border-primary/15">
            <Bot size={15} />
          </div>
          <div>
            <div className="text-[10px] font-black uppercase tracking-[0.2em] text-primary/60">
              {language === 'es' ? 'Ejecución del Agente' : 'Agent Run'}
            </div>
          </div>
        </div>
        <StatusBadge status={run.status} language={language} elapsedMs={elapsedMs} />
      </motion.div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical timeline line */}
        <div className="absolute left-[27px] top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary/20 via-primary/10 to-transparent" />

        {/* Steps */}
        <div className="space-y-2 relative z-10">
          <AnimatePresence mode="popLayout">
            {steps.map((step, i) => (
              <AgentStepCard
                key={`step-${step.index}-${i}`}
                title={step.title}
                index={step.index}
                events={step.events}
                isActive={i === steps.length - 1 && isStreaming}
                isStreaming={isStreaming}
                language={language}
              />
            ))}
          </AnimatePresence>

          {/* Orphan diffs (file changes not associated with a step) */}
          {orphanDiffs.length > 0 && (
            <div className="pl-7 space-y-2">
              {orphanDiffs.map((d, i) => (
                <FileDiffBlock key={`orphan-${i}`} path={d.path} diff={d.diff} language={language} />
              ))}
            </div>
          )}

          {/* Streaming indicator */}
          {isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 pl-4 py-2"
            >
              <motion.div
                animate={{ scale: [1, 1.4, 1], opacity: [0.4, 1, 0.4] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
                className="w-3 h-3 rounded-full bg-primary/60 shrink-0"
              />
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary/50">
                {language === 'es' ? 'Procesando...' : 'Processing...'}
              </span>
            </motion.div>
          )}
        </div>
      </div>

      {/* Run Footer (stats) */}
      {(run.status === 'completed' || run.status === 'failed') && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 flex flex-wrap items-center gap-4 px-4 py-3 rounded-xl bg-zinc-800/30 border border-zinc-700/30"
        >
          {elapsedMs > 0 && (
            <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
              <Clock size={11} className="text-zinc-500" />
              {formatDuration(elapsedMs)}
            </div>
          )}
          {run.stepCount > 0 && (
            <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
              <Zap size={11} className="text-zinc-500" />
              {run.stepCount} {language === 'es' ? 'pasos' : 'steps'}
            </div>
          )}
          {run.filesChanged.length > 0 && (
            <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
              <FileCode size={11} className="text-zinc-500" />
              {run.filesChanged.length} {language === 'es' ? 'archivos' : 'files'}
            </div>
          )}
          {run.commandsRun.length > 0 && (
            <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
              <Terminal size={11} className="text-zinc-500" />
              {run.commandsRun.length} {language === 'es' ? 'comandos' : 'commands'}
            </div>
          )}
          {totalTokens > 0 && (
            <div className="text-[10px] font-bold text-zinc-500 ml-auto tabular-nums">
              {totalTokens.toLocaleString()} tokens
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default AgentTimeline;
