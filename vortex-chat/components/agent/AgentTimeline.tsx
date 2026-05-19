import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, ChevronDown, ChevronRight, Clock, FileCode, Terminal as TerminalIcon } from 'lucide-react';
import type { AgentEvent, AgentRun, Language } from '../../types';
import StatusBadge from './StatusBadge';
import FileDiffBlock from './FileDiffBlock';
import TerminalBlock from './TerminalBlock';

interface AgentTimelineProps {
  run: AgentRun;
  isStreaming: boolean;
  language: Language;
}

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
};

const collectFileChanges = (events: AgentEvent[]): { path: string; diff: string }[] => {
  const changes = new Map<string, { path: string; diff: string }>();
  const seenByPath = new Map<string, Set<string>>();
  for (const event of events) {
    if (event.type !== 'file_change' || !event.diff.trim()) continue;
    const seen = seenByPath.get(event.path) || new Set<string>();
    if (seen.has(event.diff)) continue;
    seen.add(event.diff);
    seenByPath.set(event.path, seen);
    const existing = changes.get(event.path);
    changes.set(
      event.path,
      existing ? { path: event.path, diff: `${existing.diff}\n\n${event.diff}` } : { path: event.path, diff: event.diff },
    );
  }
  return Array.from(changes.values());
};

const collectErrors = (events: AgentEvent[]): string[] => {
  const errors: string[] = [];
  const seen = new Set<string>();
  for (const event of events) {
    if (event.type !== 'error') continue;
    const message = event.message.trim();
    if (!message || seen.has(message)) continue;
    seen.add(message);
    errors.push(message);
  }
  return errors;
};

const collectCommands = (events: AgentEvent[]): { command: string; output: string; streaming: boolean }[] => {
  const commands: { command: string; output: string; streaming: boolean }[] = [];
  let current: { command: string; output: string; streaming: boolean } | null = null;
  for (const event of events) {
    if (event.type === 'command') {
      if (current) commands.push(current);
      current = { command: event.command, output: '', streaming: false };
      continue;
    }
    if ((event.type === 'stdout' || event.type === 'stderr') && current) {
      current.output += `${current.output ? '\n' : ''}${event.chunk}`;
    }
  }
  if (current) commands.push(current);
  return commands;
};

const ChangedFilesPanel: React.FC<{
  changes: { path: string; diff: string }[];
  language: Language;
}> = ({ changes, language }) => {
  const totalLines = useMemo(
    () => changes.reduce((count, change) => count + change.diff.split('\n').length, 0),
    [changes]
  );
  const startsCollapsed = changes.length > 1 || totalLines > 80;
  const [expanded, setExpanded] = useState(!startsCollapsed);
  const label = language === 'es' ? 'Archivos cambiados' : 'Changed files';
  const fileLabel = language === 'es' ? 'archivo' : 'file';
  const filesLabel = language === 'es' ? 'archivos' : 'files';
  const lineLabel = language === 'es' ? 'lineas' : 'lines';

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-700/40 bg-zinc-900/70">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between gap-3 bg-zinc-800/50 px-4 py-3 text-left transition-colors hover:bg-zinc-800/70"
      >
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-primary/15 bg-primary/15 text-primary">
            <FileCode size={14} />
          </div>
          <div className="min-w-0">
            <p className="text-[12px] font-bold text-foreground/85">{label}</p>
            <p className="mt-0.5 truncate text-[10px] font-mono text-zinc-500">
              {changes.length} {changes.length === 1 ? fileLabel : filesLabel} · {totalLines} {lineLabel}
            </p>
          </div>
        </div>
        <div className="text-zinc-500">
          {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-2.5 p-3">
              {changes.map((change, index) => (
                <FileDiffBlock
                  key={`${change.path}-${index}`}
                  path={change.path}
                  diff={change.diff}
                  language={language}
                  defaultExpanded={changes.length === 1 && change.diff.split('\n').length <= 80}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const CommandsPanel: React.FC<{
  commands: { command: string; output: string; streaming: boolean }[];
  language: Language;
}> = ({ commands, language }) => {
  const [expanded, setExpanded] = useState(true);
  const label = language === 'es' ? 'Comandos ejecutados' : 'Executed commands';
  const commandLabel = language === 'es' ? 'comando' : 'command';
  const commandsLabel = language === 'es' ? 'comandos' : 'commands';

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-700/40 bg-zinc-900/70">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between gap-3 bg-zinc-800/50 px-4 py-3 text-left transition-colors hover:bg-zinc-800/70"
      >
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-emerald-400/15 bg-emerald-400/10 text-emerald-300">
            <TerminalIcon size={14} />
          </div>
          <div className="min-w-0">
            <p className="text-[12px] font-bold text-foreground/85">{label}</p>
            <p className="mt-0.5 truncate text-[10px] font-mono text-zinc-500">
              {commands.length} {commands.length === 1 ? commandLabel : commandsLabel}
            </p>
          </div>
        </div>
        <div className="text-zinc-500">
          {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-2.5 p-3">
              {commands.map((command, index) => (
                <TerminalBlock
                  key={`${command.command}-${index}`}
                  command={command.command}
                  output={command.output}
                  isStreaming={command.streaming}
                  language={language}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const AgentTimeline: React.FC<AgentTimelineProps> = ({ run, isStreaming, language }) => {
  const fileChanges = useMemo(() => collectFileChanges(run.events), [run.events]);
  const commands = useMemo(() => collectCommands(run.events), [run.events]);
  const errors = useMemo(() => collectErrors(run.events), [run.events]);
  const elapsedMs = run.completedAt
    ? run.completedAt - run.startedAt
    : isStreaming
      ? Date.now() - run.startedAt
      : 0;

  return (
    <div className="w-full">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-primary/15 bg-primary/15 text-primary">
            <Bot size={15} />
          </div>
          <div className="text-[10px] font-black uppercase tracking-[0.2em] text-primary/60">
            {language === 'es' ? 'Ejecucion del agente' : 'Agent run'}
          </div>
        </div>
        <StatusBadge status={run.status} language={language} elapsedMs={elapsedMs} />
      </motion.div>

      <div className="space-y-3">
        {fileChanges.length > 0 && (
          <ChangedFilesPanel changes={fileChanges} language={language} />
        )}

        {commands.length > 0 && (
          <CommandsPanel commands={commands} language={language} />
        )}

        {errors.map((error, index) => (
          <div
            key={`${error}-${index}`}
            className="rounded-lg border border-red-500/20 bg-red-500/8 px-3.5 py-2.5 font-mono text-[11px] text-red-400"
          >
            {error}
          </div>
        ))}

        {isStreaming && fileChanges.length === 0 && commands.length === 0 && errors.length === 0 && (
          <div className="flex items-center gap-3 rounded-xl border border-zinc-700/30 bg-zinc-800/30 px-4 py-3">
            <motion.div
              animate={{ scale: [1, 1.35, 1], opacity: [0.45, 1, 0.45] }}
              transition={{ repeat: Infinity, duration: 1.4 }}
              className="h-3 w-3 shrink-0 rounded-full bg-primary/60"
            />
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary/50">
              {language === 'es' ? 'Procesando...' : 'Processing...'}
            </span>
          </div>
        )}

        {(run.status === 'completed' || run.status === 'failed') && (
          <div className="flex flex-wrap items-center gap-4 rounded-xl border border-zinc-700/30 bg-zinc-800/30 px-4 py-3">
            {elapsedMs > 0 && (
              <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
                <Clock size={11} className="text-zinc-500" />
                {formatDuration(elapsedMs)}
              </div>
            )}
            {fileChanges.length > 0 && (
              <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
                <FileCode size={11} className="text-zinc-500" />
                {fileChanges.length} {language === 'es' ? 'archivos' : 'files'}
              </div>
            )}
            {commands.length > 0 && (
              <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400">
                <TerminalIcon size={11} className="text-zinc-500" />
                {commands.length} {language === 'es' ? 'comandos' : 'commands'}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentTimeline;
