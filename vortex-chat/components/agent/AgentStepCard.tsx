import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight, Search, FileText, Terminal as TerminalIcon, Wrench, GitBranch, Eye } from 'lucide-react';
import type { AgentEvent, Language } from '../../types';
import ThoughtBlock from './ThoughtBlock';
import TerminalBlock from './TerminalBlock';
import FileDiffBlock from './FileDiffBlock';
import ToolCallBlock from './ToolCallBlock';

interface AgentStepCardProps {
  title: string;
  index: number;
  events: AgentEvent[];
  isActive: boolean;
  isStreaming: boolean;
  language: Language;
}

const stepIcon = (title: string): React.ReactNode => {
  const t = title.toLowerCase();
  if (t.includes('search') || t.includes('busca')) return <Search size={14} />;
  if (t.includes('read') || t.includes('lee') || t.includes('inspect') || t.includes('revis'))
    return <Eye size={14} />;
  if (t.includes('modif') || t.includes('edit') || t.includes('write') || t.includes('escrib'))
    return <GitBranch size={14} />;
  if (t.includes('command') || t.includes('comand') || t.includes('execut') || t.includes('ejecut') || t.includes('test') || t.includes('run') || t.includes('corre'))
    return <TerminalIcon size={14} />;
  if (t.includes('tool') || t.includes('herramienta'))
    return <Wrench size={14} />;
  return <FileText size={14} />;
};

const AgentStepCard: React.FC<AgentStepCardProps> = ({
  title,
  index,
  events,
  isActive,
  isStreaming,
  language,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const groupedContent = useMemo(() => {
    const thoughts: string[] = [];
    const commands: { command: string; output: string; streaming: boolean }[] = [];
    const diffs: { path: string; diff: string }[] = [];
    const toolCalls: { tool: string; args: Record<string, unknown>; result?: { ok: boolean; output: string } }[] = [];
    const fileReads: string[] = [];
    const fileWrites: string[] = [];
    const errors: string[] = [];

    let currentCommand: { command: string; output: string; streaming: boolean } | null = null;
    let currentToolCall: { tool: string; args: Record<string, unknown>; result?: { ok: boolean; output: string } } | null = null;

    for (const event of events) {
      switch (event.type) {
        case 'thought':
          thoughts.push(event.text);
          break;
        case 'command':
          if (currentCommand) commands.push(currentCommand);
          currentCommand = { command: event.command, output: '', streaming: false };
          break;
        case 'stdout':
          if (currentCommand) {
            currentCommand.output += (currentCommand.output ? '\n' : '') + event.chunk;
            currentCommand.streaming = isActive && isStreaming;
          }
          break;
        case 'stderr':
          if (currentCommand) {
            currentCommand.output += (currentCommand.output ? '\n' : '') + event.chunk;
            currentCommand.streaming = isActive && isStreaming;
          }
          break;
        case 'file_change':
          diffs.push({ path: event.path, diff: event.diff });
          break;
        case 'file_read':
          fileReads.push(event.path);
          break;
        case 'file_write':
          fileWrites.push(event.path);
          break;
        case 'tool_call':
          if (currentToolCall) toolCalls.push(currentToolCall);
          currentToolCall = { tool: event.tool, args: event.args };
          break;
        case 'tool_result':
          if (currentToolCall && currentToolCall.tool === event.tool) {
            currentToolCall.result = { ok: event.ok, output: event.output };
            toolCalls.push(currentToolCall);
            currentToolCall = null;
          }
          break;
        case 'error':
          errors.push(event.message);
          break;
      }
    }

    if (currentCommand) commands.push(currentCommand);
    if (currentToolCall) toolCalls.push(currentToolCall);

    return { thoughts, commands, diffs, toolCalls, fileReads, fileWrites, errors };
  }, [events, isActive, isStreaming]);

  const hasContent =
    groupedContent.thoughts.length > 0 ||
    groupedContent.commands.length > 0 ||
    groupedContent.diffs.length > 0 ||
    groupedContent.toolCalls.length > 0 ||
    groupedContent.fileReads.length > 0 ||
    groupedContent.fileWrites.length > 0 ||
    groupedContent.errors.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
      className="relative"
    >
      {/* Step Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-left group ${
          isActive
            ? 'bg-primary/8 border border-primary/20 shadow-[0_4px_16px_-8px_hsl(var(--primary)/0.3)]'
            : 'bg-zinc-800/30 border border-zinc-700/30 hover:bg-zinc-800/50 hover:border-zinc-700/50'
        }`}
      >
        {/* Timeline dot */}
        <div className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 transition-all ${
          isActive
            ? 'bg-primary/20 text-primary'
            : 'bg-zinc-700/40 text-zinc-500 group-hover:text-zinc-400'
        }`}>
          {isActive && isStreaming ? (
            <motion.div
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-2 h-2 rounded-full bg-primary"
            />
          ) : (
            stepIcon(title)
          )}
        </div>

        {/* Title */}
        <div className="flex-1 min-w-0">
          <span className={`text-[12px] font-bold ${
            isActive ? 'text-primary' : 'text-foreground/70'
          }`}>
            {title}
          </span>
          {/* Inline summary of files */}
          {groupedContent.fileReads.length > 0 && !isExpanded && (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {groupedContent.fileReads.slice(0, 3).map((path, i) => (
                <span key={i} className="text-[9px] font-mono text-zinc-500 bg-zinc-800/60 rounded px-1.5 py-0.5">
                  {path.split('/').pop()}
                </span>
              ))}
              {groupedContent.fileReads.length > 3 && (
                <span className="text-[9px] font-mono text-zinc-600">+{groupedContent.fileReads.length - 3}</span>
              )}
            </div>
          )}
        </div>

        {/* Toggle */}
        {hasContent && (
          <div className={`shrink-0 transition-colors ${isActive ? 'text-primary/50' : 'text-zinc-600'}`}>
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>
        )}
      </button>

      {/* Step Content */}
      <AnimatePresence>
        {isExpanded && hasContent && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="overflow-hidden"
          >
            <div className="pl-7 pr-2 pt-2 pb-1 space-y-2.5">
              {/* Thoughts */}
              {groupedContent.thoughts.length > 0 && (
                <ThoughtBlock
                  text={groupedContent.thoughts.join(' ')}
                  isStreaming={isActive && isStreaming}
                  language={language}
                />
              )}

              {/* File reads */}
              {groupedContent.fileReads.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {groupedContent.fileReads.map((path, i) => (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="inline-flex items-center gap-1.5 text-[10px] font-mono text-zinc-400 bg-zinc-800/60 border border-zinc-700/30 rounded-lg px-2.5 py-1"
                    >
                      <Eye size={10} className="text-zinc-500" />
                      {path}
                    </motion.span>
                  ))}
                </div>
              )}

              {/* Tool calls */}
              {groupedContent.toolCalls.map((tc, i) => (
                <ToolCallBlock key={i} tool={tc.tool} args={tc.args} result={tc.result} language={language} />
              ))}

              {/* Terminal commands */}
              {groupedContent.commands.map((cmd, i) => (
                <TerminalBlock key={i} command={cmd.command} output={cmd.output} isStreaming={cmd.streaming} language={language} />
              ))}

              {/* File diffs */}
              {groupedContent.diffs.map((d, i) => (
                <FileDiffBlock key={i} path={d.path} diff={d.diff} language={language} defaultExpanded={i === 0} />
              ))}

              {/* File writes */}
              {groupedContent.fileWrites.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {groupedContent.fileWrites.map((path, i) => (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="inline-flex items-center gap-1.5 text-[10px] font-mono text-emerald-400/80 bg-emerald-500/8 border border-emerald-500/15 rounded-lg px-2.5 py-1"
                    >
                      <FileText size={10} />
                      {path}
                    </motion.span>
                  ))}
                </div>
              )}

              {/* Errors */}
              {groupedContent.errors.map((err, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="rounded-lg border border-red-500/20 bg-red-500/8 px-3.5 py-2.5 text-[11px] text-red-400 font-mono"
                >
                  {err}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default AgentStepCard;
