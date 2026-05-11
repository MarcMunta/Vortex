import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Bot } from 'lucide-react';
import type { Message, AgentRun, AgentEvent, Language, FontSize } from '../../types';
import AgentTimeline from './AgentTimeline';
import {
  createParseState,
  createAgentRun,
  parseAgentEvents,
  finalizeAgentRun,
} from '../../services/agentEventParser';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface AgentMessageWrapperProps {
  message: Message;
  isStreaming: boolean;
  language: Language;
  fontSize: FontSize;
}

const AgentMessageWrapper: React.FC<AgentMessageWrapperProps> = ({
  message,
  isStreaming,
  language,
  fontSize,
}) => {
  const parseStateRef = useRef(createParseState());
  const [agentRun, setAgentRun] = useState<AgentRun>(() => createAgentRun(message.id));
  const prevContentRef = useRef('');
  const prevThoughtRef = useRef('');

  const rebuildRunFromEvents = useCallback((events: AgentEvent[]): AgentRun => {
    const filesChanged = new Set<string>();
    const commandsRun = new Set<string>();
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let status: AgentRun['status'] = isStreaming ? 'running' : 'completed';
    let completedAt: number | undefined;

    for (const event of events) {
      if (event.type === 'file_change' || event.type === 'file_write') filesChanged.add(event.path);
      if (event.type === 'command') commandsRun.add(event.command);
      if (event.type === 'token_usage') {
        totalInputTokens += event.input;
        totalOutputTokens += event.output;
      }
      if (event.type === 'status') status = event.value;
      if (event.type === 'done') completedAt = event.ts;
    }

    return {
      id: `run-${message.id}`,
      messageId: message.id,
      status,
      events,
      startedAt: events[0]?.ts || Date.now(),
      completedAt,
      totalInputTokens,
      totalOutputTokens,
      stepCount: events.filter(e => e.type === 'step').length,
      toolCallCount: events.filter(e => e.type === 'tool_call').length,
      filesChanged: Array.from(filesChanged),
      commandsRun: Array.from(commandsRun),
      summary: message.content?.slice(0, 200),
    };
  }, [isStreaming, message.content, message.id]);

  useEffect(() => {
    if (!message.agentEvents?.length) return;
    setAgentRun(rebuildRunFromEvents(message.agentEvents));
  }, [message.agentEvents, rebuildRunFromEvents]);

  // Parse agent events from the streaming message content
  useEffect(() => {
    if (message.agentEvents?.length) return;
    const content = message.content || '';
    const thought = message.thought || '';

    // Only parse if content or thought actually changed
    if (content === prevContentRef.current && thought === prevThoughtRef.current) return;
    prevContentRef.current = content;
    prevThoughtRef.current = thought;

    const newEvents = parseAgentEvents(
      content,
      thought,
      parseStateRef.current,
      message.fileChanges,
    );

    if (newEvents.length > 0) {
      setAgentRun(prev => ({
        ...prev,
        events: [...prev.events, ...newEvents],
        stepCount: prev.events.concat(newEvents).filter(e => e.type === 'step').length,
        toolCallCount: prev.events.concat(newEvents).filter(e => e.type === 'tool_call').length,
        filesChanged: Array.from(new Set([
          ...prev.filesChanged,
          ...newEvents.filter(e => e.type === 'file_change' || e.type === 'file_write').map(e => (e as { path: string }).path),
        ])),
        commandsRun: Array.from(new Set([
          ...prev.commandsRun,
          ...newEvents.filter(e => e.type === 'command').map(e => (e as { command: string }).command),
        ])),
      }));
    }
  }, [message.agentEvents, message.content, message.thought, message.fileChanges]);

  // Finalize run when streaming completes
  useEffect(() => {
    if (message.agentEvents?.length) return;
    if (!isStreaming && agentRun.status === 'running') {
      const status = message.finishReason === 'error' ? 'failed' as const : 'completed' as const;
      setAgentRun(prev => finalizeAgentRun(prev, status, message.content?.slice(0, 200)));
    }
  }, [isStreaming, agentRun.status, message.finishReason, message.content, message.agentEvents]);

  const fontSizeClass = { small: 'text-[11px]', medium: 'text-[14px]', large: 'text-[16px]' }[fontSize];

  // Check if we have any agent events to show in timeline
  const hasTimelineEvents = agentRun.events.length > 0;

  // Extract the "final answer" text (content after agent processing)
  const finalAnswerText = useMemo(() => {
    if (isStreaming) return '';
    const content = message.content || '';
    let cleaned = content
      .replace(/^Preparando agente local\.[\s\S]*?(?=(Agente iniciado\.|No se pudo|```|$))/i, '')
      .replace(/^Preparing local agent\.[\s\S]*?(?=(Agent started\.|Agent run could not|```|$))/i, '')
      .replace(/^Agente iniciado\. Analizando el repo\.\s*/i, '')
      .replace(/^Agent started\. Analyzing the repo\.\s*/i, '')
      .trim();
    if (hasTimelineEvents || (message.fileChanges?.length || 0) > 0) {
      cleaned = cleaned
        .replace(/```file:[^\n]*\n[\s\S]*?```/gi, '')
        .replace(/```(?:diff|patch)\n[\s\S]*?```/gi, '')
        .replace(/^\s*\.\.\s*/g, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
    }
    return cleaned;
  }, [hasTimelineEvents, isStreaming, message.content, message.fileChanges?.length]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className="group flex w-full justify-start mb-3.5 accelerated"
    >
      <div className="flex max-w-[96%] md:max-w-[980px] xl:max-w-[1120px] flex-row items-start gap-3">
        {/* Avatar */}
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary/25 bg-primary/10 text-primary shadow-sm transition-all duration-500 group-hover:scale-105">
          <Bot size={14} />
        </div>

        {/* Content */}
        <div className="flex flex-col items-start min-w-0 flex-1">
          {/* Agent Timeline */}
          {hasTimelineEvents && (
            <div className="w-full mb-3">
              <AgentTimeline
                run={agentRun}
                isStreaming={isStreaming}
                language={language}
              />
            </div>
          )}

          {/* Final Answer (markdown rendered) */}
          {!isStreaming && finalAnswerText && (
            <div className={`${fontSizeClass} leading-[1.45] max-w-full markdown-content ${hasTimelineEvents ? 'mt-3 rounded-xl border border-border/50 bg-background/35 px-4 py-3' : ''}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {finalAnswerText}
              </ReactMarkdown>
            </div>
          )}

          {/* Streaming text when no events parsed yet */}
          {isStreaming && !hasTimelineEvents && message.content && (
            <div className={`${fontSizeClass} leading-[1.45] max-w-full text-foreground/80`}>
              {message.content}
              <span className="typing-cursor" />
            </div>
          )}

          {/* Timestamp */}
          <div className="mt-1.5 px-1 opacity-0 group-hover:opacity-100 transition-all duration-300">
            <span className="text-[9px] font-black uppercase tracking-[0.18em] text-muted-foreground/30">
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default AgentMessageWrapper;
