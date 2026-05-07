/**
 * Agent Event Parser
 *
 * Synthesizes structured AgentEvent objects from streamed text content.
 * Since the backend currently returns agent results as text, this parser
 * detects patterns in the stream to generate Codex-style timeline events.
 *
 * When the backend is upgraded to emit native agent events via SSE,
 * this parser will be replaced by direct event mapping.
 */

import type { AgentEvent, AgentRun, AgentRunStatus } from '../types';

interface ParseState {
  lastText: string;
  lastThought: string;
  stepIndex: number;
  seenPaths: Set<string>;
  seenCommands: Set<string>;
  hasStarted: boolean;
  toolCallCount: number;
}

export function createParseState(): ParseState {
  return {
    lastText: '',
    lastThought: '',
    stepIndex: 0,
    seenPaths: new Set(),
    seenCommands: new Set(),
    hasStarted: false,
    toolCallCount: 0,
  };
}

export function createAgentRun(messageId: string): AgentRun {
  return {
    id: `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    messageId,
    status: 'running',
    events: [],
    startedAt: Date.now(),
    totalInputTokens: 0,
    totalOutputTokens: 0,
    stepCount: 0,
    toolCallCount: 0,
    filesChanged: [],
    commandsRun: [],
  };
}

/**
 * Parse incremental text deltas from the agent SSE stream and
 * emit structured events for the timeline UI.
 */
export function parseAgentEvents(
  text: string,
  thought: string,
  state: ParseState,
  fileChanges?: { path: string; diff: string }[],
): AgentEvent[] {
  const events: AgentEvent[] = [];
  const now = Date.now();

  // Emit initial status event
  if (!state.hasStarted && (text || thought)) {
    state.hasStarted = true;
    events.push({ type: 'status', value: 'running', ts: now });
    events.push({ type: 'step', title: 'Analyzing request', index: 0, ts: now });
    state.stepIndex = 1;
  }

  // Parse thought/reasoning deltas
  if (thought && thought !== state.lastThought) {
    const newThought = thought.slice(state.lastThought.length).trim();
    if (newThought) {
      events.push({ type: 'thought', text: newThought, ts: now });
    }
    state.lastThought = thought;
  }

  // Parse text deltas for structured content
  if (text && text !== state.lastText) {
    const newText = text.slice(state.lastText.length);
    state.lastText = text;

    // Detect step transitions (markdown headers)
    const headerMatches = newText.matchAll(/^#{1,3}\s+(.+)$/gm);
    for (const match of headerMatches) {
      const title = match[1].trim();
      if (title && !title.startsWith('```')) {
        events.push({ type: 'step', title, index: state.stepIndex, ts: now });
        state.stepIndex++;
      }
    }

    // Detect agent status/action keywords
    const statusPatterns: [RegExp, string][] = [
      [/(?:Preparando|Preparing)\s+(?:agente|agent)/i, 'Preparing agent'],
      [/(?:Revisando|Inspecting|Reviewing)\s+(?:contexto|context|repo|project)/i, 'Inspecting workspace'],
      [/(?:Buscando|Searching|Looking)\s+(?:archivos|files|código|code)/i, 'Searching files'],
      [/(?:Leyendo|Reading)\s+(.+?\.\w+)/i, 'Reading files'],
      [/(?:Modificando|Modifying|Editing)\s+(.+?\.\w+)/i, 'Modifying files'],
      [/(?:Ejecutando|Running|Executing)\s+(?:comando|command|tests?)/i, 'Executing command'],
      [/(?:Validando|Validating|Testing)/i, 'Validating changes'],
      [/(?:Aplicando|Applying)\s+(?:cambios|changes|patch)/i, 'Applying changes'],
    ];

    for (const [pattern, defaultTitle] of statusPatterns) {
      const match = newText.match(pattern);
      if (match) {
        const title = match[1] ? `${defaultTitle.split(' ')[0]} ${match[1]}` : defaultTitle;
        // Avoid duplicate step titles
        const lastStep = events.filter(e => e.type === 'step').pop();
        if (!lastStep || (lastStep as { title: string }).title !== title) {
          events.push({ type: 'step', title, index: state.stepIndex, ts: now });
          state.stepIndex++;
        }
      }
    }

    // Detect terminal commands in text
    const cmdMatches = newText.matchAll(/(?:^|\n)\s*\$\s+(.+?)(?:\n|$)/g);
    for (const match of cmdMatches) {
      const cmd = match[1].trim();
      if (cmd && !state.seenCommands.has(cmd)) {
        state.seenCommands.add(cmd);
        events.push({ type: 'command', command: cmd, ts: now });
      }
    }

    // Detect command blocks
    const codeBlockMatches = newText.matchAll(/```(?:bash|shell|sh|terminal|cmd|powershell)\n([\s\S]*?)```/g);
    for (const match of codeBlockMatches) {
      const commands = match[1].trim().split('\n').filter(Boolean);
      for (const line of commands) {
        const cmd = line.replace(/^\$\s*/, '').trim();
        if (cmd && !state.seenCommands.has(cmd)) {
          state.seenCommands.add(cmd);
          events.push({ type: 'command', command: cmd, ts: now });
        }
      }
    }

    // Detect file references
    const fileRefMatches = newText.matchAll(/(?:Reading|Leyendo|Analyzing|Analizando|`)([\w/.\\-]+\.(?:ts|tsx|js|jsx|py|dart|css|html|json|yaml|yml|md|toml))`?/g);
    for (const match of fileRefMatches) {
      const path = match[1].trim();
      if (path && !state.seenPaths.has(path)) {
        state.seenPaths.add(path);
        events.push({ type: 'file_read', path, ts: now });
      }
    }

    // Detect stdout-like output (lines that look like terminal output)
    const outputLines = newText.split('\n').filter(line => {
      const trimmed = line.trim();
      return (
        trimmed.startsWith('added ') ||
        trimmed.startsWith('PASS') ||
        trimmed.startsWith('FAIL') ||
        trimmed.startsWith('running') ||
        trimmed.startsWith('✓') ||
        trimmed.startsWith('✗') ||
        trimmed.startsWith('Error:') ||
        trimmed.startsWith('error:') ||
        trimmed.match(/^\d+ passing/) ||
        trimmed.match(/^\d+ failing/)
      );
    });
    if (outputLines.length > 0) {
      events.push({ type: 'stdout', chunk: outputLines.join('\n'), ts: now });
    }
  }

  // Detect file changes from structured data
  if (fileChanges) {
    for (const change of fileChanges) {
      if (!state.seenPaths.has(`change:${change.path}`)) {
        state.seenPaths.add(`change:${change.path}`);
        events.push({ type: 'file_change', path: change.path, diff: change.diff, ts: now });
      }
    }
  }

  return events;
}

/**
 * Parse native agent events from SSE data (future backend support)
 */
export function parseNativeAgentEvent(data: Record<string, unknown>): AgentEvent | null {
  if (!data || typeof data !== 'object') return null;
  const agentEvent = data.agent_event as Record<string, unknown> | undefined;
  if (!agentEvent) return null;

  const type = String(agentEvent.type || '');
  const ts = Number(agentEvent.ts || Date.now());

  switch (type) {
    case 'thought':
      return { type: 'thought', text: String(agentEvent.text || ''), ts };
    case 'step':
      return { type: 'step', title: String(agentEvent.title || ''), index: Number(agentEvent.index || 0), ts };
    case 'command':
      return { type: 'command', command: String(agentEvent.command || ''), cwd: agentEvent.cwd ? String(agentEvent.cwd) : undefined, ts };
    case 'stdout':
      return { type: 'stdout', chunk: String(agentEvent.chunk || ''), ts };
    case 'stderr':
      return { type: 'stderr', chunk: String(agentEvent.chunk || ''), ts };
    case 'tool_call':
      return { type: 'tool_call', tool: String(agentEvent.tool || ''), args: (agentEvent.args as Record<string, unknown>) || {}, ts };
    case 'tool_result':
      return { type: 'tool_result', tool: String(agentEvent.tool || ''), ok: Boolean(agentEvent.ok), output: String(agentEvent.output || ''), ts };
    case 'file_read':
      return { type: 'file_read', path: String(agentEvent.path || ''), ts };
    case 'file_write':
      return { type: 'file_write', path: String(agentEvent.path || ''), bytes: agentEvent.bytes ? Number(agentEvent.bytes) : undefined, ts };
    case 'file_change':
      return { type: 'file_change', path: String(agentEvent.path || ''), diff: String(agentEvent.diff || ''), ts };
    case 'status':
      return { type: 'status', value: String(agentEvent.value || 'running') as 'running' | 'completed' | 'failed' | 'cancelled', ts };
    case 'token_usage':
      return { type: 'token_usage', input: Number(agentEvent.input || 0), output: Number(agentEvent.output || 0), ts };
    case 'error':
      return { type: 'error', message: String(agentEvent.message || ''), ts };
    case 'done':
      return { type: 'done', ts };
    default:
      return null;
  }
}

/**
 * Finalize an agent run with completion events
 */
export function finalizeAgentRun(
  run: AgentRun,
  status: 'completed' | 'failed' | 'cancelled',
  summary?: string,
): AgentRun {
  const now = Date.now();
  const events = [...run.events];
  events.push({ type: 'status', value: status, ts: now });
  events.push({ type: 'done', ts: now });

  const filesChanged = new Set(run.filesChanged);
  const commandsRun = new Set(run.commandsRun);
  let totalInputTokens = run.totalInputTokens;
  let totalOutputTokens = run.totalOutputTokens;

  for (const event of events) {
    if (event.type === 'file_change' || event.type === 'file_write') {
      filesChanged.add(event.path);
    }
    if (event.type === 'command') {
      commandsRun.add(event.command);
    }
    if (event.type === 'token_usage') {
      totalInputTokens += event.input;
      totalOutputTokens += event.output;
    }
  }

  return {
    ...run,
    status,
    events,
    completedAt: now,
    totalInputTokens,
    totalOutputTokens,
    stepCount: events.filter(e => e.type === 'step').length,
    toolCallCount: events.filter(e => e.type === 'tool_call').length,
    filesChanged: Array.from(filesChanged),
    commandsRun: Array.from(commandsRun),
    summary,
  };
}
