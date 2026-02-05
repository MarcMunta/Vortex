import { AppMode, LlmSettings, Message, Role, Source } from '../types';
import { chatCompletionStream, VortexChatMessage } from './vortexClient';

type StreamChunk = {
  text: string;
  thought: string;
  sources: Source[];
  groundingSupports: [];
  fileChanges: { path: string; diff: string }[];
  done: boolean;
};

const extractFileChanges = (content: string): { path: string; diff: string }[] => {
  const changes: { path: string; diff: string }[] = [];
  const codeBlockRegex = /```file:([^\n]+)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    changes.push({
      path: match[1].trim(),
      diff: match[2].trim(),
    });
  }

  return changes;
};

const parseThinking = (rawBuffer: string) => {
  let displayText = rawBuffer;
  let thoughtText = '';

  const thinkingRegex = /<thinking>([\s\S]*?)<\/thinking>/g;
  let match: RegExpExecArray | null;
  while ((match = thinkingRegex.exec(displayText)) !== null) {
    thoughtText += match[1] + '\n';
  }

  displayText = displayText.replace(thinkingRegex, '');

  const openTag = '<thinking>';
  const openTagIndex = displayText.indexOf(openTag);
  if (openTagIndex !== -1) {
    const pendingThought = displayText.substring(openTagIndex + openTag.length);
    thoughtText += pendingThought;
    displayText = displayText.substring(0, openTagIndex);
  }

  return { displayText: displayText.trim(), thoughtText: thoughtText.trim() };
};

const toOpenAiMessages = (history: Message[], prompt: string): VortexChatMessage[] => {
  const mapped = history
    .filter((m) => !!m.content)
    .map((m) => ({
      role: m.role === Role.USER ? ('user' as const) : ('assistant' as const),
      content: m.content,
    }));
  mapped.push({ role: 'user', content: prompt });
  return mapped;
};

const sourcesFromRefs = (refs: any[] | undefined): Source[] => {
  const list = Array.isArray(refs) ? refs : [];
  return list.map((ref, index) => {
    const url = String(ref || '');
    let domain = url;
    try {
      domain = new URL(url).hostname;
    } catch {}
    return {
      title: url,
      url,
      domain,
      index,
    };
  });
};

export class VortexService {
  async *generateResponseStream(opts: {
    history: Message[];
    prompt: string;
    api: LlmSettings;
    mode: AppMode;
    useInternet: boolean;
    useThinking: boolean;
    signal: AbortSignal;
  }): AsyncGenerator<StreamChunk> {
    const messages = toOpenAiMessages(opts.history, opts.prompt);
    const maxTokens = opts.useThinking ? opts.api.maxTokens : Math.min(opts.api.maxTokens, 512);
    const temperature = opts.useThinking ? opts.api.temperature : Math.min(opts.api.temperature, 0.4);

    let rawBuffer = '';
    let sources: Source[] = [];

    for await (const evt of chatCompletionStream({
      baseUrl: opts.api.baseUrl,
      token: opts.api.token,
      signal: opts.signal,
      request: {
        model: opts.api.model,
        messages,
        temperature,
        top_p: opts.api.topP,
        max_tokens: maxTokens,
        stream: true,
        include_sources: opts.useInternet,
        metadata: {
          mode: opts.mode,
          use_internet: opts.useInternet,
          use_thinking: opts.useThinking,
        },
      },
    })) {
      const choice = Array.isArray(evt?.choices) ? evt.choices[0] : undefined;
      const delta = (choice as any)?.delta?.content ?? (choice as any)?.text ?? '';

      if (Array.isArray((evt as any)?.sources)) {
        sources = sourcesFromRefs((evt as any).sources);
      }

      if (typeof delta === 'string' && delta) rawBuffer += delta;

      const { displayText, thoughtText } = parseThinking(rawBuffer);

      yield {
        text: displayText,
        thought: thoughtText,
        sources,
        groundingSupports: [],
        fileChanges: extractFileChanges(displayText),
        done: false,
      };
    }

    const { displayText, thoughtText } = parseThinking(rawBuffer);
    yield {
      text: displayText,
      thought: thoughtText,
      sources,
      groundingSupports: [],
      fileChanges: extractFileChanges(displayText),
      done: true,
    };
  }
}

export const vortexService = new VortexService();
