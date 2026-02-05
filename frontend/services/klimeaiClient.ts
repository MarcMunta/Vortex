export type KlimeAiRole = 'system' | 'user' | 'assistant';

export interface KlimeAiChatMessage {
  role: KlimeAiRole;
  content: string;
}

export interface KlimeAiModelInfo {
  id: string;
  object?: string;
  owned_by?: string;
  loaded?: boolean;
  backend?: string;
  device?: string | null;
  dtype?: string | null;
  context_length?: number | null;
  quant?: string | null;
}

export interface KlimeAiListModelsResponse {
  object: 'list';
  data: KlimeAiModelInfo[];
}

export interface KlimeAiChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface KlimeAiChatCompletionChoice {
  index: number;
  message?: { role: 'assistant'; content: string };
  delta?: { role?: 'assistant'; content?: string };
  finish_reason?: string | null;
  text?: string;
}

export interface KlimeAiChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  request_id?: string;
  choices: KlimeAiChatCompletionChoice[];
  usage?: KlimeAiChatCompletionUsage;
  sources?: any[];
}

export interface KlimeAiChatCompletionRequest {
  model?: string;
  messages: KlimeAiChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  user?: string;
  metadata?: Record<string, unknown>;
  include_sources?: boolean;
}

export class KlimeAiApiError extends Error {
  status: number;
  code?: string;
  type?: string;

  constructor(message: string, status: number, opts?: { code?: string; type?: string }) {
    super(message);
    this.name = 'KlimeAiApiError';
    this.status = status;
    this.code = opts?.code;
    this.type = opts?.type;
  }
}

const joinUrl = (baseUrl: string, path: string) => {
  const base = (baseUrl || '').trim();
  if (!base) return path;
  return base.replace(/\/+$/, '') + path;
};

const buildHeaders = (token?: string) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const trimmed = (token || '').trim();
  if (trimmed) headers['Authorization'] = `Bearer ${trimmed}`;
  return headers;
};

const readErrorFromResponse = async (res: Response) => {
  let message = `HTTP ${res.status}`;
  let code: string | undefined;
  let type: string | undefined;
  try {
    const data = await res.json();
    const err = data?.error;
    if (typeof err?.message === 'string') message = err.message;
    if (typeof err?.code === 'string') code = err.code;
    if (typeof err?.type === 'string') type = err.type;
  } catch {
    try {
      const text = await res.text();
      if (text) message = text;
    } catch {}
  }
  return new KlimeAiApiError(message, res.status, { code, type });
};

export const listModels = async (opts: { baseUrl: string; token?: string; signal?: AbortSignal }) => {
  const res = await fetch(joinUrl(opts.baseUrl, '/v1/models'), {
    method: 'GET',
    headers: buildHeaders(opts.token),
    signal: opts.signal,
  });
  if (!res.ok) throw await readErrorFromResponse(res);
  const data = (await res.json()) as KlimeAiListModelsResponse;
  return Array.isArray(data?.data) ? data.data : [];
};

export const chatCompletion = async (opts: {
  baseUrl: string;
  token?: string;
  request: KlimeAiChatCompletionRequest;
  signal?: AbortSignal;
}) => {
  const res = await fetch(joinUrl(opts.baseUrl, '/v1/chat/completions'), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({ ...opts.request, stream: false }),
    signal: opts.signal,
  });
  if (!res.ok) throw await readErrorFromResponse(res);
  return (await res.json()) as KlimeAiChatCompletionResponse;
};

function* parseSseEventDataLines(eventText: string) {
  const lines = eventText.split(/\r?\n/);
  for (const line of lines) {
    if (!line.startsWith('data:')) continue;
    yield line.slice('data:'.length).trimStart();
  }
}

export async function* chatCompletionStream(opts: {
  baseUrl: string;
  token?: string;
  request: KlimeAiChatCompletionRequest;
  signal?: AbortSignal;
}): AsyncGenerator<KlimeAiChatCompletionResponse, void, void> {
  const res = await fetch(joinUrl(opts.baseUrl, '/v1/chat/completions'), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({ ...opts.request, stream: true }),
    signal: opts.signal,
  });
  if (!res.ok) throw await readErrorFromResponse(res);
  if (!res.body) throw new KlimeAiApiError('No response body', res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sepIndex = buffer.indexOf('\n\n');
      if (sepIndex === -1) break;
      const rawEvent = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      for (const dataLine of parseSseEventDataLines(rawEvent)) {
        if (!dataLine) continue;
        if (dataLine === '[DONE]') return;
        try {
          const evt = JSON.parse(dataLine);
          if (evt && typeof evt === 'object') yield evt as KlimeAiChatCompletionResponse;
        } catch {
          continue;
        }
      }
    }
  }
}

