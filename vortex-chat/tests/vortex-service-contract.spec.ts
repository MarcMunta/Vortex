import { test, expect } from '@playwright/test';
import {
  classifyPromptIntent,
  COMPLETE_CODE_MAX_TOKENS,
  DEFAULT_CHAT_MAX_TOKENS,
  isLikelyTruncatedCode,
  resolveApiBaseUrl,
  resolveDirectApiBaseUrl,
  shouldUseSources,
  VortexService,
} from '../services/vortexService';

test('classifyPromptIntent detects Flutter complete code', () => {
  const intent = classifyPromptIntent('Crea un login básico en Flutter. Quiero código completo y sin cortar.');
  expect(intent.wantsCode).toBeTruthy();
  expect(intent.wantsCompleteCode).toBeTruthy();
  expect(intent.isFlutter).toBeTruthy();
  expect(intent.isDart).toBeTruthy();
});

test('shouldUseSources disables sources for simple code generation', () => {
  const intent = classifyPromptIntent('Crea un login básico en Flutter');
  expect(shouldUseSources('Crea un login básico en Flutter', true, intent)).toBeFalsy();
  expect(shouldUseSources('Busca en la documentación oficial de Flutter constraints', true, intent)).toBeTruthy();
});

test('isLikelyTruncatedCode detects Dart truncation', () => {
  expect(isLikelyTruncatedCode('```dart\nvoid main() {\n  runApp(const App());\n}\n```')).toBeFalsy();
  expect(isLikelyTruncatedCode('```dart\nvoid main() {\n  runApp(const App());')).toBeTruthy();
  expect(isLikelyTruncatedCode('class A {\n  void f() {\n')).toBeTruthy();
  expect(isLikelyTruncatedCode('Texto normal sin código.')).toBeFalsy();
  expect(isLikelyTruncatedCode('class A {\n  void f() {\n\nEspero que esto te ayude. Buena suerte.')).toBeFalsy();
});

test('token constants keep code budget long', () => {
  expect(DEFAULT_CHAT_MAX_TOKENS).toBeGreaterThanOrEqual(2048);
  expect(COMPLETE_CODE_MAX_TOKENS).toBeGreaterThanOrEqual(4096);
});

test('agent stream uses same-origin API proxy by default', async () => {
  expect(resolveApiBaseUrl()).toBe('');
  expect(resolveDirectApiBaseUrl()).toBe('http://127.0.0.1:8000');

  const originalFetch = globalThis.fetch;
  let seenUrl = '';
  let seenPayload: any = null;

  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    seenUrl = String(input);
    seenPayload = JSON.parse(String(init?.body || '{}'));
    const body = [
      'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"request_id":"req-1"}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":"agent-ok"},"finish_reason":null}],"request_id":"req-1"}\n\n',
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"request_id":"req-1","perf":{"file_changes":[{"path":"lib/main.dart","diff":"--- /dev/null\\n+++ b/lib/main.dart\\n@@\\n+void main() {}"}],"agent_events":[{"type":"step","title":"Ejecutando comando","index":1,"ts":1},{"type":"command","command":"python -m pytest -q","ts":2},{"type":"stdout","chunk":"1 passed","ts":3},{"type":"file_change","path":"lib/main.dart","diff":"--- /dev/null\\n+++ b/lib/main.dart\\n@@\\n+void main() {}","ts":4},{"type":"status","value":"completed","ts":5},{"type":"done","ts":5}],"tool_calls":[{"action":"run_command","args":{"command":"python -m pytest -q"},"ok":true,"output":"1 passed"}]}}\n\n',
      'data: [DONE]\n\n',
    ].join('');
    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }) as typeof fetch;

  try {
    const chunks = [];
    for await (const chunk of new VortexService().generateResponseStream([], 'haz algo', false, true, 'agent', 'es')) {
      chunks.push(chunk);
    }

    expect(seenUrl).toBe('/v1/chat/completions');
    expect(seenPayload.agent_mode).toBeTruthy();
    expect(seenPayload.vortex_mode).toBe('agent');
    expect(chunks.at(-1)?.text).toBe('agent-ok');
    expect(chunks.at(-1)?.fileChanges?.[0]?.path).toBe('lib/main.dart');
    expect(chunks.at(-1)?.fileChanges?.[0]?.diff).toContain('+void main()');
    expect(chunks.at(-1)?.agentEvents?.some((event) => event.type === 'command' && event.command === 'python -m pytest -q')).toBeTruthy();
    expect(chunks.at(-1)?.agentEvents?.some((event) => event.type === 'file_change' && event.path === 'lib/main.dart')).toBeTruthy();
    expect(chunks.at(-1)?.done).toBeTruthy();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('chat stream falls back to direct backend when proxy does not return SSE', async () => {
  const originalFetch = globalThis.fetch;
  const seenUrls: string[] = [];

  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    seenUrls.push(url);
    if (seenUrls.length === 1) {
      return new Response('<html>vite preview fallback</html>', {
        status: 200,
        headers: { 'Content-Type': 'text/html' },
      });
    }
    const body = [
      'data: {"choices":[{"index":0,"delta":{"content":"fallback-ok"},"finish_reason":null}],"request_id":"req-2"}\n\n',
      'data: [DONE]\n\n',
    ].join('');
    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }) as typeof fetch;

  try {
    const chunks = [];
    for await (const chunk of new VortexService().generateResponseStream([], 'hola', false, true, 'ask', 'es')) {
      chunks.push(chunk);
    }

    expect(seenUrls).toEqual(['/v1/chat/completions', 'http://127.0.0.1:8000/v1/chat/completions']);
    expect(chunks.at(-1)?.text).toBe('fallback-ok');
    expect(chunks.at(-1)?.done).toBeTruthy();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('status falls back to direct backend when proxy returns non-json', async () => {
  const originalFetch = globalThis.fetch;
  const seenUrls: string[] = [];

  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    seenUrls.push(url);
    if (seenUrls.length === 1) {
      return new Response('<html>frontend shell</html>', {
        status: 200,
        headers: { 'Content-Type': 'text/html' },
      });
    }
    return new Response(JSON.stringify({ ok: true, chat_ready: true, chat_mode: 'primary' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }) as typeof fetch;

  try {
    const status = await new VortexService().fetchOperationalStatus();

    expect(seenUrls).toEqual(['/v1/status', 'http://127.0.0.1:8000/v1/status']);
    expect(status?.chat_ready).toBeTruthy();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('agent stream falls back to degraded chat when agent backend returns 503', async () => {
  const originalFetch = globalThis.fetch;
  const seenPayloads: any[] = [];

  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const payload = JSON.parse(String(init?.body || '{}'));
    seenPayloads.push(payload);
    if (payload.agent_mode) {
      return new Response(JSON.stringify({ error: { message: 'model_load_failed:runtime unavailable' } }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    const body = [
      'data: {"choices":[{"index":0,"delta":{"content":"degraded-agent-ok"},"finish_reason":null}],"request_id":"req-3"}\n\n',
      'data: [DONE]\n\n',
    ].join('');
    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }) as typeof fetch;

  try {
    const chunks = [];
    for await (const chunk of new VortexService().generateResponseStream([], 'haz algo', false, true, 'agent', 'es')) {
      chunks.push(chunk);
    }

    expect(seenPayloads.some((payload) => payload.agent_mode === true)).toBeTruthy();
    expect(seenPayloads.some((payload) => payload.agent_mode === false && payload.vortex_mode === 'chat')).toBeTruthy();
    expect(String(seenPayloads.at(-1)?.messages?.at(-1)?.content || '')).toContain('Modo agente degradado');
    expect(chunks.at(-1)?.text).toBe('degraded-agent-ok');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
