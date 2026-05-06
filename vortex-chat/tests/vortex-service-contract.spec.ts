import { test, expect } from '@playwright/test';
import {
  classifyPromptIntent,
  COMPLETE_CODE_MAX_TOKENS,
  DEFAULT_CHAT_MAX_TOKENS,
  isLikelyTruncatedCode,
  resolveApiBaseUrl,
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
});

test('token constants keep code budget long', () => {
  expect(DEFAULT_CHAT_MAX_TOKENS).toBeGreaterThanOrEqual(2048);
  expect(COMPLETE_CODE_MAX_TOKENS).toBeGreaterThanOrEqual(4096);
});

test('agent stream uses same-origin API proxy by default', async () => {
  expect(resolveApiBaseUrl()).toBe('');

  const originalFetch = globalThis.fetch;
  let seenUrl = '';
  let seenPayload: any = null;

  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    seenUrl = String(input);
    seenPayload = JSON.parse(String(init?.body || '{}'));
    const body = [
      'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"request_id":"req-1"}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":"agent-ok"},"finish_reason":null}],"request_id":"req-1"}\n\n',
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
    expect(chunks.at(-1)?.done).toBeTruthy();
  } finally {
    globalThis.fetch = originalFetch;
  }
});
