import { test, expect } from '@playwright/test';
import {
  classifyPromptIntent,
  COMPLETE_CODE_MAX_TOKENS,
  DEFAULT_CHAT_MAX_TOKENS,
  isLikelyTruncatedCode,
  shouldUseSources,
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
