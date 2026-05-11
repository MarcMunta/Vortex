import {
  AppMode,
  BrowserAction,
  AgentEvent,
  ChatSession,
  GroundingSupport,
  Message,
  ObsidianStatus,
  OperationalStatus,
  Role,
  Source,
  SpatialSessionState,
  VoiceStatus,
  VoiceTranscriptionResult,
  SkillSummary,
  SkillsConfig,
  WorkspacePermissions,
  WorkspaceProject,
} from "../types";
import { parseJsonSafely, requestJson } from "./apiClient";
import { parseNativeAgentEvent } from "./agentEventParser";

export type StreamChunk = {
  text: string;
  thought: string;
  sources: Source[];
  groundingSupports: GroundingSupport[];
  fileChanges?: FileChange[];
  browserActions?: BrowserAction[];
  agentEvents?: AgentEvent[];
  requestId?: string;
  finishReason?: string | null;
  done: boolean;
};

type FileChange = { path: string; diff: string };

export type PromptIntent = {
  wantsCode: boolean;
  wantsCompleteCode: boolean;
  isFlutter: boolean;
  isDart: boolean;
  isDebugging: boolean;
  isExplanationOnly: boolean;
};

export const DEFAULT_CHAT_MAX_TOKENS = 2048;
export const CODE_CHAT_MAX_TOKENS = 3072;
export const COMPLETE_CODE_MAX_TOKENS = 4096;
export const DEFAULT_STREAM_CONNECT_TIMEOUT_MS = 120000;
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300000;

export const resolveApiBaseUrl = (): string => {
  const env = ((import.meta as any).env || {}) as Record<string, string | undefined>;
  const raw = (env.VITE_API_BASE_URL || "").trim();
  if (raw) return raw.replace(/\/+$/, "");
  return "";
};

export const resolveDirectApiBaseUrl = (): string => {
  const env = ((import.meta as any).env || {}) as Record<string, string | undefined>;
  const port = (
    env.VITE_BACKEND_PORT
    || env.VITE_API_PORT
    || "8000"
  ).trim() || "8000";
  const host = typeof window !== "undefined" ? (window.location.hostname || "127.0.0.1") : "127.0.0.1";
  return `http://${host}:${port}`;
};

export function classifyPromptIntent(prompt: string): PromptIntent {
  const text = String(prompt || "");
  const normalized = text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
  const has = (pattern: RegExp) => pattern.test(normalized);
  const isFlutter = has(/\bflutter\b|widget|scaffold|materialapp|textformfield|renderbox|renderflex|overflow|layout/);
  const isDart = isFlutter || has(/\bdart\b|void\s+main|statelesswidget|statefulwidget/);
  const wantsCompleteCode = has(/completo|funcional|sin\s+cortar|archivo\s+completo|full\s+code|complete\s+code|compilable|cerrado/);
  const isDebugging = has(/debug|error|bug|renderbox|renderflex|overflow|layout|exception|stacktrace|fall[ao]|arregla|corrige/);
  const isExplanationOnly = has(/solo\s+explica|solo\s+explicacion|sin\s+codigo|no\s+codigo|explain\s+only|no\s+code/);
  const wantsCode = !isExplanationOnly && (
    wantsCompleteCode
    || isFlutter
    || isDart
    || has(/codigo|code|implementa|crea|formulario|login|pantalla|clase|funcion|function|component|archivo|snippet|ejemplo/)
  );
  return {
    wantsCode,
    wantsCompleteCode,
    isFlutter,
    isDart,
    isDebugging,
    isExplanationOnly,
  };
}

export function shouldUseSources(prompt: string, useInternet: boolean, intent: PromptIntent): boolean {
  if (intent.wantsCode && !/fuente|documentaci[oó]n|docs|buscar|internet|oficial/i.test(prompt)) {
    return false;
  }
  return Boolean(useInternet);
}

const stripCodeFencesForBalance = (text: string): string => {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/^```[^\n]*\n?/, "").replace(/```$/, ""));
};

const balanceOf = (text: string, open: string, close: string): number => {
  let balance = 0;
  let quote: string | null = null;
  let escaped = false;
  for (const ch of text) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === "`") {
      quote = ch;
      continue;
    }
    if (ch === open) balance++;
    if (ch === close) balance--;
  }
  return balance;
};

export function isLikelyTruncatedCode(text: string): boolean {
  const value = String(text || "").trim();
  if (!value) return false;
  if (hasAssistantCompletionClosure(value)) return false;
  const fenceCount = (value.match(/```/g) || []).length;
  if (fenceCount % 2 === 1) return true;
  const hasCodeSignal = /```|class\s+\w+|void\s+main\(|Widget\s+build\(|State<|Scaffold\(|MaterialApp\(|TextFormField\(|=>|;\s*$/m.test(value);
  if (!hasCodeSignal) return false;
  const code = stripCodeFencesForBalance(value);
  if (balanceOf(code, "{", "}") > 0) return true;
  if (balanceOf(code, "(", ")") > 0) return true;
  if (balanceOf(code, "[", "]") > 0) return true;
  const tail = value.split("\n").filter((line) => line.trim()).slice(-1)[0]?.trim() || "";
  if (/^(controller|child|children|onPressed|validator|return)\s*:?\s*$/.test(tail)) return true;
  if (/(:\s*_[A-Za-z0-9_]*|=>|\.|,)$/.test(tail)) return true;
  if (/\b(class|Widget build|State<|Scaffold|MaterialApp)\b[\s\S]*$/.test(code) && balanceOf(code, "{", "}") !== 0) return true;
  return false;
}

export function hasAssistantCompletionClosure(text: string): boolean {
  const value = String(text || "").trim();
  if (!value) return false;
  const tail = value.slice(-1600).toLowerCase();
  return /(?:espero que (?:esto|te) (?:sea util|sea útil|ayude)|buena suerte|respuesta final|tarea completada|archivos actualizados|tests?: ok|validaci[oó]n|listo\b|hecho\b|done\b|completed\b)/i.test(tail);
}

const repairMojibakeText = (value: string): string => {
  if (!/[ÃÂ]/.test(value)) return value;
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder("utf-8").decode(bytes);
    return decoded.includes("\uFFFD") ? value : decoded;
  } catch {
    return value;
  }
};

const extractDomain = (rawUrl: string): string => {
  try {
    return new URL(rawUrl).hostname.replace("www.", "");
  } catch {
    return "local";
  }
};

const toSources = (refs: unknown): Source[] => {
  if (!Array.isArray(refs)) return [];
  const results: Source[] = [];
  for (let index = 0; index < refs.length; index++) {
    const r = refs[index];
    // New format: { kind: "web", ref: "https://..." }
    const isRich = r && typeof r === 'object' && 'ref' in r;
    const rawRef = isRich ? String(r.ref) : (typeof r === 'string' ? r : JSON.stringify(r));
    const rawKind = isRich ? String(r.kind || '') : '';

    // Determine kind
    const isUrl = /^https?:\/\//.test(rawRef);
    let kind: 'web' | 'file' | 'unknown';
    if (rawKind === 'web' || isUrl) kind = 'web';
    else if (rawKind === 'self_code' || /\.(py|ts|tsx|js|jsx|json|yaml|yml|md|toml)$/i.test(rawRef)) kind = 'file';
    else kind = 'unknown';

    // Filter: only show web pages and source code files
    if (kind === 'unknown') continue;
    if (kind === 'file') {
      if (/^data[\/\\]|[\/\\]data[\/\\]|episodes|feedback|\blog|checkpoint|lock|\.sqlite|\.db/i.test(rawRef)) continue;
    }

    const domain = isUrl ? extractDomain(rawRef) : 'local';
    const title = isUrl
      ? domain
      : rawRef.split(/[\/\\]/).pop() || rawRef;

    results.push({ url: rawRef, domain, title, kind, index });
  }
  return results;
};

const extractFileChanges = (content: string): FileChange[] => {
  const changes: FileChange[] = [];
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

const toFileChanges = (raw: unknown): FileChange[] => {
  if (!Array.isArray(raw)) return [];
  const changes: FileChange[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const path = String((item as { path?: unknown }).path || "").trim();
    const diff = String((item as { diff?: unknown }).diff || "").trim();
    if (!path || !diff) continue;
    changes.push({ path, diff });
  }
  return changes;
};

const mergeFileChanges = (...groups: FileChange[][]): FileChange[] => {
  const merged: FileChange[] = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const change of group) {
      const key = `${change.path}\n${change.diff}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(change);
    }
  }
  return merged;
};

const toBrowserActions = (raw: unknown): BrowserAction[] => {
  if (!Array.isArray(raw)) return [];
  const actions: BrowserAction[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const target = String((item as { target?: unknown }).target || "").trim();
    if (!target) continue;
    actions.push({
      target,
      opened: Boolean((item as { opened?: unknown }).opened),
    });
  }
  return actions;
};

const toAgentEvents = (raw: unknown): AgentEvent[] => {
  if (!Array.isArray(raw)) return [];
  const events: AgentEvent[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const parsed = parseNativeAgentEvent({ agent_event: item as Record<string, unknown> });
    if (parsed) events.push(parsed);
  }
  return events;
};

const mergeAgentEvents = (...groups: AgentEvent[][]): AgentEvent[] => {
  const merged: AgentEvent[] = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const event of group) {
      const key = JSON.stringify(event);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(event);
    }
  }
  return merged;
};

const MAX_PROMPT_HISTORY_MESSAGES = 24;
const MAX_PROMPT_HISTORY_CHARS = 24000;

const selectHistoryWindow = (history: Message[]): Message[] => {
  const selected: Message[] = [];
  let usedChars = 0;
  for (let index = history.length - 1; index >= 0; index--) {
    const message = history[index];
    const contentLength = (message.content?.length || 0) + (message.thought?.length || 0);
    if (selected.length >= MAX_PROMPT_HISTORY_MESSAGES) break;
    if (selected.length > 0 && usedChars + contentLength > MAX_PROMPT_HISTORY_CHARS) break;
    selected.unshift(message);
    usedChars += contentLength;
  }
  return selected;
};

const buildPermissionsInstruction = (
  permissions: WorkspacePermissions | undefined,
  language: "es" | "en"
) => {
  const level = permissions?.level || "none";
  const workspaceRoot = String(permissions?.workspaceRoot || "").trim();
  const projectPath = String(permissions?.projectPath || "").trim();
  const actionMode = permissions?.actionMode || "safe";

  if (level === "none") {
    return language === "es"
      ? "Permisos activos: nada. Limítate a análisis, explicación y pasos. No afirmes cambios, ejecuciones ni accesos reales."
      : "Active permissions: none. Limit yourself to analysis, explanation, and steps. Do not claim real changes, executions, or access.";
  }

  const scopeLabel = projectPath || workspaceRoot || (language === "es" ? "scope no definido" : "scope not set");
  if (level === "read") {
    return language === "es"
      ? `Permisos activos: solo lectura dentro del scope autorizado. Scope: ${scopeLabel}. Puedes usar contexto del proyecto, pero no edites, ejecutes comandos ni abras navegador.`
      : `Active permissions: read-only inside the authorized scope. Scope: ${scopeLabel}. You may use project context, but do not edit, run commands, or open a browser.`;
  }
  if (level === "edit") {
    return language === "es"
      ? `Permisos activos: lectura y edición dentro del scope autorizado. Scope: ${scopeLabel}. Puedes leer y modificar archivos dentro de esa carpeta/proyecto. No ejecutes comandos ni abras navegador.`
      : `Active permissions: read and edit inside the authorized scope. Scope: ${scopeLabel}. You may read and modify files inside that folder/project. Do not run commands or open a browser.`;
  }
  return language === "es"
    ? `Permisos activos: todo dentro del scope autorizado. Scope: ${scopeLabel}. ${actionMode === "full" ? "Puedes actuar como operador técnico completo dentro de esa carpeta/proyecto: editar archivos, lanzar comandos del proyecto, abrirlo en navegador y hacer los cambios necesarios para la tarea. No asumas acceso fuera de ese scope." : "Mantén el trabajo en modo seguro: puedes analizar el proyecto y preparar cambios, pero no afirmes ejecuciones ni modificaciones reales."}`
    : `Active permissions: full inside the authorized scope. Scope: ${scopeLabel}. ${actionMode === "full" ? "You may act as a full technical operator inside that folder/project: edit files, run project commands, open it in the browser, and make the changes needed for the task. Do not assume access outside that scope." : "Keep work in safe mode: you may analyze the project and prepare changes, but do not claim real executions or modifications."}`;
};

const buildPromptEnvelope = (
  history: Message[],
  prompt: string,
  mode: AppMode,
  useThinking: boolean,
  language: "es" | "en" = "es",
  permissions?: WorkspacePermissions,
  intent: PromptIntent = classifyPromptIntent(prompt)
) => {
  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [];
  const selectedHistory = selectHistoryWindow(history);
  const lang = language === "es" ? "Responde en espanol." : "Reply in English.";
  const tempo = intent.wantsCode
    ? (
        language === "es"
          ? "El usuario esta pidiendo codigo. Devuelve codigo completo, compilable y cerrado. No cortes bloques de codigo. No respondas solo con explicacion. Si el codigo es largo, separalo por archivos. Incluye imports, clases completas y pasos de validacion."
          : "The user is asking for code. Return complete, compilable, closed code. Do not cut code blocks. Do not answer only with explanation. If the code is long, split it by files. Include imports, complete classes, and validation steps."
      )
    : useThinking
      ? (
          language === "es"
            ? "Piensa antes de responder, pero entrega solo la respuesta util."
            : "Think before answering, but return only the useful answer."
        )
      : (
          language === "es"
            ? "Responde de forma directa."
            : "Answer directly."
        );
  const behavior = mode === "agent"
    ? (
        language === "es"
          ? "Actua como operador tecnico. Prioriza diagnostico, acciones reales, archivos tocados y validacion. No inventes diffs ni bloques ```file:path```; los cambios reales se muestran desde las herramientas."
          : "Act as a technical operator. Prioritize diagnosis, real actions, touched files, and validation. Do not invent diffs or ```file:path``` blocks; real changes are shown from tools."
      )
    : (
        language === "es"
          ? "Actua como asistente tecnico local. Da respuestas claras, grounded y sin relleno."
          : "Act as a local technical assistant. Give clear, grounded answers without filler."
      );
  const codeFormat = language === "es"
    ? `Si incluyes codigo multilinea, devuelvelo SIEMPRE dentro de bloques Markdown con triple backtick y lenguaje, por ejemplo \`\`\`dart o \`\`\`ts. No dejes codigo suelto fuera del bloque. ${mode === "ask" && intent.wantsCode ? "En modo consulta, antes del codigo incluye contexto/diagnostico y plan breve. Luego codigo y validacion." : "En modo agente, resume acciones reales, archivos tocados y validacion."}`
    : `If you include multiline code, ALWAYS return it inside Markdown triple-backtick code blocks with a language, for example \`\`\`dart or \`\`\`ts. Do not leave raw code outside the block. ${mode === "ask" && intent.wantsCode ? "In ask mode, include brief context/diagnosis and plan before code. Then code and validation." : "In agent mode, summarize real actions, touched files, and validation."}`;
  const flutterDartInstruction = intent.isFlutter || intent.isDart
    ? (
        language === "es"
          ? "Para Flutter/Dart: usa bloques ```dart```, incluye imports, usa widgets completos, cierra clases/metodos/llaves/parentesis, usa Form, GlobalKey<FormState>, TextFormField, validadores y dispose() cuando aplique, no hagas print(password), no entregues codigo inseguro, y anade flutter analyze y flutter test como validacion."
          : "For Flutter/Dart: use ```dart``` fences, include imports, use complete widgets, close classes/methods/braces/parentheses, use Form, GlobalKey<FormState>, TextFormField, validators, and dispose() when applicable, do not print(password), do not provide insecure code, and add flutter analyze and flutter test as validation."
      )
    : "";
  const noApology = language === "es"
    ? "No empieces con 'Mis disculpas' salvo que estes corrigiendo un error real previo."
    : "Do not start with 'My apologies' unless you are correcting a real prior error.";
  const permissionsInstruction = buildPermissionsInstruction(permissions, language);

  messages.push({
    role: "system",
    content: `Eres Vortex. ${lang} ${tempo} ${behavior} ${codeFormat} ${flutterDartInstruction} ${noApology} ${permissionsInstruction}`,
  });

  for (const msg of selectedHistory) {
    const content = (msg.content ?? "").trim();
    if (!content) continue;
    if (msg.role === Role.USER) messages.push({ role: "user", content });
    else messages.push({ role: "assistant", content });
  }

  messages.push({ role: "user", content: prompt });
  return {
    messages,
    contextMessageIds: selectedHistory.map((msg) => msg.id),
  };
};

const summarizeReasoning = (raw: string): string => {
  const normalized = raw.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  const parts = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [normalized];
  const brief = parts.slice(0, 3).join(" ").trim();
  const maxChars = 600;
  if (brief.length <= maxChars) return brief;
  return `${brief.slice(0, maxChars).trim()}...`;
};

/**
 * Nuclear cleanup: strip ALL leaked system prompts, context, role markers,
 * JSON blobs, and prompt echoes that small local models produce.
 *
 * Strategy: the model frequently echoes the ENTIRE prompt verbatim.
 * The real answer always comes after the LAST "assistant" role marker.
 * We find it (anywhere — start-of-line, inline, with/without colon)
 * and keep only the tail.
 */
const cleanLeakedSystemContent = (text: string): string => {
  let cleaned = text;

  // ==== 0. Quick bail: if it looks clean already, skip heavy processing ====
  const looksLeaky =
    /^\s*(system|user|assistant)\b/im.test(cleaned) ||
    /\bCONTEXT\b/i.test(cleaned) ||
    /"request_id"|"rating"|"train"/i.test(cleaned) ||
    /\\{2,}"/.test(cleaned) ||
    /\[INST\]/i.test(cleaned);
  if (!looksLeaky) return cleaned;

  // ==== 1. Detect prompt echo: find the LAST "assistant" marker anywhere ====
  //         Handles:  "assistant ", "assistant:", "\nassistant\n", inline "...user Hola assistant Según..."
  const assistantPattern = /\bassistant\b\s*:?\s*/gi;
  let lastMatch: RegExpExecArray | null = null;
  let m: RegExpExecArray | null;
  while ((m = assistantPattern.exec(cleaned)) !== null) {
    lastMatch = m;
  }
  if (lastMatch && lastMatch.index !== undefined) {
    const after = cleaned.slice(lastMatch.index + lastMatch[0].length).trim();
    // Use the tail if there's meaningful content (or even short answers > 5 chars)
    if (after.length > 5) {
      cleaned = after;
    }
  }

  // ==== 2. Strip known context / RAG blocks ====
  cleaned = cleaned.replace(/UNTRUSTED CONTEXT[\s\S]*?END_CONTEXT/gi, "");
  cleaned = cleaned.replace(/CONTEXT \(use to inform[\s\S]*?---/gi, "");
  cleaned = cleaned.replace(/\bCONTEXT:[\s\S]*?END_CONTEXT/gi, "");
  cleaned = cleaned.replace(/\bCONTEXT[\s\S]*?---/g, "");
  cleaned = cleaned.replace(/\[INST\][\s\S]*?\[\/INST\]/gi, "");

  // ==== 3. Strip JSON blobs (feedback/episode data leaked from RAG) ====
  cleaned = cleaned.replace(/^.*\\{3,}.*$/gm, "");
  // Full JSON objects with typical episode keys
  cleaned = cleaned.replace(/\{[^{}]*"(?:request_id|rating|train|episode|feedback|timestamp)"[^{}]*\}/g, "");
  // Escaped-quotes JSON
  cleaned = cleaned.replace(/\{[^{}]*\\"[^{}]*\}/g, "");
  cleaned = cleaned.replace(/\\+"/g, ""); // stray \\\" fragments
  // Stray key-value pairs from leaked JSON
  cleaned = cleaned.replace(/"(?:request_id|rating|train|episode|feedback|timestamp)"\s*:\s*"[^"]*"/g, "");

  // ==== 4. Strip leaked system prompt fragments (any language) ====
  const leakedPhrases = [
    /Eres Vortex[^.\n]*\.?/gi,
    /You are Vortex[^.\n]*\.?/gi,
    /Responde en español\.?/gi,
    /Reply in English\.?/gi,
    /Usa bloques ```diff```[^.\n]*\.?/gi,
    /Provide a brief rationale[^.\n]*\.?/gi,
    /Do not reveal chain[^.\n]*\.?/gi,
    /Keep it high-level\.?/gi,
    /Fuera de esas etiquetas[^.\n]*\.?/gi,
    /IMPORTANT[AE]?:?\s*(Si necesitas|If you need)[^.\n]*\.?/gi,
    /SIEMPRE responde[^.\n]*\.?/gi,
    /NUNCA repitas[^.\n]*\.?/gi,
    /NEVER repeat[^.\n]*\.?/gi,
    /Reply ONLY with[^.\n]*\.?/gi,
    /Responde SOLO[^.\n]*\.?/gi,
    /CONTEXT \(use to inform[^)\n]*\):/gi,
    /Never repeat system instructions[^.\n]*\.?/gi,
    /Si necesitas razonar[^.\n]*\.?/gi,
    /pon tu razonamiento DENTRO[^.\n]*\.?/gi,
    /ANTE:?\s*(Si necesitas|If you)[^.\n]*\.?/gi,
    /You are a helpful assistant[^.\n]*\.?/gi,
    /Responde siempre en[^.\n]*\.?/gi,
  ];
  for (const re of leakedPhrases) {
    cleaned = cleaned.replace(re, "");
  }

  // ==== 5. Strip role markers (only when they look like role markers, not normal words) ====
  // Beginning of line: "system ..." or "user ..." or "system: ..."
  cleaned = cleaned.replace(/^\s*(system|user)\b\s*:?\s*/gim, "");
  cleaned = cleaned.replace(/###\s*(System|User|Assistant):[^\n]*/gi, "");
  cleaned = cleaned.replace(/\[(SYSTEM|USER|ASSISTANT|INST)\][^\n]*/gi, "");
  // Q:/A: fallback markers
  cleaned = cleaned.replace(/^\s*[QA]:\s*/gm, "");

  // ==== 6. Strip lines that are just whitespace or punctuation debris ====
  cleaned = cleaned
    .split("\n")
    .filter((line) => line.trim().length > 0 && !/^[\s.,;:!?-]+$/.test(line.trim()))
    .join("\n");

  // ==== 7. Collapse whitespace ====
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n").trim();

  return cleaned;
};

const extractReasoning = (raw: string, isStreaming: boolean = false): { cleanText: string; thought: string } => {
  const parts: string[] = [];

  // 1. Extract COMPLETE tagged reasoning blocks
  const tagRegex = /<(reasoning|think)>([\s\S]*?)<\/\1>/gi;
  let cleanText = raw
    .replace(tagRegex, (_match, _tag, body) => {
      if (typeof body === "string") parts.push(body);
      return "";
    })
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  // 2. Hide UNCLOSED reasoning/think tags (still streaming)
  //    e.g. "<reasoning>partial text here" → hide it all, it'll appear in the reasoning panel
  const unclosedMatch = cleanText.match(/<(reasoning|think)>([\s\S]*)$/i);
  if (unclosedMatch) {
    // Capture the partial thought content for the reasoning panel
    if (unclosedMatch[2]) parts.push(unclosedMatch[2]);
    // Remove the unclosed tag + content from the chat text
    cleanText = cleanText.slice(0, unclosedMatch.index).trim();
  }

  // 3. Also hide standalone opening tags that might linger
  cleanText = cleanText.replace(/<(reasoning|think)>\s*/gi, "");

  // 4. Some models emit "reasoning: <text>" without tags
  const reasoningPrefixMatch = cleanText.match(/^\s*reasoning:\s*(.+)/im);
  if (reasoningPrefixMatch && parts.length === 0) {
    parts.push(reasoningPrefixMatch[1].trim());
    cleanText = cleanText.replace(/^\s*reasoning:\s*.+/im, "").trim();
  }

  // 5. Always clean leaked system/context content — even during streaming
  cleanText = cleanLeakedSystemContent(cleanText);

  const thought = repairMojibakeText(summarizeReasoning(parts.join("\n\n")));
  return { cleanText: repairMojibakeText(cleanText), thought };
};

const parseSseLines = (rawEvent: string): string[] => {
  // Supports multi-line SSE events; we only care about `data:`.
  return rawEvent
    .split(/\r?\n/)
    .map((l) => l.trimEnd())
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice("data:".length).trim());
};

const isServiceUnavailableError = (error: Error | null): boolean => {
  const message = String(error?.message || "").toLowerCase();
  return message.includes("http 503")
    || message.includes("service unavailable")
    || message.includes("model_loading")
    || message.includes("model_load_failed")
    || message.includes("server_error")
    || message.includes("stream_connect_timeout")
    || message.includes("stream_first_chunk_timeout")
    || message.includes("failed to fetch")
    || message.includes("networkerror")
    || message.includes("connection refused")
    || message.includes("net::err");
};

const resolveStreamConnectTimeoutMs = (): number => {
  const env = ((import.meta as any).env || {}) as Record<string, string | undefined>;
  const raw = Number(env.VITE_STREAM_CONNECT_TIMEOUT_MS || "");
  return Number.isFinite(raw) && raw >= 1000 ? raw : DEFAULT_STREAM_CONNECT_TIMEOUT_MS;
};

const resolveStreamIdleTimeoutMs = (): number => {
  const env = ((import.meta as any).env || {}) as Record<string, string | undefined>;
  const raw = Number(env.VITE_STREAM_IDLE_TIMEOUT_MS || "");
  return Number.isFinite(raw) && raw >= 1000 ? raw : DEFAULT_STREAM_IDLE_TIMEOUT_MS;
};

export class VortexService {
  private model: string = "auto";
  private readonly baseUrl = resolveApiBaseUrl();
  private readonly directBaseUrl = resolveDirectApiBaseUrl();

  private url(path: string): string {
    return this.baseUrl ? `${this.baseUrl}${path}` : path;
  }

  private urls(path: string): string[] {
    if (this.baseUrl) return [`${this.baseUrl}${path}`];
    return [path, `${this.directBaseUrl}${path}`];
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    let lastError: unknown = null;
    for (const endpoint of this.urls(path)) {
      try {
        const payload = await requestJson<T>(endpoint, init);
        if (payload === null || payload === undefined) {
          throw new Error("empty_json_response");
        }
        return payload;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError instanceof Error ? lastError : new Error("network_error");
  }

  private responseError(payload: unknown, fallback: string): string {
    if (payload && typeof payload === "object") {
      const body = payload as { detail?: unknown; error?: unknown };
      if (typeof body.detail === "string") return body.detail;
      if (typeof body.error === "string") return body.error;
      if (body.error && typeof body.error === "object" && typeof (body.error as { message?: unknown }).message === "string") {
        return (body.error as { message: string }).message;
      }
    }
    return fallback;
  }

  private async fetchCompletionStream(
    payload: Record<string, unknown>,
    abortSignal: AbortSignal,
  ): Promise<Response> {
    const body = JSON.stringify(payload);
    let lastError: Error | null = null;
    const connectTimeoutMs = resolveStreamConnectTimeoutMs();
    for (const endpoint of this.urls("/v1/chat/completions")) {
      const endpointController = new AbortController();
      let timedOut = false;
      const timeoutId = globalThis.setTimeout(() => {
        timedOut = true;
        endpointController.abort();
      }, connectTimeoutMs);
      const abortEndpoint = () => endpointController.abort();
      if (abortSignal.aborted) {
        abortEndpoint();
      } else {
        abortSignal.addEventListener("abort", abortEndpoint, { once: true });
      }
      try {
        const candidate = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body,
          signal: endpointController.signal,
        });
        globalThis.clearTimeout(timeoutId);

        if (!candidate.ok || !candidate.body) {
          const text = await candidate.text().catch(() => "");
          const parsed = parseJsonSafely<{ error?: { message?: string }; detail?: string }>(text);
          const detail = parsed?.error?.message || parsed?.detail || text;
          throw new Error(`HTTP ${candidate.status}${detail ? `: ${detail}` : ""}`);
        }

        const contentType = candidate.headers.get("content-type") || "";
        if (!contentType.toLowerCase().includes("text/event-stream")) {
          const text = await candidate.text().catch(() => "");
          throw new Error(text ? `non_sse_response:${text.slice(0, 120)}` : "non_sse_response");
        }

        return candidate;
      } catch (error) {
        globalThis.clearTimeout(timeoutId);
        abortSignal.removeEventListener("abort", abortEndpoint);
        if (abortSignal.aborted && !timedOut) {
          throw error instanceof Error ? error : new Error("aborted");
        }
        lastError = timedOut
          ? new Error(`stream_connect_timeout:${connectTimeoutMs}ms`)
          : error instanceof Error ? error : new Error("network_error");
      }
    }
    throw lastError || new Error("network_error");
  }

  async fetchOperationalStatus(): Promise<OperationalStatus | null> {
    try {
      const data = await this.json<OperationalStatus>("/v1/status");
      if (!data || typeof data !== "object") return null;
      return data;
    } catch {
      return null;
    }
  }

  async *generateResponseStream(
    history: Message[],
    prompt: string,
    useInternet: boolean = false,
    useThinking: boolean = true,
    mode: AppMode = "ask",
    language: "es" | "en" = "es",
    webAllowlist: string[] = [],
    memoryContext?: { accountId?: string | null; sessionId?: string | null },
    permissions?: WorkspacePermissions
  ): AsyncGenerator<StreamChunk> {
    const abortController = new AbortController();

    try {
      const intent = classifyPromptIntent(prompt);
      const promptEnvelope = buildPromptEnvelope(history, prompt, mode, useThinking, language, permissions, intent);
      const maxTokens = intent.wantsCompleteCode || intent.isFlutter || intent.isDart
        ? COMPLETE_CODE_MAX_TOKENS
        : intent.wantsCode
          ? CODE_CHAT_MAX_TOKENS
          : DEFAULT_CHAT_MAX_TOKENS;
      const includeSources = shouldUseSources(prompt, useInternet, intent);
      const clientTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || undefined;
      const clientNowIso = new Date().toISOString();
      const payload = {
          model: this.model,
          stream: true,
          agent_mode: mode === "agent",
          vortex_mode: mode === "agent" ? "agent" : "chat",
          include_sources: includeSources,
        include_perf: mode === "agent",
        web_ingest: includeSources && useInternet,
        web_allowlist: webAllowlist,
        rag_mode: includeSources ? "auto" : "off",
        grounding: includeSources,
        max_tokens: maxTokens,
        response_mode: intent.wantsCode ? "code" : "chat",
        code_language: intent.isFlutter || intent.isDart ? "dart" : undefined,
        require_complete_code: intent.wantsCompleteCode || intent.isFlutter || intent.isDart,
        temperature: intent.wantsCode ? 0.2 : (useThinking ? 0.7 : 0.2),
        messages: promptEnvelope.messages,
        client_timezone: clientTimezone,
        client_now_iso: clientNowIso,
        account_id: memoryContext?.accountId || undefined,
        session_id: memoryContext?.sessionId || undefined,
        context_message_ids: promptEnvelope.contextMessageIds,
        permissions: permissions
          ? {
              level: permissions.level,
              workspace_root: permissions.workspaceRoot,
              project_path: permissions.projectPath,
              action_mode: permissions.actionMode,
            }
          : undefined,
      };

      let resp: Response;
      try {
        resp = await this.fetchCompletionStream(payload, abortController.signal);
      } catch (error) {
        const lastError = error instanceof Error ? error : new Error("network_error");
        if (mode !== "agent" || !isServiceUnavailableError(lastError)) {
          throw lastError;
        }
        const fallbackMessages = [
          ...promptEnvelope.messages.slice(0, -1),
          {
            role: "user" as const,
            content: language === "es"
              ? `Modo agente degradado: responde como operador tecnico, sin ejecutar herramientas reales. Tarea:\n${prompt}`
              : `Degraded agent mode: answer as a technical operator without running real tools. Task:\n${prompt}`,
          },
        ];
        resp = await this.fetchCompletionStream(
          {
            ...payload,
            agent_mode: false,
            vortex_mode: "chat",
            include_perf: false,
            messages: fallbackMessages,
          },
          abortController.signal,
        );
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      const streamTimeoutMs = resolveStreamConnectTimeoutMs();
      const streamIdleTimeoutMs = resolveStreamIdleTimeoutMs();

      let buffer = "";
      let rawText = "";
      let fullText = "";
      let thought = "";
      let requestId: string | undefined;
      let finishReason: string | null | undefined;
      let sources: Source[] = [];
      let browserActions: BrowserAction[] = [];
      let perfFileChanges: FileChange[] = [];
      let receivedFirstBytes = false;
      let firstChunkTimedOut = false;
      let idleTimedOut = false;
      let idleTimeoutId: number | null = null;
      let agentEvents: AgentEvent[] = [];
      const firstChunkTimeoutId = globalThis.setTimeout(() => {
        if (!receivedFirstBytes) {
          firstChunkTimedOut = true;
          abortController.abort();
        }
      }, streamTimeoutMs);

      const clearIdleTimeout = () => {
        if (idleTimeoutId !== null) {
          globalThis.clearTimeout(idleTimeoutId);
          idleTimeoutId = null;
        }
      };

      const armIdleTimeout = () => {
        if (streamIdleTimeoutMs <= 0) return;
        clearIdleTimeout();
        idleTimeoutId = globalThis.setTimeout(() => {
          idleTimedOut = true;
          abortController.abort();
        }, streamIdleTimeoutMs);
      };

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!receivedFirstBytes) {
            receivedFirstBytes = true;
            globalThis.clearTimeout(firstChunkTimeoutId);
          }
          armIdleTimeout();
          buffer += decoder.decode(value, { stream: true });

          while (true) {
            const sepIndex = buffer.indexOf("\n\n");
            if (sepIndex === -1) break;

            const rawEvent = buffer.slice(0, sepIndex);
            buffer = buffer.slice(sepIndex + 2);

            for (const data of parseSseLines(rawEvent)) {
              if (!data) continue;
              if (data === "[DONE]") {
                // Final cleanup before finishing
                const finalExtracted = extractReasoning(rawText, false);
                fullText = finalExtracted.cleanText;
                if (finalExtracted.thought) thought = finalExtracted.thought;
                yield {
                  text: fullText,
                  thought,
                  sources: includeSources ? sources : [],
                  groundingSupports: [],
                  fileChanges: mergeFileChanges(extractFileChanges(fullText), perfFileChanges),
                  browserActions,
                  agentEvents,
                  requestId,
                  finishReason,
                  done: true,
                };
                return;
              }

              const parsed = parseJsonSafely<any>(data);
              if (!parsed) continue;

              if (typeof parsed?.request_id === "string") {
                requestId = parsed.request_id;
              }

              if (includeSources && parsed?.sources) {
                sources = toSources(parsed.sources);
              }
              if (parsed?.perf?.browser_actions) {
                browserActions = toBrowserActions(parsed.perf.browser_actions);
              }
              if (parsed?.perf?.file_changes) {
                perfFileChanges = toFileChanges(parsed.perf.file_changes);
              }
              if (parsed?.agent_event) {
                const parsedEvent = parseNativeAgentEvent(parsed);
                if (parsedEvent) agentEvents = mergeAgentEvents(agentEvents, [parsedEvent]);
              }
              if (parsed?.perf?.agent_events) {
                agentEvents = mergeAgentEvents(agentEvents, toAgentEvents(parsed.perf.agent_events));
              }
              const parsedFinishReason = parsed?.choices?.[0]?.finish_reason;
              if (typeof parsedFinishReason === "string") {
                finishReason = parsedFinishReason;
              }

              const delta = parsed?.choices?.[0]?.delta?.content;
              if (typeof delta === "string" && delta.length > 0) {
                rawText += delta;
              }

              // Streaming: extract reasoning + clean leaked content on every chunk
              const extracted = extractReasoning(rawText, /* isStreaming */ true);
              fullText = extracted.cleanText;
              if (extracted.thought) {
                thought = extracted.thought;
              }

              if (typeof fullText === "string") {
                yield {
                  text: fullText,
                  thought,
                  sources: includeSources ? sources : [],
                  groundingSupports: [],
                  fileChanges: mergeFileChanges(extractFileChanges(fullText), perfFileChanges),
                  browserActions,
                  agentEvents,
                  requestId,
                  finishReason,
                  done: false,
                };
              }
            }
          }
        }
      } catch (error) {
        if (firstChunkTimedOut) {
          throw new Error(`stream_first_chunk_timeout:${streamTimeoutMs}ms`);
        }
        if (idleTimedOut) {
          throw new Error(`stream_idle_timeout:${streamIdleTimeoutMs}ms`);
        }
        throw error;
      } finally {
        globalThis.clearTimeout(firstChunkTimeoutId);
        clearIdleTimeout();
      }

      // Final pass: full cleanup including leaked system content stripping
      const finalExtracted = extractReasoning(rawText, /* isStreaming */ false);
      fullText = finalExtracted.cleanText;
      if (finalExtracted.thought) thought = finalExtracted.thought;

      yield {
        text: fullText,
        thought,
        sources: includeSources ? sources : [],
        groundingSupports: [],
        fileChanges: mergeFileChanges(extractFileChanges(fullText), perfFileChanges),
        browserActions,
        agentEvents,
        requestId,
        finishReason,
        done: true,
      };
    } finally {
      abortController.abort();
    }
  }

  async ingestOnce(): Promise<{ ok: boolean; newDocs?: number; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; new_docs?: number }>("/v1/ingest", {
        method: "POST",
        body: JSON.stringify({}),
      });
      if (parsed?.ok) {
        return { ok: true, newDocs: parsed?.new_docs };
      }
      return { ok: false, error: this.responseError(parsed, "ingest_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async submitFeedback(
    requestId: string,
    idealResponse: string
  ): Promise<{
    ok: boolean;
    trainingEvent?: boolean;
    learningQueueItem?: {
      id?: string;
      status?: string;
      source_kind?: string;
      score?: number;
      queued_at?: number;
    } | null;
    learningQueueDepth?: number;
    quickTrainScheduled?: boolean;
    scheduledRunId?: string | null;
    queueReason?: string | null;
    error?: string;
  }> {
    const payload = {
      request_id: requestId,
      rating: "up",
      ideal_response: idealResponse,
    };
    try {
      const parsed = await this.json<{
        ok?: boolean;
        training_event?: boolean;
        learning_queue_item?: {
          id?: string;
          status?: string;
          source_kind?: string;
          score?: number;
          queued_at?: number;
        } | null;
        learning_queue_depth?: number;
        quick_train_scheduled?: boolean;
        scheduled_run_id?: string | null;
        queue_reason?: string | null;
      }>("/v1/feedback", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (parsed?.ok) {
        return {
          ok: true,
          trainingEvent: Boolean(parsed?.training_event),
          learningQueueItem: parsed?.learning_queue_item || null,
          learningQueueDepth: Number.isFinite(parsed?.learning_queue_depth) ? Number(parsed.learning_queue_depth) : undefined,
          quickTrainScheduled: Boolean(parsed?.quick_train_scheduled),
          scheduledRunId: parsed?.scheduled_run_id || null,
          queueReason: parsed?.queue_reason || null,
        };
      }
      return { ok: false, error: this.responseError(parsed, "feedback_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async proposeSelfEditFromDiff(
    diffText: string,
    title: string,
    summary: string
  ): Promise<{ ok: boolean; id?: string; status?: string; error?: string }> {
    const payload = {
      diff_text: diffText,
      title,
      summary,
      author: "frontend",
    };
    try {
      const parsed = await this.json<{ ok?: boolean; id?: string; status?: string }>("/v1/self-edits/proposals/from-diff", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (parsed?.ok) {
        return { ok: true, id: parsed?.id, status: parsed?.status };
      }
      return { ok: false, error: this.responseError(parsed, "proposal_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async generateChatTitle(
    message: string,
    language: "es" | "en" = "es"
  ): Promise<{ ok: boolean; title?: string }> {
    try {
      const data = await this.json<{ ok?: boolean; title?: string }>("/v1/chat/title", {
        method: "POST",
        body: JSON.stringify({ message, language }),
      });
      if (data?.ok && data?.title) {
        return { ok: true, title: data.title };
      }
      return { ok: false };
    } catch {
      return { ok: false };
    }
  }

  async getSpatialSession(): Promise<{ ok: boolean; session?: SpatialSessionState; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; error?: unknown }>("/v1/spatial/session");
      if (parsed?.ok && parsed?.session) {
        return { ok: true, session: parsed.session as SpatialSessionState };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_session_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async updateSpatialSession(payload: Record<string, unknown>): Promise<{ ok: boolean; session?: SpatialSessionState; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; error?: unknown }>("/v1/spatial/session", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (parsed?.ok && parsed?.session) {
        return { ok: true, session: parsed.session as SpatialSessionState };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_session_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async publishSpatialEvent(payload: Record<string, unknown>): Promise<{ ok: boolean; session?: SpatialSessionState; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; error?: unknown }>("/v1/spatial/events", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (parsed?.ok && parsed?.session) {
        return { ok: true, session: parsed.session as SpatialSessionState };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_event_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async openSpatialPanel(payload: Record<string, unknown>): Promise<{ ok: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: unknown }>("/v1/spatial/panels/open", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (parsed?.ok) {
        return { ok: true, session: parsed.session as SpatialSessionState, panel: parsed.panel as Record<string, unknown> };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_panel_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async updateSpatialPanel(panelId: string, payload: Record<string, unknown>): Promise<{ ok: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: unknown }>("/v1/spatial/panels/update", {
        method: "POST",
        body: JSON.stringify({ panel_id: panelId, ...payload }),
      });
      if (parsed?.ok) {
        return { ok: true, session: parsed.session as SpatialSessionState, panel: parsed.panel as Record<string, unknown> };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_panel_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async navigateSpatialPanel(panelId: string, delta: number): Promise<{ ok: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; session?: SpatialSessionState; panel?: Record<string, unknown>; error?: unknown }>("/v1/spatial/panels/navigate", {
        method: "POST",
        body: JSON.stringify({ panel_id: panelId, delta }),
      });
      if (parsed?.ok) {
        return { ok: true, session: parsed.session as SpatialSessionState, panel: parsed.panel as Record<string, unknown> };
      }
      return { ok: false, error: this.responseError(parsed, "spatial_panel_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async fetchVoiceStatus(): Promise<VoiceStatus | null> {
    try {
      const parsed = await this.json<VoiceStatus>("/v1/voice/status");
      if (parsed?.ok) {
        return parsed as VoiceStatus;
      }
      return null;
    } catch {
      return null;
    }
  }

  async transcribeVoice(input: Blob | string, options?: { language?: "es" | "en" }): Promise<VoiceTranscriptionResult> {
    try {
      const init: RequestInit = {
        method: "POST",
      };
      if (typeof input === "string") {
        init.headers = { "Content-Type": "application/json" };
        init.body = JSON.stringify({ text: input, language: options?.language });
      } else {
        init.headers = {
          "Content-Type": input.type || "audio/webm",
          "X-Vortex-Voice-Language": options?.language || "",
        };
        init.body = input;
      }
      const resp = await fetch(this.url("/v1/voice/transcribe"), init);
      const text = await resp.text().catch(() => "");
      const parsed = parseJsonSafely<VoiceTranscriptionResult & { error?: unknown }>(text);
      if (resp.ok && parsed?.ok) {
        return parsed as VoiceTranscriptionResult;
      }
      return { ok: false, error: this.responseError(parsed, text || `HTTP ${resp.status}`) };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async speakText(text: string, language: "es" | "en" = "es"): Promise<{ ok: boolean; audio_url?: string; fallback_browser_tts?: boolean; text?: string; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; audio_url?: string; fallback_browser_tts?: boolean; text?: string; error?: unknown }>("/v1/voice/speak", {
        method: "POST",
        body: JSON.stringify({ text, language }),
      });
      if (parsed?.ok) {
        return parsed as { ok: boolean; audio_url?: string; fallback_browser_tts?: boolean; text?: string };
      }
      return { ok: false, error: this.responseError(parsed, "voice_speak_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async fetchObsidianStatus(): Promise<ObsidianStatus | null> {
    try {
      const parsed = await this.json<ObsidianStatus>("/v1/obsidian/status");
      if (parsed?.ok) {
        return parsed as ObsidianStatus;
      }
      return null;
    } catch {
      return null;
    }
  }

  async configureObsidian(payload: { enabled?: boolean; vault_path?: string }): Promise<ObsidianStatus> {
    return this.json<ObsidianStatus>("/v1/obsidian/config", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async saveObsidianNote(payload: { folder?: string; note_type?: string; title: string; content: string; tags?: string[]; metadata?: Record<string, unknown> }): Promise<{ ok?: boolean; path?: string | null; error?: unknown }> {
    return this.json<{ ok?: boolean; path?: string | null; error?: unknown }>("/v1/obsidian/save", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async fetchChatSessions(accountId: string): Promise<{ ok: boolean; sessions?: ChatSession[]; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; sessions?: ChatSession[]; error?: unknown; detail?: unknown }>(`/v1/chat/sessions?account_id=${encodeURIComponent(accountId)}`);
      if (parsed?.ok && Array.isArray(parsed?.sessions)) {
        return { ok: true, sessions: parsed.sessions as ChatSession[] };
      }
      return { ok: false, error: this.responseError(parsed, "chat_sessions_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async syncChatSessions(accountId: string, sessions: ChatSession[]): Promise<{ ok: boolean; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; error?: unknown; detail?: unknown }>("/v1/chat/sessions/sync", {
        method: "POST",
        body: JSON.stringify({
          account_id: accountId,
          replace: true,
          sessions,
        }),
      });
      if (parsed?.ok) {
        return { ok: true };
      }
      return { ok: false, error: this.responseError(parsed, "chat_sessions_sync_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async validateWorkspaceProjects(projects: WorkspaceProject[]): Promise<{ ok: boolean; invalidIds: string[]; error?: string }> {
    if (projects.length === 0) return { ok: true, invalidIds: [] };
    try {
      const parsed = await this.json<{ ok?: boolean; projects?: Array<{ id?: string; valid?: boolean }>; error?: unknown; detail?: unknown }>("/v1/workspace/projects/validate", {
        method: "POST",
        body: JSON.stringify({ projects }),
      });
      if (parsed?.ok && Array.isArray(parsed.projects)) {
        return {
          ok: true,
          invalidIds: parsed.projects
            .filter((project) => project && project.valid === false && project.id)
            .map((project) => String(project.id)),
        };
      }
      return { ok: false, invalidIds: [], error: this.responseError(parsed, "workspace_project_validation_failed") };
    } catch (error) {
      return { ok: false, invalidIds: [], error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async listSkills(): Promise<{ ok: boolean; skills: SkillSummary[]; error?: string }> {
    try {
      const parsed = await this.json<{ object?: string; data?: SkillSummary[]; error?: unknown; detail?: unknown }>("/v1/skills");
      if (parsed && Array.isArray(parsed.data)) {
        return { ok: true, skills: parsed.data };
      }
      return { ok: false, skills: [], error: this.responseError(parsed, "skills_list_failed") };
    } catch (error) {
      return { ok: false, skills: [], error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async getSkillsConfig(): Promise<{ ok: boolean; config?: SkillsConfig; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; config?: SkillsConfig; error?: unknown; detail?: unknown }>("/v1/skills/config");
      if (parsed?.config) {
        return { ok: Boolean(parsed.ok ?? true), config: parsed.config };
      }
      return { ok: false, error: this.responseError(parsed, "skills_config_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async updateSkillsConfig(payload: Partial<SkillsConfig>): Promise<{ ok: boolean; config?: SkillsConfig; error?: string }> {
    try {
      const body: Record<string, unknown> = {};
      if (typeof payload.enabled === "boolean") body.enabled = payload.enabled;
      if (typeof payload.strict === "boolean") body.strict = payload.strict;
      if (typeof payload.max_k === "number") body.max_k = payload.max_k;
      if (typeof payload.token_budget_total === "number") body.token_budget_total = payload.token_budget_total;
      const parsed = await this.json<{ ok?: boolean; config?: SkillsConfig; error?: unknown; detail?: unknown }>("/v1/skills/config", {
        method: "POST",
        body: JSON.stringify(body),
      });
      if (parsed?.config) {
        return { ok: Boolean(parsed.ok ?? true), config: parsed.config };
      }
      return { ok: false, error: this.responseError(parsed, "skills_config_update_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async toggleSkill(ref: string, enabled: boolean): Promise<{ ok: boolean; skill?: SkillSummary; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; skill?: SkillSummary; error?: unknown; detail?: unknown }>("/v1/skills/toggle", {
        method: "POST",
        body: JSON.stringify({ ref, enabled }),
      });
      if (parsed?.skill) {
        return { ok: Boolean(parsed.ok ?? true), skill: parsed.skill };
      }
      return { ok: false, error: this.responseError(parsed, "skills_toggle_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }

  async pickWorkspaceFolder(payload?: { title?: string; initialDir?: string }): Promise<{ ok: boolean; path?: string; cancelled?: boolean; error?: string }> {
    try {
      const parsed = await this.json<{ ok?: boolean; path?: string; cancelled?: boolean; error?: unknown; detail?: unknown }>("/v1/workspace/folder-picker", {
        method: "POST",
        body: JSON.stringify({
          title: payload?.title,
          initial_dir: payload?.initialDir,
        }),
      });
      if (parsed?.ok) {
        return {
          ok: true,
          path: typeof parsed.path === "string" ? parsed.path : "",
          cancelled: Boolean(parsed.cancelled),
        };
      }
      return { ok: false, error: this.responseError(parsed, "folder_picker_failed") };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "network_error" };
    }
  }
}

export const vortexService = new VortexService();
