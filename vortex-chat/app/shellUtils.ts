import { ChatSession, Language, LocalAccount, UserSettings, WorkspacePermissions } from "../types";

export const DEFAULT_PERMISSIONS: WorkspacePermissions = {
  level: "none",
  workspaceRoot: "",
  projectPath: "",
  actionMode: "safe",
};

export const DEFAULT_SETTINGS: UserSettings = {
  categoryOrder: ["Acciones Rápidas", "Preferencias", "Interfaz", "Datos", "Chats Recientes", "Sistema"],
  codeTheme: "dark",
  fontSize: "medium",
  language: "es",
  permissions: DEFAULT_PERMISSIONS,
};

export const VIEW_INDEX = {
  chat: 0,
  spatial: 1,
  analysis: 2,
  training: 3,
  edits: 4,
  terminal: 5,
} as const;

export const MAX_LOCAL_SESSION_CACHE_SESSIONS = 12;
export const MAX_LOCAL_SESSION_CACHE_MESSAGES = 18;

export const repairMojibakeText = (value: string | null | undefined): string => {
  if (!value || !/[ÃƒÃ‚]/.test(value)) return value ?? "";
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder("utf-8").decode(bytes);
    return decoded.includes("\uFFFD") ? value : decoded;
  } catch {
    return value;
  }
};

export const normalizeSession = (rawSession: unknown): ChatSession | null => {
  if (!rawSession || typeof rawSession !== "object" || !Array.isArray((rawSession as ChatSession).messages)) {
    return null;
  }

  const session = rawSession as ChatSession;
  return {
    ...session,
    title: repairMojibakeText(session.title),
    messages: session.messages.map((message) => ({
      ...message,
      content: repairMojibakeText(message.content),
      thought: typeof message.thought === "string" ? repairMojibakeText(message.thought) : message.thought,
    })),
  };
};

export const normalizeSettings = (rawSettings: unknown): UserSettings => {
  if (!rawSettings || typeof rawSettings !== "object") return DEFAULT_SETTINGS;

  const candidate = rawSettings as Partial<UserSettings>;
  const rawPermissions = candidate.permissions && typeof candidate.permissions === "object"
    ? candidate.permissions as Partial<WorkspacePermissions>
    : {};
  return {
    ...DEFAULT_SETTINGS,
    ...candidate,
    categoryOrder: Array.isArray(candidate.categoryOrder)
      ? candidate.categoryOrder.map((entry) => repairMojibakeText(String(entry)))
      : DEFAULT_SETTINGS.categoryOrder,
    permissions: {
      ...DEFAULT_PERMISSIONS,
      ...rawPermissions,
      workspaceRoot: repairMojibakeText(String(rawPermissions.workspaceRoot || DEFAULT_PERMISSIONS.workspaceRoot)),
      projectPath: repairMojibakeText(String(rawPermissions.projectPath || DEFAULT_PERMISSIONS.projectPath)),
    },
  };
};

export const createEmptySession = (language: Language): ChatSession => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: language === "es" ? "Nueva Conversación" : "New Conversation",
  messages: [],
  updatedAt: Date.now(),
});

export const createLocalAccount = (name: string, email: string, handle?: string): LocalAccount => {
  const normalizedHandle = handle?.trim()
    ? (handle.trim().startsWith("@") ? handle.trim() : `@${handle.trim()}`)
    : `@${name.toLowerCase().replace(/[^a-z0-9]+/gi, "").slice(0, 12) || "vortex"}`;
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    email,
    handle: normalizedHandle,
    avatarHue: 198 + Math.floor(Math.random() * 24),
    createdAt: Date.now(),
    lastUsedAt: Date.now(),
  };
};

export const buildSessionCache = (sessions: ChatSession[]): ChatSession[] => (
  sessions.slice(0, MAX_LOCAL_SESSION_CACHE_SESSIONS).map((session) => ({
    ...session,
    messages: session.messages.slice(-MAX_LOCAL_SESSION_CACHE_MESSAGES),
  }))
);

export const createDefaultAccount = (): LocalAccount => createLocalAccount("Vortex Local", "local@vortex.dev", "@vortex");
export const accountSessionsKey = (accountId: string) => `chat-sessions:${accountId}`;
export const accountSettingsKey = (accountId: string) => `user-settings:${accountId}`;

export const getInitialDarkMode = (): boolean => {
  const savedMode = localStorage.getItem("dark-mode");
  if (savedMode !== null) return savedMode === "true";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
};
