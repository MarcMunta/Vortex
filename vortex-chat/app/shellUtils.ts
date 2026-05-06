import {
  ChatSession,
  Language,
  LocalAccount,
  PermissionLevel,
  UserSettings,
  WorkspacePermissions,
  WorkspaceProject,
} from "../types";

export const DEFAULT_PERMISSIONS: WorkspacePermissions = {
  level: "none",
  workspaceRoot: "",
  projectPath: "",
  actionMode: "safe",
};

export const DEFAULT_SETTINGS: UserSettings = {
  categoryOrder: ["Preferencias", "Interfaz", "Datos", "Chats Recientes"],
  codeTheme: "dark",
  fontSize: "medium",
  language: "es",
  themeMode: "system",
  permissions: DEFAULT_PERMISSIONS,
  projects: [],
  activeProjectId: null,
};

export const VIEW_INDEX = {
  chat: 0,
  spatial: 1,
} as const;

export const MAX_LOCAL_SESSION_CACHE_SESSIONS = 12;
export const MAX_LOCAL_SESSION_CACHE_MESSAGES = 48;

export const repairMojibakeText = (value: string | null | undefined): string => {
  if (!value || !/[ÃƒÆ’Ãƒâ€š]/.test(value)) return value ?? "";
  try {
    const bytes = Uint8Array.from(Array.from(value), (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder("utf-8").decode(bytes);
    return decoded.includes("\uFFFD") ? value : decoded;
  } catch {
    return value;
  }
};

const normalizePermissionLevel = (raw: unknown): PermissionLevel => {
  const value = String(raw || "none").trim().toLowerCase();
  if (value === "read" || value === "edit" || value === "full") return value;
  return "none";
};

export const permissionsFromProject = (project: WorkspaceProject | null | undefined): WorkspacePermissions => {
  if (!project) return DEFAULT_PERMISSIONS;
  return {
    level: normalizePermissionLevel(project.permissionLevel),
    workspaceRoot: repairMojibakeText(project.rootPath || ""),
    projectPath: repairMojibakeText(project.projectPath || ""),
    actionMode: project.actionMode === "full" ? "full" : "safe",
  };
};

export const normalizeProject = (raw: unknown): WorkspaceProject | null => {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Partial<WorkspaceProject>;
  const id = String(value.id || "").trim() || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const rootPath = repairMojibakeText(String(value.rootPath || ""));
  const projectPath = repairMojibakeText(String(value.projectPath || ""));
  const name = repairMojibakeText(String(value.name || projectPath.split(/[\\/]/).filter(Boolean).pop() || rootPath.split(/[\\/]/).filter(Boolean).pop() || "Workspace"));
  return {
    id,
    name,
    rootPath,
    projectPath,
    permissionLevel: normalizePermissionLevel(value.permissionLevel),
    actionMode: value.actionMode === "full" ? "full" : "safe",
    lastUsedAt: Number(value.lastUsedAt || Date.now()),
  };
};

export const normalizeSession = (rawSession: unknown): ChatSession | null => {
  if (!rawSession || typeof rawSession !== "object" || !Array.isArray((rawSession as ChatSession).messages)) {
    return null;
  }

  const session = rawSession as ChatSession;
  return {
    ...session,
    title: repairMojibakeText(session.title),
    projectId: session.projectId ? String(session.projectId) : null,
    projectName: session.projectName ? repairMojibakeText(String(session.projectName)) : null,
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
  const projects = Array.isArray(candidate.projects)
    ? candidate.projects.map(normalizeProject).filter((project): project is WorkspaceProject => project !== null)
    : [];
  const activeProjectId = projects.some((project) => project.id === candidate.activeProjectId)
    ? String(candidate.activeProjectId)
    : (projects[0]?.id || null);
  const activeProject = projects.find((project) => project.id === activeProjectId) || null;
  const fallbackPermissions: WorkspacePermissions = {
    ...DEFAULT_PERMISSIONS,
    ...rawPermissions,
    level: normalizePermissionLevel(rawPermissions.level),
    actionMode: rawPermissions.actionMode === "full" ? "full" : "safe",
    workspaceRoot: repairMojibakeText(String(rawPermissions.workspaceRoot || DEFAULT_PERMISSIONS.workspaceRoot)),
    projectPath: repairMojibakeText(String(rawPermissions.projectPath || DEFAULT_PERMISSIONS.projectPath)),
  };
  const themeMode = candidate.themeMode === "light" || candidate.themeMode === "dark" || candidate.themeMode === "system"
    ? candidate.themeMode
    : DEFAULT_SETTINGS.themeMode;

  return {
    ...DEFAULT_SETTINGS,
    ...candidate,
    themeMode,
    categoryOrder: Array.isArray(candidate.categoryOrder)
      ? candidate.categoryOrder.map((entry) => repairMojibakeText(String(entry)))
      : DEFAULT_SETTINGS.categoryOrder,
    permissions: activeProject ? permissionsFromProject(activeProject) : fallbackPermissions,
    projects,
    activeProjectId,
  };
};

export const createEmptySession = (language: Language, project?: WorkspaceProject | null): ChatSession => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: project?.name
    ? (language === "es" ? `${project.name} - nuevo chat` : `${project.name} - new chat`)
    : (language === "es" ? "Nueva Conversacion" : "New Conversation"),
  messages: [],
  projectId: project?.id || null,
  projectName: project?.name || null,
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

export const systemPrefersDark = (): boolean => window.matchMedia("(prefers-color-scheme: dark)").matches;

export const getInitialDarkMode = (): boolean => {
  const savedSettings = localStorage.getItem("user-settings");
  if (savedSettings) {
    try {
      const themeMode = normalizeSettings(JSON.parse(savedSettings)).themeMode;
      if (themeMode === "dark") return true;
      if (themeMode === "light") return false;
    } catch {
      // Fall back below.
    }
  }
  const savedMode = localStorage.getItem("dark-mode");
  if (savedMode !== null) return savedMode === "true";
  return systemPrefersDark();
};
