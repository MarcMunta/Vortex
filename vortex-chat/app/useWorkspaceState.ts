import { useCallback, useEffect, useMemo, useState } from "react";
import { vortexService } from "../services/vortexService";
import { ChatSession, LocalAccount, UserSettings } from "../types";
import {
  accountSessionsKey,
  accountSettingsKey,
  buildSessionCache,
  createDefaultAccount,
  createEmptySession,
  createLocalAccount,
  DEFAULT_SETTINGS,
  getInitialDarkMode,
  normalizeSession,
  normalizeSettings,
} from "./shellUtils";

type UseWorkspaceStateArgs = {
  isLoading: boolean;
};

type CreateAccountDraft = {
  name: string;
  email: string;
  handle?: string;
};

export const useWorkspaceState = ({ isLoading }: UseWorkspaceStateArgs) => {
  const [accounts, setAccounts] = useState<LocalAccount[]>([]);
  const [currentAccountId, setCurrentAccountId] = useState<string | null>(null);
  const [isAccountHydrated, setIsAccountHydrated] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [isDarkMode, setIsDarkMode] = useState(getInitialDarkMode());

  const currentAccount = useMemo(
    () => accounts.find((account) => account.id === currentAccountId) || accounts[0] || null,
    [accounts, currentAccountId],
  );
  const currentSession = useMemo(
    () => sessions.find((session) => session.id === currentSessionId),
    [currentSessionId, sessions],
  );

  useEffect(() => {
    const savedAccounts = localStorage.getItem("vortex-accounts");
    const savedCurrentAccountId = localStorage.getItem("vortex-current-account-id");
    let nextAccounts: LocalAccount[] = [];

    if (savedAccounts) {
      try {
        const parsed = JSON.parse(savedAccounts);
        if (Array.isArray(parsed)) {
          nextAccounts = parsed as LocalAccount[];
        }
      } catch {
        nextAccounts = [];
      }
    }

    if (nextAccounts.length === 0) {
      const fallback = createDefaultAccount();
      nextAccounts = [fallback];
      const legacySessions = localStorage.getItem("chat-sessions");
      const legacySettings = localStorage.getItem("user-settings");
      if (legacySessions && !localStorage.getItem(accountSessionsKey(fallback.id))) {
        localStorage.setItem(accountSessionsKey(fallback.id), legacySessions);
      }
      if (legacySettings && !localStorage.getItem(accountSettingsKey(fallback.id))) {
        localStorage.setItem(accountSettingsKey(fallback.id), legacySettings);
      }
    }

    const safeCurrentAccountId = nextAccounts.some((account) => account.id === savedCurrentAccountId)
      ? savedCurrentAccountId
      : nextAccounts[0].id;

    setAccounts(nextAccounts);
    setCurrentAccountId(safeCurrentAccountId);
    setIsAccountHydrated(false);
  }, []);

  useEffect(() => {
    if (!currentAccountId) return;

    const savedSessions = localStorage.getItem(accountSessionsKey(currentAccountId));
    const savedSettings = localStorage.getItem(accountSettingsKey(currentAccountId));
    let disposed = false;

    const nextSettings = savedSettings
      ? (() => {
          try {
            return normalizeSettings(JSON.parse(savedSettings));
          } catch {
            return DEFAULT_SETTINGS;
          }
        })()
      : DEFAULT_SETTINGS;

    const applyCandidateSessions = (candidate: unknown, language: UserSettings["language"]): boolean => {
      const normalizedSessions = Array.isArray(candidate)
        ? candidate
            .map((session) => normalizeSession(session))
            .filter((session): session is ChatSession => session !== null)
            .filter((session, index, all) => {
              const isEmptyDraft = session.messages.length === 0;
              if (!isEmptyDraft) return true;
              return all.findIndex((item) => item && item.title === session.title && item.messages.length === 0) === index;
            })
        : [];

      if (normalizedSessions.length > 0) {
        setSessions(normalizedSessions);
        setCurrentSessionId(normalizedSessions[0].id);
        return true;
      }

      const freshSession = createEmptySession(language);
      setSessions([freshSession]);
      setCurrentSessionId(freshSession.id);
      return false;
    };

    setSettings(nextSettings);

    let usedCache = false;
    if (savedSessions) {
      try {
        usedCache = applyCandidateSessions(JSON.parse(savedSessions), nextSettings.language);
      } catch {
        usedCache = applyCandidateSessions([], nextSettings.language);
      }
    } else {
      applyCandidateSessions([], nextSettings.language);
    }

    const finalizeHydration = () => {
      if (disposed) return;
      setAccounts((prev) => prev.map((account) => (
        account.id === currentAccountId
          ? { ...account, lastUsedAt: Date.now() }
          : account
      )));
      setIsAccountHydrated(true);
    };

    void vortexService.fetchChatSessions(currentAccountId).then((result) => {
      if (disposed) return;
      if (result.ok && Array.isArray(result.sessions) && result.sessions.length > 0) {
        applyCandidateSessions(result.sessions, nextSettings.language);
        try {
          localStorage.setItem(accountSessionsKey(currentAccountId), JSON.stringify(buildSessionCache(result.sessions)));
        } catch {
          // Remote state remains authoritative if local cache overflows.
        }
      } else if (!usedCache) {
        applyCandidateSessions([], nextSettings.language);
      }
      finalizeHydration();
    }).catch(() => {
      finalizeHydration();
    });

    return () => {
      disposed = true;
    };
  }, [currentAccountId]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode);
    document.body.classList.toggle("dark", isDarkMode);
    document.documentElement.style.colorScheme = isDarkMode ? "dark" : "light";
    document.body.style.colorScheme = isDarkMode ? "dark" : "light";
    localStorage.setItem("dark-mode", String(isDarkMode));
  }, [isDarkMode]);

  useEffect(() => {
    localStorage.setItem("vortex-accounts", JSON.stringify(accounts));
  }, [accounts]);

  useEffect(() => {
    if (currentAccountId) localStorage.setItem("vortex-current-account-id", currentAccountId);
  }, [currentAccountId]);

  useEffect(() => {
    if (isAccountHydrated && currentAccountId) {
      try {
        localStorage.setItem(accountSessionsKey(currentAccountId), JSON.stringify(buildSessionCache(sessions)));
      } catch {
        // Backend persistence remains source of truth.
      }
    }
  }, [currentAccountId, isAccountHydrated, sessions]);

  useEffect(() => {
    if (isAccountHydrated && currentAccountId) {
      localStorage.setItem(accountSettingsKey(currentAccountId), JSON.stringify(settings));
    }
  }, [currentAccountId, isAccountHydrated, settings]);

  useEffect(() => {
    if (!isAccountHydrated || !currentAccountId || isLoading) return;
    const timer = window.setTimeout(() => {
      void vortexService.syncChatSessions(currentAccountId, sessions);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [currentAccountId, isAccountHydrated, isLoading, sessions]);

  useEffect(() => {
    if (sessions.length === 0) {
      setCurrentSessionId(null);
      return;
    }
    if (!currentSessionId || !sessions.some((session) => session.id === currentSessionId)) {
      setCurrentSessionId(sessions[0].id);
    }
  }, [currentSessionId, sessions]);

  const handleSelectAccount = useCallback((accountId: string) => {
    if (accountId === currentAccountId) return;
    setIsAccountHydrated(false);
    setCurrentAccountId(accountId);
  }, [currentAccountId]);

  const handleCreateAccount = useCallback((draft: CreateAccountDraft) => {
    const nextAccount = createLocalAccount(draft.name.trim(), draft.email.trim(), draft.handle?.trim());
    const freshSettings = { ...DEFAULT_SETTINGS, language: settings.language };
    const freshSession = createEmptySession(freshSettings.language);

    localStorage.setItem(accountSettingsKey(nextAccount.id), JSON.stringify(freshSettings));
    localStorage.setItem(accountSessionsKey(nextAccount.id), JSON.stringify(buildSessionCache([freshSession])));

    setAccounts((prev) => [nextAccount, ...prev]);
    setIsAccountHydrated(false);
    setCurrentAccountId(nextAccount.id);
    setSessions([freshSession]);
    setCurrentSessionId(freshSession.id);
  }, [settings.language]);

  const handleNewChat = useCallback(() => {
    const newSession = createEmptySession(settings.language);
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
  }, [settings.language]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    const remainingSessions = sessions.filter((session) => session.id !== sessionId);
    if (remainingSessions.length === 0) {
      const replacementSession = createEmptySession(settings.language);
      setSessions([replacementSession]);
      setCurrentSessionId(replacementSession.id);
      return;
    }
    setSessions(remainingSessions);
    if (!remainingSessions.some((session) => session.id === currentSessionId)) {
      setCurrentSessionId(remainingSessions[0].id);
    }
  }, [currentSessionId, sessions, settings.language]);

  const handleClearHistory = useCallback(() => {
    const freshSession = createEmptySession(settings.language);
    setSessions([freshSession]);
    setCurrentSessionId(freshSession.id);
  }, [settings.language]);

  return {
    accounts,
    currentAccount,
    currentAccountId,
    currentSession,
    currentSessionId,
    handleClearHistory,
    handleCreateAccount,
    handleDeleteSession,
    handleNewChat,
    handleSelectAccount,
    isAccountHydrated,
    isDarkMode,
    setAccounts,
    setCurrentAccountId,
    setCurrentSessionId,
    setIsAccountHydrated,
    setIsDarkMode,
    setSessions,
    setSettings,
    sessions,
    settings,
  };
};
