export class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export const parseJsonSafely = <T = unknown>(text: string): T | null => {
  const raw = text.trim();
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
};

export const parseEventData = <T>(data: string, fallbackMessage: string): T => {
  const payload = parseJsonSafely<T>(data);
  if (payload === null) {
    throw new Error(fallbackMessage);
  }
  return payload;
};

export const requestJson = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  const text = await response.text().catch(() => "");
  const payload = parseJsonSafely<unknown>(text);
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object"
        ? String(
            (payload as { detail?: unknown; error?: { message?: unknown } | string }).detail
              || (typeof (payload as { error?: unknown }).error === "string"
                ? (payload as { error?: string }).error
                : (payload as { error?: { message?: unknown } }).error?.message)
              || `HTTP ${response.status}`,
          )
        : `HTTP ${response.status}`;
    throw new ApiError(detail, response.status, payload);
  }
  return payload as T;
};
