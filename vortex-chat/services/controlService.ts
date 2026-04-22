import {
  AutonomyStreamPayload,
  AutonomyStatus,
  ControlStatus,
  TrainingRunSummary,
  TrainingStreamPayload,
} from "../types";

const resolveBaseUrl = (): string => {
  const raw = (import.meta.env.VITE_CONTROL_BASE_URL || "").trim();
  if (raw) return raw.replace(/\/+$/, "");
  const port = (import.meta.env.VITE_CONTROL_PORT || "").trim();
  if (!port) return "";
  const host = window.location.hostname || "127.0.0.1";
  return `http://${host}:${port}`;
};

class ControlService {
  private readonly baseUrl = resolveBaseUrl();

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      ...init,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || `HTTP ${response.status}`;
      throw new Error(String(detail));
    }
    return payload as T;
  }

  async fetchStatus(): Promise<ControlStatus | null> {
    try {
      return await this.json<ControlStatus>("/control/status", { method: "GET" });
    } catch {
      return null;
    }
  }

  async bootstrap(
    force: boolean = false,
    mode: "ensure" | "rebuild" = (force ? "rebuild" : "ensure"),
  ): Promise<{ ok: boolean; started?: boolean; reason?: string; mode?: string; stage?: string; log_path?: string }> {
    return this.json("/control/bootstrap", {
      method: "POST",
      body: JSON.stringify({ force, mode }),
    });
  }

  async initModel(): Promise<{ ok: boolean; started?: boolean }> {
    return this.json("/control/model/init", { method: "POST", body: JSON.stringify({}) });
  }

  async restartRuntime(): Promise<{ ok: boolean }> {
    return this.json("/control/runtime/restart", { method: "POST", body: JSON.stringify({}) });
  }

  async reloadInstructions(): Promise<{ ok: boolean }> {
    return this.json("/control/instructions/reload", { method: "POST", body: JSON.stringify({}) });
  }

  async getAllowlist(): Promise<string[]> {
    const payload = await this.json<{ ok: boolean; domains?: string[] }>("/control/internet/allowlist", { method: "GET" });
    return Array.isArray(payload.domains) ? payload.domains : [];
  }

  async saveAllowlist(domains: string[]): Promise<string[]> {
    const payload = await this.json<{ ok: boolean; domains?: string[] }>("/control/internet/allowlist", {
      method: "POST",
      body: JSON.stringify({ domains }),
    });
    return Array.isArray(payload.domains) ? payload.domains : [];
  }

  async startTraining(
    mode: "quick" | "full",
    source?: string,
  ): Promise<{ ok: boolean; run_id?: string; status?: string; queue_reason?: string | null; reused?: boolean; error?: string }> {
    return this.json("/control/training/start", {
      method: "POST",
      body: JSON.stringify({ mode, source }),
    });
  }

  async resetTrainingState(payload?: { clear_runs?: boolean; clear_learning_queue?: boolean }) {
    return this.json<{ ok: boolean; removed_runs?: number; runs?: TrainingRunSummary[]; autonomy?: AutonomyStatus }>("/control/training/reset", {
      method: "POST",
      body: JSON.stringify({
        clear_runs: payload?.clear_runs ?? true,
        clear_learning_queue: payload?.clear_learning_queue ?? true,
      }),
    });
  }

  async getTrainingRuns(): Promise<TrainingRunSummary[]> {
    const payload = await this.json<{ ok: boolean; runs?: TrainingRunSummary[] }>("/control/training/runs", { method: "GET" });
    return Array.isArray(payload.runs) ? payload.runs : [];
  }

  async getTrainingRun(runId: string): Promise<TrainingRunSummary | null> {
    try {
      const payload = await this.json<{ ok: boolean; run?: TrainingRunSummary }>(`/control/training/runs/${encodeURIComponent(runId)}`, { method: "GET" });
      return payload.run || null;
    } catch {
      return null;
    }
  }

  async getTrainingRunEvents(runId: string) {
    const payload = await this.json<{ ok: boolean; events?: TrainingRunSummary["events"] }>(`/control/training/runs/${encodeURIComponent(runId)}/events`, { method: "GET" });
    return Array.isArray(payload.events) ? payload.events : [];
  }

  async getTrainingRunLogs(runId: string) {
    const payload = await this.json<{ ok: boolean; logs?: Record<string, string[]> }>(`/control/training/runs/${encodeURIComponent(runId)}/logs`, { method: "GET" });
    return payload.logs || {};
  }

  subscribeTrainingStream(
    onMessage: (payload: TrainingStreamPayload) => void,
    onError?: (error: Event | Error) => void,
  ): () => void {
    const source = new EventSource(`${this.baseUrl}/control/training/stream`);
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as TrainingStreamPayload;
        onMessage(payload);
      } catch (error) {
        onError?.(error instanceof Error ? error : new Error("training_stream_parse_failed"));
      }
    };
    source.onerror = (event) => {
      onError?.(event);
    };
    return () => source.close();
  }

  async getAutonomyStatus(): Promise<AutonomyStatus | null> {
    try {
      const payload = await this.json<{ ok: boolean; autonomy?: AutonomyStatus }>("/control/autonomy/status", { method: "GET" });
      return payload.autonomy || null;
    } catch {
      return null;
    }
  }

  async startAutonomy(): Promise<{ ok: boolean; enabled?: boolean }> {
    return this.json("/control/autonomy/start", {
      method: "POST",
      body: JSON.stringify({}),
    });
  }

  async stopAutonomy(): Promise<{ ok: boolean; enabled?: boolean }> {
    return this.json("/control/autonomy/stop", {
      method: "POST",
      body: JSON.stringify({}),
    });
  }

  async configureAutonomy(config: {
    enabled?: boolean;
    reflection_enabled?: boolean;
    training_enabled?: boolean;
    autoedit_enabled?: boolean;
    multi_agent_dialogue_enabled?: boolean;
    descriptive_reports_enabled?: boolean;
    live_autoedit_enabled?: boolean;
  }): Promise<{ ok: boolean; autonomy?: AutonomyStatus }> {
    return this.json("/control/autonomy/config", {
      method: "POST",
      body: JSON.stringify(config),
    });
  }

  subscribeAutonomyStream(
    onMessage: (payload: AutonomyStreamPayload) => void,
    onError?: (error: Event | Error) => void,
  ): () => void {
    const source = new EventSource(`${this.baseUrl}/control/autonomy/stream`);
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as AutonomyStreamPayload;
        onMessage(payload);
      } catch (error) {
        onError?.(error instanceof Error ? error : new Error("autonomy_stream_parse_failed"));
      }
    };
    source.onerror = (event) => {
      onError?.(event);
    };
    return () => source.close();
  }
}

export const controlService = new ControlService();
