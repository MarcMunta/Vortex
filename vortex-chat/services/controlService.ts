import {
  AutonomyStreamPayload,
  AutonomyStatus,
  MultimodalStatus,
  MultimodalStreamPayload,
  TrainingRunSummary,
  TrainingStreamPayload,
} from "../types";
import { ControlContractClient } from "../contracts.generated";
import { parseEventData, requestJson } from "./apiClient";

const resolveBaseUrl = (): string => {
  const raw = (import.meta.env.VITE_CONTROL_BASE_URL || "").trim();
  if (raw) return raw.replace(/\/+$/, "");
  const port = (import.meta.env.VITE_CONTROL_PORT || "8765").trim() || "8765";
  const host = typeof window !== "undefined" ? (window.location.hostname || "127.0.0.1") : "127.0.0.1";
  return `http://${host}:${port}`;
};

class ControlService {
  private readonly baseUrl = resolveBaseUrl();
  private readonly client = new ControlContractClient(this.baseUrl, requestJson);

  async fetchStatus() {
    try {
      return await this.client.get("GET /control/status");
    } catch {
      return null;
    }
  }

  async bootstrap(
    force: boolean = false,
    mode: "ensure" | "rebuild" = (force ? "rebuild" : "ensure"),
  ) {
    return this.client.post("POST /control/bootstrap", { force, mode });
  }

  async initModel() {
    return this.client.post("POST /control/model/init");
  }

  async restartRuntime() {
    return this.client.post("POST /control/runtime/restart");
  }

  async reloadInstructions() {
    return this.client.post("POST /control/instructions/reload");
  }

  async getAllowlist(): Promise<string[]> {
    const payload = await this.client.get("GET /control/internet/allowlist");
    return Array.isArray(payload.domains) ? payload.domains : [];
  }

  async saveAllowlist(domains: string[]): Promise<string[]> {
    const payload = await this.client.post("POST /control/internet/allowlist", { domains });
    return Array.isArray(payload.domains) ? payload.domains : [];
  }

  async getMultimodalStatus(): Promise<MultimodalStatus | null> {
    try {
      const payload = await this.client.get("GET /control/multimodal/status");
      return payload.status || null;
    } catch {
      return null;
    }
  }

  async getVoiceStatus() {
    return this.client.get("GET /control/voice/status");
  }

  async restartVoice() {
    return this.client.post("POST /control/voice/restart");
  }

  async getObsidianStatus() {
    return this.client.get("GET /control/obsidian/status");
  }

  async configureObsidian(payload: { enabled?: boolean; vault_path?: string }) {
    return this.client.post("POST /control/obsidian/config", payload);
  }

  async startTraining(
    mode: "quick" | "full",
    source?: string,
  ) {
    return this.client.post("POST /control/training/start", { mode, source });
  }

  async resetTrainingState(payload?: { clear_runs?: boolean; clear_learning_queue?: boolean }) {
    return this.client.post("POST /control/training/reset", {
        clear_runs: payload?.clear_runs ?? true,
        clear_learning_queue: payload?.clear_learning_queue ?? true,
    });
  }

  async getTrainingRuns(): Promise<TrainingRunSummary[]> {
    const payload = await this.client.get("GET /control/training/runs");
    return Array.isArray(payload.runs) ? payload.runs : [];
  }

  async getTrainingRun(runId: string): Promise<TrainingRunSummary | null> {
    try {
      const payload = await this.client.get("GET /control/training/runs/{run_id}", { run_id: runId });
      return payload.run || null;
    } catch {
      return null;
    }
  }

  async getTrainingRunEvents(runId: string) {
    const payload = await this.client.get("GET /control/training/runs/{run_id}/events", { run_id: runId });
    return Array.isArray(payload.events) ? payload.events : [];
  }

  async getTrainingRunLogs(runId: string) {
    const payload = await this.client.get("GET /control/training/runs/{run_id}/logs", { run_id: runId });
    return payload.logs || {};
  }

  subscribeTrainingStream(
    onMessage: (payload: TrainingStreamPayload) => void,
    onError?: (error: Event | Error) => void,
  ): () => void {
    const source = this.client.stream("GET /control/training/stream");
    source.onmessage = (event) => {
      try {
        const payload = parseEventData<TrainingStreamPayload>(event.data, "training_stream_parse_failed");
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
      const payload = await this.client.get("GET /control/autonomy/status");
      return payload.autonomy || null;
    } catch {
      return null;
    }
  }

  async startAutonomy() {
    return this.client.post("POST /control/autonomy/start");
  }

  async stopAutonomy() {
    return this.client.post("POST /control/autonomy/stop");
  }

  async configureAutonomy(config: {
    enabled?: boolean;
    reflection_enabled?: boolean;
    training_enabled?: boolean;
    autoedit_enabled?: boolean;
    multi_agent_dialogue_enabled?: boolean;
    descriptive_reports_enabled?: boolean;
    live_autoedit_enabled?: boolean;
  }) {
    return this.client.post("POST /control/autonomy/config", config);
  }

  subscribeAutonomyStream(
    onMessage: (payload: AutonomyStreamPayload) => void,
    onError?: (error: Event | Error) => void,
  ): () => void {
    const source = this.client.stream("GET /control/autonomy/stream");
    source.onmessage = (event) => {
      try {
        const payload = parseEventData<AutonomyStreamPayload>(event.data, "autonomy_stream_parse_failed");
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

  subscribeMultimodalStream(
    onMessage: (payload: MultimodalStreamPayload) => void,
    onError?: (error: Event | Error) => void,
  ): () => void {
    const source = this.client.stream("GET /control/multimodal/stream");
    source.onmessage = (event) => {
      try {
        const payload = parseEventData<MultimodalStreamPayload>(event.data, "multimodal_stream_parse_failed");
        onMessage(payload);
      } catch (error) {
        onError?.(error instanceof Error ? error : new Error("multimodal_stream_parse_failed"));
      }
    };
    source.onerror = (event) => {
      onError?.(event);
    };
    return () => source.close();
  }
}

export const controlService = new ControlService();
