import type {
  AutonomyStatus,
  ChatSession,
  ControlStatus,
  MultimodalStatus,
  ObsidianStatus,
  OperationalStatus,
  SpatialSessionState,
  TrainingRunSummary,
  VoiceStatus,
} from "./types";

export interface ApiContracts {
  "/v1/status": { method: "GET"; response: OperationalStatus };
  "/v1/chat/sessions": { method: "GET" | "DELETE"; response: { ok: boolean; sessions?: ChatSession[] } };
  "/v1/chat/sessions/sync": { method: "POST"; response: { ok: boolean; sessions?: ChatSession[]; count?: number } };
  "/v1/spatial/session": { method: "GET" | "POST"; response: { ok: boolean; session?: SpatialSessionState } };
  "/v1/voice/status": { method: "GET"; response: VoiceStatus };
  "/v1/obsidian/status": { method: "GET"; response: ObsidianStatus };
}

export interface ControlContracts {
  "/control/status": { method: "GET"; response: ControlStatus };
  "/control/training/runs": { method: "GET"; response: { ok: boolean; runs: TrainingRunSummary[] } };
  "/control/autonomy/status": { method: "GET"; response: { ok: boolean; autonomy: AutonomyStatus } };
  "/control/multimodal/status": { method: "GET"; response: { ok: boolean; status: MultimodalStatus } };
}
