
export enum Role {
  USER = 'user',
  AI = 'ai'
}

export type ViewType = 'chat' | 'spatial' | 'analysis' | 'training' | 'edits' | 'terminal';
export type AppMode = 'ask' | 'agent';
export type FontSize = 'small' | 'medium' | 'large';
export type Language = 'es' | 'en';
export type PermissionLevel = 'none' | 'full';
export type PermissionActionMode = 'safe' | 'full';
export type SpatialPanelKind = 'note' | 'presentation' | 'browser' | 'image' | 'obsidian' | 'sketch' | 'pdf';

export interface WorkspacePermissions {
  level: PermissionLevel;
  workspaceRoot: string;
  projectPath: string;
  actionMode: PermissionActionMode;
}

export interface BrowserAction {
  target: string;
  opened?: boolean;
}

export interface SpatialTransform {
  x: number;
  y: number;
  z: number;
  scale: number;
  rotation: number;
  skew_x: number;
  skew_y: number;
  tilt_x: number;
  tilt_y: number;
  perspective: number;
  width: number;
  height: number;
}

export interface SpatialRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SpatialPanelModel {
  id: string;
  type: SpatialPanelKind;
  title: string;
  content: string;
  source?: Record<string, unknown>;
  transform: SpatialTransform;
  page_index: number;
  page_count: number;
  selected?: boolean;
  locked?: boolean;
  created_at: number;
  updated_at: number;
}

export interface SpatialSessionState {
  session_id: string;
  selected_object_id: string | null;
  selected_region: SpatialRegion | null;
  active_panel_ids: string[];
  active_presentation_id: string | null;
  active_page_index: number;
  interaction_mode: string;
  last_voice_command?: string | null;
  last_gesture_event?: Record<string, unknown> | null;
  camera_state?: Record<string, unknown> | null;
  gesture_state?: Record<string, unknown> | null;
  focused_item?: Record<string, unknown> | null;
  recent_multimodal_summary?: string | null;
  panels: SpatialPanelModel[];
  updated_at: number;
  created_at: number;
}

export interface VoiceIntent {
  kind: string;
  panel_id?: string | null;
  query?: string;
  target?: string;
  panel_type?: string;
  delta?: number;
  transform?: Record<string, number>;
}

export interface VoiceStatus {
  ok: boolean;
  enabled: boolean;
  push_to_talk?: boolean;
  vad_enabled?: boolean;
  whisper_model?: string;
  tts_model?: string;
  asr_backend?: string;
  tts_backend?: string;
  asr_available?: boolean;
  tts_available?: boolean;
  output_dir?: string;
  error?: string;
}

export interface VoiceTranscriptionResult {
  ok: boolean;
  transcript?: string;
  detected_language?: string | null;
  intent?: VoiceIntent | null;
  action_result?: Record<string, unknown> | null;
  stored_path?: string | null;
  error?: string;
}

export interface ObsidianStatus {
  ok: boolean;
  enabled: boolean;
  vault_path?: string | null;
  resolved_vault_path?: string | null;
  available?: boolean;
  validated?: boolean;
  folders?: Record<string, string>;
  last_saved_note?: string | null;
  error?: string;
}

export interface MultimodalStatus {
  ok: boolean;
  voice?: VoiceStatus;
  spatial?: SpatialSessionState | null;
  camera?: Record<string, unknown> | null;
  gesture?: Record<string, unknown> | null;
  obsidian?: ObsidianStatus;
  fusion?: {
    enabled?: boolean;
    summary?: string | null;
    refs?: Array<Record<string, unknown>>;
  };
}

export interface MultimodalStreamPayload {
  ts: number;
  status: MultimodalStatus;
}

export interface LocalAccount {
  id: string;
  name: string;
  email: string;
  handle: string;
  avatarHue: number;
  createdAt: number;
  lastUsedAt: number;
}

export interface Source {
  title: string;
  url: string;
  domain: string;
  kind: 'web' | 'file' | 'unknown';
  index: number;
}

export interface GroundingSupport {
  segmentText: string;
  startIndex: number;
  endIndex: number;
  sourceIndices: number[];
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  thought?: string;
  requestId?: string;
  trainingEvent?: boolean;
  learningStatus?: 'queued' | 'scheduled' | 'consumed' | 'skipped' | string;
  learningQueueId?: string;
  learningRunId?: string;
  sources?: Source[];
  groundingSupports?: GroundingSupport[];
  timestamp: number;
  fileChanges?: { path: string; diff: string }[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

export interface UserSettings {
  categoryOrder: string[];
  codeTheme: 'dark' | 'light' | 'match-app';
  fontSize: FontSize;
  language: Language;
  permissions: WorkspacePermissions;
}

export interface LogEntry {
  id: string;
  timestamp: number;
  level: 'INFO' | 'LEARN' | 'SEARCH' | 'SYSTEM';
  message: string;
}

export interface OperationalStatus {
  ok: boolean;
  chat_ready?: boolean;
  chat_mode?: 'primary' | 'fallback_degraded' | 'unavailable' | string | null;
  chat_block_reason?: string | null;
  offline_ready: boolean;
  engine_ready: boolean;
  engine_kind?: string | null;
  engine_base_url?: string | null;
  model_ready: boolean;
  active_backend?: string | null;
  active_model?: string | null;
  training_ready: boolean;
  web_disabled: boolean;
  docker_ready?: boolean;
  degraded_reason?: string | null;
  offline_reason?: string | null;
  engine_reason?: string | null;
  model_reason?: string | null;
  training_reason?: string | null;
  docker_reason?: string | null;
  wsl_ready?: boolean;
  wsl_reason?: string | null;
  runtime_mode?: string | null;
  fallback_active?: boolean;
  fallback_backend?: string | null;
  instructions?: {
    digest?: string | null;
    sources?: string[];
  };
}

export interface TrainingRunEvent {
  id: string;
  ts: number;
  run_id: string;
  phase: string;
  kind: string;
  message: string;
  latest_metrics?: Record<string, unknown>;
  progress_pct?: number | null;
  metadata?: Record<string, unknown>;
}

export interface TrainingDialogueTurn {
  id: string;
  speaker: 'analyst' | 'builder' | 'system' | string;
  speaker_label?: string;
  kind?: string;
  ts?: number;
  message: string;
  cycle_id?: string | null;
}

export interface TrainingReviewSection {
  key: string;
  title: string;
  content: string;
}

export interface TrainingNotebookSection {
  id: string;
  phase: string;
  kind: string;
  title: string;
  content: string;
  ts?: number;
  metadata?: Record<string, unknown>;
}

export interface TrainingMetricPoint {
  ts: number;
  phase: string;
  metrics: Record<string, unknown>;
}

export interface TrainingGateResults {
  manual_only?: boolean;
  promoted?: boolean;
  eval_ok?: boolean;
  bench_ok?: boolean;
  smoke_check_required?: boolean;
  repo_clean_for_autoedit?: boolean;
  autoedit_scope_ok?: boolean;
  smoke_ok?: boolean;
  smoke_waited_s?: number | null;
  [key: string]: unknown;
}

export interface TrainingApplyResult {
  applied?: boolean;
  decision?: string;
  adapter_path?: string;
  requested_adapter_path?: string;
  queued_reload?: boolean;
  error?: string | null;
  reload?: Record<string, unknown>;
  smoke?: Record<string, unknown>;
}

export interface TrainingRollbackResult {
  ok?: boolean;
  reason?: string;
  requested_adapter_path?: string;
  request?: Record<string, unknown>;
  smoke?: Record<string, unknown>;
}

export interface TrainingComparison {
  parent_run_id?: string | null;
  outcome?: string | null;
  summary?: string | null;
  source_mix_delta?: Record<string, number>;
  apply_changed?: boolean;
  verification_changed?: boolean;
}

export interface TrainingCampaignSummary {
  campaign_id: string;
  objective?: string;
  started_at?: number;
  run_count?: number;
  completed_count?: number;
  rolled_back_count?: number;
  degraded_count?: number;
  active_run_id?: string | null;
  success_streak?: number;
  failure_streak?: number;
  throughput_per_hour?: number;
  last_apply?: TrainingApplyResult | Record<string, unknown> | null;
  last_rollback?: TrainingRollbackResult | Record<string, unknown> | null;
}

export interface TrainingRunSummary {
  run_id: string;
  mode: 'quick' | 'full' | string;
  status: string;
  stage?: string;
  lifecycle_state?: string;
  created_at?: number;
  updated_at?: number;
  profile?: string;
  base_model?: string;
  served_model?: string;
  dataset_hash?: string;
  adapter_dir?: string;
  log_path?: string;
  eval_log_path?: string;
  bench_log_path?: string;
  promotion?: {
    manual_only?: boolean;
    decision?: string;
    eval_ok?: boolean;
    bench_ok?: boolean;
  };
  train_result?: Record<string, unknown>;
  eval_result?: Record<string, unknown>;
  bench_result?: Record<string, unknown>;
  runtime_mode?: string | null;
  fallback_active?: boolean;
  fallback_backend?: string | null;
  progress_pct?: number;
  execution_progress_pct?: number;
  pipeline_progress_pct?: number;
  queue_reason?: string | null;
  blocked_reason?: string | null;
  blocked_since?: number | null;
  retry_in_s?: number | null;
  next_run_scheduled_at?: number | null;
  queue_diagnostics?: {
    blocking_roles?: string[];
    lock_errors?: string[];
    runtime?: Record<string, unknown>;
    [key: string]: unknown;
  } | null;
  latest_metrics?: Record<string, unknown>;
  artifacts?: Record<string, string>;
  failure?: Record<string, unknown> | null;
  dataset_manifest?: {
    queued_count?: number;
    consumed_count?: number;
    quick_threshold?: number;
    source_kinds?: Record<string, number>;
    request_ids?: string[];
    items?: Array<Record<string, unknown>>;
  };
  latest_event?: TrainingRunEvent | null;
  events?: TrainingRunEvent[];
  log_tail?: string[];
  logs?: Record<string, string[]>;
  display_name?: string;
  display_description?: string;
  objective?: string;
  learning_focus?: string[];
  campaign_id?: string | null;
  parent_run_id?: string | null;
  attempt?: number;
  run_lineage?: string[];
  agent_dialogue?: TrainingDialogueTurn[];
  review_sections?: TrainingReviewSection[];
  notebook_sections?: TrainingNotebookSection[];
  live_metrics_series?: TrainingMetricPoint[];
  gate_results?: TrainingGateResults | null;
  apply_result?: TrainingApplyResult | null;
  rollback_result?: TrainingRollbackResult | null;
  source_mix?: Record<string, number>;
  terminal_reason?: string | null;
  comparison?: TrainingComparison | null;
}

export interface TrainingStreamPayload {
  ts: number;
  active_run_id?: string | null;
  active_run?: TrainingRunSummary | null;
  phase?: string | null;
  progress_pct?: number | null;
  execution_progress_pct?: number | null;
  pipeline_progress_pct?: number | null;
  latest_metrics?: Record<string, unknown>;
  runtime_mode?: string | null;
  fallback_active?: boolean;
  fallback_backend?: string | null;
  last_event?: TrainingRunEvent | null;
  log_tail?: string[];
  campaign?: TrainingCampaignSummary | null;
  next_run_scheduled_at?: number | null;
  scheduled_followup_reason?: string | null;
  pipeline_runs?: TrainingRunSummary[];
  blocked_runs?: TrainingRunSummary[];
  runs?: TrainingRunSummary[];
}

export interface AutonomyAgentStatus {
  id: string;
  name: string;
  role: string;
  status: string;
  accent?: 'ask' | 'agent' | 'neutral';
  last_event_at?: number | null;
}

export interface AutonomyRollbackState {
  ts?: number | null;
  status?: string | null;
  target?: string | null;
  reason?: string | null;
}

export interface TrainingOutcomeSummary {
  run_id?: string | null;
  mode?: string | null;
  status?: string | null;
  stage?: string | null;
  updated_at?: number | null;
  reason?: string | null;
}

export interface AutonomyEvent {
  id: string;
  ts: number;
  agent: 'analyst' | 'builder' | 'system';
  kind: string;
  title: string;
  detail: string;
  cycle_id?: string | null;
  state?: string | null;
  metadata?: Record<string, unknown>;
}

export interface AutonomyStatus {
  enabled: boolean;
  boot_mode: string;
  state: string;
  active_agents: AutonomyAgentStatus[];
  current_cycle?: string | null;
  last_reflection_at?: number | null;
  last_train_at?: number | null;
  last_patch_at?: number | null;
  autoedit_scope: string;
  last_rollback?: AutonomyRollbackState | null;
  training_queue?: string[];
  next_cycle_at?: number | null;
  next_run_scheduled_at?: number | null;
  scheduled_run_mode?: string | null;
  scheduled_parent_run_id?: string | null;
  scheduled_followup_reason?: string | null;
  maintenance_mode?: boolean;
  runtime_drained_for_training?: boolean;
  current_dataset_mix?: Record<string, number>;
  campaign?: TrainingCampaignSummary | null;
  blocked_run_count?: number;
  blocked_runs?: TrainingRunSummary[];
  last_eval_summary?: Record<string, unknown> | null;
  last_bench_summary?: Record<string, unknown> | null;
  last_training_outcome?: TrainingOutcomeSummary | null;
  config?: {
    reflection_enabled?: boolean;
    training_enabled?: boolean;
    autoedit_enabled?: boolean;
    multi_agent_dialogue_enabled?: boolean;
    descriptive_reports_enabled?: boolean;
    live_autoedit_enabled?: boolean;
    reflection_interval_s?: number;
    quick_train_interval_s?: number;
    full_train_interval_s?: number;
    autoedit_interval_s?: number;
  };
  latest_events?: AutonomyEvent[];
  latest_dialogue?: TrainingDialogueTurn[];
}

export interface AutonomyStreamPayload {
  ts: number;
  status: AutonomyStatus;
  events: AutonomyEvent[];
  active_run_id?: string | null;
  runs?: TrainingRunSummary[];
}

export interface ControlStatus {
  ok: boolean;
  bootstrap?: {
    running?: boolean;
    stage?: string;
    message?: string;
    updated_at?: number;
    error?: unknown;
    tail?: string[];
  };
  docker?: {
    ready?: boolean;
    reason?: string;
    detail?: string;
    server_version?: string | null;
  };
  model?: {
    model_id?: string;
    cache_dir?: string;
    repo_dir?: string;
    cached?: boolean;
    snapshot_count?: number;
    last_snapshot?: string | null;
  };
  runtime?: {
    api_ready?: boolean;
    runtime_ready?: boolean;
    readyz?: { ok?: boolean };
    runtime_mode?: string | null;
    fallback_active?: boolean;
    fallback_backend?: string | null;
    status?: OperationalStatus | null;
  };
  frontend?: {
    ready?: boolean;
    port?: number;
    url?: string;
  };
  internet?: {
    allowlist?: string[];
  };
  learning_queue?: {
    queued_count?: number;
    quick_threshold?: number;
    quick_cooldown_s?: number;
    last_quick_dispatch_at?: number | null;
    last_quick_dispatch_run_id?: string | null;
    items?: Array<{
      id?: string;
      request_id?: string;
      source_kind?: string;
      score?: number;
      status?: string;
      queued_at?: number;
      consumed_by?: string | null;
    }>;
  };
  instructions?: {
    digest?: string | null;
    sources?: string[];
  };
  multimodal?: MultimodalStatus;
  autonomy?: AutonomyStatus;
  active_run_id?: string | null;
  runs?: TrainingRunSummary[];
}
