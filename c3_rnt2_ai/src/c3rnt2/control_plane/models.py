from __future__ import annotations

from pydantic import BaseModel, Field


class BootstrapRequest(BaseModel):
    force: bool = False
    mode: str = Field(default="ensure")


class TrainingStartRequest(BaseModel):
    mode: str = Field(default="quick")
    source: str | None = Field(default=None)


class TrainingResetRequest(BaseModel):
    clear_runs: bool = True
    clear_learning_queue: bool = True


class AllowlistRequest(BaseModel):
    domains: list[str] = Field(default_factory=list)


class AutonomyConfigRequest(BaseModel):
    enabled: bool | None = None
    reflection_enabled: bool | None = None
    training_enabled: bool | None = None
    autoedit_enabled: bool | None = None
    multi_agent_dialogue_enabled: bool | None = None
    descriptive_reports_enabled: bool | None = None
    live_autoedit_enabled: bool | None = None


class ObsidianConfigRequest(BaseModel):
    enabled: bool | None = None
    vault_path: str | None = None
