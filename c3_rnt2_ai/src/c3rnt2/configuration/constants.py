from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
DEFAULT_PROFILE = "rtx4080_16gb_gemma4_26b_a4b_hf"


def resolve_profile(profile: str | None = None) -> str:
    env_profile = os.getenv("C3RNT2_PROFILE")
    return profile or env_profile or DEFAULT_PROFILE
