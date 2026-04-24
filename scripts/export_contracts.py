from __future__ import annotations

import argparse
import json
from pathlib import Path

from c3rnt2.control_plane.app import create_control_app
from c3rnt2.control_plane.dependencies import ControlDependencies


class _StubState:
    api_url = "http://127.0.0.1:8000"
    _active_run_id = None

    def status(self): return {"ok": True}
    def start_bootstrap(self, **_kwargs): return {"ok": True}
    def restart_runtime(self): return {"ok": True}
    def get_allowlist(self): return []
    def set_allowlist(self, domains): return domains
    def start_training(self, *_args, **_kwargs): return {"ok": True}
    def reset_training_state(self, **_kwargs): return {"ok": True}
    def list_runs(self, **_kwargs): return []
    def get_run(self, *_args, **_kwargs): return None
    def get_run_events(self, *_args, **_kwargs): return []
    def get_run_logs(self, *_args, **_kwargs): return {}
    def _build_training_stream_payload(self): return {"ts": 0.0}
    def runtime_status(self): return {"ok": True}
    def autonomy_status(self, **_kwargs): return {"enabled": False, "boot_mode": "manual", "state": "idle", "active_agents": [], "autoedit_scope": "safe"}
    def start_autonomy(self): return {"ok": True}
    def stop_autonomy(self): return {"ok": True}
    def configure_autonomy(self, _payload): return {"ok": True}
    def _latest_autonomy_events(self, **_kwargs): return []
    def voice_status(self): return {"ok": True}
    def restart_voice(self): return {"ok": True}
    def obsidian_status(self): return {"ok": True}
    def configure_obsidian(self, _payload): return {"ok": True}
    def multimodal_status(self): return {"ok": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if generated contract differs from committed file.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = root / "vortex-chat" / "public" / "contracts" / "control-openapi.json"
    app = create_control_app(ControlDependencies.from_state(_StubState()))
    rendered = json.dumps(app.openapi(), ensure_ascii=True, indent=2) + "\n"
    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != rendered:
            raise SystemExit(f"contract drift: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
