from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


LineCallback = Callable[[str], None]


@dataclass(frozen=True)
class RuntimeCommandService:
    base_dir: Path
    compose_file: Path
    api_profile: str
    training_profile: str
    compose_actions_enabled: bool

    def compose_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("VORTEX_API_PROFILE", self.api_profile)
        if extra:
            env.update(extra)
        return env

    def compose_cmd_prefix(self) -> list[str]:
        docker_bin = shutil.which("docker")
        if docker_bin is not None:
            try:
                probe = subprocess.run(
                    [docker_bin, "compose", "version"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                    env=self.compose_env(),
                )
                if probe.returncode == 0:
                    return ["docker", "compose"]
            except OSError:
                pass
        if shutil.which("docker-compose") is not None:
            return ["docker-compose"]
        return ["docker", "compose"]

    def compose_cmd(self, *args: str) -> list[str]:
        return [*self.compose_cmd_prefix(), "-f", str(self.compose_file), *args]

    def run_compose(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        line_callback: LineCallback | None = None,
    ) -> tuple[int, str]:
        return self._run_process(
            self.compose_cmd(*args),
            env=self.compose_env(env),
            log_path=log_path,
            line_callback=line_callback,
        )

    def should_use_local_job_runner(self) -> bool:
        return (not self.compose_actions_enabled) or shutil.which("docker") is None

    def run_local_command(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        line_callback: LineCallback | None = None,
    ) -> tuple[int, str]:
        child_env = dict(os.environ)
        if env:
            child_env.update(env)
        return self._run_process(
            cmd,
            env=child_env,
            log_path=log_path,
            line_callback=line_callback,
        )

    def run_local_training_job(
        self,
        *,
        mode: str,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        parallel_runtime_training: bool = False,
        line_callback: LineCallback | None = None,
    ) -> tuple[int, str]:
        cmd = [sys.executable, "-m", "c3rnt2", "train-once", "--profile", self.training_profile]
        if mode == "quick":
            cmd.append("--reuse-dataset")
        if parallel_runtime_training:
            cmd.append("--allow-parallel-runtime")
        return self.run_local_command(
            cmd,
            env=env,
            log_path=log_path,
            line_callback=line_callback,
        )

    def _run_process(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        log_path: Path | None,
        line_callback: LineCallback | None,
    ) -> tuple[int, str]:
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.base_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        lines: list[str] = []
        sink = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = log_path.open("a", encoding="utf-8")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = line.rstrip()
                lines.append(clean)
                if sink is not None:
                    sink.write(line)
                    sink.flush()
                if line_callback is not None:
                    try:
                        line_callback(clean)
                    except Exception:
                        pass
        finally:
            if sink is not None:
                sink.close()
        return int(proc.wait()), "\n".join(lines)
