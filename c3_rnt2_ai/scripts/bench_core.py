from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings  # type: ignore[import-not-found]
from c3rnt2.model.bad_decode import bad_decode  # type: ignore[import-not-found]
from c3rnt2.model.core_transformer import CoreTransformer  # type: ignore[import-not-found]


def _rss_mb() -> float | None:
    # Best-effort RSS, no extra deps.
    try:
        if sys.platform.startswith("win"):
            import ctypes
            import ctypes.wintypes as wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            hproc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb) == 0:
                return None
            return float(counters.WorkingSetSize) / 1e6
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes.
        if rss > 10_000_000:
            return rss / 1e6
        return (rss * 1024.0) / 1e6
    except Exception:
        return None


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(len(data) - 1, f + 1)
    if f == c:
        return float(data[f])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)


def _bench_thresholds(settings: dict) -> dict:
    cfg = settings.get("bench", {}) or {}
    bench_thresholds = settings.get("bench_thresholds", {}) or {}
    autopilot = settings.get("autopilot", {}) or {}
    required_ctx = int(cfg.get("required_ctx", 0) or 0)
    max_vram = cfg.get("max_vram_peak_mb")
    try:
        max_vram = float(max_vram) if max_vram is not None else None
    except Exception:
        max_vram = None

    min_tps = cfg.get("min_tokens_per_sec", bench_thresholds.get("min_tokens_per_sec", autopilot.get("bench_min_tokens_per_sec", 0.0)))
    try:
        min_tps = float(min_tps) if min_tps is not None else 0.0
    except Exception:
        min_tps = 0.0

    max_reg_pct = cfg.get("max_regression_pct")
    if max_reg_pct is None:
        mr = bench_thresholds.get("max_regression", autopilot.get("bench_max_regression", 0.15))
        try:
            max_reg_pct = float(mr) * 100.0
        except Exception:
            max_reg_pct = 15.0
    else:
        try:
            max_reg_pct = float(max_reg_pct)
        except Exception:
            max_reg_pct = 15.0

    return {
        "required_ctx": required_ctx,
        "min_tokens_per_sec": float(min_tps),
        "max_regression_pct": float(max_reg_pct),
        "max_vram_peak_mb": max_vram,
    }


def _load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_baseline(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _prompt_for_ctx(core: CoreTransformer, ctx: int) -> tuple[str, int]:
    seed = "def f(x):\n    return x\n"
    ids, _total = core.encode_prompt(seed)
    if not ids:
        ids = [0]
    reps = (int(ctx) + len(ids) - 1) // len(ids)
    prompt_ids = (ids * max(1, reps))[: int(ctx)]
    prompt = core.decode_ids(prompt_ids, total_len=int(ctx))
    # Ensure stable ctx_len after decode+encode (Vortex-Tok should roundtrip).
    final_ids, _total2 = core.encode_prompt(prompt)
    return prompt, int(len(final_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    base_dir = Path(".").resolve()
    settings = load_settings(args.profile)
    profile_name = str(settings.get("_profile") or (args.profile or "dev_small"))
    raw_backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
    backend = "hf" if raw_backend == "hf" else "core"

    core = CoreTransformer.from_settings(settings)

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    prompt, ctx_len = _prompt_for_ctx(core, max(1, int(args.ctx)))

    lat_s: list[float] = []
    iters = max(1, int(args.iters))
    for _ in range(iters):
        start = time.time()
        _text, _stats = bad_decode(core, prompt=prompt, max_new_tokens=int(args.max_new_tokens))
        elapsed = max(1e-6, time.time() - start)
        lat_s.append(elapsed)

    avg = sum(lat_s) / max(1, len(lat_s))
    tokens_per_sec = float(args.max_new_tokens) / max(1e-6, avg)
    latency_p50_ms = _pct([x * 1000.0 for x in lat_s], 50.0)
    latency_p95_ms = _pct([x * 1000.0 for x in lat_s], 95.0)

    vram_peak_mb = None
    if torch.cuda.is_available():
        try:
            vram_peak_mb = float(torch.cuda.max_memory_allocated() / 1e6)
        except Exception:
            vram_peak_mb = None
    ram_rss_mb = _rss_mb()

    bench_dir = base_dir / "data" / "bench"
    baseline_path = bench_dir / "baseline.json"
    baseline = _load_baseline(baseline_path)

    thresholds = _bench_thresholds(settings)
    required_ctx = int(thresholds["required_ctx"])
    min_tps = float(thresholds["min_tokens_per_sec"])
    max_reg_pct = float(thresholds["max_regression_pct"])
    max_vram = thresholds.get("max_vram_peak_mb")

    base_entry = (baseline.get(profile_name, {}) or {}).get(backend) if isinstance(baseline, dict) else None
    baseline_tps = None
    if isinstance(base_entry, dict) and base_entry.get("tokens_per_sec") is not None:
        try:
            baseline_tps = float(base_entry.get("tokens_per_sec"))
        except Exception:
            baseline_tps = None

    regression_pct = None
    if baseline_tps is not None and baseline_tps > 0:
        regression_pct = max(0.0, (baseline_tps - tokens_per_sec) / baseline_tps * 100.0)

    baseline_missing = baseline_tps is None

    failures: list[str] = []
    warnings: list[str] = []

    if required_ctx and int(ctx_len) < required_ctx:
        failures.append("ctx_below_required")
    if max_vram is not None and vram_peak_mb is not None:
        try:
            if float(vram_peak_mb) > float(max_vram):
                failures.append("vram_peak_exceeded")
        except Exception:
            pass
    if regression_pct is not None and max_reg_pct > 0:
        if float(regression_pct) > float(max_reg_pct):
            failures.append("regression_exceeded")
    if baseline_tps is None and min_tps > 0 and tokens_per_sec < min_tps:
        warnings.append("below_min_tokens_per_sec_no_baseline")

    ok = not failures

    baseline_created = False
    if isinstance(baseline, dict):
        prof_entry = dict(baseline.get(profile_name, {}) or {})
        if backend not in prof_entry:
            eligible = ok and (min_tps <= 0 or tokens_per_sec >= min_tps)
            if eligible:
                prof_entry[backend] = {
                    "tokens_per_sec": float(tokens_per_sec),
                    "latency_p50_ms": float(latency_p50_ms),
                    "latency_p95_ms": float(latency_p95_ms),
                    "vram_peak_mb": vram_peak_mb,
                    "ram_rss_mb": ram_rss_mb,
                    "ctx": int(ctx_len),
                    "max_new_tokens": int(args.max_new_tokens),
                    "timestamp": time.time(),
                }
                baseline[profile_name] = prof_entry
                _write_baseline(baseline_path, baseline)
                baseline_created = True

    if baseline_missing:
        warnings.append("baseline_missing_created" if baseline_created else "baseline_missing")

    bench = {
        "ok": bool(ok),
        "profile": profile_name,
        "backend": backend,
        "ctx": int(ctx_len),
        "max_new_tokens": int(args.max_new_tokens),
        "tokens_per_sec": round(tokens_per_sec, 6),
        "latency_p50_ms": round(float(latency_p50_ms), 3),
        "latency_p95_ms": round(float(latency_p95_ms), 3),
        "vram_peak_mb": round(float(vram_peak_mb), 3) if vram_peak_mb is not None else None,
        "ram_rss_mb": round(float(ram_rss_mb), 3) if ram_rss_mb is not None else None,
        "timestamp": time.time(),
        "baseline_tokens_per_sec": baseline_tps,
        "regression_pct": round(float(regression_pct), 3) if regression_pct is not None else None,
        "thresholds": thresholds,
        "warnings": warnings or None,
        "failures": failures or None,
        "baseline_created": bool(baseline_created),
    }

    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "latest.json").write_text(json.dumps(bench, ensure_ascii=True, indent=2), encoding="utf-8")
    (bench_dir / "latest.txt").write_text(json.dumps(bench, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(bench, ensure_ascii=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
