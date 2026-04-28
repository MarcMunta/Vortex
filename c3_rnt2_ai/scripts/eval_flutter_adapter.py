from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


SYSTEM = "You are Vortex, a direct Flutter/Dart programming assistant. Give concrete fixes, code, and validation."


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def call_runtime(base_url: str, prompt: str, *, model: str | None, timeout: float) -> str:
    payload = {
        "model": model or "vortex",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False,
    }
    response = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception:
        return json.dumps(data, ensure_ascii=True)


def score_output(output: str, rubric: list[str]) -> dict:
    text = output.lower()
    hits = []
    for item in rubric:
        words = [w.lower() for w in str(item).replace("/", " ").replace("-", " ").split() if len(w) > 3]
        hits.append(any(word in text for word in words))
    return {"rubric_hits": int(sum(1 for hit in hits if hit)), "rubric_total": len(rubric), "rubric_pass": bool(hits and all(hits))}


def render_report(rows: list[dict], profile: str, base_url: str) -> str:
    passed = sum(1 for row in rows if row.get("rubric_pass"))
    lines = [
        "# Flutter Adapter Eval",
        "",
        f"Profile: `{profile}`",
        f"Runtime URL: `{base_url}`",
        f"Samples: {len(rows)}",
        f"Rubric pass: {passed}/{len(rows)}",
        "",
        "## Criteria",
        "- Technical accuracy: manual review required.",
        "- Concrete fixes: check for code/actionable steps.",
        "- Generic advice avoidance: fail answers with vague summaries only.",
        "- Constraints/layout understanding: inspect RenderBox/RenderFlex prompts.",
        "- Flutter code compilability: manual or `flutter analyze` in target repo.",
        "- Mobile/web/desktop distinction: inspect adaptive prompts.",
        "- Codex prompt quality: must name files/tests/constraints.",
        "",
        "## Results",
    ]
    for idx, row in enumerate(rows, 1):
        lines.extend(
            [
                f"### {idx}. {row.get('prompt')}",
                f"- expected_topics: {', '.join(row.get('expected_topics') or [])}",
                f"- rubric_hits: {row.get('rubric_hits')}/{row.get('rubric_total')}",
                "",
                str(row.get("output") or "")[:1800],
                "",
            ]
        )
    return "\n".join(lines)


def run_eval(eval_path: Path, out_jsonl: Path, out_md: Path, *, profile: str, base_url: str, model: str | None, limit: int | None, timeout: float) -> dict:
    prompts = read_jsonl(eval_path)
    if limit:
        prompts = prompts[:limit]
    rows = []
    for item in prompts:
        prompt = str(item.get("prompt") or "")
        started = time.time()
        try:
            output = call_runtime(base_url, prompt, model=model, timeout=timeout)
            error = None
        except Exception as exc:
            output = ""
            error = str(exc)
        scored = score_output(output, [str(x) for x in item.get("rubric") or []])
        rows.append(
            {
                **item,
                "profile": profile,
                "output": output,
                "error": error,
                "latency_s": round(time.time() - started, 3),
                **scored,
            }
        )
    write_jsonl(out_jsonl, rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_report(rows, profile, base_url), encoding="utf-8")
    return {"ok": True, "samples": len(rows), "out_jsonl": str(out_jsonl), "out_md": str(out_md)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Flutter adapter via local Vortex OpenAI-compatible runtime.")
    parser.add_argument("--profile", default="rtx4080_16gb_programming_gemma4_local")
    parser.add_argument("--eval", default="config/datasets/flutter_official_hard_eval.jsonl")
    parser.add_argument("--base-url", default=os.getenv("VORTEX_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-jsonl", default="data/bench/flutter_eval_outputs.jsonl")
    parser.add_argument("--out-md", default="data/bench/flutter_eval_report.md")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_eval(
        Path(args.eval),
        Path(args.out_jsonl),
        Path(args.out_md),
        profile=str(args.profile),
        base_url=str(args.base_url),
        model=args.model,
        limit=args.limit,
        timeout=float(args.timeout),
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
