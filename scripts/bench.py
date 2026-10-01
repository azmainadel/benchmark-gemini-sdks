#!/usr/bin/env python3
"""Run every case in an input set against each SDK/mode, then print summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

SDK_NAMES = ("google_genai", "agno", "litellm", "langchain")
MODES = ("sync", "async", "stream", "stream/async")
DEFAULT_INPUT_FILE = (
    Path(__file__).resolve().parent.parent / "inputs" / "benchmark_cases.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--runs", type=int, default=1, help="Repetitions per input/SDK/mode"
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--prompt", help="Run one custom prompt instead of the input set")
    parser.add_argument("--thinking-mode", choices=("on", "off"), default="off")
    parser.add_argument("--model", help="Override the server's GEMINI_MODEL")
    parser.add_argument("--sdk", action="append", choices=SDK_NAMES)
    parser.add_argument("--mode", action="append", choices=MODES)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def load_inputs(path: Path) -> list[dict[str, str]]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read benchmark inputs from {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"Input {index} must be a JSON object")
        case_id = item.get("id")
        prompt = item.get("prompt")
        thinking_mode = item.get("thinking_mode")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Input {index} has a missing or duplicate id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Input {case_id!r} has an empty prompt")
        if thinking_mode not in {"on", "off"}:
            raise ValueError(f"Input {case_id!r} thinking_mode must be 'on' or 'off'")
        cases.append({"id": case_id, "prompt": prompt, "thinking_mode": thinking_mode})
        seen.add(case_id)
    return cases


def stream_call(
    client: httpx.Client, url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    final: dict[str, Any] = {}
    with client.stream("POST", url, json=payload, timeout=None) as response:
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.read().decode()}")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed SSE data: {line[6:]!r}") from exc
            if event.get("type") == "done":
                final = event
            elif event.get("type") == "error":
                raise RuntimeError(event.get("error", "stream failed"))
    if not final:
        raise RuntimeError("Stream ended without a done event")
    final.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 2))
    return final


def endpoint(url: str, mode: str) -> str:
    if mode == "sync":
        return f"{url}/call"
    if mode == "async":
        return f"{url}/call/async"
    if mode == "stream":
        return f"{url}/stream"
    return f"{url}/stream/async"


def show_summary(summary: Iterable[dict[str, Any]]) -> None:
    rows = list(summary)
    if not rows:
        print("No benchmark results recorded yet.")
        return
    print(
        "\nCASE  THINK  SDK  MODE  N  MEAN ms  P50 ms  TTFT ms  OUT TOK  COST USD  PRICED"
    )
    for row in rows:
        print(
            "  ".join(
                str(value)
                for value in (
                    row.get("case_id") or "legacy",
                    row.get("thinking_mode") or "-",
                    row["sdk"],
                    row["mode"],
                    f"{row['ok']}/{row['runs']}",
                    row.get("latency_mean_ms") or "-",
                    row.get("latency_p50_ms") or "-",
                    row.get("ttft_mean_ms") or "-",
                    row.get("output_tokens_mean") or "-",
                    f"${row['estimated_cost_usd_total']:.8f}"
                    if row.get("priced_runs")
                    else "-",
                    row.get("cost_coverage", "0/0"),
                )
            )
        )


def main() -> int:
    args = parse_args()
    sdks = args.sdk or list(SDK_NAMES)
    modes = args.mode or list(MODES)
    try:
        cases = (
            [{"id": "custom", "prompt": args.prompt, "thinking_mode": args.thinking_mode}]
            if args.prompt is not None
            else load_inputs(args.input_file)
        )
    except (TypeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    failures = 0
    with httpx.Client(base_url=args.url) as client:
        health = client.get("/health")
        health.raise_for_status()
        if not health.json().get("has_api_key"):
            print("Warning: server reports no Gemini API key configured", file=sys.stderr)

        for case in cases:
            payload: dict[str, Any] = {
                "prompt": case["prompt"],
                "case_id": case["id"],
                "thinking_mode": case["thinking_mode"],
            }
            if args.model:
                payload["model"] = args.model
            for sdk in sdks:
                for mode in modes:
                    path = endpoint(f"/sdk/{sdk}", mode)
                    for run in range(args.runs):
                        try:
                            if mode.startswith("stream"):
                                result = stream_call(client, path, payload)
                            else:
                                response = client.post(path, json=payload, timeout=None)
                                if response.is_error:
                                    raise RuntimeError(
                                        f"HTTP {response.status_code}: {response.text}"
                                    )
                                result = response.json()
                            print(
                                f"{case['id']} ({case['thinking_mode']}) "
                                f"{sdk}/{mode} run {run + 1}/{args.runs}: "
                                f"{result.get('latency_ms', '?')} ms"
                            )
                        except (httpx.HTTPError, RuntimeError) as exc:
                            failures += 1
                            print(
                                f"{case['id']} {sdk}/{mode} failed: {exc}",
                                file=sys.stderr,
                            )

        try:
            summary = client.get("/results/summary")
            summary.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Could not retrieve result summary: {exc}", file=sys.stderr)
            return 1
        summary_data = summary.json()
        show_summary(summary_data.get("summary", []))
        input_set_totals = summary_data.get("cost_totals", {}).get("input_set_runs", {})
        print(
            "\nInput-set estimated cost: "
            f"${input_set_totals.get('estimated_cost_usd', 0):.8f} USD "
            f"({input_set_totals.get('priced_runs', 0)}/{input_set_totals.get('runs', 0)} "
            "priced calls)"
        )
        for item in input_set_totals.get("by_sdk", []):
            print(
                f"  {item['sdk']}: ${item['estimated_cost_usd']:.8f} "
                f"({item['priced_runs']}/{item['runs']} priced)"
            )
    if failures:
        print(f"Benchmark completed with {failures} failed call(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
