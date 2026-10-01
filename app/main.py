"""FastAPI surface: one endpoint per SDK/mode, every call recorded as JSONL."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from statistics import mean, median
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.adapters import get_adapter, registry
from app.adapters.base import Adapter, CallOptions, now_ms
from app.config import get_settings
from app.pricing import estimate_cost
from app.recorder import ResultRecorder
from app.schemas import CallRequest, CallResponse, RunRecord, Usage

settings = get_settings()
recorder = ResultRecorder(settings.results_file)

app = FastAPI(
    title="Gemini SDK benchmark",
    description=(
        "Sends the same one-shot prompt to the same Gemini model through four "
        "different SDKs, and records latency and token usage for each."
    ),
    version="0.1.0",
)


def _adapter(name: str) -> Adapter:
    try:
        return get_adapter(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _target(req: CallRequest) -> tuple[str, CallOptions]:
    return req.model or settings.model, CallOptions(
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
        thinking_mode=req.thinking_mode,
    )


def _cost_fields(
    sdk: str,
    model: str,
    usage: Usage | None,
    thinking_mode: Literal["on", "off"] | None,
) -> dict[str, Any]:
    return estimate_cost(
        sdk=sdk,
        model=model,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        reasoning_tokens=usage.reasoning_tokens if usage else None,
        thinking_mode=thinking_mode,
    ).as_dict()


def _record(
    sdk: str,
    mode: str,
    model: str,
    prompt: str,
    *,
    case_id: str | None = None,
    thinking_mode: Literal["on", "off"] | None = None,
    latency_ms: float | None = None,
    ttft_ms: float | None = None,
    client_init_ms: float | None = None,
    usage: Usage | None = None,
    output_chars: int | None = None,
    ok: bool = True,
    error: str | None = None,
) -> None:
    cost = _cost_fields(sdk, model, usage, thinking_mode)
    recorder.record(
        RunRecord(
            sdk=sdk,
            mode=mode,
            model=model,
            ok=ok,
            case_id=case_id,
            thinking_mode=thinking_mode,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            client_init_ms=client_init_ms,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
            **cost,
            output_chars=output_chars,
            error=error,
            prompt_chars=len(prompt),
        )
    )


def _response(
    sdk: str,
    mode: str,
    model: str,
    result: Any,
    latency_ms: float,
    case_id: str | None,
    thinking_mode: Literal["on", "off"],
):
    usage = result.usage
    text = result.text or ""
    return CallResponse(
        sdk=sdk,
        mode=mode,
        model=model,
        latency_ms=round(latency_ms, 2),
        case_id=case_id,
        thinking_mode=thinking_mode,
        client_init_ms=round(result.client_init_ms, 2)
        if result.client_init_ms is not None
        else None,
        output_chars=len(text),
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        reasoning_tokens=usage.reasoning_tokens if usage else None,
        **_cost_fields(sdk, model, usage, thinking_mode),
        output_preview=text[:280],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.model,
        "has_api_key": bool(settings.api_key),
    }


@app.get("/sdks")
def sdks() -> dict[str, Any]:
    return {"default_model": settings.model, "sdks": registry()}


@app.post("/sdk/{sdk}/call")
async def call(sdk: str, req: CallRequest) -> CallResponse:
    """Blocking one-shot call, executed on a worker thread."""
    adapter = _adapter(sdk)
    model, opts = _target(req)

    def run() -> tuple[Any, float]:
        t0 = now_ms()
        return adapter.call(req.prompt, model, opts), now_ms() - t0

    try:
        result, latency_ms = await asyncio.to_thread(run)
    except Exception as exc:
        _record(
            sdk,
            "sync",
            model,
            req.prompt,
            case_id=req.case_id,
            thinking_mode=req.thinking_mode,
            ok=False,
            error=repr(exc),
        )
        raise HTTPException(status_code=502, detail=repr(exc)) from exc

    _record(
        sdk,
        "sync",
        model,
        req.prompt,
        case_id=req.case_id,
        thinking_mode=req.thinking_mode,
        latency_ms=latency_ms,
        client_init_ms=result.client_init_ms,
        usage=result.usage,
        output_chars=len(result.text or ""),
    )
    return _response(
        sdk, "sync", model, result, latency_ms, req.case_id, req.thinking_mode
    )


@app.post("/sdk/{sdk}/call/async")
async def call_async(sdk: str, req: CallRequest) -> CallResponse:
    """Native async one-shot call on the event loop."""
    adapter = _adapter(sdk)
    model, opts = _target(req)
    t0 = now_ms()
    try:
        result = await adapter.call_async(req.prompt, model, opts)
    except Exception as exc:
        latency_ms = now_ms() - t0
        _record(
            sdk,
            "async",
            model,
            req.prompt,
            case_id=req.case_id,
            thinking_mode=req.thinking_mode,
            latency_ms=latency_ms,
            ok=False,
            error=repr(exc),
        )
        raise HTTPException(status_code=502, detail=repr(exc)) from exc
    latency_ms = now_ms() - t0

    _record(
        sdk,
        "async",
        model,
        req.prompt,
        case_id=req.case_id,
        thinking_mode=req.thinking_mode,
        latency_ms=latency_ms,
        client_init_ms=result.client_init_ms,
        usage=result.usage,
        output_chars=len(result.text or ""),
    )
    return _response(
        sdk, "async", model, result, latency_ms, req.case_id, req.thinking_mode
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_response(
    sdk: str,
    mode: str,
    model: str,
    prompt: str,
    case_id: str | None,
    thinking_mode: Literal["on", "off"],
    chunks: AsyncIterator,
) -> AsyncIterator[str]:
    """Time the stream and emit first_token / delta / done events."""
    start = now_ms()
    usage: Usage | None = None
    ttft_ms: float | None = None
    chars = 0
    error: str | None = None
    try:
        async for chunk in chunks:
            if chunk.text:
                if ttft_ms is None:
                    first_token_ms = now_ms() - start
                    ttft_ms = first_token_ms
                    yield _sse(
                        {"type": "first_token", "ttft_ms": round(first_token_ms, 2)}
                    )
                chars += len(chunk.text)
                yield _sse({"type": "delta", "text": chunk.text})
            if chunk.usage:
                usage = chunk.usage
    except Exception as exc:  # noqa: BLE001 - serialize SDK/provider failures as SSE
        error = repr(exc)
        yield _sse({"type": "error", "error": error})

    latency_ms = now_ms() - start
    _record(
        sdk,
        mode,
        model,
        prompt,
        case_id=case_id,
        thinking_mode=thinking_mode,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        usage=usage,
        output_chars=chars,
        ok=error is None,
        error=error,
    )
    if error is None:
        yield _sse(
            {
                "type": "done",
                "sdk": sdk,
                "mode": mode,
                "model": model,
                "case_id": case_id,
                "thinking_mode": thinking_mode,
                "latency_ms": round(latency_ms, 2),
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "output_chars": chars,
                "usage": usage.as_dict() if usage else None,
                "cost": _cost_fields(sdk, model, usage, thinking_mode),
            }
        )


def _sync_to_async(iterator: Iterator) -> AsyncIterator:
    return iterate_in_threadpool(iterator)


@app.post("/sdk/{sdk}/stream")
async def stream(sdk: str, req: CallRequest) -> StreamingResponse:
    """Blocking SDK, streamed over SSE."""
    adapter = _adapter(sdk)
    model, opts = _target(req)
    iterator = adapter.stream(req.prompt, model, opts)
    return StreamingResponse(
        _stream_response(
            sdk,
            "stream_sync",
            model,
            req.prompt,
            req.case_id,
            req.thinking_mode,
            _sync_to_async(iterator),
        ),
        media_type="text/event-stream",
    )


@app.post("/sdk/{sdk}/stream/async")
async def stream_async(sdk: str, req: CallRequest) -> StreamingResponse:
    """Native async SDK, streamed over SSE."""
    adapter = _adapter(sdk)
    model, opts = _target(req)
    return StreamingResponse(
        _stream_response(
            sdk,
            "stream_async",
            model,
            req.prompt,
            req.case_id,
            req.thinking_mode,
            adapter.stream_async(req.prompt, model, opts),
        ),
        media_type="text/event-stream",
    )


def _with_cost(row: dict[str, Any]) -> dict[str, Any]:
    if "estimated_cost_usd" not in row:
        row.update(
            estimate_cost(
                sdk=row["sdk"],
                model=row["model"],
                input_tokens=row.get("input_tokens"),
                output_tokens=row.get("output_tokens"),
                reasoning_tokens=row.get("reasoning_tokens"),
                thinking_mode=row.get("thinking_mode"),
            ).as_dict()
        )
    return row


@app.get("/results")
def results() -> dict[str, Any]:
    rows = [_with_cost(row) for row in recorder.read_all()]
    return {"count": len(rows), "runs": rows}


@app.get("/results/summary")
def summary() -> dict[str, Any]:
    """Aggregate latency and estimated costs by case, SDK and call mode."""
    all_rows = [_with_cost(row) for row in recorder.read_all()]
    buckets: dict[tuple[str, str, str | None, str | None], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (row["sdk"], row["mode"], row.get("case_id"), row.get("thinking_mode"))
        buckets.setdefault(key, []).append(row)

    out = []
    for (sdk, mode, case_id, thinking_mode), rows in buckets.items():
        ok_rows = [r for r in rows if r.get("ok")]
        lat = [r["latency_ms"] for r in ok_rows if r.get("latency_ms")]
        ttft = [r["ttft_ms"] for r in ok_rows if r.get("ttft_ms")]
        init = [r["client_init_ms"] for r in ok_rows if r.get("client_init_ms")]
        priced = [r for r in ok_rows if r.get("estimated_cost_usd") is not None]
        input_costs = [
            r["input_cost_usd"] for r in priced if r.get("input_cost_usd") is not None
        ]
        output_costs = [
            r["output_cost_usd"] for r in priced if r.get("output_cost_usd") is not None
        ]
        costs = [r["estimated_cost_usd"] for r in priced]
        out.append(
            {
                "sdk": sdk,
                "mode": mode,
                "case_id": case_id,
                "thinking_mode": thinking_mode,
                "runs": len(rows),
                "ok": len(ok_rows),
                "failed": len(rows) - len(ok_rows),
                "latency_mean_ms": round(mean(lat), 2) if lat else None,
                "latency_p50_ms": round(median(lat), 2) if lat else None,
                "latency_min_ms": round(min(lat), 2) if lat else None,
                "ttft_mean_ms": round(mean(ttft), 2) if ttft else None,
                "client_init_mean_ms": round(mean(init), 2) if init else None,
                "priced_runs": len(priced),
                "cost_coverage": f"{len(priced)}/{len(ok_rows)}",
                "input_cost_usd_total": round(sum(input_costs), 12)
                if input_costs
                else 0.0,
                "output_cost_usd_total": round(sum(output_costs), 12)
                if output_costs
                else 0.0,
                "estimated_cost_usd_total": round(sum(costs), 12) if costs else 0.0,
                "estimated_cost_usd_mean": round(mean(costs), 12) if costs else None,
                "output_tokens_mean": round(
                    mean([r["output_tokens"] for r in ok_rows if r.get("output_tokens")]),
                    1,
                )
                if any(r.get("output_tokens") for r in ok_rows)
                else None,
            }
        )

    def cost_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
        priced = [row for row in rows if row.get("estimated_cost_usd") is not None]
        by_sdk: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_sdk.setdefault(row["sdk"], []).append(row)
        return {
            "runs": len(rows),
            "priced_runs": len(priced),
            "missing_cost_runs": len(rows) - len(priced),
            "estimated_cost_usd": round(
                sum(row["estimated_cost_usd"] for row in priced), 12
            ),
            "by_sdk": [
                {
                    "sdk": sdk,
                    "runs": len(sdk_rows),
                    "priced_runs": sum(
                        row.get("estimated_cost_usd") is not None for row in sdk_rows
                    ),
                    "estimated_cost_usd": round(
                        sum(
                            row["estimated_cost_usd"]
                            for row in sdk_rows
                            if row.get("estimated_cost_usd") is not None
                        ),
                        12,
                    ),
                }
                for sdk, sdk_rows in sorted(by_sdk.items())
            ],
        }

    case_rows = [row for row in all_rows if row.get("case_id")]
    return {
        "summary": sorted(out, key=lambda r: (r["mode"], r["latency_mean_ms"] or 1e9)),
        "cost_totals": {
            "all_recorded": cost_totals(all_rows),
            "input_set_runs": cost_totals(case_rows),
        },
    }
