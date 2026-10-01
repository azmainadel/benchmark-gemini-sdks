"""Candidate 1: Google's official SDK (`google-genai`)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from app.adapters.base import CallOptions, now_ms, thinking_config
from app.schemas import CallResult, Chunk, Usage

try:  # imported once at startup; not part of per-call latency
    from google import genai
except ImportError:  # pragma: no cover - dependency guard
    genai = None  # type: ignore[assignment]


def _sdk() -> Any:
    if genai is None:  # pragma: no cover - dependency guard
        raise RuntimeError("google-genai not installed. Run: uv sync")
    return genai


def _config(opts: CallOptions, model: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "temperature": opts.temperature,
        "max_output_tokens": opts.max_output_tokens,
        "thinking_config": thinking_config(model, opts.thinking_mode),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def _usage(metadata: Any) -> Usage | None:
    if metadata is None:
        return None
    return Usage(
        input_tokens=getattr(metadata, "prompt_token_count", None),
        output_tokens=getattr(metadata, "candidates_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
        reasoning_tokens=getattr(metadata, "thoughts_token_count", None),
    )


def _text(response: Any) -> str:
    return getattr(response, "text", None) or ""


class GoogleGenAIAdapter:
    name = "google_genai"
    label = "google-genai (official)"

    def call(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        sdk = _sdk()
        t0 = now_ms()
        client = sdk.Client()
        init_ms = now_ms() - t0
        response = client.models.generate_content(
            model=model, contents=prompt, config=_config(opts, model)
        )
        return CallResult(
            text=_text(response),
            usage=_usage(getattr(response, "usage_metadata", None)),
            client_init_ms=init_ms,
        )

    async def call_async(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        sdk = _sdk()
        t0 = now_ms()
        client = sdk.Client()
        init_ms = now_ms() - t0
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=prompt, config=_config(opts, model)
            )
        finally:
            await client.aio.aclose()
        return CallResult(
            text=_text(response),
            usage=_usage(getattr(response, "usage_metadata", None)),
            client_init_ms=init_ms,
        )

    def stream(self, prompt: str, model: str, opts: CallOptions) -> Iterator[Chunk]:
        sdk = _sdk()
        client = sdk.Client()
        stream = client.models.generate_content_stream(
            model=model, contents=prompt, config=_config(opts, model)
        )
        last: Any = None
        for event in stream:
            last = event
            text = _text(event)
            if text:
                yield Chunk(text=text)
        yield Chunk(usage=_usage(getattr(last, "usage_metadata", None)))

    async def stream_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> AsyncIterator[Chunk]:
        sdk = _sdk()
        client = sdk.Client()
        try:
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=prompt, config=_config(opts, model)
            )
            last: Any = None
            async for event in stream:
                last = event
                text = _text(event)
                if text:
                    yield Chunk(text=text)
            yield Chunk(usage=_usage(getattr(last, "usage_metadata", None)))
        finally:
            await client.aio.aclose()
