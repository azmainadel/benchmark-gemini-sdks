"""Candidate 3: LiteLLM unified gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from app.adapters.base import CallOptions, thinking_config
from app.schemas import CallResult, Chunk, Usage

try:  # imported once at startup; not part of per-call latency
    import litellm
except ImportError:  # pragma: no cover - dependency guard
    litellm = None  # type: ignore[assignment]


def _sdk() -> Any:
    if litellm is None:  # pragma: no cover - dependency guard
        raise RuntimeError("litellm not installed. Run: uv sync")
    return litellm


def _routed(model: str) -> str:
    """LiteLLM routes on a provider/model string."""
    return model if "/" in model else f"gemini/{model}"


def _config(opts: CallOptions, model: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "temperature": opts.temperature,
        "max_tokens": opts.max_output_tokens,
        "thinking_config": thinking_config(model, opts.thinking_mode),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    completion = getattr(raw, "completion_tokens", None)
    reasoning = None
    details = getattr(raw, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
    return Usage(
        input_tokens=getattr(raw, "prompt_tokens", None),
        output_tokens=completion,
        total_tokens=getattr(raw, "total_tokens", None),
        reasoning_tokens=reasoning,
    )


def _text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0].message, "content", None) or ""


def _delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0].delta, "content", None) or ""


class LiteLLMAdapter:
    name = "litellm"
    label = "litellm.completion"

    def call(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        response = _sdk().completion(
            model=_routed(model), messages=_messages(prompt), **_config(opts, model)
        )
        return CallResult(
            text=_text(response), usage=_usage(getattr(response, "usage", None))
        )

    async def call_async(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        response = await _sdk().acompletion(
            model=_routed(model), messages=_messages(prompt), **_config(opts, model)
        )
        return CallResult(
            text=_text(response), usage=_usage(getattr(response, "usage", None))
        )

    def stream(self, prompt: str, model: str, opts: CallOptions) -> Iterator[Chunk]:
        stream = _sdk().completion(
            model=_routed(model),
            messages=_messages(prompt),
            stream=True,
            stream_options={"include_usage": True},
            **_config(opts, model),
        )
        for chunk in stream:
            text = _delta_text(chunk)
            usage = _usage(getattr(chunk, "usage", None))
            if usage is not None:
                yield Chunk(usage=usage)
            elif text:
                yield Chunk(text=text)

    async def stream_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> AsyncIterator[Chunk]:
        stream = await _sdk().acompletion(
            model=_routed(model),
            messages=_messages(prompt),
            stream=True,
            stream_options={"include_usage": True},
            **_config(opts, model),
        )
        async for chunk in stream:
            text = _delta_text(chunk)
            usage = _usage(getattr(chunk, "usage", None))
            if usage is not None:
                yield Chunk(usage=usage)
            elif text:
                yield Chunk(text=text)
