"""Candidate 2: Agno agent framework, one-shot via `Agent.run()`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from app.adapters.base import CallOptions, now_ms, thinking_config
from app.config import get_settings
from app.schemas import CallResult, Chunk, Usage

try:  # imported once at startup; not part of per-call latency
    from agno.agent import Agent
    from agno.models.google import Gemini as AgnoGemini
except ImportError:  # pragma: no cover - dependency guard
    Agent = None  # type: ignore[assignment,misc]
    AgnoGemini = None  # type: ignore[assignment,misc]


def _agent(model: str, opts: CallOptions) -> Any:
    if Agent is None or AgnoGemini is None:  # pragma: no cover - guard
        raise RuntimeError("agno not installed. Run: uv sync")
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "id": model,
        "api_key": settings.api_key,
    }
    if opts.temperature is not None:
        kwargs["temperature"] = opts.temperature
    if opts.max_output_tokens is not None:
        kwargs["max_output_tokens"] = opts.max_output_tokens
    kwargs.update(thinking_config(model, opts.thinking_mode))
    return Agent(model=AgnoGemini(**kwargs), telemetry=False)


def _usage(metrics: Any) -> Usage | None:
    if metrics is None:
        return None
    return Usage(
        input_tokens=getattr(metrics, "input_tokens", None),
        output_tokens=getattr(metrics, "output_tokens", None),
        total_tokens=getattr(metrics, "total_tokens", None),
        reasoning_tokens=getattr(metrics, "reasoning_tokens", None),
    )


def _event_usage(event: Any) -> Usage | None:
    """RunContentEvent carries metrics too; only use the completed event."""
    if getattr(event, "event", "") != "RunCompleted":
        return None
    return _usage(getattr(event, "metrics", None))


class AgnoAdapter:
    name = "agno"
    label = "agno agent (Agent.run)"

    def call(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        t0 = now_ms()
        agent = _agent(model, opts)
        init_ms = now_ms() - t0
        run = agent.run(prompt)
        return CallResult(
            text=getattr(run, "content", None) or "",
            usage=_usage(getattr(run, "metrics", None)),
            client_init_ms=init_ms,
        )

    async def call_async(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        t0 = now_ms()
        agent = _agent(model, opts)
        init_ms = now_ms() - t0
        run = await agent.arun(prompt)
        return CallResult(
            text=getattr(run, "content", None) or "",
            usage=_usage(getattr(run, "metrics", None)),
            client_init_ms=init_ms,
        )

    def stream(self, prompt: str, model: str, opts: CallOptions) -> Iterator[Chunk]:
        agent = _agent(model, opts)
        for event in agent.run(prompt, stream=True):
            usage = _event_usage(event)
            text = getattr(event, "content", None)
            if usage is not None:
                yield Chunk(usage=usage)
            elif text:
                yield Chunk(text=text)

    async def stream_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> AsyncIterator[Chunk]:
        agent = _agent(model, opts)
        async for event in agent.arun(prompt, stream=True):
            usage = _event_usage(event)
            text = getattr(event, "content", None)
            if usage is not None:
                yield Chunk(usage=usage)
            elif text:
                yield Chunk(text=text)
