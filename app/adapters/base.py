"""Adapter contract: every SDK is normalized to the same four call shapes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol, runtime_checkable

from app.schemas import CallResult, Chunk


def now_ms() -> float:
    """Monotonic clock in milliseconds."""
    return perf_counter() * 1000


def thinking_config(model: str, mode: Literal["on", "off"]) -> dict[str, object]:
    """Translate the shared thinking setting into Gemini's model-specific API.

    A zero thinking budget disables thinking. Gemini 3 uses a thinking level
    for the enabled case; Gemini 2.x uses a bounded thinking-token budget.
    """
    if mode == "off":
        return {"thinking_budget": 0}
    if model.lower().startswith("gemini-3"):
        return {"thinking_level": "high"}
    return {"thinking_budget": 1024}


@dataclass(slots=True)
class CallOptions:
    """Optional per-request knobs; None means 'let the SDK default apply'."""

    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_mode: Literal["on", "off"] = "off"


@runtime_checkable
class Adapter(Protocol):
    """Four entry points per SDK, so every mode is comparable.

    Implementations construct their client/agent inside the call, so SDK setup
    cost lands inside the measurement — that is a real part of the overhead
    being compared. Each call reports its own ``client_init_ms`` share.
    """

    name: str
    label: str

    def call(self, prompt: str, model: str, opts: CallOptions) -> CallResult: ...

    async def call_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> CallResult: ...

    def stream(self, prompt: str, model: str, opts: CallOptions) -> Iterator[Chunk]: ...

    def stream_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> AsyncIterator[Chunk]: ...
