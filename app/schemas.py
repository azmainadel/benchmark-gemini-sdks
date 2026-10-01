"""Wire formats shared by adapters, endpoints and the result log."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_PROMPT = (
    "Explain how rainbows form to a curious 10-year-old. Use three short "
    "bullet points and one friendly analogy. Keep it under 100 words."
)


class CallRequest(BaseModel):
    """A benchmark call: one prompt, one model, no conversation state."""

    prompt: str = Field(
        default=DEFAULT_PROMPT,
        description="Prompt sent verbatim by every SDK.",
    )
    case_id: str | None = Field(default=None, description="Benchmark input-set case ID.")
    thinking_mode: Literal["on", "off"] = Field(
        default="off", description="Requested Gemini thinking level for this case."
    )
    model: str | None = Field(
        default=None, description="Override the server default model."
    )
    temperature: float | None = Field(
        default=0.0, description="Shared sampling temperature sent to every SDK."
    )
    max_output_tokens: int | None = Field(
        default=1024, description="Shared output-token ceiling sent to every SDK."
    )


@dataclass(slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """One streamed piece. ``usage`` rides on the final chunk when available."""

    text: str | None = None
    usage: Usage | None = None


@dataclass(slots=True)
class CallResult:
    text: str
    usage: Usage | None = None
    client_init_ms: float | None = None


@dataclass(slots=True)
class RunRecord:
    """One row of results.jsonl."""

    sdk: str
    mode: str
    model: str
    ok: bool
    case_id: str | None = None
    thinking_mode: Literal["on", "off"] | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    client_init_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    billable_output_tokens: int | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    reasoning_price_per_million: float | None = None
    pricing_source: str | None = None
    pricing_retrieved_at: str | None = None
    cost_status: str = "missing_usage"
    output_chars: int | None = None
    error: str | None = None
    prompt_chars: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CallResponse(BaseModel):
    """Response body for the non-streaming endpoints."""

    sdk: str
    mode: str
    model: str
    latency_ms: float
    case_id: str | None = None
    thinking_mode: Literal["on", "off"] | None = None
    client_init_ms: float | None = None
    output_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    billable_output_tokens: int | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    reasoning_price_per_million: float | None = None
    pricing_source: str | None = None
    pricing_retrieved_at: str | None = None
    cost_status: str = "missing_usage"
    output_preview: str = ""
    run_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
