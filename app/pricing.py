"""Token-based estimated costs from the checked-in models.dev price snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

from app.config import PROJECT_ROOT

PRICING_FILE = PROJECT_ROOT / "pricing.json"


@dataclass(slots=True)
class CostEstimate:
    estimated_cost_usd: float | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    billable_output_tokens: int | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    reasoning_price_per_million: float | None = None
    pricing_source: str | None = None
    pricing_retrieved_at: str | None = None
    cost_status: str = "missing_usage"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _pricing() -> dict[str, Any]:
    try:
        content = PRICING_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot load pricing snapshot at {PRICING_FILE}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(
        data.get("prices_per_million_tokens"), dict
    ):
        raise TypeError(f"Invalid pricing snapshot at {PRICING_FILE}")
    return data


def _billable_output_tokens(
    sdk: str,
    output_tokens: int,
    reasoning_tokens: int | None,
    thinking_mode: Literal["on", "off"] | None,
) -> int | None:
    """Normalize SDK usage fields to billable generated tokens.

    In the saved Gemini Developer API data, google-genai and Agno report
    candidate/output tokens separately from thoughts, so add thoughts once.
    LiteLLM completion_tokens and LangChain output_tokens already include
    thoughts. If Google/Agno ran with thinking enabled but omitted thought
    usage, or the thinking mode is unknown, return unknown instead of silently
    underestimating the bill.
    """
    if sdk in {"google_genai", "agno"}:
        if reasoning_tokens is None and thinking_mode != "off":
            return None
        return output_tokens + (reasoning_tokens or 0)
    return output_tokens


def estimate_cost(
    *,
    sdk: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    thinking_mode: Literal["on", "off"] | None,
) -> CostEstimate:
    """Return cost components, or an explicit status when usage is insufficient."""
    price = _pricing()
    if model.casefold() != str(price["model"]).casefold():
        return CostEstimate(cost_status="unpriced_model")

    prices = price["prices_per_million_tokens"]
    result = CostEstimate(
        input_price_per_million=float(prices["input"]),
        output_price_per_million=float(prices["output"]),
        reasoning_price_per_million=float(prices["reasoning"]),
        pricing_source=str(price["source"]),
        pricing_retrieved_at=str(price["retrieved_at"]),
    )
    if input_tokens is None or output_tokens is None:
        result.cost_status = "missing_token_usage"
        return result

    billable_output = _billable_output_tokens(
        sdk, output_tokens, reasoning_tokens, thinking_mode
    )
    if billable_output is None:
        result.cost_status = "missing_reasoning_usage"
        return result

    divisor = Decimal(1_000_000)
    input_cost = Decimal(input_tokens) * Decimal(str(prices["input"])) / divisor
    output_cost = Decimal(billable_output) * Decimal(str(prices["output"])) / divisor
    result.input_cost_usd = round(float(input_cost), 12)
    result.output_cost_usd = round(float(output_cost), 12)
    result.estimated_cost_usd = round(float(input_cost + output_cost), 12)
    result.billable_output_tokens = billable_output
    result.cost_status = "estimated"
    return result
