"""Adapter registry: the four benchmark candidates, keyed by name."""

from __future__ import annotations

from app.adapters.agno_adapter import AgnoAdapter
from app.adapters.base import Adapter
from app.adapters.google_genai import GoogleGenAIAdapter
from app.adapters.langchain_adapter import LangChainAdapter
from app.adapters.litellm_adapter import LiteLLMAdapter

_ADAPTERS: dict[str, Adapter] = {
    a.name: a
    for a in (
        GoogleGenAIAdapter(),
        AgnoAdapter(),
        LiteLLMAdapter(),
        LangChainAdapter(),
    )
}

DEFAULT_ORDER = list(_ADAPTERS)


def get_adapter(name: str) -> Adapter:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown sdk {name!r}. Available: {', '.join(DEFAULT_ORDER)}"
        ) from None


def registry() -> list[dict[str, str]]:
    return [{"name": a.name, "label": a.label} for a in _ADAPTERS.values()]


__all__ = ["DEFAULT_ORDER", "Adapter", "get_adapter", "registry"]
