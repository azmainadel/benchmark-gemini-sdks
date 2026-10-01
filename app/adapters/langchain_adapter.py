"""Candidate 4: LangChain (`langchain-google-genai` chat model)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.adapters.base import CallOptions, now_ms, thinking_config
from app.schemas import CallResult, Chunk, Usage


def _llm(model: str, opts: CallOptions) -> ChatGoogleGenerativeAI:
    kwargs: dict[str, Any] = {"model": model}
    if opts.temperature is not None:
        kwargs["temperature"] = opts.temperature
    if opts.max_output_tokens is not None:
        kwargs["max_output_tokens"] = opts.max_output_tokens
    kwargs.update(thinking_config(model, opts.thinking_mode))
    return ChatGoogleGenerativeAI(**kwargs)


def _usage(response: Any) -> Usage | None:
    metadata = getattr(response, "usage_metadata", None)
    if not metadata:
        return None
    details = metadata.get("input_token_details") or {}
    return Usage(
        input_tokens=metadata.get("input_tokens"),
        output_tokens=metadata.get("output_tokens"),
        total_tokens=metadata.get("total_tokens"),
        reasoning_tokens=details.get("reasoning"),
    )


def _text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


class LangChainAdapter:
    name = "langchain"
    label = "langchain (ChatGoogleGenerativeAI)"

    def call(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        t0 = now_ms()
        llm = _llm(model, opts)
        init_ms = now_ms() - t0
        response = llm.invoke([HumanMessage(content=prompt)])
        return CallResult(
            text=_text(response),
            usage=_usage(response),
            client_init_ms=init_ms,
        )

    async def call_async(self, prompt: str, model: str, opts: CallOptions) -> CallResult:
        t0 = now_ms()
        llm = _llm(model, opts)
        init_ms = now_ms() - t0
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return CallResult(
            text=_text(response),
            usage=_usage(response),
            client_init_ms=init_ms,
        )

    def stream(self, prompt: str, model: str, opts: CallOptions) -> Iterator[Chunk]:
        llm = _llm(model, opts)
        last: Any = None
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            last = chunk
            text = _text(chunk)
            if text:
                yield Chunk(text=text)
        yield Chunk(usage=_usage(last))

    async def stream_async(
        self, prompt: str, model: str, opts: CallOptions
    ) -> AsyncIterator[Chunk]:
        llm = _llm(model, opts)
        last: Any = None
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            last = chunk
            text = _text(chunk)
            if text:
                yield Chunk(text=text)
        yield Chunk(usage=_usage(last))
