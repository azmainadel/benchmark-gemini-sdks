"""Environment-driven configuration for the benchmark harness."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class Settings:
    """Snapshot of the env config, resolved once at startup."""

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        # google-genai reads GEMINI_API_KEY; agno/litellm prefer GOOGLE_API_KEY.
        self.api_key: str | None = (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
        )
        self.model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.results_file: Path = Path(
            os.getenv("BENCH_RESULTS_FILE", RESULTS_DIR / "runs.jsonl")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
