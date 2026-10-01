"""Append-only JSONL result log, safe for concurrent requests."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.schemas import RunRecord

_lock = threading.Lock()


class ResultRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: RunRecord) -> None:
        row = {"ts": _now_iso(), **record.as_dict()}
        line = json.dumps(row, ensure_ascii=False)
        with _lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with _lock, self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
