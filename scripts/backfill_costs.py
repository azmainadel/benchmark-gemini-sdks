#!/usr/bin/env python3
"""Add estimated USD costs to an existing JSONL run log without API calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.config import RESULTS_DIR
from app.pricing import estimate_cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=RESULTS_DIR / "runs.jsonl",
        help="JSONL file to update",
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="Recompute all cost fields using the current local pricing snapshot.",
    )
    args = parser.parse_args()
    path: Path = args.file
    if not path.exists():
        parser.error(f"No results file found: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
        if args.recalculate or "estimated_cost_usd" not in row:
            row.update(
                estimate_cost(
                    sdk=row["sdk"],
                    model=row["model"],
                    input_tokens=row.get("input_tokens"),
                    output_tokens=row.get("output_tokens"),
                    reasoning_tokens=row.get("reasoning_tokens"),
                    thinking_mode=row.get("thinking_mode"),
                ).as_dict()
            )
        rows.append(row)

    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

    priced = sum(row.get("estimated_cost_usd") is not None for row in rows)
    total = sum(row.get("estimated_cost_usd") or 0 for row in rows)
    print(
        f"Updated {len(rows)} rows; cost available for {priced}; priced total ${total:.8f} USD"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
