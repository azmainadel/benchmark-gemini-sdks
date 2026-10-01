set shell := ["bash", "-cu"]

default:
    @just --list

# Start the FastAPI benchmark server with auto-reload.
run:
    uv run uvicorn app.main:app --reload

# Install/sync project dependencies.
install:
    uv sync

# Start a temporary server and run every case in inputs/benchmark_cases.json.
bench runs="1" port="8001":
    ./scripts/run_bench.sh "{{runs}}" "{{port}}"

# Add price estimates to existing JSONL results (local-only; no model calls).
backfill-costs:
    uv run python scripts/backfill_costs.py

# Run static checks and offline unit tests.
check:
    uvx pyright app scripts tests
    uvx ruff check app scripts tests
    uv run python -m unittest discover -s tests

# Open the FastAPI OpenAPI docs in the default browser.
docs:
    open http://127.0.0.1:8000/docs
