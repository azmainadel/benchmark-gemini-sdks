#!/usr/bin/env bash
set -euo pipefail

runs="${1:-1}"
port="${2:-8001}"
host=127.0.0.1
log="$(mktemp)"
uv run uvicorn app.main:app --host "$host" --port "$port" >"$log" 2>&1 &
server_pid=$!

cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    rm -f "$log"
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Benchmark server exited during startup:" >&2
        tail -40 "$log" >&2
        exit 1
    fi
    if curl -fsS "http://$host:$port/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 0.5
done

if [[ "$ready" != 1 ]]; then
    echo "Timed out waiting for benchmark server:" >&2
    tail -40 "$log" >&2
    exit 1
fi

uv run python scripts/bench.py --url "http://$host:$port" --runs "$runs"
