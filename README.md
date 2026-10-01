# Gemini SDK one-shot benchmark

Small FastAPI harness that sends one prompt to the same Gemini model through four SDKs and records latency/token usage:

- `google_genai` — official `google-genai` SDK
- `agno` — `Agent.run()` / `Agent.arun()`
- `litellm` — `completion()` / `acompletion()`
- `langchain` — `ChatGoogleGenerativeAI`

It exposes blocking and native async one-shot calls, plus blocking and native async streaming (SSE). Each request is independent; there is no conversation/session state.

## Setup

Python 3.12 is managed with `uv`:

```bash
uv sync
cp .env.example .env
# Set GEMINI_API_KEY in .env
just run
```

Open <http://127.0.0.1:8000/docs> for interactive API docs. `just install`, `just bench`, `just backfill-costs`, `just check`, and `just docs` are also available.

`GEMINI_MODEL` defaults to `gemini-3.8-flash`; change it in `.env` if that model ID is unavailable to your account. Every SDK receives the same model ID and prompt. `GOOGLE_API_KEY` is also accepted as a key fallback.

## Endpoints

| Endpoint | Behavior |
|---|---|
| `GET /health` | Health and API-key presence (never returns the key) |
| `GET /sdks` | SDK names and default model |
| `POST /sdk/{sdk}/call` | Blocking one-shot call (run in a worker thread) |
| `POST /sdk/{sdk}/call/async` | Native async one-shot call |
| `POST /sdk/{sdk}/stream` | Blocking SDK stream over SSE |
| `POST /sdk/{sdk}/stream/async` | Native async stream over SSE |
| `GET /results` | Recorded JSONL rows |
| `GET /results/summary` | Aggregates by input case, SDK, and mode |

Valid `{sdk}` values: `google_genai`, `agno`, `litellm`, `langchain`.

Example request:

```bash
curl -s http://127.0.0.1:8000/sdk/google_genai/call \
  -H 'content-type: application/json' \
  -d '{"prompt":"Reply with exactly: benchmark ok"}'
```

Stream endpoints return SSE `data:` events: `first_token` (TTFT), `delta` (text), then `done` (total latency and usage), or `error`.

## Run a comparison

`just bench` starts a temporary local server, runs every case in `inputs/benchmark_cases.json` once through every SDK/mode, prints per-case summaries, and stops the server automatically. There are six varied cases (simple explanation, lesson plan, constrained planning, algorithmic tasks) with thinking on/off variations. That is 96 model calls by default; `just bench 2` repeats each case twice. It defaults to port 8001; pass a port as the second argument if needed (`just bench 1 8010`). Alternatively, start the server yourself and run the script directly:

```bash
uv run python scripts/bench.py --runs 1
uv run python scripts/bench.py --runs 2 --sdk google_genai --sdk agno --mode sync --mode async
uv run python scripts/bench.py --prompt 'Say hello in one word' --thinking-mode off
uv run python scripts/bench.py --input-file inputs/benchmark_cases.json
```

The runner reads each `{id, prompt, thinking_mode}` case from `inputs/benchmark_cases.json`, sends that same prompt/settings to every SDK, and calls candidates sequentially to avoid self-induced concurrency. For thinking control, `off` uses a zero thinking-token budget; `on` uses high thinking for Gemini 3 models or a 1,024-token thinking budget for Gemini 2.x. Summary rows are grouped by case ID, thinking mode, SDK, and call mode. Results append to `results/runs.jsonl` (override with `BENCH_RESULTS_FILE`). Each row includes timestamp, SDK/mode/model, success/error, latency, TTFT, token usage, prompt/output sizes, initialization time, billable output tokens, cost components, estimated USD total, pricing source/rates, and whether cost could be calculated. `just backfill-costs` adds estimates to older JSONL rows without contacting Gemini.

## Pricing and cost accounting

`pricing.json` is a dated snapshot from models.dev for the direct Google Gemini 3.8 Flash row: $0.75/M input and $3.75/M output tokens. Google's output price includes thinking tokens, which use the output rate. The SDKs expose thinking usage differently: Google GenAI and Agno report thought tokens separately, so those are added to candidate output; LiteLLM and LangChain report output counts that already include thoughts, so they are not double-counted. If a stream omits token usage, its cost is `null` and the summary reports cost coverage instead of inventing a number. Prices are an estimate for standard paid-tier usage; account free-tier, discounts, caching, or provider routing may change the invoice.

## Benchmark caveats

- Network, provider load, model routing, and rate limits dominate one-shot latency. Alternate or randomize SDK order and run multiple repetitions before comparing.
- Token counts are whatever each SDK/provider reports; streaming APIs may omit usage metadata. Missing values are `null`, not zero.
- Client/agent construction is included in total latency. `client_init_ms` is separately meaningful for adapters that can isolate construction; compare it cautiously across SDKs.
- This is a quick harness, not a statistically controlled benchmark. It does not normalize retries, SDK defaults, or provider-side generation settings beyond the options explicitly sent.
