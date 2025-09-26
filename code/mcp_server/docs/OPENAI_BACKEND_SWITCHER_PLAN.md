# SIM-ONE — OpenAI Backend Switcher Implementation Plan

## Objectives

- Allow SIM-ONE to run with either the local MVLM (HF/LoRA) or the OpenAI API.
- Keep a single, stable engine interface so protocols do not change.
- Enforce JSON-first prompting, with parse → retry → fallback across both backends.
- Preserve security (RBAC, quotas, rate limits) and observability (audit + metrics).

## Configuration (.env)

- `NEURAL_ENGINE_BACKEND`: `mvlm | openai | mock | local`
- `OPENAI_API_KEY`: API key (required for `openai` backend)
- `OPENAI_MODEL`: e.g., `gpt-4o-mini` (default) or `gpt-4o`
- `OPENAI_API_BASE` (optional): Azure/OpenAI compatible endpoint
- `OPENAI_TIMEOUT_SECONDS`: default `30`
- `OPENAI_MAX_RETRIES`: default `2`
- `OPENAI_JSON_ONLY`: `true|false` (enable OpenAI JSON mode or force JSON via prompts)

MVLM controls remain unchanged (temperature/top-p/k, repetition penalty, no-repeat n-gram, LoRA adapter path, etc.).

## Architecture Overview

- Continue to use `mcp_server/neural_engine/neural_engine.py` as the public factory. It already switches on `NEURAL_ENGINE_BACKEND`.
- Extend `OpenAIEngine` to support:
  - Configurable model, base URL, timeouts, and retries with backoff.
  - JSON-only generation via OpenAI Chat Completions response_format OR via strict system prompts.
  - A helper for JSON-first calls used by protocols.
- Add two tiny utilities:
  - `prompt_adapters.py`: standardized builders for system/user messages for JSON and bullets across providers.
  - `json_guard.py`: parse → retry (tightened prompt + shorter tokens + higher penalties) → safe fallback JSON.

## File-Level Changes

1) `code/mcp_server/neural_engine/neural_engine.py`
   - OpenAIEngine additions:
     - Read `OPENAI_MODEL`, `OPENAI_API_BASE`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, `OPENAI_JSON_ONLY`.
     - Implement `generate_text/json_only=True` to set JSON mode in OpenAI when supported, else prepend strict JSON system prompts.
     - Backoff on 429/5xx with jitter up to `OPENAI_MAX_RETRIES`.
   - Keep the singleton proxy + `refresh_neural_engine()` for runtime backend/model switches.

2) `code/mcp_server/neural_engine/prompt_adapters.py` (new)
   - `build_json_prompt(schema_hint: str, task_text: str)` → returns (system, user) messages for OpenAI and a plain string for MVLM.
   - `build_bullets_prompt(count: int, task_text: str)` → same idea but for fixed bullet schemas; encourages `{ "bullets": [...] }` JSON.

3) `code/mcp_server/neural_engine/json_guard.py` (new)
   - `ensure_json(engine_call, primary_prompt, tight_prompt=None) -> dict`:
     - Call once, try `json.loads`.
     - On failure, retry with tight prompt ("Return ONLY valid JSON" + shorter tokens; increase penalty/no-repeat if MVLM).
     - On second failure, return safe default JSON (empty bullets, neutral sentiment, empty entities, etc.) and log a governance warning.

4) Protocol wiring (incremental)
   - Update one protocol end-to-end (Summarizer) to use `prompt_adapters` + `json_guard`.
   - Confirm output is always valid JSON and agent-usable.
   - Repeat for ESL/MTP/REP once validated.

## Prompting & Decoding

- MVLM backend: keep existing env controls (temperature/top-p/k, repetition penalty, no-repeat, max tokens) to reduce drift.
- OpenAI backend:
  - If `OPENAI_JSON_ONLY=true`, set `response_format={"type":"json_object"}` on Chat Completions and provide clear schema hints.
  - Else, enforce strict JSON via system prompts: "You are a JSON generator. Return ONLY JSON, no prose." + schema.
  - Keep max tokens low (e.g., 96–160) and stop early where possible.

## Error Handling & Retries

- OpenAIEngine: apply timeouts and backoff retries on 429/5xx (up to `OPENAI_MAX_RETRIES`). Redact prompts from logs; log hashes and key metadata only.
- `json_guard`: parse failure → tighten prompt/decoding → retry once → return safe default.
- Increment governance counters on retries/fallbacks and attach warnings into context.

## Security & Compliance

- Do not log prompts/outputs in audit logs (metadata only). Use `audit` logger events:
  - `model_selected` with backend + model + latency_ms + ok
  - `json_retry` when parse fails
- Respect existing RBAC + rate limits + per-API-key quotas.
- Only enable OpenAI backend when `OPENAI_API_KEY` is present; else fallback to MVLM or `mock`.

## Observability

- Prometheus counters (add alongside governance metrics):
  - `simone_openai_calls_total`
  - `simone_openai_errors_total`
  - `simone_json_retries_total`
- Keep existing governance summary in `/execute`.

## Tests

- Unit (`code/mcp_server/tests/neural_engine/`):
  - `test_openai_engine_config.py`: env mapping for model/base/timeouts/retries.
  - `test_openai_engine_json_mode.py`: mock OpenAI client to verify JSON mode and parsable JSON.
  - `test_json_guard.py`: simulate bad → good → default paths.
- E2E:
  - Toggle `NEURAL_ENGINE_BACKEND=openai` with mocked OpenAI responses and assert `/execute` returns valid protocol JSON and governance summary.

## Docs

- `.env.example`:
  - Add `OPENAI_MODEL`, `OPENAI_API_BASE`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, `OPENAI_JSON_ONLY`.
- README (Security & Governance / Execution Controls):
  - How to switch backend: `NEURAL_ENGINE_BACKEND=openai` vs `mvlm`.
  - JSON-first behavior and parse-retry-fallback guarantee.

## Rollout Plan

1. Phase 1 (1 day)
   - Extend `OpenAIEngine` (config, timeouts, retries, JSON mode).
   - Add `prompt_adapters.py` and `json_guard.py` utilities.
   - Update `.env.example` + README.

2. Phase 2 (0.5 day)
   - Wire SummarizerProtocol to use adapters + json_guard.
   - Unit tests with mocked OpenAI client.

3. Phase 3 (0.5–1 day)
   - Wire ESL/MTP/REP.
   - E2E test toggling backends; verify governance summary + safe defaults.

4. Phase 4 (optional)
   - Add provider registry to support Azure OpenAI or other APIs (Bedrock/Anthropic) behind same interface.

## Risks & Mitigations

- Cost spikes with API usage → per-key quotas, short prompts, low max tokens, backoff with limits.
- JSON failures → json_guard parse-retry-fallback + governance alerts.
- Latency → timeouts, minimal tokens, retries with backoff, optional caching for deterministic prompts.

## Example .env for OpenAI

```
NEURAL_ENGINE_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_JSON_ONLY=true
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_RETRIES=2
```

## Example .env for MVLM (unchanged)

```
NEURAL_ENGINE_BACKEND=mvlm
MVLM_MODEL_DIRS=main:models/mvlm_gpt2/mvlm_final
ACTIVE_MVLM_MODEL=main
```

