# OpenAI Backend Switcher — TODOs to Production

This checklist drives the implementation of the OpenAI backend switcher per the plan in `OPENAI_BACKEND_SWITCHER_PLAN.md`, and brings the system to production readiness with JSON‑first agent outputs, retries, metrics, and docs.

Legend: [ ] pending · [~] in progress · [x] done · Priority: P0/P1/P2

## Phase 0 — Preflight (P0)

- [ ] Add .env keys (code/mcp_server/.env.example)
  - [ ] `OPENAI_MODEL`, `OPENAI_API_BASE`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, `OPENAI_JSON_ONLY`
  - [ ] Add comments indicating safe defaults and usage examples
- [ ] Add CI toggle for OpenAI tests (skip when `OPENAI_API_KEY` not present)
- [ ] Decide initial cost guardrails (document): short prompts, `max_tokens<=160`, retries<=2

## Phase 1 — Engine + Utilities (P0)

- [ ] Extend `OpenAIEngine` (code/mcp_server/neural_engine/neural_engine.py)
  - [ ] Read env: model, api base, timeout, retries, json_only
  - [ ] Implement JSON‑only path (OpenAI response_format or strict system prompt fallback)
  - [ ] Add retry with backoff (429/5xx) up to `OPENAI_MAX_RETRIES`
  - [ ] Redact prompts in logs; audit `model_selected` with backend/model/latency
- [ ] Add prompt adapters (new: `code/mcp_server/neural_engine/prompt_adapters.py`)
  - [ ] `build_json_prompt(schema_hint, task_text)` → (system,user) for OpenAI and plain string for MVLM
  - [ ] `build_bullets_prompt(count, task_text)` → same pattern, returns JSON schema for bullets
- [ ] Add JSON guard (new: `code/mcp_server/neural_engine/json_guard.py`)
  - [ ] `ensure_json(engine_call, primary_prompt, tight_prompt=None)`
  - [ ] Retry once with tightened prompt/decoding; fallback to safe JSON and log warning
- [ ] Metrics (reusing Prometheus setup)
  - [ ] Add counters: `simone_openai_calls_total`, `simone_openai_errors_total`, `simone_json_retries_total`

### Phase 1 Tests (P0)

- [ ] Unit: `tests/neural_engine/test_openai_engine_config.py` (env mapping)
- [ ] Unit: `tests/neural_engine/test_openai_engine_json_mode.py` (mock OpenAI; validates JSON response mode)
- [ ] Unit: `tests/neural_engine/test_json_guard.py` (bad→good→fallback paths)

### Phase 1 Docs (P0)

- [ ] README: add “Backend Switcher: MVLM vs OpenAI” with sample .env and behavior
- [ ] Update `.env.example` with new keys + comments

## Phase 2 — Wire Summarizer (P0)

- [ ] SummarizerProtocol: use `prompt_adapters` + `json_guard`
  - [ ] Force fixed schema `{ "bullets": [5 strings] }`
  - [ ] Respect decoding envs (MVLM) or JSON mode (OpenAI)
- [ ] Unit: mocked engine returns non‑JSON → guard retries → returns valid JSON
- [ ] E2E: with backend=openai (mocked client), `/execute` returns valid bullets JSON and governance_summary present

## Phase 3 — Wire ESL/MTP/REP (P0/P1)

- [ ] ESL: strict schema `{ "valence": string, "intensity": number, "detected_emotions": [...] }`
- [ ] MTP: strict schema `{ "entities": [...], "relations": [...] }`
- [ ] REP: strict schema `{ "conclusions": [...], "reasoning": [...], "valid": boolean }`
- [ ] Tests: unit + E2E for each protocol ensuring JSON validity + parse‑retry fallback

## Phase 4 — Observability & Security (P1)

- [ ] Audit events: `model_selected`, `json_retry` (metadata only)
- [ ] Prometheus counters wired into `/metrics/prometheus`
- [ ] Quotas: confirm per‑API‑key quotas in `/execute` apply to OpenAI usage
- [ ] Runbook: add OpenAI ops section (timeouts, retries, quotas, costs) to README/SECURITY

## Phase 5 — Documentation & Examples (P1)

- [ ] Add prompt schema snippets to `docs` so users know expected JSON contracts per protocol
- [ ] Add example .env configs for MVLM vs OpenAI
- [ ] Add a short “switch backend at runtime” note (use admin endpoints or env + restart)

## Phase 6 — Optional Provider Registry (P2)

- [ ] Abstract OpenAI provider so Azure OpenAI or other APIs can be added behind same interface
- [ ] Add config keys and tests for alternate providers

## Acceptance Criteria

- [ ] `NEURAL_ENGINE_BACKEND=openai` produces valid JSON for Summarizer/ESL/MTP/REP with a single `/execute` call
- [ ] JSON parse success rate ≥ 95% on a small validation set (guard handles the rest)
- [ ] Governance counters track retries/fallbacks; audit logs include model selection
- [ ] README + .env.example document backend switching and JSON‑first behavior
- [ ] CI green: unit + E2E with mocked OpenAI; secure defaults; no sensitive data in logs

## Validation Steps

Commands (set `OPENAI_API_KEY` for local test or mock in CI):

```bash
# Summarizer (OpenAI)
NEURAL_ENGINE_BACKEND=openai OPENAI_MODEL=gpt-4o-mini OPENAI_JSON_ONLY=true \
  python -m unittest mcp_server.tests.e2e.test_workflow_with_governance

# Unit tests for engine + guard
python -m unittest discover mcp_server/tests/neural_engine/

# Smoke test with curl
curl -sS -X POST http://localhost:8000/execute \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"protocol_names":["SummarizerProtocol"],"initial_data":{"user_input":"Explain SIM-ONE governance."}}' | jq .
```

## Risk Log

- Cost/latency: mitigate with quotas, short prompts, low max_tokens, retries limited to 2
- JSON drift: guard ensures parse‑retry‑fallback; governance flags annotate warnings
- Security: ensure API key is only read from env; no prompt/response logging in audit

