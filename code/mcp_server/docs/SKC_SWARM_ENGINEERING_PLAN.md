SIM-ONE — Specialized Knowledge Centers (SKC) Swarm Engineering Plan

Overview
- Purpose: Add modular, curated “Specialized Knowledge Centers” (SKCs) that complement the Truth Foundation MVLM with focused capabilities (e.g., code, summarization, verification) without growing a single monolith.
- Status: Design document for post‑deployment phase. Implementation will be phased to minimize risk and preserve deterministic behavior.

Goals
- Modularity over scale: compose small/adapter models + retrieval packs per capability/domain.
- Governance‑driven learning: route, compare, and promote SKCs using quality/coherence metrics already in place.
- Operational clarity: simple config, clear audit, predictable latency/cost, strict RBAC.

Principles
- Truth Foundation: the main MVLM remains the source of general capability and grounding; SKCs are specialists.
- Energy Stewardship: choose the smallest SKC that meets quality; escalate only if needed.
- Deterministic Reliability: apply coherence/quality thresholds; retry once; fallback; abort when required.
- Safety First: sanitize inputs, restrict providers by RBAC, log decisions (not contents).

Architecture
1) Swarm Backend
   - New backend in the neural engine factory: `NEURAL_ENGINE_BACKEND="swarm"`.
   - `SwarmEngine` orchestrates model selection and optional fan‑out + governance ranking.

2) Model Registry
   - Static JSON registry (`mcp_server/neural_engine/model_registry.json`) describing SKCs:
     - id, provider (openai|local|mvlm), model_name/path
     - task_tags (e.g., ["code.synthesize","python"]) and per‑protocol weights
     - constraints (max_tokens, latency_hint_ms, cost_tier)
     - RBAC (allowed_roles, environments) and rollout flags (shadow_eval_rate)
     - retrieval_pack (optional): identifier for bound KB/context

3) Routing & Strategy
   - Inputs: protocol/agent name, `task_tags`, constraints (latency_budget_ms, cost_tier), session/governance context.
   - Modes:
     - best‑of (default): choose one SKC by weights and constraints
     - fan‑out (K=2..N): run in parallel; aggregate by governance quality; pick top‑1
     - shadow eval (telemetry only): occasionally sample an alternative without affecting response
   - Fallback chain: primary → secondary (same capability) → baseline MVLM → mock/local

4) Adapters
   - Per‑SKC prompt/IO adapters to normalize style, schema, and constraints (e.g., low temperature for linting/security).
   - Light formatting/validation to return uniform text/JSON for governance scoring.

5) Governance Integration
   - Use existing `QualityScorer` + coherence checks to rank fan‑out outputs.
   - Update per‑SKC weights via moving averages of governance quality, coherence, latency SLO adherence.
   - Respect `GOV_MIN_QUALITY` and `GOV_REQUIRE_COHERENCE` during selection (retry next SKC then fallback/abort).

6) Telemetry & Learning
   - Capture `model_selected`, `swarm_retry`, `swarm_aggregate`, `governance_scores`, `latency_ms`, success signals.
   - Store compact telemetry (no sensitive content) for weight updates and A/B promotions.
   - Optional shadow eval (p<0.05) to keep routing fresh.

Configuration
- `.env` additions:
  - `NEURAL_ENGINE_BACKEND=swarm`
  - `SWARM_REGISTRY_PATH=mcp_server/neural_engine/model_registry.json`
  - `SWARM_MAX_FANOUT=2`
  - `SWARM_DEFAULT_STRATEGY=best` (best|fanout)
  - `SWARM_LATENCY_BUDGET_MS=2000`
  - Provider credentials as needed (OpenAI/Local/MVLM)

Security & Compliance
- RBAC at registry level: restrict high‑cost or external SKCs.
- Input validation/sanitization for all SKC calls (reuse existing middleware/validator).
- Audit logging of routing/aggregation decisions; never log prompts/outputs.

Initial SKC Set (Pilot)
- Reasoning & Verification
  - Logic Verifier: contradiction detection, simple proofs; boosts REP validity.
  - Fact Verifier: claim decomposition + cross‑check (KB/local sources; web when allowed).
- Coding
  - Code Understanding: control/data flow, symbol mapping, contracts.
  - Code Synthesis (python): function/class generation with typing and style.
  - Refactorer: safe API migrations, dead‑code removal.
  - Test Synthesizer: unit/property tests; edge cases.
  - Static/Security Analysis: OWASP/CWE patterns, taint reasoning, safe remediation.
- Language & Editing
  - Summarizer: long→short with high determinism.
  - Technical Editor: structure/clarity/terminology compliance.
- Extraction & Structuring
  - NER + Relations: high‑precision extraction; ontology mapping.
  - Tabular/Schema Mapper: PDF/HTML→tables/JSON; schema alignment.
- Safety & Compliance
  - PII/PHI Redactor; License/Policy Classifier.

Registry Seed (example snippets)
```
[
  {
    "id": "truth.mvlm.main",
    "provider": "mvlm",
    "model": "mvlm-main",
    "task_tags": ["general"],
    "weights": {"default": 0.5},
    "constraints": {"latency_hint_ms": 1500, "cost_tier": "medium"}
  },
  {
    "id": "code.synth.py.v1",
    "provider": "local",
    "model": "llama-3-8b-lora-code.gguf",
    "task_tags": ["code.synthesize","python"],
    "weights": {"default": 0.7, "DrafterProtocol": 0.8},
    "constraints": {"latency_hint_ms": 900, "cost_tier": "low"}
  },
  {
    "id": "summarizer.v1",
    "provider": "mvlm",
    "model": "mvlm-main",
    "task_tags": ["summarize"],
    "weights": {"default": 0.8, "SummarizerProtocol": 0.9},
    "constraints": {"latency_hint_ms": 800, "cost_tier": "low"}
  }
]
```

Routing Policy (v1)
- Default mode: best‑of routing using (weight × capability match) subject to constraints.
- Retry policy: on quality < `GOV_MIN_QUALITY` or `is_coherent=False` → try next preferred SKC once.
- Fan‑out (optional, K<=`SWARM_MAX_FANOUT`): run top‑K; use governance to rank; pick top‑1.
- Fallback: baseline MVLM; then MockEngine for determinism in tests.

Audit Events (additions)
- `model_selected`: {skc_id, provider, task_tags, latency_budget_ms}
- `swarm_retry`: {previous_skc_id, next_skc_id, reason}
- `swarm_aggregate`: {candidates: [{skc_id, quality}], winner}

Testing Strategy
1) Unit Tests
   - Registry load/validation; filter by tags/constraints/RBAC.
   - Router selection (best‑of) and fallback order.
   - Aggregator ranking (governance‑mocked) for fan‑out.
   - Weight updates from telemetry (moving averages).
2) Integration Tests
   - Writing team: Summarizer routed to summarizer SKC; governance score present.
   - Coding flow: code.synthesize SKC chosen; verify linter/compile hooks (dry‑run).
   - Policy: with `GOV_REQUIRE_COHERENCE=true`, ensure retry‑then‑abort on persistent incoherence.
3) Performance Tests
   - Verify latency under `SWARM_LATENCY_BUDGET_MS`; K=2 fan‑out overhead bounded.

Rollout Plan
- Phase 1 — Scaffold (best‑of only)
  - Add `SwarmEngine` + registry + factory switch.
  - Seed with main MVLM + `summarizer.v1` + `code.synth.py.v1`.
  - Emit routing audit; add basic unit/e2e tests.
- Phase 2 — Code Core
  - Add refactorer, test synthesizer, static/security SKCs.
  - Introduce compile/lint/test hooks to feed governance quality.
- Phase 3 — Fan‑out & Learning
  - Enable K=2 fan‑out for selected tasks (security/test synthesis).
  - Add telemetry store + periodic weight updates; shadow eval at 5%.
- Phase 4 — Capsules & Promotion
  - Package small LoRA/adapters + RAG packs as versioned capsules.
  - A/B under governance; promote on sustained wins; document promotion criteria.

Files & Changes (planned)
- Add: `mcp_server/neural_engine/swarm_engine.py`
- Add: `mcp_server/neural_engine/model_registry.json`
- Update: `mcp_server/neural_engine/neural_engine.py` (backend="swarm")
- Update: protocol calls optionally attach `task_tags` in context (non‑breaking)
- Tests: `tests/neural_engine/test_swarm_engine.py`, `tests/e2e/test_swarm_routing.py`

Risks & Mitigations
- Latency growth with fan‑out → cap K, strict budgets, prefer best‑of by default.
- Cost creep → RBAC for expensive SKCs, per‑model rate limits, audit monitoring.
- Drift in routing weights → periodic reset/bounds; hold‑out validations.
- Security/PII exposure → centralized validation/sanitization; audit only metadata.

Acceptance Criteria (Phase 1)
- Swarm backend selectable; registry loads; routing chooses expected SKC by tags.
- Governance scores present and used for ranking when fan‑out is enabled in tests.
- Audit `model_selected` and `swarm_retry` events visible in `security_events.log`.
