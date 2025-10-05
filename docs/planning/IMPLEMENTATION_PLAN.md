SIM-ONE MCP Server — Governance & MVLM Implementation Plan

Overview
- Goal: Deliver a production-ready cognitive governance layer and MVLM backend, fully integrated with the orchestration pipeline, maintaining existing security guarantees and test coverage.
- Target Readiness: 95%+ after MVLM integration, governance runtime wiring, and end-to-end validation.
- Scope: Backend (`code/mcp_server/*`) with selective test and docs updates. Infrastructure work tracked but implemented in a later phase.

Definition of Done
- Governance orchestrator executes coherence, quality, error recovery, and metacognitive checks per phase and per step.
- Orchestration pipeline calls governance after each protocol output and aggregates diagnostics in `/execute` responses.
- MVLM backend implemented and selectable via `NEURAL_ENGINE_BACKEND="mvlm"`; deterministic fallback when unavailable.
- Comprehensive unit + integration tests for governance and MVLM paths; existing security tests pass.
- Documentation updated (configuration, runtime behavior, troubleshooting).

Key Workstreams
1) Governance Orchestrator (Core)
- Implement `code/mcp_server/cognitive_governance_engine/governance_orchestrator.py`:
  - Inputs: session context, protocol step result, runtime metrics.
  - Invokes: coherence validator, quality scorer, error recovery hooks, metacognitive self-optimizer.
  - Outputs: `GovernanceReport` (coherence score, quality score, actions, warnings, remediation hints).
  - Policies: minimal and fast path by default (Law 4), deterministic decisions (Law 5).

2) Metacognitive & Adaptive Modules
- Flesh out:
  - `code/mcp_server/cognitive_governance_engine/metacognitive_engine/self_optimizer.py`
  - `code/mcp_server/cognitive_governance_engine/metacognitive_engine/strategy_selector.py`
  - (Optional) `adaptive_learning/*` if present for parameter auto-tuning.
- Capabilities: adjust prompt budgets, sampling temperature, and step retries based on governance metrics.

3) Pipeline Integration
- Update `code/mcp_server/orchestration_engine/orchestration_engine.py`:
  - After each protocol execution, call orchestrator and collect `GovernanceReport`.
  - If incoherent/low-quality: trigger recovery (retry with adjusted strategy; or skip with rationale) per configured policy.
  - Aggregate governance diagnostics into the workflow result.
- Update `code/mcp_server/main.py`:
  - Include summarized governance diagnostics in `/execute` responses (non-sensitive).

4) MVLM Backend
- Add `code/mcp_server/neural_engine/mvlm_engine.py` with a clean interface parity to current engines.
- Extend engine factory (e.g., `code/mcp_server/neural_engine/__init__.py` or `factory.py`) to support `NEURAL_ENGINE_BACKEND="mvlm"`.
- Configuration:
  - `NEURAL_ENGINE_BACKEND="mvlm"`
  - `LOCAL_MODEL_PATH` (optional)
- Behavior: deterministic mock-compatible paths for tests when MVLM assets are unavailable.

5) Testing & Validation
- Unit tests:
  - `mcp_server/tests/governance/test_governance_orchestrator.py`
  - `mcp_server/tests/governance/test_metacognitive_adjustments.py`
  - `mcp_server/tests/neural_engine/test_mvlm_engine.py`
- Integration/E2E:
  - `mcp_server/tests/e2e/test_workflow_with_governance.py` (REP/ESL/MTP + governance + MVLM/mock)
  - Failure scenarios: incoherence triggers recovery; low quality triggers adjustment; ensure deterministic reliability.
- Security regression: run existing security suite unchanged.

6) Documentation & Ops
- Update `README.md` and `code/mcp_server/project_status.md` with governance/MVLM details.
- Add configuration docs: `.env` keys, backend selection, troubleshooting, performance notes.
- Make governance diagnostics visible but sanitized (no secrets, no internals leakage).

Milestones & Acceptance Criteria
M1 — Interfaces & Scaffolding (Day 1)
- Governance orchestrator, report schema, and MVLM engine stubs landed.
- Factory selection and env parsing in place.
- AC: Stubs import without runtime errors; unit tests compile.

M2 — Governance Core Behavior (Day 2–3)
- Coherence and quality pathways wired; error recovery and metacognitive adjustments minimally functional.
- AC: Unit tests for orchestrator pass; recovery triggers on simulated incoherence; deterministic decisions validated.

M3 — MVLM Integration (Day 3–4)
- MVLM engine operational with provided assets; deterministic test mode available.
- AC: `test_mvlm_engine` passes under both mock and MVLM configs; throughput within acceptable bounds.

M4 — Pipeline Wiring & API Diagnostics (Day 4–5)
- Orchestration pipeline invokes governance; `/execute` returns diagnostics (scores, decisions, retries count).
- AC: E2E tests pass; security tests remain green.

M5 — Hardening & Docs (Day 5)
- Performance tuning; clear documentation; updated `project_status.md`.
- AC: Bandit/Safety scans clean; production startup validated; readiness at 95%+.

Testing Commands (reference)
- Security:
  - `bandit -r mcp_server/`
  - `safety check`
  - `python -m unittest discover mcp_server/tests/security/`
- Governance & Protocols:
  - `python -m unittest mcp_server.tests.test_esl_protocol`
  - `python -m unittest mcp_server.tests.test_mtp_protocol`
  - `python -m unittest mcp_server.tests.test_main`
  - `python -m unittest mcp_server.tests.test_memory_consolidation`
- New governance/MVLM tests (to be added):
  - `python -m unittest mcp_server.tests.governance.test_governance_orchestrator`
  - `python -m unittest mcp_server.tests.governance.test_metacognitive_adjustments`
  - `python -m unittest mcp_server.tests.neural_engine.test_mvlm_engine`
  - `python -m unittest mcp_server.tests.e2e.test_workflow_with_governance`

Design Notes & Integration Points
- Governance Orchestrator API (proposed):
  - `orchestrator.evaluate(step_name, input_data, output_data, session_ctx) -> GovernanceReport`
  - `GovernanceReport`: `{coherence: float, quality: float, actions: list[str], warnings: list[str], adjustments: dict}`
- Error Recovery Policy (minimal):
  - Retry once with adjusted parameters if coherence < threshold OR quality < threshold.
  - Else continue, annotate diagnostics.
- Metacognition:
  - Maintain rolling metrics in session (e.g., moving averages) to guide temperature/prompts/retries.

Risks & Mitigations
- MVLM availability/tuning risk → Provide deterministic fallback and mock-compatible test mode.
- Performance overhead → Keep checks lightweight; allow configurable sampling frequency.
- Security regression → Run full security suite on each milestone.
- Interface drift → Centralize types in a `schemas.py` file under governance engine if missing.

Dependencies
- MVLM model/assets (to be provided).
- Redis for session metrics (already required by project).
- `.env` configuration keys present and documented.

Out of Scope (this phase)
- Full containerization and Kubernetes manifests (tracked separately).
- Advanced observability (OTel) and enterprise secrets managers.

Next Steps (Immediate)
- Create orchestrator + report schema stubs and MVLM engine scaffold.
- Add minimal tests to exercise imports and basic flows.

