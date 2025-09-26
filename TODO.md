SIM-ONE MCP Server — Production Readiness TODO

Legend: [ ] pending, [~] in progress, [x] done | Priority: P0/P1/P2

Governance Engine (P0)
- [x] Implement governance orchestrator core
  - File: `code/mcp_server/cognitive_governance_engine/governance_orchestrator.py`
  - Deliver: `GovernanceReport` schema + evaluate() entrypoint
- [x] Wire coherence + quality checks
  - Use: `coherence_validator/coherence_checker.py`, `quality_assurance/quality_scorer.py`
- [x] Add error recovery policy hooks
  - Minimal retry with metacognitive adjustments
- [x] Add unit tests
  - `mcp_server/tests/governance/test_governance_orchestrator.py`
  - `mcp_server/tests/governance/test_governance_policy.py`

Metacognition & Adaptive Tuning (P1)
- [ ] Implement `metacognitive_engine/self_optimizer.py`
- [ ] Implement `metacognitive_engine/strategy_selector.py`
- [ ] Unit tests for adjustments
  - `mcp_server/tests/governance/test_metacognitive_adjustments.py`

Pipeline Integration (P0)
- [x] Invoke governance after each protocol step
  - File: `code/mcp_server/orchestration_engine/orchestration_engine.py`
- [x] Surface diagnostics in `/execute` responses
  - File: `code/mcp_server/main.py`
- [x] E2E test covering governance flow
  - `mcp_server/tests/e2e/test_workflow_with_governance.py`

MVLM Backend (P0)
- [x] Add `neural_engine/mvlm_engine.py` (HF local engine + stubs)
- [x] Extend engine factory to support `NEURAL_ENGINE_BACKEND="mvlm"`
- [x] Deterministic test mode when assets absent
- [x] Unit tests
  - `mcp_server/tests/neural_engine/test_mvlm_engine.py`
  - Docs: `code/mcp_server/docs/DEVELOPMENT_NOTES.md`

Security Regression (P0)
- [ ] Run full security suite unchanged
  - `bandit -r mcp_server/` | `safety check`
  - `python -m unittest discover mcp_server/tests/security/`
- [x] Ensure governance diagnostics reveal no sensitive data
  - Implemented governance_summary with aggregate scores only

Docs & Config (P1)
- [x] Update `README.md` with governance, rate limits, audit logs
- [ ] Add MVLM backend usage to README
- [ ] Update `code/mcp_server/project_status.md` progress and roadmap
- [x] Document `.env` keys for MVLM and governance tuning (`DEVELOPMENT_NOTES.md`, `.env.example` pending)

Performance & Reliability (P1)
- [ ] Add basic performance budget and thresholds
- [x] Validate rate limiting and overhead (keep checks lightweight)

Production Validation (P0)
- [ ] Gunicorn startup with governance enabled
- [ ] E2E flows pass with MVLM + mock
- [ ] All tests green (security + governance + e2e)

Infrastructure (Tracked, separate phase)
- [ ] Docker compose (dev/prod) with MVLM toggle
- [ ] CI pipeline: bandit, safety, unit, e2e, health checks
- [ ] Kubernetes manifests (later)

Blocking Dependencies
- [x] MVLM model/assets from owner (main model available)
  - Configure via `MVLM_MODEL_DIRS` + `ACTIVE_MVLM_MODEL` or `LOCAL_MODEL_PATH`

Quick-Run Checklist
- [ ] Set `.env`: `NEURAL_ENGINE_BACKEND`, `MVLM_MODEL_DIRS`/`ACTIVE_MVLM_MODEL` or `LOCAL_MODEL_PATH`, `VALID_API_KEYS`, CORS
- [ ] Initialize DB and ensure Redis running
- [ ] Dev run: `uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000 --reload`
- [ ] Tests: unit + security + e2e

New Items (Post‑Governance)
- [ ] Redis enhancements (session TTL, clustering, key‑based rate limits)
- [ ] Admin: runtime model switch endpoint (RBAC + audit)
- [ ] Structured metrics for governance and recovery (Prometheus)
- [ ] Threat model update for MVLM local execution
