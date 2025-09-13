SIM-ONE MCP Server — Production Readiness TODO

Legend: [ ] pending, [~] in progress, [x] done | Priority: P0/P1/P2

Governance Engine (P0)
- [ ] Implement governance orchestrator core
  - File: `code/mcp_server/cognitive_governance_engine/governance_orchestrator.py`
  - Deliver: `GovernanceReport` schema + evaluate() entrypoint
- [ ] Wire coherence + quality checks
  - Use: `coherence_validator/coherence_checker.py`, `quality_assurance/quality_scorer.py`
- [ ] Add error recovery policy hooks
  - Minimal retry with metacognitive adjustments
- [ ] Add unit tests
  - `mcp_server/tests/governance/test_governance_orchestrator.py`

Metacognition & Adaptive Tuning (P1)
- [ ] Implement `metacognitive_engine/self_optimizer.py`
- [ ] Implement `metacognitive_engine/strategy_selector.py`
- [ ] Unit tests for adjustments
  - `mcp_server/tests/governance/test_metacognitive_adjustments.py`

Pipeline Integration (P0)
- [ ] Invoke governance after each protocol step
  - File: `code/mcp_server/orchestration_engine/orchestration_engine.py`
- [ ] Surface diagnostics in `/execute` responses
  - File: `code/mcp_server/main.py`
- [ ] E2E test covering governance flow
  - `mcp_server/tests/e2e/test_workflow_with_governance.py`

MVLM Backend (P0)
- [ ] Add `neural_engine/mvlm_engine.py` (parity interface)
- [ ] Extend engine factory to support `NEURAL_ENGINE_BACKEND="mvlm"`
- [ ] Deterministic test mode when assets absent
- [ ] Unit tests
  - `mcp_server/tests/neural_engine/test_mvlm_engine.py`

Security Regression (P0)
- [ ] Run full security suite unchanged
  - `bandit -r mcp_server/` | `safety check`
  - `python -m unittest discover mcp_server/tests/security/`
- [ ] Ensure governance diagnostics reveal no sensitive data

Docs & Config (P1)
- [ ] Update `README.md` with MVLM backend and governance usage
- [ ] Update `code/mcp_server/project_status.md` progress and roadmap
- [ ] Document `.env` keys for MVLM and governance tuning

Performance & Reliability (P1)
- [ ] Add basic performance budget and thresholds
- [ ] Validate rate limiting and overhead (keep checks lightweight)

Production Validation (P0)
- [ ] Gunicorn startup with governance enabled
- [ ] E2E flows pass with MVLM + mock
- [ ] All tests green (security + governance + e2e)

Infrastructure (Tracked, separate phase)
- [ ] Docker compose (dev/prod) with MVLM toggle
- [ ] CI pipeline: bandit, safety, unit, e2e, health checks
- [ ] Kubernetes manifests (later)

Blocking Dependencies
- [ ] MVLM model/assets from owner (you)

Quick-Run Checklist
- [ ] Set `.env`: `NEURAL_ENGINE_BACKEND`, `LOCAL_MODEL_PATH`, `VALID_API_KEYS`, CORS
- [ ] Initialize DB and ensure Redis running
- [ ] Dev run: `uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000 --reload`
- [ ] Tests: unit + security + e2e

