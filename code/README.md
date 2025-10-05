# SIM-ONE Cognitive Control Protocol (mCP) Server

⚠️ **Important Naming Note**: The `mcp_server` directory predates the industry-standard "Model Context Protocol" (MCP). In this codebase, "mcp_server" refers to SIM-ONE's **"Multi-Protocol Cognitive Platform"** or **"Modular Cognitive Platform"** - the core orchestrator and agent system. This is NOT an MCP tool registry in the modern sense. See [../MIGRATION_PLAN.md](../MIGRATION_PLAN.md) for future renaming strategy.

---

## Project Overview

Welcome to the SIM-ONE mCP Server, a sophisticated, multi-protocol cognitive architecture designed to simulate advanced reasoning, emotional intelligence, and metacognitive governance. This server is the backbone of the SIM-ONE framework, providing a powerful platform for developing and orchestrating autonomous AI agents that can perform complex cognitive tasks.

The server is built on a modular, protocol-based architecture, allowing for dynamic loading and execution of various cognitive functions. From deep reasoning and emotional analysis to advanced entity extraction and self-governance, the mCP Server provides the tools to build truly intelligent systems.

## Key Features

*   **Modular Protocol Architecture**: Dynamically load and chain "Cognitive Protocols" to create complex workflows.
*   **Advanced Reasoning (REP)**: A rule-based engine for deductive, inductive, and abductive reasoning.
*   **Emotional Intelligence (ESL)**: Sophisticated, rule-based emotional analysis to understand user sentiment and intent.
*   **Entity & Relationship Extraction (MTP)**: Advanced entity recognition (people, places, organizations) and detection of the relationships between them.
*   **Cognitive Governance Engine**: A powerful metacognitive layer that validates the coherence, quality, and resilience of all cognitive processes.
*   **Persistent Memory**: A memory manager that allows the system to learn and recall information across sessions.
*   **Secure and Scalable**: Built with FastAPI, including features like API key authentication, rate limiting, and a configurable setup for production deployment.

## Quick Start Guide

Get the server up and running in 5 minutes.

**1. Clone the repository:**
```bash
git clone [repository-url]
cd SIM-ONE
```

**2. Set up a Python virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables:**
Create a `.env` file in the root directory and add the required variables. See the [Configuration Guide](./docs/CONFIGURATION.md) for details. A minimal example:
```
MCP_API_KEY="your-secret-api-key"
```

**5. Run the server:**
```bash
uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000
```
The server is now running and accessible at `http://localhost:8000`.

---

## Using SIM-ONE Protocols as Standalone Tools

**Each protocol is available as a CLI tool in `/tools/` for integration with autonomous agents:**

### Individual Protocol Tools
- `run_rep_tool.py` - Reasoning & Explanation (deductive, inductive, abductive, analogical, causal)
- `run_esl_tool.py` - Emotional State Analysis (multi-dimensional emotion detection)
- `run_vvp_tool.py` - Validation & Verification (input validation and logical checking)
- (see [tools/README.md](tools/README.md) for complete list)

### Governance Tools
- `run_five_laws_validator.py` - **Validate any response against Five Laws** ⭐
- `run_governed_response.py` - Generate governed AI responses (planned)
- `run_cognitive_workflow.py` - Multi-step governed processing (planned)

### Example Usage

```bash
# Validate AI response against Five Laws
python tools/run_five_laws_validator.py --text "response to check"

# Use REP for reasoning
python tools/run_rep_tool.py --reasoning-type deductive \
  --facts "Socrates is a man" "All men are mortal" \
  --rules '[["Socrates is a man", "All men are mortal"], "Socrates is mortal"]'

# Chain protocols for governed workflow
python tools/run_rep_tool.py --json '{...}' | \
  python tools/run_vvp_tool.py | \
  python tools/run_five_laws_validator.py
```

**📖 Full Tool Documentation**: [tools/README.md](tools/README.md)
**🔧 Integration Guide**: [../PAPER2AGENT_INTEGRATION.md](../PAPER2AGENT_INTEGRATION.md)
**📋 Tool Manifest**: [tools/tools_manifest.json](tools/tools_manifest.json)

---

## Architecture Overview

The mCP Server consists of several key components that work in tandem:

*   **API Gateway (`main.py`)**: The public-facing interface of the server, handling incoming requests, authentication, and routing.
*   **Orchestration Engine**: The core component that executes workflows by managing and sequencing Cognitive Protocols.
*   **Protocol Manager**: Responsible for dynamically discovering and loading available protocols (e.g., REP, ESL, MTP).
*   **Cognitive Protocols**: Individual modules that perform specific cognitive tasks. Each protocol is self-contained and exposes a standard interface.
*   **Cognitive Governance Engine**: A metacognitive layer that monitors, validates, and governs the outputs of all protocols to ensure coherence and quality.
*   **Memory Manager**: Interfaces with a Redis database to provide persistent memory and session management.

For a more detailed breakdown, please see the full [Architecture Documentation](./docs/ARCHITECTURE.md).

## Installation

For detailed, step-by-step installation instructions for various platforms and production environments, please refer to our comprehensive [Installation Guide](./docs/INSTALLATION.md).

## Configuration

The server is configured entirely through environment variables. For a full list of all required and optional variables, their purpose, and example values, please see the [Configuration Guide](./docs/CONFIGURATION.md).

## Security & Governance

The mCP Server includes a governance layer and structured audit logging. Enable and tune these with environment flags:

- Governance
  - `GOV_ENABLE` (default: `true`): Toggles governance evaluation.
  - `GOV_MIN_QUALITY` (default: `0.6`): Minimum acceptable protocol quality score; lower values are flagged.
  - `GOV_REQUIRE_COHERENCE` (default: `false`): If `true`, incoherence triggers a single retry of the step; persistent incoherence aborts the workflow with an error.
- Rate limiting (per endpoint)
  - `RATE_LIMIT_EXECUTE` (default: `20/minute`)
  - `RATE_LIMIT_PROTOCOLS` (default: `60/minute`)
  - `RATE_LIMIT_TEMPLATES` (default: `60/minute`)
  - `RATE_LIMIT_SESSION` (default: `30/minute`)
  - `RATE_LIMIT_METRICS` (default: `10/minute`)

Audit logs are emitted in JSON format by a dedicated `audit` logger:

- File: `security_events.log` (rotated daily; 14 backups)
- Location: server working directory (when running from this repo, see `code/security_events.log`)
- Event examples: `recovery_decision`, `governance_incoherence_detected`, `governance_abort`, `execute_completed`
- No sensitive content is logged; governance summaries include aggregate scores and booleans only.

Tip: Responses from `/execute` include a `governance_summary` field with `quality_scores` and `is_coherent` for quick inspection.

Admin model management (MVLM local)
- List available model aliases and show active alias (admin): `GET /admin/models`
- Activate a model by alias (admin): `POST /admin/models/activate` with body `{ "alias": "main" }`
- Configuration via `.env`:
  - `MVLM_MODEL_DIRS=main:models/mvlm_gpt2/mvlm_final,enhanced:/opt/models/next`
- `ACTIVE_MVLM_MODEL=main`

Execution Controls
- Timeouts and concurrency
  - `PROTOCOL_TIMEOUT_MS` (default: 10000): Max time per protocol step.
  - `MAX_PARALLEL_PROTOCOLS` (default: 4): Cap for parallel step execution.
  - `PROTOCOL_TIMEOUTS_MS`: Optional per‑protocol overrides as `name:ms` pairs, comma‑separated (e.g., `ReasoningAndExplanationProtocol:15000,EmotionalStateLayerProtocol:8000`).
- Quotas and rate limiting
  - API/IP rate limiting configured via `RATE_LIMIT_*` envs.
  - Optional per‑API‑key quota: `API_KEY_QUOTA_PER_MINUTE` (default: 0 disables).
- Metrics
  - JSON metrics: `GET /metrics` (admin)
  - Prometheus metrics: `GET /metrics/prometheus` (admin unless `METRICS_PUBLIC=true`)
  - Prometheus example config: see `code/config/prometheus.yml`.
  - Grafana example dashboard: `code/mcp_server/docs/monitoring/grafana_governance_dashboard.json`.

## Local Monitoring

Run a local Prometheus + Grafana stack to visualize governance/recovery metrics.

- Start services (single node app + Redis + Prometheus + Grafana):
  - `docker-compose -f code/docker-compose.override.yml up`
- Endpoints:
  - App: `http://localhost:8000`
  - Prometheus: `http://localhost:9090` (scrapes `/metrics/prometheus`)
  - Grafana: `http://localhost:3000` (default admin password: `admin`)
- Import dashboard:
  - In Grafana, go to Dashboards → Import
  - Upload `code/mcp_server/docs/monitoring/grafana_governance_dashboard.json`
  - Select the Prometheus data source and save
- Notes:
  - The app exposes Prometheus metrics at `/metrics/prometheus`; set `METRICS_PUBLIC=true` for local testing (already set in override compose).
  - Governance counters include coherence failures, governance aborts, recovery retries, and fallbacks.

## Model Test CLI

Quickly test a local GPT‑2–style MVLM without running the server.

- Script: `code/tools/mvlm_textgen.py`
- Requirements: included in `code/requirements.txt` (`transformers`, `torch`, `safetensors`).
- Usage examples:
  - Project MVLMEngine (simple defaults)
    - `python code/tools/mvlm_textgen.py --model-dir models/mvlm_gpt2/mvlm_final --prompt "Give three bullet points on SIM-ONE governance:"`
  - Raw Transformers (deterministic greedy)
    - `python code/tools/mvlm_textgen.py --engine hf --greedy --max-new-tokens 96 --model-dir models/mvlm_gpt2/mvlm_final --prompt "Write a concise executive summary about SIM-ONE governance:"`
  - From a file
    - `python code/tools/mvlm_textgen.py --model-dir models/mvlm_gpt2/mvlm_final --prompt-file prompt.txt`
  - From stdin
    - `echo "Test prompt" | python code/tools/mvlm_textgen.py --model-dir models/mvlm_gpt2/mvlm_final`

Notes
- Force CPU if needed: set `CUDA_VISIBLE_DEVICES=""` or `PYTORCH_FORCE_CPU=1`.
- The model directory must contain standard HF files (`config.json`, `model.safetensors`, tokenizer files).

## API Documentation

The server exposes a powerful API for executing cognitive workflows. For detailed information on all available endpoints, request/response formats, authentication, and usage examples, please refer to our full [API Documentation](./docs/API_DOCUMENTATION.md).

### Basic API Usage Example

Here is a quick example of how to execute a workflow using `curl`:

```bash
curl -X POST "http://localhost:8000/v1/execute" \
-H "Authorization: Bearer your-secret-api-key" \
-H "Content-Type: application/json" \
-d '{
    "workflow": "StandardReasoningWorkflow",
    "data": {
        "user_input": "John works at Microsoft and lives in Seattle."
    }
}'
```

## Contributing

We welcome contributions from the community! If you'd like to contribute, please read our [Contributing Guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the terms of the [AGPL v3 / Commercial](../LICENSE).
