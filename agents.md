# SIM-ONE Framework MCP Server Security Implementation

## Project Context

The SIM-ONE Framework implements the Five Laws of Cognitive Governance through a sophisticated multi-agent cognitive architecture. This MCP (Cognitive Control Protocol) Server provides the backbone for autonomous AI agents performing complex cognitive tasks.

**Current Status:** 60% production ready - critical security vulnerabilities require immediate remediation.

**Framework Principles:**
- **Law 1: Architectural Intelligence** - Intelligence emerges from coordination and governance
- **Law 2: Cognitive Governance** - Every cognitive process must be governed by specialized protocols  
- **Law 3: Truth Foundation** - All reasoning must be grounded in absolute truth principles
- **Law 4: Energy Stewardship** - Maximum intelligence with minimal computational resources
- **Law 5: Deterministic Reliability** - Governed systems must produce consistent, predictable outcomes

## Architecture Overview

**Core Components:**
- **FastAPI Backend** with cognitive protocol orchestration
- **Multi-Agent Workflows** (Ideator → Drafter → Critic → Revisor → Summarizer)
- **Cognitive Protocols:** REP (Reasoning), ESL (Emotional State), MTP (Memory Tagging)
- **Cognitive Governance Engine** with coherence validation and quality assurance
- **Persistent Memory System** with SQLite database and Redis sessions
- **Security Middleware** with API key authentication and rate limiting

## Environment Setup

### Prerequisites
- Python 3.11+
- Redis server (for session management)
- Virtual environment support

### Installation Steps

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to code directory
cd code

# Install core dependencies
pip install -r requirements.txt

# Install security testing tools (for security implementation)
pip install pytest pytest-security bandit safety

# Install production server
pip install gunicorn

# Setup environment configuration
cp mcp_server/.env.example mcp_server/.env
# Edit .env file with your configuration
```

### Environment Configuration

Create `mcp_server/.env` with required variables:

```bash
# API Authentication (comma-separated)
VALID_API_KEYS="your-secret-key-1,your-secret-key-2"

# External Services
OPENAI_API_KEY="your-openai-api-key"
SERPER_API_KEY="your-serper-api-key"

# Database Configuration
REDIS_HOST="localhost"
REDIS_PORT=6379

# Neural Engine Backend
NEURAL_ENGINE_BACKEND="openai"
LOCAL_MODEL_PATH="models/llama-3.1-8b.gguf"
```

### Database Initialization

```bash
# Initialize SQLite database for memory management
cd code
python -m mcp_server.database.memory_database
```

## Running the Server

### Development Mode
```bash
cd code
uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
```bash
cd code
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mcp_server.main:app -b 0.0.0.0:8000
```

## Testing Framework

### Core Testing (unittest framework)
```bash
# Run all existing tests
cd code
python -m unittest discover mcp_server/tests/

# Run specific test modules
python -m unittest mcp_server.tests.test_main
python -m unittest mcp_server.tests.test_esl_protocol
python -m unittest mcp_server.tests.test_mtp_protocol
```

### Security Testing (after security implementation)
```bash
# Security vulnerability scanning
bandit -r mcp_server/
safety check

# Security-specific tests
python -m unittest mcp_server.tests.security.test_cors_security
python -m unittest mcp_server.tests.security.test_endpoint_auth
python -m unittest mcp_server.tests.security.test_error_handling
```

## SIM-ONE Cognitive Protocols

### Available Protocols
- **ReasoningAndExplanationProtocol (REP)** - Deductive, inductive, and abductive reasoning
- **EmotionalStateLayerProtocol (ESL)** - Sophisticated emotional analysis and sentiment detection
- **MemoryTaggerProtocol (MTP)** - Entity extraction and relationship mapping
- **IdeatorProtocol** - Creative ideation and concept generation
- **DrafterProtocol** - Content drafting and structuring
- **CriticProtocol** - Critical analysis and feedback
- **RevisorProtocol** - Content revision and improvement
- **SummarizerProtocol** - Final summarization and output generation

### Workflow Templates
```bash
# Test individual protocols
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"protocol_names": ["ReasoningAndExplanationProtocol"], "initial_data": {"facts": ["test fact"]}}'

# Test workflow templates
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "writing_team", "initial_data": {"topic": "AI safety"}}'

# Available templates: analyze_only, full_reasoning, writing_team
curl -X GET "http://localhost:8000/templates" -H "X-API-Key: your-key"
```

## Cognitive Governance Engine

### Critical Components
- **Coherence Validator** - Ensures logical consistency across protocol outputs
- **Quality Assurance** - Validates output quality and relevance
- **Error Recovery** - Handles cognitive failures and provides resilience
- **Metacognitive Engine** - Self-monitoring and performance optimization

### Governance Validation
```bash
# Test coherence validation
python -c "from mcp_server.cognitive_governance_engine.coherence_validator.coherence_checker import validate_coherence; print('Coherence validation operational')"

# Test quality assurance
python -c "from mcp_server.cognitive_governance_engine.quality_assurance.quality_scorer import assess_quality; print('Quality assessment operational')"
```

## Security Implementation Context

### Current Security Status
- ✅ **Environment Variables** - Hardcoded secrets eliminated
- ✅ **Basic Authentication** - API key validation implemented
- ❌ **CORS Security** - Wildcard origins allow any domain (CRITICAL)
- ❌ **Endpoint Protection** - /protocols, /templates, /session unprotected (HIGH)
- ❌ **Error Handling** - Internal information disclosed in errors (HIGH)
- ❌ **Security Infrastructure** - Missing .gitignore, security headers (MEDIUM)

### Critical Security Endpoints
- **`/execute`** - Main cognitive workflow endpoint (PROTECTED)
- **`/protocols`** - Exposes cognitive architecture (NEEDS PROTECTION)
- **`/templates`** - Reveals workflow capabilities (NEEDS PROTECTION)
- **`/session/{id}`** - Memory and session access (NEEDS AUTHORIZATION)
- **`/`** - Status endpoint (PUBLIC - OK)

### Security Implementation Requirements

**CORS Configuration:**
```python
# CURRENT (INSECURE)
allow_origins=["*"]  # Allows any domain
allow_credentials=True  # Dangerous with wildcards

# REQUIRED (SECURE)
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
allow_credentials=True
```

**Endpoint Protection:**
```python
# ADD AUTHENTICATION TO:
@app.get("/protocols", dependencies=[Depends(get_api_key)])
@app.get("/templates", dependencies=[Depends(get_api_key)])
@app.get("/session/{session_id}", dependencies=[Depends(get_api_key)])
```

## SIM-ONE Framework Compliance

### Security Must Preserve Cognitive Governance
When implementing security fixes, ensure:
- **Cognitive protocol functionality** remains intact
- **Workflow orchestration** continues to operate
- **Memory management** security doesn't break persistence
- **Governance engine** validation processes remain active

### Framework Compliance Testing
```bash
# Test cognitive governance after security changes
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -d '{"template_name": "full_reasoning", "initial_data": {"facts": ["Security test"]}}'

# Verify protocol discovery
curl -X GET "http://localhost:8000/protocols" -H "X-API-Key: your-key"

# Test memory persistence
curl -X GET "http://localhost:8000/session/test-session" -H "X-API-Key: your-key"
```

### Five Laws Compliance Validation
- **Law 2 Compliance:** Security processes must be governed by clear protocols
- **Law 5 Compliance:** Security behavior must be deterministic and reliable

## Production Deployment

### Production Requirements
- **Gunicorn + Uvicorn workers** (not standalone uvicorn)
- **Redis cluster** for session management
- **Load balancer** for high availability
- **GPU support** for local neural models (if using LOCAL_MODEL_PATH)
- **Firewall configuration** with only necessary ports open

### Production Validation
```bash
# Test production server startup
gunicorn -w 2 -k uvicorn.workers.UvicornWorker mcp_server.main:app -b 0.0.0.0:8000

# Verify all cognitive protocols work in production
curl -X GET "http://localhost:8000/protocols"

# Test workflow execution under load
for i in {1..10}; do
  curl -X POST "http://localhost:8000/execute" \
    -H "X-API-Key: your-key" \
    -d '{"template_name": "analyze_only", "initial_data": {"text": "test"}}' &
done
wait
```

## Development Guidelines

### Code Quality Standards
- **No placeholders** in security-related code
- **Comprehensive error handling** for all security functions
- **Clear documentation** for all security decisions
- **Unit tests** for all security components using unittest framework

### SIM-ONE Security Principles
- **Fail securely** - Default to denying access
- **Preserve cognitive governance** - Security must not break cognitive processes
- **Deterministic behavior** - Security responses must be predictable
- **Energy efficient** - Security overhead must be minimal

### Testing Requirements
```bash
# Security implementation must pass:
python -m unittest mcp_server.tests.security.test_cors_security
python -m unittest mcp_server.tests.security.test_endpoint_auth
python -m unittest mcp_server.tests.security.test_cognitive_protocols_with_auth

# Cognitive governance must remain functional:
python -m unittest mcp_server.tests.test_esl_protocol
python -m unittest mcp_server.tests.test_mtp_protocol
python -m unittest mcp_server.tests.test_main
```

## Troubleshooting

### Common Issues
- **Redis Connection Failed:** Ensure Redis server is running on configured host/port
- **API Key Invalid:** Check VALID_API_KEYS environment variable format
- **Protocol Not Found:** Verify protocol files exist in mcp_server/protocols/
- **Memory Database Error:** Run database initialization script

### Debug Commands
```bash
# Check environment configuration
cd code && python -c "from mcp_server.config import settings; print(settings.model_dump_json(indent=2))"

# Test API key parsing
cd code && python -c "from mcp_server.config import settings; print(settings.get_valid_api_keys())"

# Verify protocol loading
cd code && python -c "from mcp_server.protocol_manager.protocol_manager import ProtocolManager; pm = ProtocolManager(); print(list(pm.protocols.keys()))"
```

## Security Implementation Mission

Transform the MCP server from development prototype to production-ready system while maintaining full cognitive governance functionality and SIM-ONE Framework compliance.

**Remember:** "In structure there is freedom" - Proper security structure provides the freedom to deploy confidently in production environments while preserving the cognitive capabilities that make the SIM-ONE Framework revolutionary.