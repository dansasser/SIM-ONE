# SIM-ONE Framework MCP Server Security Implementation

## Project Context

The SIM-ONE Framework implements the Five Laws of Cognitive Governance through a sophisticated multi-agent cognitive architecture. This MCP (Cognitive Control Protocol) Server provides the backbone for autonomous AI agents performing complex cognitive tasks.

**Current Status:** 85% production ready - See [project_status.md](/code/mcp_server/project_status.md) for details on project status of the backend project.

**Framework Principles:**
- **Law 1: Architectural Intelligence** - Intelligence emerges from coordination and governance
- **Law 2: Cognitive Governance** - Every cognitive process must be governed by specialized protocols  
- **Law 3: Truth Foundation** - All reasoning must be grounded in absolute truth principles
- **Law 4: Energy Stewardship** - Maximum intelligence with minimal computational resources
- **Law 5: Deterministic Reliability** - Governed systems must produce consistent, predictable outcomes

## Architecture Overview

**Code Base Location**
- `/code/` (code root)
- `/code/mcp_server/` (backend)
- `/code/astro-chat-interface/` (frontend)

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

# Install security testing tools
pip install pytest bandit safety

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

# Security Configuration
ALLOWED_ORIGINS="http://localhost:3000,https://yourdomain.com"

# Neural Engine Backend
NEURAL_ENGINE_BACKEND="openai"
LOCAL_MODEL_PATH="models/llama-3.1-8b.gguf"

# Logging
LOG_LEVEL="INFO"
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
python -m unittest mcp_server.tests.test_memory_consolidation
python -m unittest mcp_server.tests.test_production_setup
```

### Security Testing
```bash
# Security vulnerability scanning
bandit -r mcp_server/
safety check

# Security-specific tests
python -m unittest mcp_server.tests.security.test_cors_security
python -m unittest mcp_server.tests.security.test_endpoint_auth
python -m unittest mcp_server.tests.security.test_error_handling
python -m unittest mcp_server.tests.security.test_cognitive_protocols_with_auth

# Run all security tests
python -m unittest discover mcp_server/tests/security/
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

## Current Security Implementation Status

### ✅ **IMPLEMENTED SECURITY FEATURES**
- **✅ Environment Variables** - Hardcoded secrets eliminated, configurable via .env
- **✅ Advanced Authentication** - Hashed API key system with role-based access control (admin/user/read-only)
- **✅ CORS Security** - Configurable origins via ALLOWED_ORIGINS (no wildcards)
- **✅ Endpoint Protection** - All sensitive endpoints protected with RBAC
- **✅ Secure Error Handling** - Sanitized error messages, no information disclosure
- **✅ Security Headers** - CSP, X-Frame-Options, X-Content-Type-Options implemented
- **✅ Rate Limiting** - IP-based rate limiting on all endpoints
- **✅ Input Validation** - Advanced input validation and sanitization
- **✅ Security Test Suite** - Comprehensive security test coverage
- **✅ Session Isolation** - User-specific session management with authorization
- **✅ Audit Logging** - Security events logged for monitoring

### 🔄 **IN PROGRESS**
- **🔄 Containerization** - Docker and docker-compose configurations
- **🔄 CI/CD Pipeline** - Automated security testing and deployment
- **🔄 Production Deployment** - Kubernetes manifests and deployment guides

### 📋 **PLANNED ENHANCEMENTS**
- **📋 PostgreSQL Support** - Production database alongside SQLite
- **📋 Secrets Management** - HashiCorp Vault or cloud secret managers
- **📋 Distributed Tracing** - OpenTelemetry for cognitive workflow tracing
- **📋 Advanced Monitoring** - Prometheus metrics and Grafana dashboards

### Critical Security Endpoints Status

| Endpoint | Status | Protection Level | Notes |
|----------|--------|------------------|-------|
| `/` | ✅ Public | None Required | Status endpoint |
| `/health` | ✅ Public | None Required | Health check endpoint |
| `/health/detailed` | ✅ Public | None Required | Detailed health status |
| `/execute` | ✅ Protected | Admin/User RBAC | Main cognitive workflow endpoint |
| `/protocols` | ✅ Protected | Admin/User/Read-only RBAC | Cognitive architecture discovery |
| `/templates` | ✅ Protected | Admin/User/Read-only RBAC | Workflow template access |
| `/session/{id}` | ✅ Protected | User isolation + RBAC | Session management with ownership |
| `/metrics` | ✅ Protected | Admin-only RBAC | System metrics (admin access) |

## Security Configuration Details

### CORS Configuration (SECURE)
```python
# CURRENT IMPLEMENTATION (SECURE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Configurable, no wildcards
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Environment Configuration
ALLOWED_ORIGINS="http://localhost:3000,https://yourdomain.com"
```

### Authentication & Authorization
```python
# RBAC Implementation
@app.get("/protocols", dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
@app.get("/templates", dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
@app.post("/execute", dependencies=[Depends(RoleChecker(["admin", "user"]))])
@app.get("/session/{session_id}", dependencies=[Depends(RoleChecker(["admin", "user"]))])
@app.get("/metrics", dependencies=[Depends(RoleChecker(["admin"]))])

# API Key Management
# - Hashed storage with individual salts
# - Role-based key assignment
# - Secure key validation
```

### Security Headers
```python
# Implemented Security Headers
'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; upgrade-insecure-requests;"
'X-Frame-Options': 'DENY'
'X-Content-Type-Options': 'nosniff'
'Referrer-Policy': 'strict-origin-when-cross-origin'
'Permissions-Policy': "geolocation=(), microphone=(), camera=()"
```

### Rate Limiting
```python
# IP-based rate limiting
limiter = Limiter(key_func=get_remote_address)
@limiter.limit("20/minute")  # Configurable per endpoint
```

## SIM-ONE Framework Compliance

### Security Preserves Cognitive Governance
The security implementation maintains full compliance with the Five Laws:

- **✅ Architectural Intelligence (Law 1)** - Security enhances coordination through proper access controls
- **✅ Cognitive Governance (Law 2)** - Security protocols govern access to cognitive processes
- **✅ Truth Foundation (Law 3)** - Security ensures authentic user identity for reasoning context
- **✅ Energy Stewardship (Law 4)** - Efficient security middleware with minimal overhead
- **✅ Deterministic Reliability (Law 5)** - Consistent, predictable security behavior

### Framework Compliance Testing
```bash
# Test cognitive governance after security changes
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "full_reasoning", "initial_data": {"facts": ["Security test"]}}'

# Verify protocol discovery
curl -X GET "http://localhost:8000/protocols" -H "X-API-Key: your-key"

# Test memory persistence with session isolation
curl -X GET "http://localhost:8000/session/test-session" -H "X-API-Key: your-key"
```

## Production Deployment

### Production Requirements
- **✅ Gunicorn + Uvicorn workers** (implemented and tested)
- **✅ Redis cluster** for session management (configured)
- **🔄 Load balancer** configuration (in progress)
- **✅ Security headers** and CORS protection (implemented)
- **📋 GPU support** for local neural models (planned)
- **📋 Firewall configuration** with only necessary ports (documentation pending)

### Production Validation
```bash
# Test production server startup
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mcp_server.main:app -b 0.0.0.0:8000

# Verify all cognitive protocols work in production
curl -X GET "http://localhost:8000/protocols" -H "X-API-Key: your-key"

# Test workflow execution under load
for i in {1..10}; do
  curl -X POST "http://localhost:8000/execute" \
    -H "X-API-Key: your-key" \
    -H "Content-Type: application/json" \
    -d '{"template_name": "analyze_only", "initial_data": {"text": "test"}}' &
done
wait

# Health check validation
curl -X GET "http://localhost:8000/health/detailed"
```

## Development Guidelines

### Code Quality Standards
- **✅ No placeholders** in security-related code (implemented)
- **✅ Comprehensive error handling** for all security functions (implemented)
- **✅ Clear documentation** for all security decisions (documented)
- **✅ Unit tests** for all security components using unittest framework (complete)

### SIM-ONE Security Principles
- **✅ Fail securely** - Default to denying access (implemented)
- **✅ Preserve cognitive governance** - Security enhances cognitive processes (validated)
- **✅ Deterministic behavior** - Security responses are predictable (tested)
- **✅ Energy efficient** - Security overhead is minimal (optimized)

### Testing Requirements
```bash
# Security implementation validation:
python -m unittest mcp_server.tests.security.test_cors_security
python -m unittest mcp_server.tests.security.test_endpoint_auth
python -m unittest mcp_server.tests.security.test_error_handling
python -m unittest mcp_server.tests.security.test_cognitive_protocols_with_auth

# Cognitive governance functionality validation:
python -m unittest mcp_server.tests.test_esl_protocol
python -m unittest mcp_server.tests.test_mtp_protocol
python -m unittest mcp_server.tests.test_main
python -m unittest mcp_server.tests.test_memory_consolidation

# Production readiness validation:
python -m unittest mcp_server.tests.test_production_setup
```

## Infrastructure & Deployment (Next Phase)

### Containerization (In Development)
```bash
# Docker development environment
docker-compose -f docker-compose.dev.yml up

# Production containerization
docker-compose -f docker-compose.prod.yml up

# Kubernetes deployment
kubectl apply -f k8s/
```

### CI/CD Pipeline (Planned)
```yaml
# Automated testing and deployment
- Security scanning (bandit, safety)
- Unit test execution
- Integration testing
- Production deployment
- Health check validation
```

## Troubleshooting

### Common Issues
- **Redis Connection Failed:** Ensure Redis server is running on configured host/port
- **API Key Invalid:** Check VALID_API_KEYS environment variable format
- **Protocol Not Found:** Verify protocol files exist in mcp_server/protocols/
- **Memory Database Error:** Run database initialization script
- **CORS Errors:** Check ALLOWED_ORIGINS configuration in .env file

### Debug Commands
```bash
# Check environment configuration
cd code && python -c "from mcp_server.config import settings; print(settings.model_dump_json(indent=2))"

# Test API key system
cd code && python -c "from mcp_server.security.key_manager import validate_api_key; print('Key validation:', validate_api_key('test-key'))"

# Verify protocol loading
cd code && python -c "from mcp_server.protocol_manager.protocol_manager import ProtocolManager; pm = ProtocolManager(); print('Protocols:', list(pm.protocols.keys()))"

# Test security configuration
cd code && python -c "from mcp_server.config import settings; print('CORS Origins:', settings.ALLOWED_ORIGINS)"
```

### Security Event Monitoring
```bash
# Monitor security logs
tail -f security_events.log

# Check rate limiting status
curl -X GET "http://localhost:8000/health/detailed"

# Validate security headers
curl -I "http://localhost:8000/"
```

## Security Implementation Achievement

**Mission Accomplished:** The MCP server has been transformed from development prototype to production-ready system while maintaining full cognitive governance functionality and SIM-ONE Framework compliance.

**Current Status: 85% Production Ready**
- ✅ Security hardening complete
- ✅ Comprehensive test coverage
- ✅ Production-ready authentication and authorization
- 🔄 Infrastructure automation in progress
- 📋 Advanced enterprise features planned

**Remember:** "In structure there is freedom" - The implemented security structure provides the freedom to deploy confidently in production environments while preserving and enhancing the cognitive capabilities that make the SIM-ONE Framework revolutionary.

## Next Phase: Infrastructure & Enterprise Features

See [project_status.md](/code/mcp_server/project_status.md) for the comprehensive roadmap of remaining infrastructure and enterprise feature implementation.