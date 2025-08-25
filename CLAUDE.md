# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Server
```bash
# Start the mCP server (from /var/www/SIM-ONE/code/)
cd /var/www/SIM-ONE/code
uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
cd /var/www/SIM-ONE/code
python -m unittest discover mcp_server/tests/

# Run specific test file
python -m unittest mcp_server.tests.test_main
python -m unittest mcp_server.tests.test_esl_protocol

# Run individual test
python test_advanced_rep.py

# Note: Tests include role-based access control, security headers, and advanced input validation
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies include: FastAPI, uvicorn, redis, openai, llama-cpp-python, python-dotenv
# Security: python-jose for JWT, slowapi for rate limiting
```

## Architecture Overview

SIM-ONE is a **Multi-agent Cognitive Protocol (mCP) Server** that orchestrates cognitive workflows through modular protocols. The architecture follows a hub-and-spoke pattern with the Orchestration Engine at the center.

### Core Components

**API Gateway** (`mcp_server/main.py`)
- FastAPI-based entry point with comprehensive security features
- Role-based access control (admin, user, read-only roles)
- Advanced input validation and security headers middleware  
- Rate limiting and API key authentication
- Handles workflow execution requests and routing

**Orchestration Engine** (`mcp_server/orchestration_engine/`)
- Executes workflows by chaining cognitive protocols sequentially or in parallel
- Manages data context flow between protocols
- Integrates with Cognitive Governance Engine for validation

**Protocol Manager** (`mcp_server/protocol_manager/`)
- Dynamically discovers and loads cognitive protocols from `mcp_server/protocols/`
- Each protocol has a `protocol.json` configuration file and main implementation

**Cognitive Governance Engine** (`mcp_server/cognitive_governance_engine/`)
- Metacognitive layer that validates protocol outputs
- Includes coherence checking, quality assurance, and error recovery
- Implements adaptive learning and performance monitoring

### Key Cognitive Protocols

Located in `mcp_server/protocols/`, each protocol is self-contained:

- **REP** (Reasoning and Explanation): Rule-based logical reasoning system
- **ESL** (Emotional State Layer): Sophisticated emotion detection and analysis
- **MTP** (Memory Tagger): Named entity recognition and relationship extraction
- **Ideator/Critic/Revisor**: Multi-agent writing and content creation workflow
- **HIP** (Hierarchical Information Processing): Advanced information structuring

### Configuration

All configuration via environment variables with `.env` file support (see `mcp_server/config.py`):
- `VALID_API_KEYS`: Comma-separated list of API keys with role-based access
- `REDIS_HOST`/`REDIS_PORT`: Redis for session and memory persistence
- `NEURAL_ENGINE_BACKEND`: "openai" or "local" for LLM backend
- `OPENAI_API_KEY`: Required if using OpenAI backend
- `LOCAL_MODEL_PATH`: Path to GGUF model if using local backend
- `ALLOWED_ORIGINS`: CORS configuration for frontend origins
- `SERPER_API_KEY`: Google Search API key for RAG functionality

### Workflow System

Workflows defined in `mcp_server/workflow_templates.json`:
- **Sequential execution**: Protocols run one after another, passing context
- **Parallel execution**: Protocols run simultaneously
- **Loop structures**: Support for iterative protocol execution (e.g., critic/revisor loops)
- **Custom workflows**: Can define complex multi-step cognitive processes

### Memory and Session Management

- **Redis-based persistence**: Session history and tagged memories
- **Memory Manager**: Interfaces with Redis for persistent storage
- **Session Manager**: Tracks conversation context with user-scoped session isolation
- **Role-based session access**: Users can only access their own sessions, admins can access all

## Development Guidelines

- **Protocol Development**: New protocols go in `mcp_server/protocols/<name>/` with `protocol.json` config
- **Testing**: Use unittest framework with comprehensive security and RBAC tests
- **Security**: Advanced input validation blocks SQL injection, XSS, and command injection attacks
- **Role-Based Access**: Three roles (admin, user, read-only) with different endpoint permissions
- **Key Management**: API keys managed through `mcp_server/security/key_manager.py` 
- **Error Handling**: Cognitive Governance Engine provides error recovery strategies
- **Logging**: Structured logging throughout the system for debugging and monitoring

## Working Directory

Primary development occurs in `/var/www/SIM-ONE/code/` - this contains all the implementation files, documentation, and tests for the mCP server.

## Documentation Structure

- **`code/docs/`** - Complete technical documentation for developers
  - `API_DOCUMENTATION.md` - Complete API reference with authentication
  - `ARCHITECTURE.md` - System architecture and component details  
  - `CONFIGURATION.md` - Environment setup and configuration
  - `MVLM_INTEGRATION.md` - Neural engine backend setup
  - `PROTOCOL_DEVELOPMENT.md` - Guide for creating custom protocols
  - `PROTOCOLS.md` - Documentation for all cognitive protocols
  - `SECURITY.md` - Security implementation guide
  - `DEPLOYMENT.md` - Production deployment guide
  - `INSTALLATION.md` - Setup instructions
  - `TROUBLESHOOTING.md` - Common issues and solutions

- **Root Level Documentation** - Project-level information
  - `SECURITY.md` - Security policy for vulnerability reporting
  - `CLAUDE.md` - This file for Claude Code guidance
  - `DOCUMENTATION_PLAN.md` - Overall documentation planning