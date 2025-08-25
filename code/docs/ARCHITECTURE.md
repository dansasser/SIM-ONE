# SIM-ONE mCP Server Architecture

This document provides a comprehensive overview of the SIM-ONE Multi-agent Cognitive Protocol (mCP) Server's technical architecture, including all major components, data flows, security layers, and scalability considerations.

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Core Components](#core-components)
3. [Cognitive Governance Engine](#cognitive-governance-engine)
4. [Security Architecture](#security-architecture)
5. [Data Flow & Communication](#data-flow--communication)
6. [Database Schema](#database-schema)
7. [Scalability Design](#scalability-design)
8. [Extension Points](#extension-points)

---

## System Architecture Overview

### High-Level Architecture

The SIM-ONE mCP Server follows a **layered, microservice-inspired architecture** with a central orchestration engine coordinating multiple cognitive protocols. The system is designed for modularity, security, and scalability.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
├─────────────────────┬─────────────────┬─────────────────────────┤
│   Web Applications  │   Mobile Apps   │    Direct API Clients   │
└─────────────────────┴─────────────────┴─────────────────────────┘
                                 │
                          ┌──────┴──────┐
                          │   HTTPS     │
                          │ (TLS 1.3)   │
                          └──────┬──────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  • Security Headers Middleware     • Input Validation          │
│  • CORS Protection                 • Rate Limiting              │
│  • API Key Authentication          • RBAC Authorization         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY                                │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application (main.py)                                  │
│  • Request/Response Handling       • Error Management          │
│  • Endpoint Routing                • Session Management        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATION ENGINE                            │
├─────────────────────────────────────────────────────────────────┤
│  • Workflow Execution              • Context Management        │
│  • Protocol Coordination           • Error Recovery            │
│  • Parallel/Sequential Processing  • Resource Monitoring       │
└───────┬─────────────────────┬─────────────────────┬─────────────┘
        │                     │                     │
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐
│   PROTOCOL    │    │ COGNITIVE       │    │  RESOURCE   │
│   MANAGER     │    │ GOVERNANCE      │    │  MANAGER    │
│               │    │ ENGINE          │    │             │
├───────────────┤    ├─────────────────┤    ├─────────────┤
│• Discovery    │    │• Quality        │    │• CPU/Memory │
│• Loading      │    │  Assurance      │    │  Monitoring │
│• Instantiation│    │• Coherence      │    │• Performance│
│               │    │  Validation     │    │  Profiling  │
│               │    │• Error Recovery │    │             │
│               │    │• Adaptive       │    │             │
│               │    │  Learning       │    │             │
└───────┬───────┘    └─────────────────┘    └─────────────┘
        │
        │
┌─────────────────────────────────────────────────────────────────┐
│                 COGNITIVE PROTOCOLS                             │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│   REP   │   ESL   │   MTP   │   HIP   │ Ideator │ Critic  │ ... │
│(Reason) │(Emotion)│(Memory) │(Hierarc)│(Create) │(Review) │     │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Redis (Session & Memory)       │  SQLite (Persistent Memory)   │
│  • Session History              │  • Entity Relationships       │
│  • User Context                 │  • Memory Graphs              │
│  • Temporary Data               │  • Long-term Knowledge        │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                   NEURAL ENGINE LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  OpenAI Backend                 │  Local MVLM Backend           │
│  • GPT-4/GPT-3.5               │  • Local GGUF Models          │
│  • Remote API Calls            │  • llama-cpp-python           │
│  • Rate Limited                │  • GPU Acceleration           │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

1. **Modularity**: Each component has a single responsibility and clear interfaces
2. **Extensibility**: New protocols can be added without modifying core components
3. **Security-First**: Multi-layered security with defense in depth
4. **Scalability**: Designed for horizontal scaling and resource efficiency
5. **Observability**: Comprehensive logging, monitoring, and error tracking
6. **Fault Tolerance**: Graceful degradation and error recovery mechanisms

---

## Core Components

### API Gateway (`mcp_server/main.py`)

**Purpose**: Single entry point for all external communication with comprehensive security and routing.

**Architecture**:
```python
# FastAPI Application Structure
app = FastAPI(title="mCP Server", version=settings.APP_VERSION)

# Middleware Stack (order matters)
app.add_middleware(SecurityHeadersMiddleware)      # Security headers
app.add_middleware(CORSMiddleware)                 # CORS protection  
app.state.limiter = limiter                       # Rate limiting
add_exception_handlers(app)                       # Custom error handling
```

**Key Features**:
- **FastAPI Framework**: High-performance async web framework
- **Role-Based Access Control**: Admin, user, read-only roles
- **Rate Limiting**: 20 req/min on execute endpoint using slowapi
- **Input Validation**: Advanced pattern matching for injection attacks
- **Security Headers**: CSP, X-Frame-Options, HSTS, etc.
- **Error Handling**: Structured error responses with proper HTTP codes

**Endpoints**:
- `POST /execute` - Execute cognitive workflows
- `GET /protocols` - List available protocols  
- `GET /templates` - List workflow templates
- `GET /session/{id}` - Retrieve session history
- `GET /` - Health check

### Orchestration Engine (`mcp_server/orchestration_engine/`)

**Purpose**: Central coordinator that executes cognitive workflows by managing protocol sequences and data context flow.

**Architecture**:
```python
class OrchestrationEngine:
    def __init__(self, protocol_manager, resource_manager, memory_manager):
        self.protocol_manager = protocol_manager
        self.resource_manager = resource_manager  
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor()  # For parallel execution
```

**Workflow Execution Modes**:
1. **Sequential Processing**: Protocols execute one after another
2. **Parallel Processing**: Multiple protocols execute concurrently
3. **Loop Structures**: Iterative protocol execution (e.g., critic/revisor loops)
4. **Conditional Logic**: Branch execution based on context

**Context Management**:
```python
# Context flow through workflow
context = {
    "user_input": "...",           # Initial user input
    "session_id": "...",          # Session identifier
    "batch_memory": [...],        # Retrieved memories
    "ProtocolName": {...},        # Protocol outputs
    "error": "..."                # Error handling
}
```

**Key Features**:
- **Dynamic Workflow Definition**: JSON-based workflow templates
- **Context Propagation**: Seamless data flow between protocols
- **Error Recovery**: Graceful handling of protocol failures
- **Resource Monitoring**: CPU/memory usage tracking
- **Batch Memory Access**: Efficient memory retrieval for sessions

### Protocol Manager (`mcp_server/protocol_manager/`)

**Purpose**: Discovers, loads, and manages the lifecycle of cognitive protocols through a plugin-like architecture.

**Discovery Process**:
```python
def scan_protocols(self):
    """Scans mcp_server/protocols/ directory"""
    for root, _, files in os.walk(self.protocol_dir):
        if "protocol.json" in files:
            manifest = json.load(open("protocol.json"))
            protocol_name = manifest.get("name")
            self.protocols[protocol_name] = manifest
```

**Protocol Structure**:
```
mcp_server/protocols/
├── protocol_name/
│   ├── __init__.py
│   ├── protocol.json          # Metadata and configuration
│   ├── protocol_name.py       # Main implementation
│   └── additional_modules.py  # Supporting code
```

**Protocol Registration**:
```json
// protocol.json structure
{
    "name": "ProtocolName", 
    "version": "1.0.0",
    "description": "Protocol description",
    "dependencies": ["other_protocol"],
    "configuration": {
        "timeout": 30,
        "max_retries": 3
    }
}
```

### Memory Manager (`mcp_server/memory_manager/`)

**Purpose**: Manages persistent memory storage and retrieval with semantic search capabilities.

**Architecture**:
- **SQLite Database**: Long-term persistent memory storage
- **Redis Integration**: Session-based temporary memory
- **Semantic Search**: Memory retrieval by content similarity
- **Entity Relationship Mapping**: Graph-based knowledge storage

**Memory Types**:
1. **Session Memory**: Temporary conversation context
2. **Entity Memory**: Persistent entity and relationship data
3. **Episodic Memory**: Event-based memory storage
4. **Semantic Memory**: Concept and knowledge storage

**Memory Operations**:
```python
class MemoryManager:
    def store_memory(session_id: str, memory_data: dict) -> bool
    def retrieve_memories(session_id: str, query: str) -> List[dict]
    def get_all_memories(session_id: str) -> List[dict]
    def semantic_search(query: str, limit: int) -> List[dict]
```

### Resource Manager (`mcp_server/resource_manager/`)

**Purpose**: Monitors and manages system resources with performance profiling capabilities.

**Monitoring Capabilities**:
```python
@contextmanager
def profile(self, name: str = "operation"):
    """Profiles CPU and memory usage of operations"""
    # Tracks: CPU percentage, memory usage, execution time
    # Logs performance metrics for optimization
```

**Resource Tracking**:
- **CPU Usage**: Per-process and system-wide monitoring
- **Memory Usage**: RAM consumption tracking
- **Execution Time**: Protocol and operation timing
- **Resource Alerts**: Threshold-based notifications

---

## Cognitive Governance Engine

### Architecture Overview

The Cognitive Governance Engine provides **metacognitive oversight** of all protocol operations, ensuring quality, coherence, and adaptive improvement.

```
┌─────────────────────────────────────────────────────────────────┐
│                COGNITIVE GOVERNANCE ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │    ADAPTIVE     │    │     QUALITY     │    │   COHERENCE │  │
│  │    LEARNING     │    │   ASSURANCE     │    │ VALIDATION  │  │
│  │                 │    │                 │    │             │  │
│  │ • Pattern       │    │ • Completeness  │    │ • Logic     │  │
│  │   Recognition   │    │ • Relevance     │    │   Consistency│ │
│  │ • Performance   │    │ • Quality       │    │ • Sentiment │  │
│  │   Optimization  │    │   Scoring       │    │   Alignment │  │
│  │ • Learning      │    │ • Validation    │    │ • Cross-    │  │
│  │   Adaptation    │    │   Rules         │    │   Protocol  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ METACOGNITIVE   │    │ ERROR RECOVERY  │    │ PERFORMANCE │  │
│  │    ENGINE       │    │                 │    │  ASSESSOR   │  │
│  │                 │    │ • Error         │    │             │  │
│  │ • Self-Monitor  │    │   Classification│    │ • Metrics   │  │
│  │ • Self-Optimize │    │ • Recovery      │    │   Collection│  │
│  │ • Strategy      │    │   Strategies    │    │ • Benchmarks│  │
│  │   Selection     │    │ • Fallback      │    │ • Reporting │  │
│  │ • Performance   │    │   Management    │    │ • Analysis  │  │
│  │   Assessment    │    │ • Resilience    │    │             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   GOVERNANCE            │
                    │   ORCHESTRATOR          │
                    │                         │
                    │ • Component Coordination│
                    │ • Decision Making       │
                    │ • Policy Enforcement    │
                    └─────────────────────────┘
```

### Core Subsystems

#### Quality Assurance (`cognitive_governance_engine/quality_assurance/`)
- **Completeness Validator**: Ensures protocol outputs contain required fields
- **Quality Monitor**: Tracks output quality metrics over time
- **Quality Scorer**: Assigns quality scores based on multiple criteria
- **Relevance Analyzer**: Validates output relevance to input context

#### Coherence Validation (`cognitive_governance_engine/coherence_validator/`)
- **Logic Consistency**: Checks for contradictions between protocol outputs
- **Sentiment Alignment**: Ensures emotional analysis aligns with reasoning results
- **Cross-Protocol Validation**: Validates compatibility between protocol results

#### Error Recovery (`cognitive_governance_engine/error_recovery/`)
- **Error Classifier**: Categorizes and prioritizes different types of errors
- **Recovery Strategist**: Implements recovery strategies for different error types
- **Fallback Manager**: Provides alternative processing paths
- **Resilience Monitor**: Tracks system resilience and failure patterns

#### Adaptive Learning (`cognitive_governance_engine/adaptive_learning/`)
- **Pattern Recognizer**: Identifies patterns in protocol performance
- **Performance Tracker**: Monitors long-term performance trends
- **Learning Optimizer**: Optimizes protocol parameters based on performance
- **Adaptation Engine**: Implements adaptive changes to improve performance

### Governance Integration

**Protocol Execution Pipeline**:
```python
async def execute_protocol_with_governance(protocol_name, context):
    # 1. Pre-execution validation
    governance_engine.validate_input(context)
    
    # 2. Execute protocol with monitoring
    result = await protocol.execute(context)
    
    # 3. Quality assurance
    quality_score = governance_engine.assess_quality(result)
    
    # 4. Coherence validation
    coherence_check = governance_engine.validate_coherence(result, context)
    
    # 5. Adaptive learning
    governance_engine.record_performance(protocol_name, quality_score)
    
    # 6. Error recovery if needed
    if quality_score < threshold:
        result = governance_engine.attempt_recovery(result, context)
    
    return result
```

---

## Security Architecture

### Multi-Layered Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 7: APPLICATION                     │
├─────────────────────────────────────────────────────────────────┤
│  • Input Validation (SQL injection, XSS, Command injection)    │
│  • Business Logic Security   • Protocol-level Validation       │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 6: SESSION                         │
├─────────────────────────────────────────────────────────────────┤
│  • Session Isolation         • User-scoped Data Access         │
│  • Session Timeout           • Cross-session Prevention        │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                      LAYER 5: AUTHORIZATION                     │
├─────────────────────────────────────────────────────────────────┤
│  • Role-Based Access Control (Admin, User, Read-only)          │
│  • Endpoint-level Authorization  • Resource-level Permissions  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                      LAYER 4: AUTHENTICATION                    │
├─────────────────────────────────────────────────────────────────┤
│  • API Key Authentication    • SHA-256 Hashed Keys             │
│  • Unique Salt per Key       • Timing Attack Prevention        │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 3: TRANSPORT                       │
├─────────────────────────────────────────────────────────────────┤
│  • HTTPS/TLS 1.3             • Security Headers                │
│  • CORS Protection           • Content Security Policy         │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 2: NETWORK                         │
├─────────────────────────────────────────────────────────────────┤
│  • Rate Limiting              • IP-based Restrictions          │
│  • DDoS Protection           • Firewall Rules                  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                       LAYER 1: INFRASTRUCTURE                   │
├─────────────────────────────────────────────────────────────────┤
│  • OS-level Security          • Container Security             │
│  • Network Segmentation      • Hardware Security               │
└─────────────────────────────────────────────────────────────────┘
```

### Security Components

#### Authentication System
```python
# SHA-256 with unique salts
def hash_api_key(api_key: str, salt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(salt.encode('utf-8'))
    hasher.update(api_key.encode('utf-8'))
    return hasher.hexdigest()
```

#### Input Validation Pipeline
```python
# Multi-pattern validation
ATTACK_PATTERNS = {
    'sqli': [r"(\s*')\s*OR\s+'\d+'\s*=\s*'\d+'", ...],
    'xss': [r"<script.*?>.*?</script>", ...],
    'cmd_injection': [r";\s*(ls|cat|whoami)", ...]
}
```

#### Security Monitoring
- **Audit Logging**: All security events logged to `security_events.log`
- **Intrusion Detection**: Pattern-based suspicious activity detection
- **Rate Limit Monitoring**: Per-user and per-IP tracking
- **Session Anomaly Detection**: Unusual session behavior identification

---

## Data Flow & Communication

### Request Processing Flow

```
Client Request
      │
      ▼
┌─────────────────┐
│ Security        │
│ Middleware      │ ──── Authentication, Input Validation, Rate Limiting
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ API Gateway     │ ──── Request parsing, Routing, Error handling
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Orchestration   │ ──── Workflow management, Context preparation
│ Engine          │
└─────────────────┘
      │
      ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Protocol        │ │ Memory          │ │ Resource        │
│ Manager         │ │ Manager         │ │ Manager         │
└─────────────────┘ └─────────────────┘ └─────────────────┘
      │                       │                       │
      ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Protocol Execution                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │   REP   │  │   ESL   │  │   MTP   │  │   ...   │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────┐
│ Cognitive       │ ──── Quality validation, Coherence check
│ Governance      │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Response        │ ──── Result aggregation, Error handling
│ Assembly        │
└─────────────────┘
      │
      ▼
Client Response
```

### Protocol Communication Patterns

#### Sequential Execution
```python
# Linear protocol chain
workflow = [
    {"step": "EmotionalStateLayerProtocol"},
    {"step": "ReasoningAndExplanationProtocol"}, 
    {"step": "MemoryTaggerProtocol"}
]
# Context flows: Input → ESL → REP → MTP → Output
```

#### Parallel Execution
```python
# Concurrent protocol execution
workflow = [
    {"parallel": [
        {"step": "EmotionalStateLayerProtocol"},
        {"step": "MemoryTaggerProtocol"}
    ]}
]
# Context flows: Input → (ESL || MTP) → Output
```

#### Loop Structures
```python
# Iterative refinement
workflow = [
    {"step": "IdeatorProtocol"},
    {"loop": 3, "steps": [
        {"step": "CriticProtocol"},
        {"step": "RevisorProtocol"}
    ]}
]
```

### Inter-Component Communication

#### Message Passing
```python
# Standardized context structure
context = {
    "user_input": str,           # Original input
    "session_id": str,           # Session identifier  
    "batch_memory": List[dict],  # Retrieved memories
    "metadata": dict,            # Request metadata
    "ProtocolName": dict         # Protocol results
}
```

#### Event System
```python
# Event-driven architecture for monitoring
events = {
    "protocol.start": {"protocol": name, "timestamp": time},
    "protocol.complete": {"protocol": name, "duration": ms},
    "protocol.error": {"protocol": name, "error": message},
    "governance.quality_check": {"score": float, "threshold": float}
}
```

---

## Database Schema

### Redis Schema (Session & Temporary Data)

```redis
# Session Management
session:<session_id>:metadata -> {
    "owner": "user_id",
    "created_at": timestamp,
    "last_activity": timestamp,
    "ip_address": "client_ip"
}

session:<session_id>:history -> [
    {"user_request": {...}, "server_response": {...}},
    ...
]

# Rate Limiting
rate_limit:<user_id>:<endpoint> -> count (TTL: 60s)
rate_limit:<ip_address> -> count (TTL: 60s)

# Caching
cache:<hash>:response -> json_response (TTL: 3600s)
```

### SQLite Schema (Persistent Memory)

```sql
-- Memory Storage
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    content TEXT,
    emotional_context TEXT,
    entities TEXT,
    timestamp DATETIME,
    memory_type TEXT
);

-- Entity Relationships  
CREATE TABLE entity_relationships (
    id INTEGER PRIMARY KEY,
    entity1 TEXT,
    relationship TEXT,
    entity2 TEXT,
    confidence REAL,
    source_session TEXT,
    created_at DATETIME
);

-- Performance Metrics
CREATE TABLE protocol_performance (
    id INTEGER PRIMARY KEY,
    protocol_name TEXT,
    execution_time REAL,
    quality_score REAL,
    timestamp DATETIME,
    context_hash TEXT
);
```

### Data Persistence Strategy

**Session Data Lifecycle**:
1. **Creation**: Session metadata stored in Redis
2. **Active Use**: History and context maintained in Redis
3. **Persistence**: Important memories migrated to SQLite
4. **Cleanup**: Redis sessions expire after inactivity

**Memory Management**:
- **Hot Data**: Recent session data in Redis (fast access)
- **Warm Data**: Recent memories in SQLite (indexed access)  
- **Cold Data**: Archived memories (compressed storage)

---

## Scalability Design

### Horizontal Scaling Architecture

```
                        Load Balancer
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ mCP Server      │ │ mCP Server      │ │ mCP Server      │
│ Instance 1      │ │ Instance 2      │ │ Instance N      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Shared Redis    │
                    │ Cluster         │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Shared SQLite   │
                    │ (or PostgreSQL) │
                    └─────────────────┘
```

### Performance Optimization

#### Connection Pooling
```python
# Database connection management
class DatabasePool:
    def __init__(self, max_connections=10):
        self.pool = asyncio.Queue(maxsize=max_connections)
        # Pre-populate connection pool
```

#### Caching Strategy
```python
# Multi-level caching
@lru_cache(maxsize=1000)
def cached_protocol_execution(context_hash):
    # Memory cache for frequent requests
    
@redis_cache(ttl=3600)  
def cached_neural_engine_call(prompt_hash):
    # Redis cache for expensive operations
```

#### Resource Management
- **CPU Optimization**: Thread pool for parallel execution
- **Memory Management**: Context size limits and cleanup
- **Network Optimization**: Connection reuse and pooling
- **Storage Optimization**: Data compression and indexing

### Container Deployment

#### Docker Configuration
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mcp-server
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

---

## Extension Points

### Adding New Protocols

#### Protocol Interface
```python
class CognitiveProtocol:
    """Base class for all cognitive protocols"""
    
    def __init__(self, config: dict):
        self.config = config
    
    async def execute(self, context: dict) -> dict:
        """Main execution method - must be implemented"""
        raise NotImplementedError
    
    def validate_input(self, context: dict) -> bool:
        """Input validation - optional override"""
        return True
    
    def get_metadata(self) -> dict:
        """Protocol metadata - optional override"""
        return {"name": self.__class__.__name__}
```

#### Protocol Registration
```python
# Automatic discovery via protocol.json
{
    "name": "CustomProtocol",
    "version": "1.0.0", 
    "description": "Custom cognitive protocol",
    "class": "CustomProtocol",
    "module": "custom_protocol",
    "dependencies": [],
    "configuration": {
        "timeout": 30,
        "retries": 3
    }
}
```

### Adding New Neural Backends

#### Neural Engine Interface
```python
class NeuralBackend:
    """Interface for neural engine backends"""
    
    def generate_text(self, prompt: str, model: str) -> str:
        raise NotImplementedError
    
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError
    
    def health_check(self) -> bool:
        raise NotImplementedError
```

### Adding New Governance Components

#### Governance Component Interface
```python
class GovernanceComponent:
    """Base class for governance components"""
    
    def evaluate(self, protocol_result: dict, context: dict) -> dict:
        """Evaluate protocol output"""
        raise NotImplementedError
    
    def suggest_improvement(self, evaluation: dict) -> dict:
        """Suggest improvements based on evaluation"""
        return {}
```

### Middleware Extension

#### Custom Middleware
```python
class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response

# Registration
app.add_middleware(CustomMiddleware)
```

---

## Support Resources

### Related Documentation
- [API Documentation](./API_DOCUMENTATION.md) - Complete API reference
- [Security Implementation](./SECURITY.md) - Security architecture details  
- [Configuration Guide](./CONFIGURATION.md) - Environment and deployment config
- [Protocol Development Guide](./PROTOCOL_DEVELOPMENT.md) - Creating custom protocols

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework
- [Redis Documentation](https://redis.io/documentation) - Caching and session storage
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation

---

*Last updated: August 25, 2025*