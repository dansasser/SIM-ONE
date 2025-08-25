# SIM-ONE mCP Server

A sophisticated, multi-protocol cognitive architecture server designed to simulate advanced reasoning, emotional intelligence, and metacognitive governance. The mCP Server provides a powerful platform for orchestrating autonomous AI agents that perform complex cognitive tasks.

## Quick Start

### Terminal Usage

**1. Install Dependencies**
```bash
cd /var/www/SIM-ONE/code
pip install -r requirements.txt
```

**2. Configure Environment**
Create a `.env` file in the `/var/www/SIM-ONE/code` directory:
```bash
# Required
VALID_API_KEYS="your-admin-key,your-user-key"
OPENAI_API_KEY="sk-proj-your-openai-key"

# Optional
REDIS_HOST="localhost"
REDIS_PORT=6379
NEURAL_ENGINE_BACKEND="openai"  # or "local"
ALLOWED_ORIGINS="http://localhost:3000"
SERPER_API_KEY="your-serper-key"  # For web search functionality
```

**3. Start the Server**
```bash
uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000
```

**4. Test with curl**
```bash
# Execute a reasoning workflow
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_names": ["ReasoningAndExplanationProtocol"],
    "initial_data": {
      "user_input": "John works at Microsoft and lives in Seattle."
    }
  }'

# List available protocols
curl -H "X-API-Key: your-user-key" "http://localhost:8000/protocols"

# List workflow templates
curl -H "X-API-Key: your-user-key" "http://localhost:8000/templates"
```

### SDK Integration

**Python SDK Example**
```python
import httpx
import asyncio

class McpClient:
    def __init__(self, base_url="http://localhost:8000", api_key="your-user-key"):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        
    async def execute_workflow(self, protocol_names=None, template_name=None, 
                             initial_data=None, coordination_mode="Sequential"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/execute",
                headers=self.headers,
                json={
                    "protocol_names": protocol_names,
                    "template_name": template_name,
                    "initial_data": initial_data or {},
                    "coordination_mode": coordination_mode
                }
            )
            return response.json()
    
    async def list_protocols(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/protocols", headers=self.headers)
            return response.json()
    
    async def get_session(self, session_id):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/session/{session_id}", headers=self.headers)
            return response.json()

# Usage example
async def main():
    client = McpClient()
    
    # Execute emotional analysis and entity extraction
    result = await client.execute_workflow(
        protocol_names=["EmotionalStateLayerProtocol", "MemoryTaggerProtocol"],
        initial_data={"user_input": "I'm really excited about the new project at work!"},
        coordination_mode="Parallel"
    )
    print(f"Session ID: {result['session_id']}")
    print(f"Results: {result['results']}")
    
    # Use a predefined workflow template
    result = await client.execute_workflow(
        template_name="full_reasoning",
        initial_data={"user_input": "Analyze the implications of remote work on productivity."}
    )
    print(f"Reasoning Results: {result['results']}")

asyncio.run(main())
```

**JavaScript/Node.js SDK Example**
```javascript
class McpClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = 'your-user-key') {
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }
    
    async executeWorkflow({ protocolNames, templateName, initialData, coordinationMode = 'Sequential' }) {
        const response = await fetch(`${this.baseUrl}/execute`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                protocol_names: protocolNames,
                template_name: templateName,
                initial_data: initialData || {},
                coordination_mode: coordinationMode
            })
        });
        return response.json();
    }
    
    async listProtocols() {
        const response = await fetch(`${this.baseUrl}/protocols`, { headers: this.headers });
        return response.json();
    }
    
    async getSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}`, { headers: this.headers });
        return response.json();
    }
}

// Usage
const client = new McpClient();

// Execute a writing workflow with critique loop
client.executeWorkflow({
    templateName: 'writing_team',
    initialData: { topic: 'The future of artificial intelligence' }
}).then(result => {
    console.log('Session:', result.session_id);
    console.log('Generated content:', result.results);
});
```

## Key Features

- **Modular Protocol Architecture**: Dynamically load and chain cognitive protocols
- **Advanced Reasoning (REP)**: Rule-based deductive, inductive, and abductive reasoning
- **Emotional Intelligence (ESL)**: Sophisticated emotion analysis and sentiment detection
- **Entity Extraction (MTP)**: Advanced NER and relationship detection
- **Role-Based Access Control**: Admin, user, and read-only access levels
- **Persistent Memory**: Redis-based session and memory management
- **Security**: Advanced input validation, rate limiting, and security headers

## Available Cognitive Protocols

- **REP** - Reasoning and Explanation Protocol
- **ESL** - Emotional State Layer Protocol  
- **MTP** - Memory Tagger Protocol (Entity Extraction)
- **Ideator** - Creative content generation
- **Critic** - Content analysis and critique
- **Revisor** - Content improvement and revision
- **HIP** - Hierarchical Information Processing
- **VVP** - Verification and Validation Protocol

## Predefined Workflow Templates

- **analyze_only**: Parallel emotional and entity analysis
- **full_reasoning**: Sequential reasoning with summarization
- **writing_team**: Multi-agent writing workflow with critique loops

## API Endpoints

- `POST /execute` - Execute cognitive workflows (admin, user)
- `GET /protocols` - List available protocols (all roles)
- `GET /templates` - List workflow templates (all roles)
- `GET /session/{id}` - Get session history (admin, user - own sessions only)
- `GET /` - Server status check (public)

## Testing

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test
python -m unittest tests.test_main
```

## Architecture

The server follows a hub-and-spoke architecture with the Orchestration Engine coordinating multiple cognitive protocols. Each protocol is self-contained and implements specific cognitive functions, while the Cognitive Governance Engine ensures quality and coherence across all outputs.

For detailed architecture documentation, see the `docs/` directory.