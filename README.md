# mCP Server

The mCP (Cognitive Control Protocol) Server is a Python-based application for orchestrating and governing cognitive workflows, based on the principles of the SIM-ONE framework. It is designed to be a modular, extensible, and stateful platform for building conversational AI systems that are transparent, reliable, and resource-aware.

This implementation serves as a comprehensive proof-of-concept for the core principles of the SIM-ONE framework, including the Five Laws of Cognitive Governance.

## Features

- **Dynamic Protocol Loading:** Automatically discovers and loads cognitive protocols from the filesystem.
- **Stateful Session Management:** Tracks conversational history and context using a Redis backend.
- **Dual Orchestration Modes:** Supports both `Sequential` and `Parallel` execution of protocol workflows.
- **Real-time Streaming API:** Provides a WebSocket endpoint (`/ws/execute`) for streaming results in real-time.
- **Resource Monitoring:** Implements the "Energy Stewardship" law by profiling the CPU and memory usage of every protocol.
- **Workflow Templating:** Allows clients to execute complex, pre-defined workflows by name.
- **Hybrid Symbolic-Generative Workflows:** Capable of running workflows that combine symbolic logic protocols (like REP) and generative protocols (like SP).

## Prerequisites

- Python 3.10+
- Redis (running on `localhost:6379`)
- An OpenAI API key (optional, for the Summarizer Protocol)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **OpenAI API Key (Optional):**
    To enable the `SummarizerProtocol` with a real LLM, set the following environment variable:
    ```bash
    export OPENAI_API_KEY='your-openai-api-key'
    ```
    If this key is not set, the protocol will use a mock response.

2.  **Redis:**
    The `SessionManager` expects a Redis server to be running on `localhost:6379`. If your Redis instance is on a different host or port, you will need to modify the connection details in `mcp_server/session_manager/session_manager.py`.

## Running the Server

To run the mCP server, use the following command from the root of the project directory:

```bash
PYTHONPATH=. uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://0.0.0.0:8000`.

## How to Use (API Guide)

The server exposes a RESTful API for executing workflows.

### 1. Execute a Workflow via HTTP POST

You can execute a workflow by sending a POST request to the `/execute` endpoint. You can either specify a `template_name` or provide the `protocol_names` and `coordination_mode` directly.

**Example: Using the "full_reasoning" template**

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "template_name": "full_reasoning",
  "initial_data": {
    "facts": ["Socrates is a man", "All men are mortal"],
    "rules": [
      [["Socrates is a man", "All men are mortal"], "Socrates is mortal"]
    ]
  }
}' http://0.0.0.0:8000/execute
```

### 2. Execute a Workflow via WebSocket (Streaming)

For a real-time experience, you can connect to the `/ws/execute` endpoint.

**Example: Python client for WebSocket streaming**

```python
import asyncio
import websockets
import json

async def run_streaming_workflow():
    uri = "ws://localhost:8000/ws/execute"
    async with websockets.connect(uri) as websocket:
        request = {
            "protocol_names": ["ReasoningAndExplanationProtocol", "SummarizerProtocol"],
            "initial_data": {
                "facts": ["Socrates is a man", "All men are mortal"],
                "rules": [[["Socrates is a man", "All men are mortal"], "Socrates is mortal"]]
            }
        }
        await websocket.send(json.dumps(request))

        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received: {data}")
            if data.get("status") == "complete":
                break

if __name__ == "__main__":
    asyncio.run(run_streaming_workflow())
```

### Available Endpoints

-   `POST /execute`: Execute a workflow.
-   `GET /protocols`: List all available protocols.
-   `GET /templates`: List all available workflow templates.
-   `GET /session/{session_id}`: Retrieve the history for a given session.
-   `WS /ws/execute`: Execute a workflow over a WebSocket connection for streaming results.

## Available Protocols

-   **ReasoningAndExplanationProtocol (REP):** A rule-based symbolic logic engine.
-   **ValidationAndVerificationProtocol (VVP):** Validates the format of workflow inputs.
-   **EmotionalStateLayerProtocol (ESL):** A keyword-based sentiment analyzer.
-   **MemoryTaggerProtocol (MTP):** A simple entity extractor for conversational memory.
-   **SummarizerProtocol (SP):** A generative protocol that uses an LLM to create summaries.

## Available Workflow Templates

-   **analyze_only:** Runs ESL and MTP in parallel to analyze user input.
-   **full_reasoning:** Runs REP and SP sequentially to perform reasoning and summarize the result.
