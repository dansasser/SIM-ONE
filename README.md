# SIM-ONE Cognitive Control Protocol (mCP) Server

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

This project is licensed under the terms of the [MIT License](./LICENSE).
