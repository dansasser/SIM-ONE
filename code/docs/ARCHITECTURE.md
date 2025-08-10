# Technical Architecture

This document provides a detailed overview of the SIM-ONE mCP Server's technical architecture, its major components, and the data flows between them.

## Table of Contents
1.  [System Architecture](#system-architecture)
2.  [Component Documentation](#component-documentation)
3.  [Protocol Documentation](#protocol-documentation)
4.  [Database Schema](#database-schema)
5.  [Data Flow Diagram](#data-flow-diagram)

---

## System Architecture

The mCP Server is designed as a modular, service-oriented system built around a central **Orchestration Engine**. This engine executes cognitive workflows composed of one or more **Cognitive Protocols**. The entire system is exposed via a secure **API Gateway**.

### High-Level Diagram

```
+---------------------+      +--------------------------+
|   External Client   |----->|      API Gateway         |
| (e.g., SDK, curl)   |      | (main.py, auth, limiter) |
+---------------------+      +-------------+------------+
                                           |
                                           v
+--------------------------------------------------------------------+
|                         Orchestration Engine                         |
|    (Executes workflows, manages data context, handles errors)      |
+--------------------------+---------------------+-------------------+
                           |                     ^
                           v                     |
+--------------------------+---------------------+-------------------+
|                     Protocol Manager                               |
| (Discovers and provides access to all available Cognitive Protocols)|
+--------------------------+-------------------------------------------+
                           |
                           v
+--------------------------+-------------------------------------------+
|      Cognitive Protocols (Self-contained cognitive modules)          |
| +--------+  +--------+  +--------+  +--------+  +--------+  +--------+ |
| |  REP   |  |  ESL   |  |  MTP   |  | Ideator|  | Critic |  | ...etc | |
| +--------+  +--------+  +--------+  +--------+  +--------+  +--------+ |
+----------------------------------------------------------------------+
```

---

## Component Documentation

### API Gateway (`mcp_server/main.py`)
*   **Purpose**: Serves as the single entry point for all external communication. It is responsible for handling HTTP requests, deserializing data, and routing requests to the appropriate internal services.
*   **Key Features**:
    *   Built with **FastAPI**.
    *   Secures endpoints using **API Key Authentication**.
    *   Implements **rate limiting** to prevent abuse.
    *   Provides endpoints for workflow execution, status checks, and listing available protocols/templates.

### Orchestration Engine (`mcp_server/orchestration_engine/`)
*   **Purpose**: The brain of the server. It receives a workflow definition (a sequence of protocols) and a data context, and executes the protocols in the specified order.
*   **Key Features**:
    *   Manages the data context, passing the output of one protocol as the input to the next.
    *   Supports both **Sequential** and **Parallel** execution of protocols.
    *   Interfaces with the `CognitiveGovernanceEngine` (when implemented) to validate results after each step.
    *   Handles errors during protocol execution and can trigger recovery strategies.

### Protocol Manager (`mcp_server/protocol_manager/`)
*   **Purpose**: To dynamically discover, load, and provide access to all available Cognitive Protocols.
*   **Key Features**:
    *   Scans the `mcp_server/protocols/` directory on startup.
    *   Loads each protocol's metadata from its `protocol.json` file.
    *   Instantiates the protocol's main class, making it available to the Orchestration Engine.
    *   This design allows new protocols to be added to the system just by adding a new subdirectory, without changing the core engine code.

### Cognitive Governance Engine (`mcp_server/cognitive_governance_engine/`)
*   **Purpose**: A metacognitive layer that monitors and validates the outputs of all protocols to ensure coherence, quality, and logical consistency.
*   **Key Features**:
    *   **Coherence Validation**: Checks for contradictions between protocol outputs (e.g., ensuring emotional sentiment from ESL aligns with the logical conclusion from REP).
    *   **Quality Assurance**: Scores protocol outputs on metrics like completeness and relevance.
    *   **Error Recovery**: Provides strategies for handling protocol failures, such as retrying or using fallback data.

---

## Protocol Documentation

Cognitive Protocols are the building blocks of workflows. Each is a self-contained Python module designed to perform one specific cognitive function.

### REP (Reasoning and Explanation Protocol)
*   **Purpose**: To perform logical reasoning and generate explanations.
*   **Algorithm**: Implements a rule-based system for deductive, inductive, and abductive reasoning. It can identify premises, draw conclusions, and assess its own confidence.
*   **Input**: A dictionary containing text to be analyzed (e.g., `user_input`).
*   **Output**: A structured dictionary containing `premises`, `conclusion`, `reasoning_type`, and a `confidence` score.

### ESL (Emotional State Layer Protocol)
*   **Purpose**: To analyze text and determine its emotional content.
*   **Algorithm**: Uses a sophisticated, rule-based engine with regex patterns to detect a wide range of primary, social, and cognitive emotions. It also analyzes context to handle negation and intensifiers.
*   **Input**: A dictionary containing `user_input`.
*   **Output**: A detailed report including the overall `valence` (positive/negative/neutral), `intensity`, and a list of all `detected_emotions` with their individual scores.

### MTP (Memory Tagger Protocol)
*   **Purpose**: To perform Named Entity Recognition (NER) and relationship extraction.
*   **Algorithm**: Uses a rule-based system with regex patterns to identify entities like people, organizations, and places. It also detects the relationships between these entities (e.g., "works_at", "located_in").
*   **Input**: A dictionary containing `user_input` and the `emotional_context` from the ESL.
*   **Output**: A structured report containing a list of `extracted_entities`, detected `entity_relationships`, and their association with the overall emotional context.

---

## Database Schema

The server uses a **Redis** database for two primary purposes:

1.  **Session Management**:
    *   **Key**: `session:<session_id>`
    *   **Type**: Hash
    *   **Purpose**: Stores the interaction history for a given session, allowing for conversational context to be maintained.

2.  **Persistent Memory**:
    *   **Key**: `memory:<session_id>`
    *   **Type**: List
    *   **Purpose**: Stores memories tagged by the MTP protocol. Each element in the list is a JSON string representing a memory object, which typically contains an entity, its associated emotional context, and the source input. This allows the system to build a knowledge base over time.

---

## Data Flow Diagram

A typical "StandardReasoningWorkflow" (`[ESL -> REP -> MTP]`) execution flow:

1.  **Client Request**: `POST /execute` with `user_input`.
2.  **Orchestrator starts**: Receives `user_input`.
3.  **Execute ESL**:
    *   Input: `{"user_input": "..."}`
    *   Output: `{"valence": "...", "detected_emotions": [...]}`
4.  **Orchestrator updates context**: Merges ESL output into the main data context.
5.  **Execute REP**:
    *   Input: `{"user_input": "...", "valence": "...", ...}`
    *   Output: `{"conclusion": "...", "confidence": 0.9}`
6.  **Orchestrator updates context**: Merges REP output.
7.  **Execute MTP**:
    *   Input: `{"user_input": "...", "emotional_context": {"valence": ...}, ...}`
    *   Output: `{"extracted_entities": [...], ...}`
8.  **Orchestrator finalizes**: Merges MTP output into the final result.
9.  **API Gateway responds**: Returns the final, aggregated data context to the client.
