# API Documentation

Welcome to the SIM-ONE mCP Server API. This document provides a detailed reference for all available endpoints, authentication mechanisms, and data formats.

## Table of Contents
1.  [API Overview](#api-overview)
    *   [Base URL](#base-url)
    *   [Authentication](#authentication)
    *   [Error Handling](#error-handling)
2.  [Endpoint Documentation](#endpoint-documentation)
    *   [POST /execute](#post-execute)
    *   [GET /](#get-)
    *   [GET /protocols](#get-protocols)
    *   [GET /templates](#get-templates)
    *   [GET /session/{session_id}](#get-session-session_id)

---

## API Overview

### Base URL
All API endpoints are relative to the server's base URL. If you are running the server locally on port 8000, the base URL is:
`http://localhost:8000`

### Authentication
The API uses a bearer token authentication scheme. All requests to protected endpoints must include an `Authorization` header containing a valid API key.

**Header Format:**
`Authorization: Bearer <your-mcp-api-key>`

You can set your valid API key(s) in the `.env` file using the `MCP_API_KEY` variable. Requests without a valid key will receive a `401 Unauthorized` response.

### Error Handling
The API uses standard HTTP status codes to indicate the success or failure of a request. In case of an error, the JSON response body will typically contain an `error` field with a descriptive message.

*   `200 OK`: The request was successful.
*   `401 Unauthorized`: The request is missing a valid API key.
*   `422 Unprocessable Entity`: The request body is malformed or missing required fields.
*   `429 Too Many Requests`: The client has exceeded the rate limit.
*   `500 Internal Server Error`: An unexpected error occurred on the server.

---

## Endpoint Documentation

### POST /execute
The primary endpoint for executing a cognitive workflow. It can run a predefined workflow from a template or a custom sequence of protocols.

*   **HTTP Method**: `POST`
*   **URL**: `/execute`
*   **Authentication**: Required (API Key)

#### Request Body
```json
{
  "template_name": "string (optional)",
  "protocol_names": ["string", "... (optional)"],
  "coordination_mode": "string (optional, 'Sequential' or 'Parallel', default: 'Sequential')",
  "initial_data": {
    "user_input": "string (required)",
    "...": "any other initial data"
  },
  "session_id": "string (optional)"
}
```
**Parameters:**
*   `template_name`: The name of a predefined workflow from `workflow_templates.json`.
*   `protocol_names`: A list of protocol names to execute in sequence.
*   `coordination_mode`: If using `protocol_names`, specifies whether to run them sequentially or in parallel.
*   `initial_data`: An object containing the initial data for the workflow. `user_input` is typically required.
*   `session_id`: An existing session ID to continue a conversation. If omitted, a new session is created.

*Note: You must provide either `template_name` or `protocol_names`.*

#### Response Body (Success)
```json
{
  "session_id": "string",
  "results": {
    "...": "final output data from the workflow"
  },
  "error": null,
  "execution_time_ms": "float"
}
```

#### Usage Example (`curl`)
```bash
curl -X POST "http://localhost:8000/execute" \
-H "Authorization: Bearer your-mcp-api-key" \
-H "Content-Type: application/json" \
-d '{
    "template_name": "StandardReasoningWorkflow",
    "initial_data": {
        "user_input": "John works at Microsoft and lives in Seattle."
    }
}'
```

---

### GET /
A simple health check endpoint to verify that the server is running.

*   **HTTP Method**: `GET`
*   **URL**: `/`
*   **Authentication**: Not required

#### Response Body (Success)
```json
{
  "message": "mCP Server is running."
}
```

#### Usage Example (`curl`)
```bash
curl http://localhost:8000/
```

---

### GET /protocols
Lists all cognitive protocols that have been discovered and loaded by the Protocol Manager.

*   **HTTP Method**: `GET`
*   **URL**: `/protocols`
*   **Authentication**: Not required

#### Response Body (Success)
An object where keys are protocol names and values are their metadata.
```json
{
  "REP": { "name": "Reasoning and Explanation Protocol", "version": "1.2.0" },
  "ESL": { "name": "Emotional State Layer Protocol", "version": "2.0.0" },
  "MTP": { "name": "Memory Tagger Protocol", "version": "2.1.0" }
}
```

#### Usage Example (`curl`)
```bash
curl http://localhost:8000/protocols
```

---

### GET /templates
Lists all available workflow templates from `workflow_templates.json`.

*   **HTTP Method**: `GET`
*   **URL**: `/templates`
*   **Authentication**: Not required

#### Response Body (Success)
An object where keys are template names and values are their definitions.
```json
{
  "StandardReasoningWorkflow": {
    "description": "A standard workflow for reasoning about user input.",
    "workflow": [
      { "step": "ESL" },
      { "step": "REP" },
      { "step": "MTP" }
    ]
  }
}
```

#### Usage Example (`curl`)
```bash
curl http://localhost:8000/templates
```

---

### GET /session/{session_id}
Retrieves the interaction history for a given session ID.

*   **HTTP Method**: `GET`
*   **URL**: `/session/{session_id}`
*   **Authentication**: Not required

#### Response Body (Success)
```json
{
  "session_id": "string",
  "history": [
    {
      "user_request": { "..." },
      "server_response": { "..." }
    }
  ]
}
```

#### Usage Example (`curl`)
```bash
curl http://localhost:8000/session/some-session-id-123
```
