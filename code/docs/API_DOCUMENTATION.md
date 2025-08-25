# SIM-ONE mCP Server API Documentation

Welcome to the SIM-ONE mCP Server API. This document provides a complete reference for all available endpoints, authentication mechanisms, role-based access control, and data formats.

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication & Security](#authentication--security)  
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [Endpoint Documentation](#endpoint-documentation)
6. [Data Models](#data-models)
7. [Examples](#examples)

---

## API Overview

### Base URL
All API endpoints are relative to the server's base URL. For local development on port 8000:
```
http://localhost:8000
```

### API Version
Current API version: **1.5.0**

### Content Type
All requests and responses use JSON format:
```
Content-Type: application/json
```

---

## Authentication & Security

### Authentication Method
The API uses **API Key Authentication** via the `X-API-Key` header (NOT Bearer tokens).

**Required Header:**
```
X-API-Key: your-api-key-here
```

### Role-Based Access Control (RBAC)
The server supports three user roles with different permissions:

- **`admin`**: Full access to all endpoints and all user sessions
- **`user`**: Can execute workflows and view own sessions only  
- **`read-only`**: Can only view protocols and templates

### API Key Configuration
Configure valid API keys in your `.env` file:
```bash
VALID_API_KEYS="admin-key,user-key,readonly-key"
```

### Security Features
- Advanced input validation (blocks SQL injection, XSS, command injection)
- Security headers middleware (CSP, X-Frame-Options, etc.)
- CORS protection with configurable origins
- Rate limiting per client IP address

---

## Rate Limiting

Rate limits are applied per client IP address:

- **POST /execute**: 20 requests per minute
- **All other endpoints**: Standard rate limits apply

When rate limit is exceeded, you'll receive:
```http
HTTP/1.1 429 Too Many Requests
```

---

## Error Handling

The API uses standard HTTP status codes with descriptive JSON error responses:

### Success Codes
- `200 OK`: Request successful
- `201 Created`: Resource created successfully

### Error Codes  
- `400 Bad Request`: Invalid request format or missing required fields
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions for the requested resource
- `404 Not Found`: Resource not found (template, session, etc.)
- `422 Unprocessable Entity`: Request validation failed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Unexpected server error

### Error Response Format
```json
{
  "detail": "Error description"
}
```

---

## Endpoint Documentation

### POST /execute

Execute cognitive workflows using protocols or predefined templates.

- **URL**: `/execute`
- **Method**: `POST`
- **Authentication**: Required (`admin`, `user` roles only)
- **Rate Limit**: 20/minute

#### Request Schema
```json
{
  "template_name": "string (optional)",
  "protocol_names": ["string"] (optional),
  "coordination_mode": "Sequential|Parallel" (optional, default: "Sequential"),
  "initial_data": {
    "user_input": "string",
    "...": "additional data"
  },
  "session_id": "string (optional)"
}
```

#### Parameters
- **`template_name`**: Name of predefined workflow template
- **`protocol_names`**: Array of protocol names to execute
- **`coordination_mode`**: Execution mode ("Sequential" or "Parallel")
- **`initial_data`**: Input data for the workflow (must include required fields)
- **`session_id`**: Existing session ID to continue conversation

*Note: Either `template_name` OR `protocol_names` must be provided*

#### Response Schema
```json
{
  "session_id": "string",
  "results": {
    "...": "workflow execution results"
  },
  "error": "string|null",
  "execution_time_ms": "number"
}
```

#### Example Request
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_names": ["EmotionalStateLayerProtocol", "MemoryTaggerProtocol"],
    "coordination_mode": "Parallel",
    "initial_data": {
      "user_input": "I am excited about the new project!"
    }
  }'
```

#### Example Response
```json
{
  "session_id": "sess_abc123",
  "results": {
    "EmotionalStateLayerProtocol": {
      "valence": "positive",
      "detected_emotions": ["excitement", "enthusiasm"],
      "intensity": 0.85
    },
    "MemoryTaggerProtocol": {
      "extracted_entities": ["project"],
      "entity_relationships": []
    }
  },
  "error": null,
  "execution_time_ms": 1250.5
}
```

---

### GET /

Health check endpoint to verify server status.

- **URL**: `/`
- **Method**: `GET` 
- **Authentication**: Not required
- **Rate Limit**: Standard

#### Response Schema
```json
{
  "message": "mCP Server is running."
}
```

#### Example Request
```bash
curl -X GET "http://localhost:8000/"
```

---

### GET /protocols

List all available cognitive protocols and their metadata.

- **URL**: `/protocols`
- **Method**: `GET`
- **Authentication**: Required (`admin`, `user`, `read-only` roles)
- **Rate Limit**: Standard

#### Response Schema
```json
{
  "protocol_name": {
    "name": "string",
    "version": "string",
    "description": "string",
    "...": "additional metadata"
  }
}
```

#### Example Request
```bash
curl -X GET "http://localhost:8000/protocols" \
  -H "X-API-Key: your-user-key"
```

#### Example Response
```json
{
  "ReasoningAndExplanationProtocol": {
    "name": "Reasoning and Explanation Protocol",
    "version": "1.2.0",
    "description": "Advanced logical reasoning engine"
  },
  "EmotionalStateLayerProtocol": {
    "name": "Emotional State Layer Protocol", 
    "version": "2.0.0",
    "description": "Emotion detection and analysis"
  },
  "MemoryTaggerProtocol": {
    "name": "Memory Tagger Protocol",
    "version": "2.1.0", 
    "description": "Entity extraction and relationship mapping"
  }
}
```

---

### GET /templates

List all available workflow templates.

- **URL**: `/templates`
- **Method**: `GET`
- **Authentication**: Required (`admin`, `user`, `read-only` roles)
- **Rate Limit**: Standard

#### Response Schema
```json
{
  "template_name": {
    "description": "string",
    "protocols": ["string"],
    "mode": "Sequential|Parallel",
    "workflow": [...] (optional - advanced workflow definition)
  }
}
```

#### Example Request
```bash
curl -X GET "http://localhost:8000/templates" \
  -H "X-API-Key: your-readonly-key"
```

#### Example Response
```json
{
  "analyze_only": {
    "description": "Analyzes user input for emotional state and entities.",
    "protocols": ["EmotionalStateLayerProtocol", "MemoryTaggerProtocol"],
    "mode": "Parallel"
  },
  "full_reasoning": {
    "description": "Performs reasoning and summarizes the result.",
    "protocols": ["ReasoningAndExplanationProtocol", "SummarizerProtocol"],
    "mode": "Sequential"
  },
  "writing_team": {
    "description": "A full multi-agent writing team with a revise/critique loop.",
    "workflow": [
      { "step": "IdeatorProtocol" },
      { "step": "DrafterProtocol" },
      {
        "loop": 2,
        "steps": [
          { "step": "CriticProtocol" },
          { "step": "RevisorProtocol" }
        ]
      },
      { "step": "SummarizerProtocol" }
    ]
  }
}
```

---

### GET /session/{session_id}

Retrieve session history for a specific session.

- **URL**: `/session/{session_id}`
- **Method**: `GET`
- **Authentication**: Required (`admin`, `user` roles)
- **Authorization**: Users can only access their own sessions; admins can access all
- **Rate Limit**: Standard

#### Path Parameters
- **`session_id`**: The session identifier to retrieve

#### Response Schema
```json
{
  "session_id": "string",
  "history": [
    {
      "user_request": {
        "template_name": "string",
        "initial_data": {...}
      },
      "server_response": {
        "...": "workflow results"
      }
    }
  ]
}
```

#### Example Request
```bash
curl -X GET "http://localhost:8000/session/sess_abc123" \
  -H "X-API-Key: your-user-key"
```

#### Example Response  
```json
{
  "session_id": "sess_abc123",
  "history": [
    {
      "user_request": {
        "template_name": "analyze_only",
        "initial_data": {
          "user_input": "I am excited about the new project!"
        }
      },
      "server_response": {
        "EmotionalStateLayerProtocol": {
          "valence": "positive",
          "detected_emotions": ["excitement"]
        },
        "MemoryTaggerProtocol": {
          "extracted_entities": ["project"]
        }
      }
    }
  ]
}
```

#### Error Responses
```json
// Session not found (admin only)
{
  "detail": "Session not found."
}

// Unauthorized access (non-admin trying to access other user's session)  
{
  "detail": "Not authorized to view this session."
}
```

---

## Data Models

### WorkflowRequest
```typescript
interface WorkflowRequest {
  template_name?: string;
  protocol_names?: string[];
  coordination_mode?: "Sequential" | "Parallel";
  initial_data: Record<string, any>;
  session_id?: string;
}
```

### WorkflowResponse
```typescript
interface WorkflowResponse {
  session_id: string;
  results: Record<string, any>;
  error: string | null;
  execution_time_ms: number;
}
```

---

## Examples

### Example 1: Execute Template-Based Workflow
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "full_reasoning",
    "initial_data": {
      "user_input": "What are the implications of remote work?"
    }
  }'
```

### Example 2: Execute Custom Protocol Sequence
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_names": [
      "ReasoningAndExplanationProtocol", 
      "EmotionalStateLayerProtocol"
    ],
    "coordination_mode": "Sequential",
    "initial_data": {
      "user_input": "I think artificial intelligence will change everything.",
      "context": "technology discussion"
    }
  }'
```

### Example 3: Continue Existing Session  
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "analyze_only",
    "session_id": "sess_abc123",
    "initial_data": {
      "user_input": "Tell me more about that topic."
    }
  }'
```

### Example 4: Error Response
```bash
# Invalid authentication
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: invalid-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {}}'

# Response:  
# HTTP/1.1 401 Unauthorized
# {
#   "detail": "Invalid API key"
# }
```

---

## Support

For issues, questions, or feature requests:
- Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)
- Review the [Configuration Documentation](./CONFIGURATION.md)  
- Consult the [Architecture Overview](./ARCHITECTURE.md)

---

*Last updated: August 25, 2025*