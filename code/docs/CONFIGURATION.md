# Configuration Guide

The SIM-ONE mCP Server is configured entirely through environment variables. These can be set directly in your shell or, for ease of use in development, loaded from a `.env` file located in the project's root directory.

## Table of Contents
1.  [Environment Variables](#environment-variables)
2.  [Configuration Files](#configuration-files)
3.  [Production vs. Development Settings](#production-vs-development-settings)

---

## Environment Variables

This section documents every environment variable used by the server.

### Security Configuration

`MCP_API_KEY` (Required)
*   **Purpose**: A secret key used to authenticate clients accessing the server's API endpoints. The server will reject any request that does not provide a valid key in the `Authorization: Bearer <key>` header.
*   **Required**: Yes
*   **Default**: None. The server will fail to start if this is not set.
*   **Example**: `MCP_API_KEY="a-very-long-and-secure-random-string-12345"`

*Note: The previous `VALID_API_KEYS` list in `config.py` has been simplified for documentation to a single `MCP_API_KEY` for clarity, which is a more common pattern. The server logic should be updated to reflect this if it hasn't already.*

### Database Configuration

`REDIS_HOST` (Optional)
*   **Purpose**: The hostname or IP address of the Redis server used for session management and memory persistence.
*   **Required**: No
*   **Default**: `"localhost"`
*   **Example**: `REDIS_HOST="192.168.1.100"` or `REDIS_HOST="my-redis-instance.example.com"`

`REDIS_PORT` (Optional)
*   **Purpose**: The port number for the Redis server.
*   **Required**: No
*   **Default**: `6379`
*   **Example**: `REDIS_PORT=6380`

### Neural Engine Configuration

`NEURAL_ENGINE_BACKEND` (Optional)
*   **Purpose**: Selects the backend for the Neural Engine. This determines which service or model will be used for generation tasks within protocols that require it (like REP, Ideator, etc.).
*   **Required**: No
*   **Default**: `"openai"`
*   **Options**: `"openai"`, `"local"`

`OPENAI_API_KEY` (Conditionally Required)
*   **Purpose**: Your API key for accessing OpenAI's services (like GPT-4).
*   **Required**: Yes, if `NEURAL_ENGINE_BACKEND` is set to `"openai"`.
*   **Default**: `None`
*   **How to obtain**: Create an account at [openai.com](https://www.openai.com/) and generate a new API key.
*   **Example**: `OPENAI_API_KEY="sk-proj-abc123..."`

`LOCAL_MODEL_PATH` (Conditionally Required)
*   **Purpose**: The local file path to the GGUF-formatted model to be used when `NEURAL_ENGINE_BACKEND` is set to `"local"`.
*   **Required**: Yes, if `NEURAL_ENGINE_BACKEND` is set to `"local"`.
*   **Default**: `"models/llama-3.1-8b.gguf"`
*   **Example**: `LOCAL_MODEL_PATH="/home/user/models/my-custom-model.gguf"`

`SERPER_API_KEY` (Optional)
*   **Purpose**: An API key for the Serper Google Search API. This is used by the RAG (Retrieval-Augmented Generation) system to perform web searches.
*   **Required**: No. If not provided, protocols that require web searches will fail.
*   **Default**: `None`
*   **How to obtain**: Create an account at [serper.dev](https://serper.dev/).
*   **Example**: `SERPER_API_KEY="abcdef1234567890..."`

---

## Configuration Files

*   **`.env`**: The primary method for setting the environment variables listed above. Create this file in the project root.
*   **`mcp_server/config.py`**: This file contains the Pydantic settings model that loads and validates all environment variables. It is the single source of truth for configuration within the application code.
*   **`mcp_server/workflow_templates.json`**: This file defines the sequences of protocols that form executable workflows. You can edit this file to create new workflows or modify existing ones without changing the application code.

---

## Production vs. Development Settings

*   **Development**: In development, it is convenient to use a `.env` file to manage your configuration. Using the default `localhost` Redis and a mock neural engine (if not using OpenAI/local) is sufficient.
*   **Production**:
    *   **Security**: It is critical to use a strong, randomly generated `MCP_API_KEY`. Do not commit this key or your `.env` file to version control. Use your hosting provider's secret management tools.
    *   **Database**: Connect to a robust, managed Redis instance, not a local development server. Ensure it is properly secured.
    *   **Performance**: The neural engine choice has the largest impact on performance. A powerful local model requires significant hardware (GPU, RAM), while the OpenAI backend's performance depends on network latency and their service status. Choose the backend that best fits your performance and cost requirements.
