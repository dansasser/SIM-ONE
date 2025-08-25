# Troubleshooting Guide

This guide is designed to help you diagnose and resolve common issues you may encounter when setting up or running the mCP Server.

## Table of Contents
1.  [Common Issues](#common-issues)
2.  [Error Messages Explained](#error-messages-explained)
3.  [Debugging Guide](#debugging-guide)
4.  [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

---

## Common Issues

### Server Fails to Start

*   **Symptom**: The `uvicorn` or `gunicorn` command exits immediately with an error.
*   **Possible Causes & Solutions**:
    1.  **Missing Dependencies**: You may have forgotten to install the required packages.
        *   **Fix**: Make sure your virtual environment is activated and run `pip install -r requirements.txt`.
    2.  **Incorrect Python Version**: The code may be incompatible with your Python version.
        *   **Fix**: Ensure you are using Python 3.10 or newer. Check with `python3 --version`.
    3.  **Port Conflict**: Another application is already using the port the server is trying to bind to (e.g., port 8000).
        *   **Fix**: Stop the other application or run the server on a different port: `uvicorn mcp_server.main:app --port 8001`.
    4.  **Missing Required Environment Variable**: A critical environment variable like `MCP_API_KEY` is not set.
        *   **Fix**: Ensure your `.env` file is present in the project root and contains all required variables. See the [Configuration Guide](./CONFIGURATION.md).

### Database Connection Issues

*   **Symptom**: The server starts but logs a `ConnectionError` related to Redis. Protocol executions that rely on memory may fail.
*   **Possible Causes & Solutions**:
    1.  **Redis Server is Not Running**: The Redis database is down.
        *   **Fix**: Start your Redis server. On Linux, you can check its status with `sudo systemctl status redis-server`.
    2.  **Incorrect Host or Port**: The server is configured to connect to the wrong Redis address.
        *   **Fix**: Verify the `REDIS_HOST` and `REDIS_PORT` variables in your `.env` file are correct.
    3.  **Firewall Issues**: A firewall is blocking the connection between the server and the Redis database.
        *   **Fix**: Ensure your firewall rules allow traffic on the Redis port (default 6379) from the server's IP address.

---

## Error Messages Explained

*   **`401 Unauthorized`**
    *   **Meaning**: Your API request is missing a valid API key.
    *   **Resolution**: Ensure you are including the `Authorization: Bearer <your-key>` header and that the key matches the `MCP_API_KEY` set on the server.

*   **`422 Unprocessable Entity`**
    *   **Meaning**: The JSON body of your request is malformed or missing required fields.
    *   **Resolution**: Carefully check your request body against the schema defined in the [API Documentation](./API_DOCUMENTATION.md). Common mistakes include missing `initial_data` or `user_input`.

*   **`429 Too Many Requests`**
    *   **Meaning**: You have exceeded the API rate limit.
    *   **Resolution**: Slow down your request rate. The default limit is 20 requests per minute per IP address.

*   **`"error": "Either 'template_name' or 'protocol_names' must be provided."`**
    *   **Meaning**: Your request to the `/execute` endpoint did not specify which workflow to run.
    *   **Resolution**: Your request body must contain either a `template_name` key with the name of a valid workflow template or a `protocol_names` key with a list of protocols to run.

---

## Debugging Guide

### Enable Debug Logging
To get more detailed logs, you can change the logging level in the application. While not exposed as an environment variable by default, you can modify `mcp_server/main.py` for debugging purposes.

Change this line:
`logging.basicConfig(level=logging.INFO, ...)`
to:
`logging.basicConfig(level=logging.DEBUG, ...)`

This will provide much more verbose output, including logs from the individual protocols and governance modules.

### Interpreting Logs
*   **INFO**: Standard operational messages, such as which protocol is being executed.
*   **WARNING**: Indicates a potential issue that did not prevent the operation from completing (e.g., a low quality score from the governance engine).
*   **ERROR**: A serious issue that likely caused an operation to fail. Look for these messages to pinpoint the source of a problem.

---

## Frequently Asked Questions (FAQ)

*   **Q: Can I add my own Cognitive Protocols?**
    *   **A**: Yes! The system is designed for it. Simply create a new subdirectory in `mcp_server/protocols/`, create your protocol class, and add a `protocol.json` metadata file. The Protocol Manager will automatically discover and load it on the next server restart.

*   **Q: Do I need a GPU to run this server?**
    *   **A**: It depends on your `NEURAL_ENGINE_BACKEND` configuration. If you are using the `"openai"` backend, no GPU is needed. If you configure it to use a `"local"` model, a powerful GPU is highly recommended for acceptable performance.

*   **Q: Is the server stateless?**
    *   **A**: No. The server uses Redis to maintain session history and persistent memory, making it stateful. If you need stateless behavior, you must either disable the memory features or ensure you generate a new `session_id` for every request.
