# Installation Guide

This guide provides detailed, step-by-step instructions for installing and verifying the SIM-ONE mCP Server.

## Table of Contents
1.  [System Requirements](#system-requirements)
2.  [Installation Steps](#installation-steps)
3.  [Troubleshooting](#troubleshooting)

---

## System Requirements

### Software Dependencies
*   **Operating System**: The server is developed and tested on Linux (Debian-based). It should be compatible with other POSIX-compliant systems like macOS and other Linux distributions. Windows is not officially supported but may work using WSL2.
*   **Python**: Python `3.10` or newer is required.
*   **Redis**: Redis is required for session management and the memory system. Version `6.0` or newer is recommended.
*   **Git**: Required for cloning the repository.

### Hardware Requirements
*   **Disk Space**: A minimum of 1GB of free disk space is recommended for the repository and dependencies.
*   **Memory (RAM)**: A minimum of 2GB of RAM is recommended for basic operation. More complex workflows may require more memory.
*   **Network**: A stable internet connection is required for downloading dependencies during installation.

---

## Installation Steps

Follow these steps to set up the mCP Server in a development environment.

### Step 1: Environment Preparation
Ensure you have Python 3.10+ and Git installed on your system. You can verify this by running:
```bash
python3 --version
git --version
```

### Step 2: Repository Cloning
Clone the official SIM-ONE repository from GitHub:
```bash
git clone [repository-url]
cd SIM-ONE
```

### Step 3: Python Environment Setup
It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with system-wide packages.

Create the virtual environment:
```bash
python3 -m venv venv
```

Activate the virtual environment:
*   On **Linux/macOS**:
    ```bash
    source venv/bin/activate
    ```
*   On **Windows**:
    ```bash
    .\venv\Scripts\activate
    ```
Your command prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

### Step 4: Dependency Installation
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
This command will download and install all necessary libraries, such as FastAPI, Uvicorn, and others.

### Step 5: Database Setup (Redis)
The server requires a running Redis instance. If you do not have one, you can install it locally.

*   **On Debian/Ubuntu**:
    ```bash
    sudo apt-get update
    sudo apt-get install redis-server
    ```
*   **On macOS (using Homebrew)**:
    ```bash
    brew install redis
    brew services start redis
    ```
*   **Using Docker (Recommended for ease of use)**:
    ```bash
    docker run --name mcp-redis -p 6379:6379 -d redis
    ```
By default, the server will try to connect to Redis at `localhost:6379`. If your Redis instance is running elsewhere, you will need to configure the environment variables accordingly.

### Step 6: Configuration File Setup
The server is configured using environment variables. Create a `.env` file in the root of the project directory (`/SIM-ONE/.env`).

```bash
touch .env
```

Open the `.env` file and add the required configuration. At a minimum, you need to set an API key for server security.

```dotenv
# .env file
MCP_API_KEY="your-super-secret-and-long-api-key"

# If your Redis server is not on localhost, specify its location
# REDIS_HOST=your-redis-host
# REDIS_PORT=your-redis-port
```
For a full list of all configuration options, see the [Configuration Guide](./CONFIGURATION.md).

### Step 7: Initial Testing and Verification
Once everything is set up, you can run the server:
```bash
uvicorn mcp_server.main:app --host 0.0.0.0 --port 8000
```
You should see output indicating the server has started, similar to this:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
You can now send a request to the server to verify it's working. See the [API Documentation](./API_DOCUMENTATION.md) for request examples.

### Step 8: Production Deployment Considerations
For production deployment, it is **not recommended** to use the `uvicorn` development server directly. Instead, use a production-grade ASGI server like Gunicorn to manage Uvicorn workers.

Example using Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mcp_server.main:app
```
For more detailed information, please refer to the [Deployment Guide](./DEPLOYMENT.md).

---

## Troubleshooting

*   **Issue**: `ModuleNotFoundError` when running the server.
    *   **Solution**: Ensure your Python virtual environment is activated. If it is, try re-installing dependencies with `pip install -r requirements.txt`.

*   **Issue**: Server fails to start with a `ConnectionError` related to Redis.
    *   **Solution**: Make sure your Redis server is running and accessible from where you are running the mCP Server. Check the `REDIS_HOST` and `REDIS_PORT` environment variables to ensure they are pointing to the correct address.

*   **Issue**: `pip install` fails on a specific package.
    *   **Solution**: Some packages may have system-level dependencies (like build tools or development libraries). Check the error message for details. On Debian/Ubuntu, you may need to install `python3-dev` or `build-essential`.

*   **Issue**: Getting `401 Unauthorized` errors from the API.
    *   **Solution**: Ensure you have set the `MCP_API_KEY` in your `.env` file and that you are sending it correctly in the `Authorization: Bearer <your-key>` header of your API requests.
