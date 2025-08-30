# Production Deployment Guide

This guide provides instructions and best practices for deploying the SIM-ONE mCP Server to a production environment.

## Table of Contents
1.  [Production Requirements](#production-requirements)
2.  [Deployment Method: Gunicorn](#deployment-method-gunicorn)
3.  [Security Configuration](#security-configuration)
4.  [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Production Requirements

Before deploying to production, ensure your environment meets the following requirements.

### Hardware
*   **CPU**: 2+ vCPUs are recommended.
*   **Memory (RAM)**: A minimum of 4GB of RAM. If using a large local neural model, a powerful GPU with sufficient VRAM is required.
*   **Storage**: A fast SSD is recommended for optimal performance.

### Network
*   Ensure the server is behind a firewall, with only the necessary ports open (e.g., port 80 for HTTP, 443 for HTTPS).
*   For high availability, consider using a load balancer to distribute traffic across multiple instances of the server.

### Database
*   Use a managed, production-grade Redis instance or a self-hosted Redis cluster. Do not use a local, unsecured Redis instance intended for development.
*   Ensure the Redis database is backed up regularly.

---

## Deployment Method: Gunicorn

While `uvicorn` is a great development server, it is not recommended for production on its own. For production, you should use a process manager and WSGI/ASGI server like **Gunicorn**. Gunicorn will manage Uvicorn workers, providing robustness, scalability, and performance.

### Step 1: Install Gunicorn
If you haven't already, add `gunicorn` to your `requirements.txt` and install it:
```bash
pip install gunicorn
```

### Step 2: Run the Server with Gunicorn
To start the server, run the following command from the root directory of the project (`/SIM-ONE`):

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mcp_server.main:app -b 0.0.0.0:8000
```
**Command Breakdown:**
*   `gunicorn`: The command to start the Gunicorn server.
*   `-w 4`: Specifies the number of worker processes. A good starting point is `(2 * number_of_cpu_cores) + 1`. Adjust this based on your server's hardware.
*   `-k uvicorn.workers.UvicornWorker`: Tells Gunicorn to use Uvicorn to handle the asynchronous application.
*   `mcp_server.main:app`: The path to the FastAPI application instance (`app` in `mcp_server/main.py`).
*   `-b 0.0.0.0:8000`: Binds the server to listen on port 8000 on all network interfaces.

### Step 3: Set up as a System Service (Optional but Recommended)
To ensure the server runs continuously and restarts automatically on failure or reboot, you should run it as a system service (e.g., using `systemd` on Linux).

1.  Create a service file at `/etc/systemd/system/mcp-server.service`:
    ```ini
    [Unit]
    Description=mCP Server Gunicorn Instance
    After=network.target

    [Service]
    User=your-user
    Group=your-group
    WorkingDirectory=/path/to/your/SIM-ONE
    Environment="PATH=/path/to/your/SIM-ONE/venv/bin"
    ExecStart=/path/to/your/SIM-ONE/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker mcp_server.main:app -b 0.0.0.0:8000

    [Install]
    WantedBy=multi-user.target
    ```
    *Replace `your-user`, `your-group`, and the paths with your actual user, group, and project path.*

2.  Reload the systemd daemon, enable, and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl start mcp-server
    sudo systemctl enable mcp-server
    ```

---

## Security Configuration

*   **API Keys**:
    *   **NEVER** hardcode API keys or commit them to version control.
    *   Use a strong, randomly generated string for your `MCP_API_KEY`.
    *   In a production environment, load secrets from a secure vault (like HashiCorp Vault, AWS Secrets Manager, etc.) or from environment variables managed by your hosting provider, not from a `.env` file.

*   **HTTPS/TLS**:
    *   Do not expose the Gunicorn server directly to the internet.
    *   Place a reverse proxy like **Nginx** or **Caddy** in front of the Gunicorn server.
    *   Configure the reverse proxy to handle HTTPS/TLS termination, providing secure, encrypted communication for your API.

*   **Firewall**:
    *   Configure your firewall to only allow incoming traffic on the ports your reverse proxy is listening on (typically port 443 for HTTPS).
    *   Block all other ports, including the port Gunicorn is running on (e.g., 8000), from public access.

---

## Monitoring and Maintenance

### Health Checks
The server has a root endpoint `GET /` that can be used as a simple health check. Configure your load balancer or monitoring system to ping this endpoint to ensure the server is responsive.

### Logging
*   By default, the server logs to standard output.
*   In a production environment, you should configure Gunicorn and your system service to redirect these logs to a dedicated log file or a centralized logging service (like ELK Stack, Splunk, or Datadog) for easier monitoring and analysis.

### Backups
*   Regularly back up your **Redis database**. The method will depend on your hosting solution (e.g., snapshots for managed databases).
*   Your application code should be managed via Git, so it is already version-controlled.

### Updates
To update the server:
1.  Stop the `mcp-server` service.
2.  Navigate to the project directory and pull the latest changes from your Git repository.
3.  Re-install dependencies in case they have changed: `pip install -r requirements.txt`.
4.  Restart the `mcp-server` service.
