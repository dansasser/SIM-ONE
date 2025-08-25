# Gunicorn configuration file

import os

# Worker Processes
workers = int(os.environ.get('GUNICORN_PROCESSES', '2'))
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'uvicorn.workers.UvicornWorker')

# Logging
loglevel = os.environ.get('GUNICORN_LOGLEVEL', 'info')
accesslog = '-' # Log to stdout
errorlog = '-' # Log to stderr

# Server Mechanics
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8000')
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '5'))

# For Uvicorn worker
forwarded_allow_ips = '*'
proxy_protocol = True
proxy_allow_ips = '*'

print(f"Gunicorn config loaded: {workers} workers, binding to {bind}")
