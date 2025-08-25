#!/bin/sh

# Production Startup Script for mCP Server

echo "--- Starting mCP Server in Production Mode ---"

# --- Environment Variable Validation ---
required_vars="VALID_API_KEYS ALLOWED_ORIGINS REDIS_HOST REDIS_PORT"
missing_vars=0

for var in $required_vars; do
  if [ -z "$(eval echo \$$var)" ]; then
    echo "Error: Required environment variable $var is not set."
    missing_vars=1
  fi
done

# Specific check for OpenAI or Local Model config
if [ "$NEURAL_ENGINE_BACKEND" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: NEURAL_ENGINE_BACKEND is set to 'openai', but OPENAI_API_KEY is not set."
  missing_vars=1
elif [ "$NEURAL_ENGINE_BACKEND" = "local" ] && [ -z "$LOCAL_MODEL_PATH" ]; then
  echo "Error: NEURAL_ENGINE_BACKEND is set to 'local', but LOCAL_MODEL_PATH is not set."
  missing_vars=1
fi


if [ $missing_vars -eq 1 ]; then
  echo "--- Startup Aborted: Missing required environment variables. ---"
  exit 1
fi

echo "Environment variables validated successfully."

# --- Start Gunicorn Server ---
# The working directory is expected to be /app/code
echo "Starting Gunicorn..."
exec gunicorn -c gunicorn.conf.py mcp_server.main:app
