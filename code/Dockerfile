# Multi-stage Dockerfile for SIM-ONE Framework MCP Server
# Optimized for production deployment with security and efficiency

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="SIM-ONE MCP Server" \
      org.label-schema.description="Cognitive Control Protocol Server for the SIM-ONE Framework" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and use a non-root user for building
RUN useradd --create-home --shell /bin/bash build
USER build
WORKDIR /home/build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for running the application
RUN groupadd -r simone && useradd --no-log-init -r -g simone simone

# Create application directory with proper permissions
RUN mkdir -p /app && chown simone:simone /app

# Copy Python packages from builder stage
COPY --from=builder /home/build/.local /home/simone/.local

# Switch to non-root user
USER simone
WORKDIR /app

# Add local Python packages to PATH
ENV PATH="/home/simone/.local/bin:$PATH"

# Copy application code with proper ownership
COPY --chown=simone:simone mcp_server/ ./mcp_server/
COPY --chown=simone:simone gunicorn.conf.py .
COPY --chown=simone:simone *.py .

# Create necessary directories for logs and data
RUN mkdir -p logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Security: Remove any potential sensitive files
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command - can be overridden
CMD ["gunicorn", "-c", "gunicorn.conf.py", "mcp_server.main:app"]