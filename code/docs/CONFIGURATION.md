# Configuration Guide

The SIM-ONE mCP Server is configured entirely through environment variables with `.env` file support. This guide provides comprehensive documentation for all configuration options, validation procedures, and deployment scenarios.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Variables Reference](#environment-variables-reference)
3. [Configuration Templates](#configuration-templates)
4. [Validation & Testing](#validation--testing)
5. [Security Considerations](#security-considerations)
6. [Deployment Scenarios](#deployment-scenarios)
7. [Migration Guide](#migration-guide)

---

## Quick Start

### Minimal Configuration

Create a `.env` file in `/var/www/SIM-ONE/code/`:

```bash
# Required - API Authentication
VALID_API_KEYS="admin-key-123,user-key-456,readonly-key-789"

# Required - Choose Neural Engine Backend
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-your-openai-key-here"

# Optional - Defaults will be used
REDIS_HOST="localhost"
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000"
```

### Validation Test
```bash
cd /var/www/SIM-ONE/code
python -c "from mcp_server.config import settings; print('✅ Configuration loaded successfully'); print(f'Backend: {settings.NEURAL_ENGINE_BACKEND}')"
```

---

## Environment Variables Reference

### Security Configuration

#### `VALID_API_KEYS` (Required)
- **Purpose**: Comma-separated list of valid API keys for authentication
- **Format**: `"key1,key2,key3"`
- **Role Assignment**: Managed via `mcp_server.security.key_manager`
- **Example**: `VALID_API_KEYS="admin-prod-key,user-app-key,readonly-monitor-key"`
- **Security**: Use strong, randomly generated keys (32+ characters)

### Neural Engine Configuration

#### `NEURAL_ENGINE_BACKEND` (Required)
- **Purpose**: Selects the AI backend for protocol processing
- **Options**: 
  - `"openai"`: Use OpenAI's API (GPT models)
  - `"local"`: Use local GGUF model via llama-cpp-python
- **Default**: `"openai"`
- **Example**: `NEURAL_ENGINE_BACKEND="local"`

#### `OPENAI_API_KEY` (Conditionally Required)
- **Purpose**: API key for OpenAI services
- **Required When**: `NEURAL_ENGINE_BACKEND="openai"`
- **Format**: `sk-proj-...` or `sk-...`
- **Obtain From**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Example**: `OPENAI_API_KEY="sk-proj-abc123def456..."`
- **Validation**: Automatically tested on startup if backend is OpenAI

#### `LOCAL_MODEL_PATH` (Conditionally Required)
- **Purpose**: File path to GGUF format model file
- **Required When**: `NEURAL_ENGINE_BACKEND="local"`
- **Format**: Absolute or relative path to `.gguf` file
- **Default**: `"models/llama-3.1-8b.gguf"`
- **Example**: `LOCAL_MODEL_PATH="/opt/models/llama-2-13b-chat.Q5_K_M.gguf"`
- **Validation**: File existence checked on startup

### Database Configuration

#### `REDIS_HOST` (Optional)
- **Purpose**: Redis server hostname/IP for session and memory storage
- **Default**: `"localhost"`
- **Production**: Use managed Redis service hostname
- **Example**: `REDIS_HOST="redis.internal.company.com"`

#### `REDIS_PORT` (Optional)
- **Purpose**: Redis server port number
- **Default**: `6379`
- **Range**: 1-65535
- **Example**: `REDIS_PORT=6380`

### Web Server Configuration

#### `ALLOWED_ORIGINS` (Optional)
- **Purpose**: CORS allowed origins for frontend applications
- **Format**: Comma-separated list of URLs
- **Default**: `"http://localhost:3000"`
- **Production**: Include all legitimate frontend domains
- **Example**: `ALLOWED_ORIGINS="https://app.company.com,https://admin.company.com,http://localhost:3000"`

### External Services

#### `SERPER_API_KEY` (Optional)
- **Purpose**: Google Search API key for RAG functionality
- **Obtain From**: [Serper.dev](https://serper.dev)
- **Used By**: RAG protocols for web search capabilities
- **Example**: `SERPER_API_KEY="abc123def456..."`
- **Fallback**: RAG protocols will disable search if not provided

### Application Metadata

#### `APP_VERSION` (Read-Only)
- **Purpose**: Application version number
- **Set In**: `mcp_server/config.py`
- **Current**: `"1.5.0"`
- **Usage**: API responses, logging, monitoring

---

## Configuration Templates

### Development Environment
```bash
# .env for local development
VALID_API_KEYS="dev-admin-key,dev-user-key,dev-readonly-key"
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-dev-key-here"
REDIS_HOST="localhost"
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:4321"
SERPER_API_KEY="dev-serper-key"
```

### Local Model Development
```bash
# .env for local model testing
VALID_API_KEYS="dev-admin-key,dev-user-key"
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="./models/llama-2-7b-chat.Q4_K_M.gguf"
REDIS_HOST="localhost"
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000"
```

### Staging Environment
```bash
# .env for staging deployment
VALID_API_KEYS="staging-admin-key,staging-app-key,staging-test-key"
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-staging-key-here"
REDIS_HOST="redis-staging.internal.company.com"
REDIS_PORT=6379
ALLOWED_ORIGINS="https://staging-app.company.com,https://staging-admin.company.com"
SERPER_API_KEY="staging-serper-key"
```

### Production Environment
```bash
# .env for production deployment
VALID_API_KEYS="prod-admin-key-2024,prod-app-key-2024,prod-monitor-key"
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="/opt/simone/models/llama-2-13b-chat.Q5_K_M.gguf"
REDIS_HOST="redis-cluster.internal.company.com"
REDIS_PORT=6379
ALLOWED_ORIGINS="https://simone.company.com,https://api.company.com"
SERPER_API_KEY="prod-serper-key-2024"
```

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-server:
    build: .
    environment:
      VALID_API_KEYS: "${VALID_API_KEYS}"
      NEURAL_ENGINE_BACKEND: "${NEURAL_ENGINE_BACKEND:-openai}"
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      LOCAL_MODEL_PATH: "${LOCAL_MODEL_PATH:-/app/models/model.gguf}"
      REDIS_HOST: "redis"
      REDIS_PORT: "6379"
      ALLOWED_ORIGINS: "${ALLOWED_ORIGINS:-http://localhost:3000}"
      SERPER_API_KEY: "${SERPER_API_KEY}"
    volumes:
      - ./models:/app/models:ro
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-server-config
data:
  NEURAL_ENGINE_BACKEND: "local"
  LOCAL_MODEL_PATH: "/models/llama-2-13b-chat.Q5_K_M.gguf"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  ALLOWED_ORIGINS: "https://app.company.com,https://admin.company.com"
---
apiVersion: v1
kind: Secret
metadata:
  name: mcp-server-secrets
type: Opaque
stringData:
  VALID_API_KEYS: "prod-admin-key,prod-user-key"
  OPENAI_API_KEY: "sk-proj-production-key"
  SERPER_API_KEY: "production-serper-key"
```

---

## Validation & Testing

### Configuration Validation Script
```bash
#!/bin/bash
# validate-config.sh

echo "🔍 Validating SIM-ONE mCP Server Configuration..."

cd /var/www/SIM-ONE/code

# Test configuration loading
echo "1. Testing configuration loading..."
python -c "
from mcp_server.config import settings
print(f'✅ Configuration loaded successfully')
print(f'   Backend: {settings.NEURAL_ENGINE_BACKEND}')
print(f'   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}')
print(f'   Origins: {len(settings.ALLOWED_ORIGINS)} configured')
" || { echo "❌ Configuration loading failed"; exit 1; }

# Test neural engine
echo "2. Testing neural engine initialization..."
python -c "
from mcp_server.neural_engine.neural_engine import NeuralEngine
try:
    engine = NeuralEngine()
    response = engine.generate_text('Configuration test')
    print(f'✅ Neural engine working: {type(engine).__name__}')
    print(f'   Response preview: {response[:50]}...')
except Exception as e:
    print(f'❌ Neural engine failed: {e}')
    exit(1)
" || { echo "❌ Neural engine validation failed"; exit 1; }

# Test Redis connection
echo "3. Testing Redis connection..."
python -c "
import redis
from mcp_server.config import settings
try:
    r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'⚠️  Redis connection failed: {e}')
    print('   Session persistence will be disabled')
"

# Test API keys format
echo "4. Testing API keys format..."
python -c "
import os
keys = os.getenv('VALID_API_KEYS', '').split(',')
valid_keys = [k.strip() for k in keys if k.strip()]
if len(valid_keys) == 0:
    print('❌ No valid API keys configured')
    exit(1)
elif len(valid_keys) < 2:
    print('⚠️  Only one API key configured (recommended: admin + user keys)')
else:
    print(f'✅ {len(valid_keys)} API keys configured')

for key in valid_keys:
    if len(key) < 16:
        print(f'⚠️  API key too short: {key[:8]}... (recommended: 32+ chars)')
"

echo "✅ Configuration validation complete!"
```

### Environment Testing Commands
```bash
# Test specific configurations
cd /var/www/SIM-ONE/code

# Test OpenAI backend
NEURAL_ENGINE_BACKEND="openai" OPENAI_API_KEY="your-key" \
python -c "from mcp_server.neural_engine.neural_engine import NeuralEngine; print(NeuralEngine().generate_text('test'))"

# Test local backend
NEURAL_ENGINE_BACKEND="local" LOCAL_MODEL_PATH="./models/model.gguf" \
python -c "from mcp_server.neural_engine.neural_engine import NeuralEngine; print(NeuralEngine().generate_text('test'))"

# Test API endpoint
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-test-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "config test"}}'
```

---

## Security Considerations

### API Key Management

#### Generation Best Practices
```bash
# Generate secure API keys
python -c "import secrets; print('admin-' + secrets.token_urlsafe(32))"
python -c "import secrets; print('user-' + secrets.token_urlsafe(32))"
python -c "import secrets; print('readonly-' + secrets.token_urlsafe(32))"
```

#### Key Rotation Strategy
```bash
# 1. Generate new keys
NEW_ADMIN_KEY=$(python -c "import secrets; print('admin-2024-' + secrets.token_urlsafe(32))")
NEW_USER_KEY=$(python -c "import secrets; print('user-2024-' + secrets.token_urlsafe(32))")

# 2. Update configuration (both old and new keys temporarily)
VALID_API_KEYS="old-admin-key,old-user-key,$NEW_ADMIN_KEY,$NEW_USER_KEY"

# 3. Update client applications to use new keys

# 4. Remove old keys after migration
VALID_API_KEYS="$NEW_ADMIN_KEY,$NEW_USER_KEY"
```

### Environment Isolation

#### File Permissions
```bash
# Secure .env file
chmod 600 .env
chown app:app .env

# Secure model files
chmod 644 models/*.gguf
chown app:app models/*.gguf
```

#### Environment Separation
```bash
# Development
cp .env.template .env.development

# Staging  
cp .env.template .env.staging

# Production
cp .env.template .env.production

# Load environment-specific config
ln -sf .env.production .env  # For production
```

### Secrets Management

#### Docker Secrets
```yaml
# docker-compose.yml with secrets
version: '3.8'
services:
  mcp-server:
    environment:
      VALID_API_KEYS_FILE: /run/secrets/api_keys
      OPENAI_API_KEY_FILE: /run/secrets/openai_key
    secrets:
      - api_keys
      - openai_key

secrets:
  api_keys:
    external: true
  openai_key:
    external: true
```

#### Kubernetes Secrets
```yaml
# Create secret
kubectl create secret generic mcp-secrets \
  --from-literal=VALID_API_KEYS="admin-key,user-key" \
  --from-literal=OPENAI_API_KEY="sk-proj-key"

# Use in deployment
spec:
  containers:
  - name: mcp-server
    envFrom:
    - secretRef:
        name: mcp-secrets
```

---

## Deployment Scenarios

### Single Server Deployment
```bash
# Simple production deployment
VALID_API_KEYS="prod-admin-key,prod-user-key"
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="/opt/simone/models/llama-2-13b-chat.Q5_K_M.gguf"
REDIS_HOST="localhost"
ALLOWED_ORIGINS="https://app.company.com"
```

### Load Balanced Deployment
```bash
# Multiple server instances with shared Redis
VALID_API_KEYS="lb-admin-key,lb-user-key-1,lb-user-key-2"
NEURAL_ENGINE_BACKEND="openai"  # Consistent across instances
OPENAI_API_KEY="sk-proj-loadbalanced-key"
REDIS_HOST="redis-cluster.internal.company.com"
ALLOWED_ORIGINS="https://app.company.com,https://api.company.com"
```

### Hybrid Cloud Deployment
```bash
# Primary: Local models for sensitive data
VALID_API_KEYS="hybrid-admin-key,hybrid-local-key"
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="/secure/models/llama-2-13b-chat.Q5_K_M.gguf"
REDIS_HOST="secure-redis.internal.company.com"

# Fallback: OpenAI for overflow traffic
VALID_API_KEYS="hybrid-admin-key,hybrid-openai-key"
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-overflow-key"
REDIS_HOST="redis-public.company.com"
```

### Multi-Tenant Deployment
```bash
# Tenant A configuration
VALID_API_KEYS="tenant-a-admin,tenant-a-user-1,tenant-a-user-2"
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="/tenants/tenant-a/models/custom-model.gguf"
REDIS_HOST="redis-tenant-a.internal.company.com"
ALLOWED_ORIGINS="https://tenant-a.platform.com"

# Tenant B configuration
VALID_API_KEYS="tenant-b-admin,tenant-b-user-1"
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-tenant-b-key"
REDIS_HOST="redis-tenant-b.internal.company.com"
ALLOWED_ORIGINS="https://tenant-b.platform.com"
```

---

## Migration Guide

### From Development to Production

1. **Security Hardening**
   ```bash
   # Generate production API keys
   PROD_ADMIN_KEY=$(python -c "import secrets; print('prod-admin-2024-' + secrets.token_urlsafe(32))")
   PROD_USER_KEY=$(python -c "import secrets; print('prod-user-2024-' + secrets.token_urlsafe(32))")
   
   # Update configuration
   VALID_API_KEYS="$PROD_ADMIN_KEY,$PROD_USER_KEY"
   ```

2. **Backend Selection**
   ```bash
   # For high-volume production: Switch to local models
   NEURAL_ENGINE_BACKEND="local"
   LOCAL_MODEL_PATH="/opt/models/production-model.gguf"
   
   # For variable workload: Stay with OpenAI
   NEURAL_ENGINE_BACKEND="openai"
   OPENAI_API_KEY="sk-proj-production-key"
   ```

3. **Infrastructure Updates**
   ```bash
   # Production Redis
   REDIS_HOST="redis-prod.company.com"
   REDIS_PORT=6379
   
   # Production domains
   ALLOWED_ORIGINS="https://simone.company.com"
   ```

### Configuration Version Control

```bash
# .env.template (committed to git)
VALID_API_KEYS="admin-key-placeholder,user-key-placeholder"
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="your-openai-key-here"
REDIS_HOST="localhost"
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000"
SERPER_API_KEY="your-serper-key-here"

# .env (gitignored - contains real secrets)
# Copy .env.template to .env and fill in real values
```

### Backup and Recovery

```bash
# Backup current configuration
cp .env .env.backup.$(date +%Y%m%d)

# Configuration restore
cp .env.backup.20241201 .env

# Validate restored configuration
./validate-config.sh
```

---

## Troubleshooting

### Common Configuration Issues

**Problem**: `ModuleNotFoundError: No module named 'dotenv'`
```bash
# Solution: Install python-dotenv
pip install python-dotenv
```

**Problem**: Configuration not loading from .env
```bash
# Check file location
ls -la /var/www/SIM-ONE/code/.env

# Check file permissions  
chmod 644 .env

# Verify format (no spaces around =)
cat .env | grep -E '^[A-Z_]+='
```

**Problem**: Redis connection refused
```bash
# Check Redis status
redis-cli ping

# Verify host/port
telnet $REDIS_HOST $REDIS_PORT

# Check firewall
sudo ufw status
```

**Problem**: API keys not working
```bash
# Verify key format
echo $VALID_API_KEYS | tr ',' '\n'

# Check for hidden characters
od -c .env | grep API_KEYS
```

### Debug Commands
```bash
# Print all configuration
cd /var/www/SIM-ONE/code
python -c "from mcp_server.config import settings; import json; print(json.dumps(settings.model_dump(), indent=2))"

# Test individual settings
python -c "from mcp_server.config import settings; print(f'Backend: {settings.NEURAL_ENGINE_BACKEND}')"
python -c "from mcp_server.config import settings; print(f'Keys: {len(settings.VALID_API_KEYS.split(\",\"))} configured')"
```

---

## Support Resources

- [API Documentation](./API_DOCUMENTATION.md) - Using configured endpoints
- [MVLM Integration Guide](./MVLM_INTEGRATION.md) - Neural engine setup
- [Security Implementation Guide](./SECURITY.md) - Security configuration
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues

---

*Last updated: August 25, 2025*