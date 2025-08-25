# Security Implementation Guide

This guide provides comprehensive documentation for the security features implemented in the SIM-ONE mCP Server, including authentication, authorization, input validation, and security monitoring.

## Table of Contents
1. [Security Overview](#security-overview)
2. [API Key Management](#api-key-management)
3. [Role-Based Access Control](#role-based-access-control)
4. [Input Validation & Attack Prevention](#input-validation--attack-prevention)
5. [Security Headers & CORS](#security-headers--cors)
6. [Session Security](#session-security)
7. [Security Monitoring](#security-monitoring)
8. [Production Security Checklist](#production-security-checklist)
9. [Incident Response](#incident-response)

---

## Security Overview

### Security Architecture

The SIM-ONE mCP Server implements a multi-layered security approach:

```
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│   Client App    │ -> │  Security Headers  │ -> │  Authentication  │
└─────────────────┘    └────────────────────┘    └──────────────────┘
                                  |                         |
                                  v                         v
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ Input Validation│ <- │   Rate Limiting    │ <- │  Authorization   │
└─────────────────┘    └────────────────────┘    └──────────────────┘
                                  |
                                  v
┌─────────────────┐    ┌────────────────────┐
│ Session Manager │ <- │ Cognitive Protocols│
└─────────────────┘    └────────────────────┘
```

### Security Features

- ✅ **API Key Authentication**: SHA-256 hashed keys with unique salts
- ✅ **Role-Based Access Control**: Admin, user, read-only roles
- ✅ **Advanced Input Validation**: SQL injection, XSS, command injection prevention
- ✅ **Security Headers**: CSP, X-Frame-Options, HSTS, etc.
- ✅ **Rate Limiting**: 20 req/min on execute endpoint
- ✅ **Session Isolation**: Users can only access their own sessions
- ✅ **CORS Protection**: Configurable allowed origins
- ✅ **Audit Logging**: Security event tracking

---

## API Key Management

### Key Architecture

API keys are managed through a secure hashing system in `mcp_server.security.key_manager`:

```python
# Key storage format (api_keys.json)
{
  "hash": "sha256_hash_of_key",
  "salt": "unique_16byte_salt", 
  "role": "admin|user|read-only",
  "user_id": "unique_user_identifier"
}
```

### Key Generation Best Practices

#### Generate Secure Keys
```bash
# Generate cryptographically secure API keys
python -c "import secrets; print('admin-' + secrets.token_urlsafe(32))"
python -c "import secrets; print('user-' + secrets.token_urlsafe(32))"  
python -c "import secrets; print('readonly-' + secrets.token_urlsafe(32))"

# Example output:
# admin-kJ7XzQ9_5mR8vN3wP1qY2eT6uI0oL4sA9bC3dE8fG1hK5mN7pQ2rS4tU
# user-a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6a7B8c9
# readonly-x1Y2z3A4b5C6d7E8f9G0h1I2j3K4l5M6n7O8p9Q0r1S2t3U4v5W6x7Y8z9
```

#### Role Assignment Logic
```python
# Automatic role assignment based on key prefix
def assign_role(api_key: str) -> str:
    if "admin" in api_key.lower():
        return "admin"
    elif "readonly" in api_key.lower():
        return "read-only"
    else:
        return "user"
```

### Key Lifecycle Management

#### Initial Setup
```bash
# Set environment variable
VALID_API_KEYS="admin-secure-key-123,user-app-key-456,readonly-monitor-key-789"

# Initialize key store (run once on startup)
cd /var/www/SIM-ONE/code
python -c "from mcp_server.security.key_manager import initialize_api_keys; initialize_api_keys()"
```

#### Key Rotation Strategy
```bash
#!/bin/bash
# rotate-api-keys.sh

echo "🔄 Starting API key rotation..."

# 1. Generate new keys
NEW_ADMIN_KEY="admin-$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
NEW_USER_KEY="user-$(python -c "import secrets; print(secrets.token_urlsafe(32))")"
NEW_READONLY_KEY="readonly-$(python -c "import secrets; print(secrets.token_urlsafe(32))")"

# 2. Temporarily accept both old and new keys
CURRENT_KEYS=$(grep VALID_API_KEYS .env | cut -d= -f2 | tr -d '"')
VALID_API_KEYS="$CURRENT_KEYS,$NEW_ADMIN_KEY,$NEW_USER_KEY,$NEW_READONLY_KEY"

# 3. Update configuration
sed -i "s/VALID_API_KEYS=.*/VALID_API_KEYS=\"$VALID_API_KEYS\"/" .env

# 4. Restart server to load new keys
systemctl restart simone-mcp

echo "✅ New keys available:"
echo "Admin: $NEW_ADMIN_KEY"
echo "User: $NEW_USER_KEY" 
echo "ReadOnly: $NEW_READONLY_KEY"

echo "🔔 Update client applications, then run:"
echo "VALID_API_KEYS=\"$NEW_ADMIN_KEY,$NEW_USER_KEY,$NEW_READONLY_KEY\""
```

#### Key Validation
```bash
# Test API key validity
curl -X GET "http://localhost:8000/protocols" \
  -H "X-API-Key: your-key-here" \
  -w "HTTP Status: %{http_code}\n"

# Valid key returns 200, invalid returns 401
```

### Key Security Features

#### Hashing Algorithm
```python
def hash_api_key(api_key: str, salt: str) -> str:
    """SHA-256 with unique salt per key"""
    hasher = hashlib.sha256()
    hasher.update(salt.encode('utf-8'))
    hasher.update(api_key.encode('utf-8'))
    return hasher.hexdigest()
```

#### Security Benefits
- **Salted Hashes**: Each key has unique 16-byte salt
- **No Plaintext Storage**: Keys never stored in readable format
- **Timing Attack Resistant**: Constant-time comparison
- **Role Isolation**: Each key bound to specific role and user ID

---

## Role-Based Access Control

### Role Definitions

| Role | Permissions | Use Cases |
|------|-------------|-----------|
| **admin** | Full access to all endpoints and all user sessions | System administration, monitoring, debugging |
| **user** | Execute workflows, view own sessions | Application users, API clients |
| **read-only** | View protocols and templates only | Monitoring systems, documentation tools |

### Endpoint Access Matrix

| Endpoint | admin | user | read-only |
|----------|-------|------|-----------|
| `GET /` | ✅ | ✅ | ✅ |
| `POST /execute` | ✅ | ✅ | ❌ |
| `GET /protocols` | ✅ | ✅ | ✅ |
| `GET /templates` | ✅ | ✅ | ✅ |
| `GET /session/{id}` | ✅ (all) | ✅ (own) | ❌ |

### RBAC Implementation

#### Authorization Decorator
```python
# Usage in endpoints
@app.post("/execute", dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def execute_workflow(user: dict = Depends(get_api_key)):
    # user["role"] contains the authenticated role
    # user["user_id"] contains unique user identifier
    pass

# Session authorization check
@app.get("/session/{session_id}", dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_session_history(session_id: str, user: dict = Depends(get_api_key)):
    session_owner = session_manager.get_session_owner(session_id)
    
    # Users can only access their own sessions
    if user["role"] != "admin" and user["user_id"] != session_owner:
        raise HTTPException(status_code=403, detail="Not authorized")
```

#### Role Management
```python
# Add new API key with specific role
from mcp_server.security.key_manager import add_api_key

add_api_key("new-admin-key-123", "admin", "admin_user_001")
add_api_key("new-user-key-456", "user", "app_user_002")  
add_api_key("monitor-key-789", "read-only", "monitor_001")
```

### Session Isolation

#### User-Scoped Sessions
```python
# Sessions are created with user context
session_id = session_manager.create_session(user_id=user["user_id"])

# Session ownership verification
def get_session_owner(session_id: str) -> str:
    """Returns the user_id that owns this session"""
    return session_metadata.get(session_id, {}).get("owner")

# Access control enforcement
if user["role"] != "admin":
    if session_owner != user["user_id"]:
        raise HTTPException(status_code=403)
```

---

## Input Validation & Attack Prevention

### Advanced Input Validator

The server implements comprehensive input validation in `mcp_server.security.advanced_validator`:

#### Attack Pattern Detection

**SQL Injection Prevention**
```python
SQLI_PATTERNS = [
    r"(\s*')\s*OR\s+'\d+'\s*=\s*'\d+'",  # Classic OR injection
    r"(\s*)\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b(\s*)",  # SQL keywords
    r"(\s*)--",  # SQL comments
    r"(\s*);",   # Statement terminators
]
```

**Command Injection Prevention**
```python
CMD_INJECTION_PATTERNS = [
    r";\s*(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)",  # Command chaining
    r"&&\s*(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)",  # Command chaining
    r"`(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)`",     # Command substitution
    r"\$\((ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)\)", # Command substitution
]
```

**Cross-Site Scripting (XSS) Prevention**
```python
XSS_PATTERNS = [
    r"<script.*?>.*?</script>",  # Script tags
    r"javascript:",              # JavaScript URLs
    r"onerror=",                # Event handlers
    r"onload=",                 # Event handlers
    r"<iframe.*?>",             # Iframe injection
]
```

#### Validation Implementation
```python
def advanced_validate_input(data: any):
    """Recursively validates all string values in data structure"""
    if isinstance(data, dict):
        for k, v in data.items():
            advanced_validate_input(v)
    elif isinstance(data, list):
        for item in data:
            advanced_validate_input(item)
    elif isinstance(data, str):
        check_for_patterns(data, SQLI_PATTERNS, "Potential SQL Injection")
        check_for_patterns(data, CMD_INJECTION_PATTERNS, "Potential Command Injection")
        check_for_patterns(data, XSS_PATTERNS, "Potential XSS")
```

### Input Sanitization Best Practices

#### Safe Input Handling
```python
# Example: Protocol input validation
def validate_user_input(user_input: str) -> str:
    # 1. Length limits
    if len(user_input) > 10000:
        raise ValueError("Input too long")
    
    # 2. Character whitelist for critical fields
    import re
    if not re.match(r'^[a-zA-Z0-9\s\.,!?-]+$', user_input):
        raise ValueError("Invalid characters detected")
    
    # 3. Advanced pattern matching
    advanced_validate_input({"input": user_input})
    
    return user_input.strip()
```

#### Testing Attack Vectors
```bash
# Test SQL injection detection
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "test; DROP TABLE users;--"}}'

# Expected response: HTTP 400 - "Malicious input detected: Potential SQL Injection"

# Test XSS detection  
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "<script>alert(\"xss\")</script>"}}'

# Expected response: HTTP 400 - "Malicious input detected: Potential XSS"

# Test command injection detection
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "test; ls -la"}}'

# Expected response: HTTP 400 - "Malicious input detected: Potential Command Injection"
```

---

## Security Headers & CORS

### Security Headers Implementation

The `SecurityHeadersMiddleware` adds comprehensive security headers to all responses:

```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "upgrade-insecure-requests;"
        )
        
        # Additional security headers
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = "geolocation=(), microphone=(), camera=()"
        
        return response
```

### Header Explanations

| Header | Purpose | Value |
|--------|---------|-------|
| **Content-Security-Policy** | Prevents XSS, code injection | `default-src 'self'; script-src 'self'; ...` |
| **X-Frame-Options** | Prevents clickjacking | `DENY` |
| **X-Content-Type-Options** | Prevents MIME sniffing | `nosniff` |
| **Referrer-Policy** | Controls referrer information | `strict-origin-when-cross-origin` |
| **Permissions-Policy** | Restricts browser features | `geolocation=(), microphone=(), camera=()` |

### CORS Configuration

#### Secure CORS Setup
```python
# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Explicit whitelist
    allow_credentials=True,                   # Allow authentication
    allow_methods=["GET", "POST"],           # Restricted methods
    allow_headers=["X-API-Key", "Content-Type"],  # Restricted headers
)
```

#### Environment Configuration
```bash
# Development
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:4321"

# Staging  
ALLOWED_ORIGINS="https://staging-app.company.com,https://staging-admin.company.com"

# Production
ALLOWED_ORIGINS="https://simone.company.com,https://api.company.com"
```

#### Security Testing
```bash
# Test CORS policy
curl -H "Origin: https://malicious-site.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: X-API-Key" \
     -X OPTIONS "http://localhost:8000/execute"

# Should return 400 if origin not in ALLOWED_ORIGINS
```

---

## Session Security

### Session Management Architecture

```python
# Session creation with user context
def create_session(user_id: str) -> str:
    session_id = secrets.token_urlsafe(32)
    session_metadata[session_id] = {
        "owner": user_id,
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "ip_address": request_ip,  # Track origin
    }
    return session_id
```

### Session Security Features

#### User Isolation
```python
# Session access control
def get_session_history(session_id: str, user: dict):
    session_owner = session_manager.get_session_owner(session_id)
    
    # Admin can access all sessions
    if user["role"] == "admin":
        return session_manager.get_history(session_id)
    
    # Users can only access their own sessions
    if session_owner != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this session")
    
    return session_manager.get_history(session_id)
```

#### Session Expiration
```python
# Configure session timeout (if implemented)
SESSION_TIMEOUT = 3600  # 1 hour in seconds

def is_session_expired(session_id: str) -> bool:
    metadata = session_metadata.get(session_id)
    if not metadata:
        return True
    
    last_activity = metadata["last_activity"]
    return (datetime.now() - last_activity).seconds > SESSION_TIMEOUT
```

### Redis Security

#### Connection Security
```bash
# Production Redis configuration
REDIS_HOST="redis-cluster.internal.company.com"  # Internal network only
REDIS_PORT=6379

# Redis AUTH (if configured)
REDIS_PASSWORD="secure-redis-password"
REDIS_SSL=true
```

#### Data Encryption
```python
# Optional: Encrypt sensitive session data
import cryptography.fernet

def encrypt_session_data(data: dict, key: bytes) -> str:
    """Encrypt session data before Redis storage"""
    f = Fernet(key)
    json_data = json.dumps(data).encode()
    encrypted = f.encrypt(json_data)
    return encrypted.decode()

def decrypt_session_data(encrypted_data: str, key: bytes) -> dict:
    """Decrypt session data from Redis"""
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_data.encode())
    return json.loads(decrypted.decode())
```

---

## Security Monitoring

### Audit Logging

#### Security Event Logging
```python
import logging

security_logger = logging.getLogger('security')
security_handler = logging.FileHandler('security_events.log')
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.WARNING)

# Log security events
def log_security_event(event_type: str, details: dict):
    security_logger.warning(f"{event_type}: {json.dumps(details)}")

# Usage examples
log_security_event("AUTH_FAILURE", {
    "ip": request.client.host,
    "api_key_prefix": api_key[:8] + "...",
    "endpoint": request.url.path
})

log_security_event("INPUT_VALIDATION_FAILURE", {
    "ip": request.client.host,
    "attack_type": "SQL_INJECTION",
    "payload": malicious_input[:100]
})
```

#### Log Analysis
```bash
# Monitor security events
tail -f security_events.log

# Analyze attack patterns
grep "INPUT_VALIDATION_FAILURE" security_events.log | \
  jq '.attack_type' | sort | uniq -c

# Track authentication failures
grep "AUTH_FAILURE" security_events.log | \
  jq '.ip' | sort | uniq -c | sort -rn
```

### Rate Limiting Monitoring

#### Current Implementation
```python
# Rate limiting with slowapi
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/execute")
@limiter.limit("20/minute")  # 20 requests per minute per IP
async def execute_workflow(request: Request, ...):
    pass
```

#### Enhanced Rate Limiting
```python
# Custom rate limiter with user tracking
def enhanced_rate_limiter(user_id: str, endpoint: str) -> bool:
    """Enhanced rate limiting per user per endpoint"""
    key = f"rate_limit:{user_id}:{endpoint}"
    current = redis_client.get(key) or 0
    
    if int(current) >= get_rate_limit(endpoint):
        log_security_event("RATE_LIMIT_EXCEEDED", {
            "user_id": user_id,
            "endpoint": endpoint,
            "current_count": current
        })
        return False
    
    redis_client.incr(key)
    redis_client.expire(key, 60)  # 1 minute window
    return True
```

### Intrusion Detection

#### Suspicious Activity Patterns
```python
def detect_suspicious_activity(user_id: str, request_data: dict):
    """Basic intrusion detection patterns"""
    
    # Pattern 1: Rapid successive requests
    request_key = f"requests:{user_id}:last_minute"
    request_count = redis_client.incr(request_key)
    redis_client.expire(request_key, 60)
    
    if request_count > 100:  # 100 requests per minute
        log_security_event("SUSPICIOUS_ACTIVITY", {
            "user_id": user_id,
            "pattern": "RAPID_REQUESTS",
            "count": request_count
        })
    
    # Pattern 2: Multiple validation failures
    validation_key = f"validation_failures:{user_id}:last_hour"
    if redis_client.exists(validation_key):
        failure_count = redis_client.incr(validation_key)
        if failure_count > 10:
            log_security_event("SUSPICIOUS_ACTIVITY", {
                "user_id": user_id,
                "pattern": "REPEATED_VALIDATION_FAILURES",
                "count": failure_count
            })
    
    # Pattern 3: Unusual payload sizes
    payload_size = len(json.dumps(request_data))
    if payload_size > 100000:  # 100KB
        log_security_event("SUSPICIOUS_ACTIVITY", {
            "user_id": user_id,
            "pattern": "LARGE_PAYLOAD",
            "size": payload_size
        })
```

---

## Production Security Checklist

### Pre-Deployment Security Audit

#### Environment Security
- [ ] **Strong API Keys**: All keys are 32+ characters, cryptographically random
- [ ] **Secret Management**: API keys not committed to version control
- [ ] **Environment Isolation**: Separate keys for dev/staging/production
- [ ] **File Permissions**: `.env` file has 600 permissions, owned by app user
- [ ] **HTTPS Only**: All endpoints served over TLS in production
- [ ] **Internal Networks**: Redis and other internal services not publicly accessible

#### Configuration Security  
- [ ] **CORS Policy**: `ALLOWED_ORIGINS` restricted to legitimate frontend domains
- [ ] **Rate Limiting**: Appropriate limits set for production traffic
- [ ] **Session Security**: Sessions isolated by user, proper expiration
- [ ] **Input Validation**: All endpoints protected by advanced validator
- [ ] **Security Headers**: CSP, HSTS, and other headers properly configured
- [ ] **Logging Enabled**: Security event logging active and monitored

#### Infrastructure Security
- [ ] **Firewall Rules**: Only necessary ports open (80, 443, SSH)
- [ ] **SSH Security**: Key-based authentication, no password access
- [ ] **System Updates**: OS and dependencies up to date
- [ ] **Monitoring**: Log aggregation and alerting system active
- [ ] **Backup Security**: Encrypted backups with access control
- [ ] **Network Segmentation**: Application and database in separate subnets

### Security Monitoring Setup

#### Log Aggregation
```bash
# ELK Stack configuration for security logs
# Logstash pattern for security events
input {
  file {
    path => "/var/log/simone/security_events.log"
    type => "security"
  }
}

filter {
  if [type] == "security" {
    json {
      source => "message"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "simone-security-%{+YYYY.MM.dd}"
  }
}
```

#### Alerting Rules
```yaml
# Prometheus alerting rules
groups:
- name: simone-security
  rules:
  - alert: HighAuthenticationFailures
    expr: increase(security_auth_failures_total[5m]) > 10
    for: 2m
    annotations:
      summary: "High authentication failure rate detected"
      
  - alert: InputValidationAttacks
    expr: increase(security_validation_failures_total[5m]) > 5
    for: 1m
    annotations:
      summary: "Input validation attacks detected"
      
  - alert: RateLimitExceeded
    expr: increase(security_rate_limit_exceeded_total[5m]) > 20
    for: 2m
    annotations:
      summary: "Rate limits being exceeded frequently"
```

### Security Testing

#### Automated Security Tests
```bash
#!/bin/bash
# security-test-suite.sh

echo "🔍 Running Security Test Suite..."

# Test 1: Authentication bypass attempts
echo "Testing authentication..."
curl -s -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "test"}}' | \
  grep -q "401" && echo "✅ Authentication required" || echo "❌ Authentication bypass possible"

# Test 2: SQL injection detection
echo "Testing SQL injection protection..."
curl -s -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: $TEST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "test; DROP TABLE users;--"}}' | \
  grep -q "Malicious input detected" && echo "✅ SQL injection blocked" || echo "❌ SQL injection possible"

# Test 3: XSS detection
echo "Testing XSS protection..."
curl -s -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: $TEST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "<script>alert(1)</script>"}}' | \
  grep -q "Malicious input detected" && echo "✅ XSS blocked" || echo "❌ XSS possible"

# Test 4: Rate limiting
echo "Testing rate limiting..."
for i in {1..25}; do
  curl -s -X POST "http://localhost:8000/execute" \
    -H "X-API-Key: $TEST_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"template_name": "analyze_only", "initial_data": {"user_input": "rate test"}}' > /dev/null
done
curl -s -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: $TEST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "rate test"}}' | \
  grep -q "429" && echo "✅ Rate limiting active" || echo "❌ Rate limiting not working"

echo "✅ Security test suite complete"
```

---

## Incident Response

### Security Incident Classification

| Severity | Description | Response Time | Actions |
|----------|-------------|---------------|---------|
| **Critical** | Active exploitation, data breach | < 15 minutes | Immediate shutdown, forensics |
| **High** | Authentication bypass, privilege escalation | < 1 hour | Block attack, patch vulnerability |
| **Medium** | Input validation bypass, DoS attempt | < 4 hours | Monitor, rate limit, investigate |
| **Low** | Suspicious activity, reconnaissance | < 24 hours | Log analysis, monitoring |

### Incident Response Procedures

#### Immediate Response (Critical/High)
```bash
# 1. Isolate the system
sudo ufw deny from <attacker_ip>

# 2. Stop the service if necessary  
sudo systemctl stop simone-mcp

# 3. Preserve evidence
cp /var/log/simone/*.log /backup/incident-$(date +%Y%m%d)/
cp api_keys.json /backup/incident-$(date +%Y%m%d)/

# 4. Rotate all API keys immediately
./rotate-api-keys.sh --emergency

# 5. Review security logs
grep -A5 -B5 "<attacker_pattern>" /var/log/simone/security_events.log
```

#### Investigation Phase
```bash
# Analyze attack patterns
cat security_events.log | jq -r 'select(.level=="WARNING") | .message' | \
  grep "$(date +%Y-%m-%d)" | sort | uniq -c

# Check for data exfiltration
grep -i "session.*admin" /var/log/simone/*.log

# Review authentication logs
grep "AUTH_FAILURE" security_events.log | tail -100
```

#### Recovery Phase
```bash
# 1. Apply security patches
git pull origin main  # If security update available
pip install -r requirements.txt --upgrade

# 2. Update security configuration
# Strengthen rate limits, update validation patterns, etc.

# 3. Restart with enhanced monitoring
sudo systemctl start simone-mcp
tail -f /var/log/simone/security_events.log

# 4. Notify stakeholders
echo "Security incident resolved. System restored at $(date)" | \
  mail -s "SIM-ONE Security Incident - Resolved" admin@company.com
```

### Post-Incident Analysis

#### Security Review Template
```markdown
# Security Incident Report - [Date]

## Incident Summary
- **Severity**: Critical/High/Medium/Low
- **Type**: SQL Injection / XSS / Auth Bypass / DoS / Other
- **Duration**: [Start time] - [Resolution time]
- **Impact**: [Data accessed, services affected, users impacted]

## Timeline
- **Detection**: How and when the incident was discovered
- **Response**: Actions taken and by whom
- **Resolution**: How the incident was resolved

## Root Cause
- **Vulnerability**: Technical details of the security flaw
- **Attack Vector**: How the attacker gained access
- **Contributing Factors**: Configuration issues, missing patches, etc.

## Lessons Learned
- **What Worked**: Effective detection/response mechanisms
- **What Failed**: Gaps in security controls
- **Improvements**: Specific recommendations for prevention

## Action Items
- [ ] **Immediate**: Critical fixes needed right away
- [ ] **Short-term**: Improvements to implement within 2 weeks  
- [ ] **Long-term**: Strategic security enhancements
```

---

## Support Resources

### Security Documentation
- [API Documentation](./API_DOCUMENTATION.md) - Authentication and authorization
- [Configuration Guide](./CONFIGURATION.md) - Secure configuration setup  
- [MVLM Integration Guide](./MVLM_INTEGRATION.md) - Neural engine security
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Security troubleshooting

### External Security Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Web application security risks
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/) - Framework security features
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) - Security standards

### Security Tools
- **Static Analysis**: `bandit` for Python security scanning
- **Dependency Check**: `safety` for vulnerable dependency detection
- **Penetration Testing**: `nmap`, `nikto`, `sqlmap` for security testing
- **Log Analysis**: ELK Stack, Splunk, or similar SIEM solutions

---

*Last updated: August 25, 2025*