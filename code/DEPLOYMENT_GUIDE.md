# SIM-ONE Framework MCP Server Deployment Guide

## Overview

This guide covers deployment options for the SIM-ONE Framework MCP Server, from development to enterprise production environments.

## Prerequisites

### System Requirements
- **CPU**: 2+ cores (4+ recommended for production)
- **Memory**: 4GB RAM minimum (8GB+ for production)  
- **Storage**: 20GB+ available space
- **Network**: HTTPS/SSL support for production

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- Redis 7.0+ (or use Docker)
- Nginx (for production load balancing)

## Deployment Options

### 1. Development Environment

**Quick Start:**
```bash
# Clone repository
git clone https://github.com/dansasser/SIM-ONE.git
cd SIM-ONE/code

# Copy environment template
cp .env.production.example .env
# Edit .env with your development values

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f mcp-server
```

**Development Features:**
- Hot reload for code changes
- Redis Commander UI at http://localhost:8081
- Debug logging enabled
- Development API keys pre-configured

### 2. Production Docker Deployment

**Step 1: Environment Configuration**
```bash
# Copy production template
cp .env.production.example .env.production

# Edit with your production values
nano .env.production
```

**Required Environment Variables:**
```bash
# Security (CRITICAL)
VALID_API_KEYS=your-super-secure-admin-key,your-user-key,your-readonly-key
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# External Services
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key

# Database
REDIS_HOST=redis-master
REDIS_PORT=6379

# Monitoring
GRAFANA_PASSWORD=your-secure-grafana-password
```

**Step 2: SSL Certificate Setup**
```bash
# Create SSL directory
mkdir -p config/ssl

# Option A: Self-signed certificates (development/testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/ssl/key.pem \
  -out config/ssl/cert.pem

# Option B: Let's Encrypt (recommended for production)
certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem config/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem config/ssl/key.pem
```

**Step 3: Production Deployment**
```bash
# Load environment
export $(cat .env.production | xargs)

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d

# Monitor deployment
docker-compose -f docker-compose.prod.yml logs -f

# Verify health
curl -f https://yourdomain.com/health
```

### 3. Kubernetes Deployment

**Step 1: Create Namespace**
```bash
kubectl create namespace simone-prod
```

**Step 2: Create Secrets**
```bash
# Create API keys secret
kubectl create secret generic simone-api-keys \
  --from-literal=keys="admin-key,user-key,readonly-key" \
  -n simone-prod

# Create external service secrets
kubectl create secret generic simone-external-apis \
  --from-literal=openai-key="your-openai-key" \
  --from-literal=serper-key="your-serper-key" \
  -n simone-prod
```

**Step 3: Deploy Components**
```yaml
# k8s/redis-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-master
  namespace: simone-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-master
  template:
    metadata:
      labels:
        app: redis-master
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-master
  namespace: simone-prod
spec:
  selector:
    app: redis-master
  ports:
  - port: 6379
    targetPort: 6379
```

**Step 4: Deploy Application**
```yaml
# k8s/mcp-server-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simone-mcp-server
  namespace: simone-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: simone-mcp-server
  template:
    metadata:
      labels:
        app: simone-mcp-server
    spec:
      containers:
      - name: mcp-server
        image: ghcr.io/dansasser/sim-one:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-master"
        - name: VALID_API_KEYS
          valueFrom:
            secretKeyRef:
              name: simone-api-keys
              key: keys
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: simone-external-apis
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Security Configuration

### API Key Management

**Secure API Key Generation:**
```bash
# Generate secure API keys
admin_key=$(openssl rand -hex 32)
user_key=$(openssl rand -hex 32)
readonly_key=$(openssl rand -hex 32)

echo "Admin Key: $admin_key"
echo "User Key: $user_key"
echo "Read-only Key: $readonly_key"
```

**Role Assignment:**
- **Admin**: Full access to all endpoints including metrics
- **User**: Execute workflows, access protocols and templates  
- **Read-only**: View protocols and templates only

### Network Security

**Firewall Configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

**Docker Network Isolation:**
```bash
# Create isolated network for production
docker network create simone-prod-network --driver bridge
```

### SSL/TLS Configuration

**Nginx SSL Best Practices:**
```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS header
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

## Monitoring and Observability

### Health Checks

**Basic Health Check:**
```bash
curl -f https://yourdomain.com/health
```

**Detailed Health Check:**
```bash
curl -f https://yourdomain.com/health/detailed
```

**Expected Response:**
```json
{
  "status": "ok",
  "services": {
    "database": "ok",
    "redis": "ok"
  }
}
```

### Metrics Collection

**Prometheus Metrics:**
- Access at `http://localhost:9090` (if deployed)
- Metrics endpoint: `https://yourdomain.com/metrics` (admin only)

**Grafana Dashboards:**
- Access at `http://localhost:3000` (if deployed)
- Default login: admin / [GRAFANA_PASSWORD]

**Key Metrics to Monitor:**
- Request rate and response times
- Error rates by endpoint
- Redis connection status
- Memory and CPU usage
- Cognitive protocol execution times

### Log Management

**Application Logs:**
```bash
# Docker Compose logs
docker-compose -f docker-compose.prod.yml logs mcp-server-1

# Kubernetes logs
kubectl logs -l app=simone-mcp-server -n simone-prod --tail=100 -f
```

**Security Event Logs:**
- Location: `security_events.log`
- Contains authentication failures, rate limiting events
- Rotate logs regularly to prevent disk space issues

## Scaling and Performance

### Horizontal Scaling

**Docker Compose Scaling:**
```bash
# Scale to 4 MCP server instances
docker-compose -f docker-compose.prod.yml up -d --scale mcp-server-1=2 --scale mcp-server-2=2
```

**Kubernetes Scaling:**
```bash
# Scale deployment
kubectl scale deployment simone-mcp-server --replicas=5 -n simone-prod
```

### Performance Optimization

**Memory Management:**
```bash
# Tune Redis memory settings
echo "maxmemory 1gb" >> config/redis-master.conf
echo "maxmemory-policy allkeys-lru" >> config/redis-master.conf
```

**Worker Configuration:**
```python
# gunicorn.conf.py optimization
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

## Backup and Recovery

### Redis Data Backup

**Manual Backup:**
```bash
# Backup Redis data
docker exec simone-redis-master redis-cli BGSAVE
docker cp simone-redis-master:/data/dump.rdb ./backup/redis-$(date +%Y%m%d).rdb
```

**Automated Backup:**
```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh
```

### Application State Backup

**Database Backup:**
```bash
# SQLite backup
cp data/memory.db backup/memory-$(date +%Y%m%d).db

# Compress and archive
tar -czf backup/simone-backup-$(date +%Y%m%d).tar.gz data/ logs/ config/
```

## Troubleshooting

### Common Issues

**1. Redis Connection Failed**
```bash
# Check Redis status
docker-compose -f docker-compose.prod.yml exec redis-master redis-cli ping

# Check network connectivity
docker-compose -f docker-compose.prod.yml exec mcp-server-1 ping redis-master
```

**2. SSL Certificate Issues**
```bash
# Verify certificate
openssl x509 -in config/ssl/cert.pem -text -noout

# Test SSL configuration
curl -vI https://yourdomain.com/health
```

**3. Authentication Problems**
```bash
# Test API key validation
curl -H "X-API-Key: your-key" https://yourdomain.com/protocols

# Check key configuration
docker-compose -f docker-compose.prod.yml exec mcp-server-1 printenv VALID_API_KEYS
```

### Performance Debugging

**High Memory Usage:**
```bash
# Check container resource usage
docker stats simone-mcp-server-1

# Monitor Redis memory
docker exec simone-redis-master redis-cli info memory
```

**Slow Response Times:**
```bash
# Check application logs for errors
docker-compose -f docker-compose.prod.yml logs mcp-server-1 | grep -i error

# Monitor request patterns
tail -f logs/nginx/access.log | grep -E "POST|GET"
```

## Maintenance

### Updates and Patches

**Security Updates:**
```bash
# Update base images
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Verify update
curl -f https://yourdomain.com/health/detailed
```

**Application Updates:**
```bash
# Deploy new version
export VERSION=1.6.0
docker-compose -f docker-compose.prod.yml up -d

# Monitor rollout
docker-compose -f docker-compose.prod.yml logs -f mcp-server-1
```

### Routine Maintenance Tasks

**Daily:**
- Check health endpoints
- Review error logs
- Monitor resource usage

**Weekly:**
- Update security patches
- Backup data and configuration
- Review security event logs

**Monthly:**
- Certificate renewal (if using Let's Encrypt)
- Performance review and optimization
- Security audit and penetration testing

## Support and Resources

### Documentation
- [SIM-ONE Framework Documentation](README.md)
- [Security Implementation](agents.md)
- [API Reference](code/mcp_server/docs/)

### Monitoring Dashboards
- Application Health: `https://yourdomain.com/health`
- Metrics: `https://yourdomain.com/metrics` (admin only)
- Grafana: `http://localhost:3000` (if deployed)

### Emergency Contacts
- Update this section with your team's contact information
- Include escalation procedures for critical issues
- Document rollback procedures for emergency situations