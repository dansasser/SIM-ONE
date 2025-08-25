# MVLM Integration Guide

This guide provides comprehensive instructions for integrating Multi-modal Large Language Models (MVLMs) with the SIM-ONE mCP Server, covering both OpenAI and local model backends.

## Table of Contents
1. [Backend Selection Guide](#backend-selection-guide)
2. [OpenAI Backend Setup](#openai-backend-setup)
3. [Local MVLM Backend Setup](#local-mvlm-backend-setup)
4. [Configuration Examples](#configuration-examples)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Migration Guide](#migration-guide)

---

## Backend Selection Guide

### Comparison Overview

| Feature | OpenAI Backend | Local MVLM Backend |
|---------|----------------|---------------------|
| **Setup Complexity** | Simple | Moderate to High |
| **Initial Cost** | Low | High (Hardware) |
| **Ongoing Cost** | Per-token pricing | Hardware/Power only |
| **Performance** | Consistent, High | Variable (Hardware-dependent) |
| **Privacy** | Data sent to OpenAI | Complete data control |
| **Internet Required** | Yes | No |
| **Model Updates** | Automatic | Manual |
| **Customization** | Limited | Full control |
| **Scalability** | Auto-scaling | Hardware limited |

### Use Case Recommendations

#### Choose **OpenAI Backend** when:
- 🚀 **Rapid Prototyping**: Need quick setup and testing
- 💰 **Low Initial Investment**: Limited hardware budget
- 📊 **Variable Usage**: Unpredictable or low-volume workloads
- 🌐 **Always Online**: Consistent internet connectivity
- 🔄 **Latest Models**: Want access to newest model versions
- 👥 **Small Teams**: Limited DevOps resources

#### Choose **Local MVLM Backend** when:
- 🔒 **Data Privacy**: Sensitive data that cannot leave your infrastructure
- 💲 **Cost Control**: High-volume usage where per-token costs are prohibitive
- ⚡ **Low Latency**: Need minimal response times
- 🏢 **Compliance**: Regulatory requirements for data residency
- 🎛️ **Customization**: Need fine-tuned models for specific domains
- 📶 **Offline Operation**: Air-gapped or limited connectivity environments

### Performance Characteristics

#### OpenAI Backend
- **Latency**: 500-2000ms (network dependent)
- **Throughput**: Rate-limited (varies by plan)
- **Reliability**: 99.9% uptime SLA
- **Model Quality**: State-of-the-art performance

#### Local MVLM Backend  
- **Latency**: 100-5000ms (hardware dependent)
- **Throughput**: Hardware limited (parallel processing possible)
- **Reliability**: Depends on your infrastructure
- **Model Quality**: Varies by model selection

---

## OpenAI Backend Setup

### Prerequisites
- OpenAI API account
- Valid API key with sufficient credits
- Internet connectivity

### Configuration Steps

1. **Obtain API Key**
   ```bash
   # Visit https://platform.openai.com/api-keys
   # Create new API key and copy it
   ```

2. **Environment Configuration**
   ```bash
   # Add to your .env file
   NEURAL_ENGINE_BACKEND="openai"
   OPENAI_API_KEY="sk-proj-your-openai-key-here"
   ```

3. **Verify Setup**
   ```bash
   # Test configuration
   cd /var/www/SIM-ONE/code
   python -c "from mcp_server.neural_engine.neural_engine import NeuralEngine; engine = NeuralEngine(); print(engine.generate_text('Hello world'))"
   ```

### Supported Models
- **GPT-4 Turbo**: Best quality, higher cost
- **GPT-4**: Balanced performance and cost  
- **GPT-3.5 Turbo**: Fastest, most economical
- **Custom**: Specify model in protocol configurations

### Cost Management
```bash
# Monitor usage at: https://platform.openai.com/usage
# Set usage limits in OpenAI dashboard
# Configure rate limiting in mCP server (already set to 20/min)
```

---

## Local MVLM Backend Setup

### System Requirements

#### Minimum Requirements
- **CPU**: 8 cores, 3.0GHz+
- **RAM**: 16GB (8GB model) / 32GB (13B model)
- **Storage**: 50GB+ SSD
- **GPU**: Optional but recommended

#### Recommended for Production
- **CPU**: 16+ cores, 3.5GHz+
- **RAM**: 64GB+ 
- **Storage**: 500GB+ NVMe SSD
- **GPU**: NVIDIA RTX 4090 or A100 (24GB+ VRAM)

### Supported Model Formats

The server supports **GGUF format models** via `llama-cpp-python`:

#### Popular Model Sources
1. **Hugging Face Hub**
   ```bash
   # Search for GGUF models
   # Example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
   ```

2. **Recommended Models**
   | Model | Size | RAM Required | Use Case |
   |-------|------|--------------|----------|
   | Llama-2-7B-Chat | 4GB | 8GB | Development/Testing |
   | Llama-2-13B-Chat | 7GB | 16GB | Production |
   | Code Llama-13B | 7GB | 16GB | Code Generation |
   | Mistral-7B-Instruct | 4GB | 8GB | Balanced Performance |

### Installation Steps

1. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install build-essential cmake python3-dev

   # macOS
   brew install cmake

   # For GPU support (NVIDIA)
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Install Python Dependencies**
   ```bash
   cd /var/www/SIM-ONE/code
   pip install llama-cpp-python
   # For GPU support:
   pip install llama-cpp-python[cuda]  # NVIDIA
   pip install llama-cpp-python[metal] # Apple Silicon
   ```

3. **Download Model**
   ```bash
   # Create models directory
   mkdir -p /var/www/SIM-ONE/code/models

   # Example: Download Llama-2-7B-Chat
   cd /var/www/SIM-ONE/code/models
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
   ```

4. **Configure Environment**
   ```bash
   # Add to your .env file
   NEURAL_ENGINE_BACKEND="local"
   LOCAL_MODEL_PATH="/var/www/SIM-ONE/code/models/llama-2-7b-chat.Q4_K_M.gguf"
   ```

5. **Verify Installation**
   ```bash
   cd /var/www/SIM-ONE/code
   python -c "from mcp_server.neural_engine.neural_engine import NeuralEngine; engine = NeuralEngine(); print(engine.generate_text('Hello world'))"
   ```

### Advanced Local Setup

#### GPU Acceleration
```bash
# NVIDIA GPU
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Apple Silicon
export CMAKE_ARGS="-DLLAMA_METAL=on"  
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

#### Memory Optimization
```python
# In local_engine.py, adjust parameters:
self.client = Llama(
    model_path=self.model_path,
    n_ctx=2048,        # Context length
    n_batch=512,       # Batch size
    n_threads=8,       # CPU threads
    n_gpu_layers=35,   # GPU layers (if GPU enabled)
    verbose=False
)
```

---

## Configuration Examples

### Development Environment (.env)
```bash
# Development with OpenAI
NEURAL_ENGINE_BACKEND="openai"
OPENAI_API_KEY="sk-proj-dev-key-here"
VALID_API_KEYS="dev-admin-key,dev-user-key"
REDIS_HOST="localhost" 
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:4321"
```

### Local Development (.env)
```bash
# Development with Local Model
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="./models/llama-2-7b-chat.Q4_K_M.gguf"
VALID_API_KEYS="dev-admin-key,dev-user-key"
REDIS_HOST="localhost"
REDIS_PORT=6379
ALLOWED_ORIGINS="http://localhost:3000"
```

### Production Environment (.env)
```bash
# Production with Local High-Performance Setup
NEURAL_ENGINE_BACKEND="local"
LOCAL_MODEL_PATH="/opt/simone/models/llama-2-13b-chat.Q5_K_M.gguf"
VALID_API_KEYS="prod-admin-key,prod-user-key-1,prod-user-key-2"
REDIS_HOST="redis.internal.company.com"
REDIS_PORT=6379
ALLOWED_ORIGINS="https://simone.company.com,https://api.company.com"
SERPER_API_KEY="serper-prod-key-here"
```

### Docker Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  simone-mcp:
    build: .
    environment:
      NEURAL_ENGINE_BACKEND: "local"
      LOCAL_MODEL_PATH: "/app/models/llama-2-13b-chat.Q4_K_M.gguf"
    volumes:
      - ./models:/app/models:ro
      - model-cache:/tmp/llama_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  model-cache:
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simone-mcp-local
spec:
  replicas: 1
  selector:
    matchLabels:
      app: simone-mcp
  template:
    metadata:
      labels:
        app: simone-mcp
    spec:
      containers:
      - name: simone-mcp
        image: simone/mcp-server:latest
        env:
        - name: NEURAL_ENGINE_BACKEND
          value: "local"
        - name: LOCAL_MODEL_PATH
          value: "/models/llama-2-13b-chat.Q4_K_M.gguf"
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
          limits:
            memory: "64Gi" 
            cpu: "16"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

---

## Performance Optimization

### Local Model Optimization

#### Model Selection Guidelines
```bash
# For Development (4-8GB RAM)
llama-2-7b-chat.Q4_K_M.gguf      # 4GB, good quality/speed balance

# For Production (16-32GB RAM) 
llama-2-13b-chat.Q5_K_M.gguf     # 9GB, higher quality
code-llama-13b-instruct.Q5_K_M.gguf  # 9GB, code-specialized

# For High-Performance (32GB+ RAM)
llama-2-13b-chat.Q8_0.gguf       # 14GB, highest quality
```

#### Runtime Parameters
```python
# Optimize for speed
Llama(
    model_path=model_path,
    n_ctx=1024,        # Smaller context for speed
    n_batch=256,       # Smaller batch size
    n_threads=16,      # Use all CPU cores
    n_gpu_layers=50,   # Offload to GPU if available
)

# Optimize for quality  
Llama(
    model_path=model_path,
    n_ctx=4096,        # Larger context for better understanding
    n_batch=512,       # Larger batch size
    n_threads=8,       # Leave CPU resources for other processes
    n_gpu_layers=35,   # Partial GPU offload
)
```

### OpenAI Backend Optimization

#### Request Optimization
```python
# In protocols, optimize API calls:
response = self.client.chat.completions.create(
    model="gpt-3.5-turbo",  # Use faster model when appropriate
    messages=messages,
    temperature=0.7,        # Balance creativity vs consistency
    max_tokens=500,         # Limit response length
    top_p=0.9,             # Nucleus sampling
    frequency_penalty=0.1,  # Reduce repetition
)
```

#### Caching Strategy
```python
# Implement response caching for repeated queries
import hashlib
import redis

def cached_generate_text(self, prompt: str, model: str = "gpt-3.5-turbo"):
    cache_key = hashlib.md5(f"{prompt}:{model}".encode()).hexdigest()
    cached = redis_client.get(cache_key)
    
    if cached:
        return cached.decode()
    
    response = self.generate_text(prompt, model)
    redis_client.setex(cache_key, 3600, response)  # 1-hour cache
    return response
```

---

## Troubleshooting

### Common Issues

#### Local Model Issues

**Problem**: `ModuleNotFoundError: No module named 'llama_cpp'`
```bash
# Solution: Install llama-cpp-python
pip install llama-cpp-python
# For GPU support:
pip install llama-cpp-python[cuda]
```

**Problem**: `FileNotFoundError: Model file not found`
```bash
# Check model path
ls -la /var/www/SIM-ONE/code/models/
# Verify path in .env file
echo $LOCAL_MODEL_PATH
# Download missing model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

**Problem**: `Out of Memory (OOM) errors`
```bash
# Check available RAM
free -h
# Use smaller model or reduce context size
# Add swap space if necessary
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Problem**: Slow inference on CPU
```bash
# Enable optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Consider GPU acceleration or smaller model
```

#### OpenAI Backend Issues

**Problem**: `Authentication Error`
```bash
# Verify API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
# Check key permissions and credits
```

**Problem**: `Rate Limit Exceeded`
```bash
# Check current usage at: https://platform.openai.com/usage  
# Reduce request frequency or upgrade plan
# Implement exponential backoff
```

**Problem**: `Connection timeout`
```bash
# Check internet connectivity
ping api.openai.com
# Increase timeout in client configuration
# Consider proxy settings if behind corporate firewall
```

#### General Issues

**Problem**: High memory usage
```bash
# Monitor memory usage
htop
# Restart service periodically
sudo systemctl restart simone-mcp
# Implement memory limits in Docker/Kubernetes
```

**Problem**: Poor response quality
```bash
# For local models: Try larger/better quantized model
# For OpenAI: Try different model (GPT-4 vs GPT-3.5)
# Adjust generation parameters (temperature, top_p)
```

### Debugging Commands

```bash
# Check neural engine status
cd /var/www/SIM-ONE/code
python -c "
from mcp_server.neural_engine.neural_engine import NeuralEngine
engine = NeuralEngine()
response = engine.generate_text('Test prompt')
print(f'Backend: {type(engine).__name__}')
print(f'Response: {response}')
"

# Test model loading (local only)
python -c "
from mcp_server.neural_engine.local_engine import LocalModelEngine
from mcp_server.config import settings
engine = LocalModelEngine(settings.LOCAL_MODEL_PATH)
print('Model loaded successfully')
"

# Verify configuration
python -c "
from mcp_server.config import settings
print(f'Backend: {settings.NEURAL_ENGINE_BACKEND}')
print(f'OpenAI Key: {'Set' if settings.OPENAI_API_KEY else 'Not Set'}')
print(f'Local Model: {settings.LOCAL_MODEL_PATH}')
"
```

### Performance Monitoring

```bash
# Monitor system resources
htop              # CPU, RAM usage
nvidia-smi        # GPU usage (if applicable)
iotop             # I/O usage
nethogs           # Network usage

# Application logs
tail -f mcp_server.log | grep "Neural\|Error"

# API response times
curl -w "@curl-format.txt" -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"template_name": "analyze_only", "initial_data": {"user_input": "test"}}'
```

---

## Migration Guide

### From OpenAI to Local Model

1. **Prepare Environment**
   ```bash
   # Install local model dependencies
   pip install llama-cpp-python
   
   # Download model
   mkdir -p models
   wget -O models/llama-2-7b-chat.Q4_K_M.gguf \
     https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
   ```

2. **Update Configuration**
   ```bash
   # Change .env file
   NEURAL_ENGINE_BACKEND="local"
   LOCAL_MODEL_PATH="./models/llama-2-7b-chat.Q4_K_M.gguf"
   # OPENAI_API_KEY can remain set (will be ignored)
   ```

3. **Test Migration**
   ```bash
   # Restart server
   uvicorn mcp_server.main:app --reload
   
   # Test endpoint
   curl -X POST "http://localhost:8000/execute" \
     -H "X-API-Key: your-key" \
     -H "Content-Type: application/json" \
     -d '{"template_name": "analyze_only", "initial_data": {"user_input": "migration test"}}'
   ```

### From Local Model to OpenAI

1. **Update Configuration**
   ```bash
   # Change .env file
   NEURAL_ENGINE_BACKEND="openai" 
   OPENAI_API_KEY="sk-proj-your-key-here"
   # LOCAL_MODEL_PATH can remain set (will be ignored)
   ```

2. **Restart and Test**
   ```bash
   # Restart server  
   uvicorn mcp_server.main:app --reload
   
   # Verify OpenAI connection
   curl -X POST "http://localhost:8000/execute" \
     -H "X-API-Key: your-key" \
     -H "Content-Type: application/json" \
     -d '{"protocol_names": ["ReasoningAndExplanationProtocol"], "initial_data": {"user_input": "openai test"}}'
   ```

---

## Support and Resources

### Documentation Links
- [API Documentation](./API_DOCUMENTATION.md)
- [Configuration Guide](./CONFIGURATION.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

### External Resources
- [Llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Hugging Face GGUF Models](https://huggingface.co/models?library=gguf)

### Community Support
- Check existing issues and solutions
- Performance benchmarks and recommendations
- Model compatibility matrix

---

*Last updated: August 25, 2025*