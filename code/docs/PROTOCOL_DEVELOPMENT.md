# Protocol Development Guide

This guide provides comprehensive documentation for developing custom cognitive protocols for the SIM-ONE mCP Server, including interface specifications, development workflows, testing procedures, and best practices.

## Table of Contents
1. [Protocol Architecture](#protocol-architecture)
2. [Development Environment Setup](#development-environment-setup)
3. [Protocol Interface Specification](#protocol-interface-specification)
4. [Development Workflow](#development-workflow)
5. [Protocol Configuration](#protocol-configuration)
6. [Testing Framework](#testing-framework)
7. [Integration & Deployment](#integration--deployment)
8. [Best Practices](#best-practices)
9. [Advanced Features](#advanced-features)

---

## Protocol Architecture

### What is a Cognitive Protocol?

A **Cognitive Protocol** is a self-contained, pluggable module that performs a specific cognitive function within the SIM-ONE mCP Server. Each protocol:

- **Receives** structured input data (context dictionary)
- **Processes** that data using cognitive algorithms
- **Returns** structured output data
- **Integrates** seamlessly with the orchestration engine
- **Supports** both synchronous and asynchronous execution

### Protocol Types by Function

| Type | Purpose | Examples | Resource Usage |
|------|---------|----------|----------------|
| **Analysis** | Data analysis and extraction | ESL (emotion), MTP (entities) | Low CPU, Low Memory |
| **Logical** | Reasoning and inference | REP (reasoning), VVP (validation) | Medium CPU, Medium Memory |
| **Creative** | Content generation | Ideator, Drafter | High CPU, High Memory |
| **Evaluative** | Quality assessment | Critic, Quality Scorer | Medium CPU, Low Memory |
| **Transformative** | Data transformation | Revisor, Summarizer | Medium CPU, Medium Memory |

### Protocol Lifecycle

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DISCOVERY     │ -> │   LOADING       │ -> │  INSTANTIATION  │
│                 │    │                 │    │                 │
│ • protocol.json │    │ • Import module │    │ • Create        │
│ • Metadata      │    │ • Validate deps │    │   instance      │
│   extraction    │    │ • Check config  │    │ • Initialize    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TERMINATION   │ <- │   EXECUTION     │ <- │  REGISTRATION   │
│                 │    │                 │    │                 │
│ • Cleanup       │    │ • Input         │    │ • Add to        │
│ • Resource      │    │   validation    │    │   protocol      │
│   release       │    │ • Processing    │    │   manager       │
│                 │    │ • Output        │    │ • Ready for     │
│                 │    │   generation    │    │   execution     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Development Environment Setup

### Prerequisites

```bash
# Python environment
python >= 3.11
pip >= 23.0

# Development tools
git
code editor (VS Code recommended)
```

### Development Setup

1. **Clone and Setup Repository**
   ```bash
   cd /var/www/SIM-ONE/code
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

2. **Development Dependencies**
   ```bash
   # Additional development tools
   pip install pytest pytest-asyncio
   pip install black flake8 mypy
   pip install pre-commit
   ```

3. **Environment Configuration**
   ```bash
   # Create development .env
   cp .env.template .env.development
   # Configure for protocol development
   NEURAL_ENGINE_BACKEND="openai"  # or "local" 
   OPENAI_API_KEY="your-dev-key"
   VALID_API_KEYS="dev-admin-key,dev-user-key"
   ```

### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "./venv/bin/pytest"
}
```

---

## Protocol Interface Specification

### Base Protocol Interface

Every protocol must implement this interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class CognitiveProtocol(ABC):
    """
    Base class for all cognitive protocols in the SIM-ONE mCP Server.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the protocol with configuration.
        
        Args:
            config: Protocol-specific configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - MUST be implemented by all protocols.
        
        Args:
            context: Input context dictionary containing data and metadata
            
        Returns:
            Dictionary containing protocol execution results
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If execution fails
        """
        raise NotImplementedError("Protocol must implement execute method")
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context before execution.
        
        Args:
            context: Input context to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If validation fails with specific error
        """
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Required field '{field}' missing from context")
        return True
    
    def get_required_fields(self) -> List[str]:
        """
        Return list of required input fields.
        
        Returns:
            List of required field names
        """
        return ["user_input"]  # Default requirement
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return protocol metadata.
        
        Returns:
            Dictionary containing protocol metadata
        """
        return {
            "name": self.__class__.__name__,
            "version": getattr(self, "VERSION", "1.0.0"),
            "type": getattr(self, "PROTOCOL_TYPE", "Unknown"),
            "description": self.__doc__ or "No description provided"
        }
    
    def cleanup(self) -> None:
        """
        Cleanup method called when protocol is destroyed.
        Override to implement custom cleanup logic.
        """
        pass
```

### Context Structure

#### Input Context
```python
context = {
    # Required fields
    "user_input": str,              # Original user input
    "session_id": str,              # Session identifier
    
    # Optional fields
    "batch_memory": List[dict],     # Retrieved memories
    "metadata": {                   # Request metadata
        "timestamp": str,
        "user_id": str,
        "ip_address": str
    },
    
    # Protocol outputs from previous protocols
    "ProtocolName": dict,           # Results from other protocols
    
    # Configuration
    "config": dict                  # Runtime configuration
}
```

#### Output Structure
```python
output = {
    # Protocol-specific results
    "result_field": Any,            # Main results
    "confidence": float,            # Confidence score (0.0-1.0)
    "processing_time": float,       # Execution time in seconds
    
    # Optional metadata
    "metadata": {
        "algorithm": str,           # Algorithm used
        "version": str,             # Protocol version
        "warnings": List[str]       # Any warnings generated
    },
    
    # Error handling
    "error": Optional[str],         # Error message if any
    "partial_result": bool          # True if partial results returned
}
```

---

## Development Workflow

### Step 1: Protocol Planning

#### Define Protocol Scope
```python
# Protocol specification template
PROTOCOL_SPEC = {
    "name": "CustomProtocol",
    "purpose": "Specific cognitive function description",
    "input_requirements": [
        "user_input",
        "additional_required_fields"
    ],
    "output_format": {
        "primary_result": "description",
        "confidence": "0.0-1.0 confidence score",
        "metadata": "additional information"
    },
    "dependencies": ["other_protocols"],
    "resource_requirements": {
        "cpu": "estimated CPU usage",
        "memory": "estimated memory usage",
        "external_apis": "any external dependencies"
    }
}
```

#### Algorithm Design
```python
# Algorithm pseudocode template
def algorithm_pseudocode():
    """
    1. Input validation
       - Check required fields
       - Validate data types
       - Apply business rules
    
    2. Preprocessing  
       - Clean and normalize input
       - Extract relevant features
       - Prepare data structures
    
    3. Core processing
       - Apply main algorithm
       - Generate intermediate results
       - Calculate confidence scores
    
    4. Post-processing
       - Format output
       - Add metadata
       - Validate results
    
    5. Error handling
       - Handle exceptions gracefully  
       - Provide meaningful error messages
       - Return partial results if possible
    """
    pass
```

### Step 2: Directory Structure Creation

```bash
# Create protocol directory
mkdir -p mcp_server/protocols/my_protocol

# Create required files
touch mcp_server/protocols/my_protocol/__init__.py
touch mcp_server/protocols/my_protocol/protocol.json
touch mcp_server/protocols/my_protocol/my_protocol.py

# Optional supporting files
touch mcp_server/protocols/my_protocol/config.py
touch mcp_server/protocols/my_protocol/utils.py
touch mcp_server/protocols/my_protocol/README.md
```

### Step 3: Protocol Implementation

#### Basic Protocol Template
```python
# mcp_server/protocols/my_protocol/my_protocol.py
import logging
from typing import Dict, Any, List
import time

from mcp_server.protocols.base import CognitiveProtocol

logger = logging.getLogger(__name__)

class MyProtocol(CognitiveProtocol):
    """
    Custom cognitive protocol for [specific function].
    
    This protocol [detailed description of functionality].
    """
    
    VERSION = "1.0.0"
    PROTOCOL_TYPE = "Analysis"  # Analysis, Logical, Creative, Evaluative, Transformative
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize protocol-specific components
        self.algorithm_config = config.get("algorithm", {})
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Load any required models or data
        self._load_resources()
    
    def _load_resources(self) -> None:
        """Load protocol-specific resources."""
        # Load models, lookup tables, etc.
        self.logger.info(f"Loading resources for {self.__class__.__name__}")
    
    def get_required_fields(self) -> List[str]:
        """Define required input fields."""
        return ["user_input"]  # Add protocol-specific requirements
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the protocol's main functionality.
        
        Args:
            context: Input context dictionary
            
        Returns:
            Protocol execution results
        """
        start_time = time.time()
        
        try:
            # 1. Input validation
            self.validate_input(context)
            
            # 2. Extract input data
            user_input = context["user_input"]
            session_id = context.get("session_id")
            
            # 3. Core processing
            result = self._process_input(user_input, context)
            
            # 4. Calculate confidence
            confidence = self._calculate_confidence(result, context)
            
            # 5. Prepare output
            output = {
                "result": result,
                "confidence": confidence,
                "processing_time": time.time() - start_time,
                "metadata": {
                    "algorithm": self.algorithm_config.get("name", "default"),
                    "version": self.VERSION,
                    "warnings": []
                }
            }
            
            self.logger.info(f"Protocol executed successfully in {output['processing_time']:.3f}s")
            return output
            
        except Exception as e:
            self.logger.error(f"Protocol execution failed: {e}")
            return {
                "error": str(e),
                "partial_result": False,
                "processing_time": time.time() - start_time
            }
    
    def _process_input(self, user_input: str, context: Dict[str, Any]) -> Any:
        """
        Core processing logic - implement your algorithm here.
        
        Args:
            user_input: The user's input text
            context: Full context dictionary
            
        Returns:
            Processed result
        """
        # TODO: Implement your core algorithm
        # Example:
        result = {
            "analysis": f"Processed: {user_input}",
            "features": [],
            "categories": []
        }
        return result
    
    def _calculate_confidence(self, result: Any, context: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the result.
        
        Args:
            result: Processing result
            context: Input context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # TODO: Implement confidence calculation
        # Example: base confidence with adjustments
        base_confidence = 0.7
        
        # Adjust based on input quality, result completeness, etc.
        confidence = min(1.0, max(0.0, base_confidence))
        
        return confidence
    
    def cleanup(self) -> None:
        """Clean up protocol resources."""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        # TODO: Release any resources (models, connections, etc.)
```

### Step 4: Protocol Configuration

#### protocol.json Configuration
```json
{
    "name": "MyProtocol",
    "version": "1.0.0",
    "entryPoint": "mcp_server.protocols.my_protocol.my_protocol.MyProtocol",
    "type": "Analysis",
    "description": "Custom protocol for specific cognitive function",
    "dependencies": [],
    "resourceRequirements": {
        "cpu": "100m",
        "memory": "128Mi"
    },
    "configuration": {
        "confidence_threshold": 0.5,
        "algorithm": {
            "name": "default",
            "parameters": {}
        }
    },
    "metadata": {
        "author": "Developer Name",
        "created": "2024-08-25",
        "tags": ["analysis", "custom"]
    }
}
```

---

## Protocol Configuration

### Configuration Hierarchy

Protocol configuration follows this hierarchy (highest priority first):
1. **Runtime Configuration** - Passed during execution
2. **Environment Variables** - System-level overrides
3. **protocol.json** - Default protocol configuration
4. **Code Defaults** - Hard-coded fallbacks

### Configuration Examples

#### Simple Configuration
```json
{
    "name": "SimpleProtocol",
    "version": "1.0.0",
    "entryPoint": "mcp_server.protocols.simple.simple.SimpleProtocol",
    "configuration": {
        "threshold": 0.7,
        "max_iterations": 10
    }
}
```

#### Advanced Configuration
```json
{
    "name": "AdvancedProtocol",
    "version": "2.1.0",
    "entryPoint": "mcp_server.protocols.advanced.advanced.AdvancedProtocol",
    "dependencies": ["ReasoningAndExplanationProtocol"],
    "configuration": {
        "algorithm": {
            "name": "neural_enhanced",
            "model_path": "./models/protocol_model.pkl",
            "parameters": {
                "learning_rate": 0.01,
                "epochs": 100,
                "batch_size": 32
            }
        },
        "preprocessing": {
            "normalize": true,
            "remove_stopwords": true,
            "min_length": 5
        },
        "output": {
            "include_metadata": true,
            "confidence_threshold": 0.8,
            "max_results": 50
        }
    },
    "resourceRequirements": {
        "cpu": "500m",
        "memory": "1Gi",
        "gpu": "optional"
    }
}
```

### Environment Variable Overrides
```bash
# Override configuration via environment variables
MY_PROTOCOL_THRESHOLD=0.9
MY_PROTOCOL_ALGORITHM_NAME="enhanced"
MY_PROTOCOL_MAX_ITERATIONS=20
```

---

## Testing Framework

### Test Structure

```
mcp_server/protocols/my_protocol/
├── __init__.py
├── protocol.json
├── my_protocol.py
└── tests/
    ├── __init__.py
    ├── test_my_protocol.py
    ├── test_integration.py
    ├── fixtures/
    │   ├── input_samples.json
    │   └── expected_outputs.json
    └── conftest.py
```

### Unit Testing Template

```python
# tests/test_my_protocol.py
import pytest
import json
from pathlib import Path
from mcp_server.protocols.my_protocol.my_protocol import MyProtocol

class TestMyProtocol:
    """Unit tests for MyProtocol."""
    
    @pytest.fixture
    def protocol(self):
        """Create protocol instance for testing."""
        config = {
            "confidence_threshold": 0.5,
            "algorithm": {"name": "test"}
        }
        return MyProtocol(config)
    
    @pytest.fixture
    def sample_context(self):
        """Sample input context for testing."""
        return {
            "user_input": "Test input text",
            "session_id": "test_session_123",
            "metadata": {"timestamp": "2024-08-25T12:00:00Z"}
        }
    
    def test_protocol_initialization(self, protocol):
        """Test protocol initialization."""
        assert protocol is not None
        assert protocol.VERSION == "1.0.0"
        assert protocol.PROTOCOL_TYPE == "Analysis"
    
    def test_required_fields(self, protocol):
        """Test required fields specification."""
        required = protocol.get_required_fields()
        assert "user_input" in required
    
    def test_input_validation_success(self, protocol, sample_context):
        """Test successful input validation."""
        assert protocol.validate_input(sample_context) is True
    
    def test_input_validation_failure(self, protocol):
        """Test input validation with missing fields."""
        invalid_context = {"session_id": "test"}
        
        with pytest.raises(ValueError, match="Required field 'user_input' missing"):
            protocol.validate_input(invalid_context)
    
    def test_execute_success(self, protocol, sample_context):
        """Test successful protocol execution."""
        result = protocol.execute(sample_context)
        
        # Verify output structure
        assert "result" in result
        assert "confidence" in result
        assert "processing_time" in result
        assert "error" not in result
        
        # Verify data types
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["processing_time"], float)
        assert result["processing_time"] >= 0.0
    
    def test_execute_with_invalid_input(self, protocol):
        """Test execution with invalid input."""
        invalid_context = {}
        result = protocol.execute(invalid_context)
        
        assert "error" in result
        assert "partial_result" in result
        assert result["partial_result"] is False
    
    @pytest.mark.parametrize("input_text,expected_confidence", [
        ("High quality input", 0.8),
        ("Medium input", 0.6),
        ("Low", 0.3),
    ])
    def test_confidence_calculation(self, protocol, input_text, expected_confidence):
        """Test confidence calculation for different inputs."""
        context = {"user_input": input_text, "session_id": "test"}
        result = protocol.execute(context)
        
        assert abs(result["confidence"] - expected_confidence) < 0.2  # Allow some variance
    
    def test_metadata_generation(self, protocol):
        """Test metadata generation."""
        metadata = protocol.get_metadata()
        
        assert metadata["name"] == "MyProtocol"
        assert metadata["version"] == "1.0.0"
        assert metadata["type"] == "Analysis"
        assert "description" in metadata
    
    def test_cleanup(self, protocol):
        """Test protocol cleanup."""
        # Should not raise any exceptions
        protocol.cleanup()

# Integration tests
class TestMyProtocolIntegration:
    """Integration tests for MyProtocol."""
    
    def test_with_orchestration_engine(self):
        """Test protocol integration with orchestration engine."""
        # TODO: Implement integration test
        pass
    
    def test_with_other_protocols(self):
        """Test protocol interaction with other protocols."""
        # TODO: Implement multi-protocol test
        pass
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import json
from pathlib import Path

@pytest.fixture
def sample_inputs():
    """Load sample input data."""
    fixtures_path = Path(__file__).parent / "fixtures" / "input_samples.json"
    with open(fixtures_path) as f:
        return json.load(f)

@pytest.fixture
def expected_outputs():
    """Load expected output data."""
    fixtures_path = Path(__file__).parent / "fixtures" / "expected_outputs.json"
    with open(fixtures_path) as f:
        return json.load(f)
```

### Running Tests

```bash
# Run protocol-specific tests
cd /var/www/SIM-ONE/code
python -m pytest mcp_server/protocols/my_protocol/tests/ -v

# Run with coverage
python -m pytest mcp_server/protocols/my_protocol/tests/ --cov=mcp_server.protocols.my_protocol

# Run integration tests
python -m pytest mcp_server/protocols/my_protocol/tests/test_integration.py -v
```

---

## Integration & Deployment

### Protocol Registration

Once your protocol is complete, it will be automatically discovered by the Protocol Manager on server startup. No manual registration is required.

#### Verification Steps
```bash
# 1. Start the server
cd /var/www/SIM-ONE/code
uvicorn mcp_server.main:app --reload

# 2. Verify protocol discovery
curl -H "X-API-Key: your-key" http://localhost:8000/protocols

# 3. Test protocol execution
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_names": ["MyProtocol"],
    "initial_data": {"user_input": "test input"}
  }'
```

### Workflow Integration

#### Adding to Templates
```json
// workflow_templates.json
{
  "custom_workflow": {
    "description": "Workflow using custom protocol",
    "protocols": ["EmotionalStateLayerProtocol", "MyProtocol"],
    "mode": "Sequential"
  }
}
```

#### Dynamic Workflow Usage
```bash
# Use in dynamic workflow
curl -X POST "http://localhost:8000/execute" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_names": ["EmotionalStateLayerProtocol", "MyProtocol", "MemoryTaggerProtocol"],
    "coordination_mode": "Sequential",
    "initial_data": {"user_input": "Complex analysis request"}
  }'
```

### Performance Monitoring

```python
# Add performance logging to your protocol
import time
from mcp_server.resource_manager.resource_manager import ResourceManager

class MyProtocol(CognitiveProtocol):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        with ResourceManager().profile("MyProtocol.execute"):
            # Your protocol execution code
            result = self._process_input(context["user_input"], context)
            return result
```

---

## Best Practices

### Code Quality

#### Type Hints
```python
from typing import Dict, Any, List, Optional, Union, Tuple

class MyProtocol(CognitiveProtocol):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Use specific return type annotations
        return self._create_response(processed_data)
    
    def _process_input(self, text: str, options: Dict[str, Any]) -> Tuple[List[str], float]:
        # Clear type hints for internal methods
        tokens: List[str] = text.split()
        confidence: float = self._calculate_score(tokens)
        return tokens, confidence
```

#### Error Handling
```python
class MyProtocol(CognitiveProtocol):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self._process_input(context["user_input"], context)
            return self._create_success_response(result)
            
        except ValueError as e:
            self.logger.warning(f"Input validation error: {e}")
            return self._create_error_response(f"Invalid input: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
            return self._create_error_response("Internal processing error")
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "error": error_msg,
            "partial_result": False,
            "processing_time": 0.0
        }
```

#### Logging Best Practices
```python
import logging

class MyProtocol(CognitiveProtocol):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Use protocol-specific logger
        self.logger = logging.getLogger(f"protocols.{self.__class__.__name__}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id", "unknown")
        self.logger.info(f"Starting execution for session {session_id}")
        
        # Log important processing steps
        self.logger.debug(f"Processing input length: {len(context['user_input'])}")
        
        # Log performance metrics
        start_time = time.time()
        result = self._process_input(context["user_input"], context)
        processing_time = time.time() - start_time
        
        self.logger.info(f"Execution completed in {processing_time:.3f}s")
        
        return result
```

### Performance Optimization

#### Memory Management
```python
class MyProtocol(CognitiveProtocol):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Use __slots__ for memory efficiency
        self._cache = {}
        self._max_cache_size = config.get("cache_size", 100)
    
    def _process_input(self, text: str, context: Dict[str, Any]) -> Any:
        # Implement caching for expensive operations
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._expensive_operation(text)
        
        # Maintain cache size
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
        return result
    
    def cleanup(self) -> None:
        # Clear cache on cleanup
        self._cache.clear()
```

#### Async Support
```python
import asyncio
from typing import Dict, Any

class AsyncMyProtocol(CognitiveProtocol):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution for I/O-bound protocols."""
        try:
            # For CPU-bound work, use thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._cpu_intensive_task, context["user_input"]
            )
            
            # For I/O-bound work, use async operations
            external_data = await self._fetch_external_data(context)
            
            return self._combine_results(result, external_data)
            
        except Exception as e:
            self.logger.error(f"Async execution failed: {e}")
            return {"error": str(e), "partial_result": False}
    
    async def _fetch_external_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Example async I/O operation."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/data") as response:
                return await response.json()
```

### Security Considerations

#### Input Sanitization
```python
import re
from typing import Dict, Any

class SecureProtocol(CognitiveProtocol):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Define input validation patterns
        self.safe_text_pattern = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\-\'\"]+$')
        self.max_input_length = config.get("max_input_length", 10000)
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        user_input = context.get("user_input", "")
        
        # Length validation
        if len(user_input) > self.max_input_length:
            raise ValueError(f"Input too long: {len(user_input)} > {self.max_input_length}")
        
        # Content validation
        if not self.safe_text_pattern.match(user_input):
            raise ValueError("Input contains potentially unsafe characters")
        
        # Business logic validation
        if self._contains_sensitive_data(user_input):
            raise ValueError("Input contains sensitive information")
        
        return True
    
    def _contains_sensitive_data(self, text: str) -> bool:
        """Check for sensitive data patterns."""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                return True
        return False
```

### Documentation Standards

#### Docstring Format
```python
class MyProtocol(CognitiveProtocol):
    """
    Custom cognitive protocol for specific analysis tasks.
    
    This protocol implements [algorithm/approach] to perform [specific function].
    It is optimized for [use case] and provides [key features].
    
    Attributes:
        VERSION (str): Protocol version
        PROTOCOL_TYPE (str): Protocol category
        
    Example:
        >>> protocol = MyProtocol({"threshold": 0.8})
        >>> result = protocol.execute({"user_input": "test text"})
        >>> print(result["confidence"])
        0.85
    """
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the protocol's main cognitive function.
        
        Args:
            context (Dict[str, Any]): Input context containing:
                - user_input (str): Text to analyze
                - session_id (str): Session identifier
                - metadata (Dict): Additional metadata
                
        Returns:
            Dict[str, Any]: Execution results containing:
                - result (Any): Main processing result
                - confidence (float): Confidence score (0.0-1.0)
                - processing_time (float): Execution time in seconds
                - metadata (Dict): Additional result metadata
                
        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing encounters an error
            
        Example:
            >>> context = {"user_input": "analyze this", "session_id": "123"}
            >>> result = protocol.execute(context)
            >>> assert "result" in result
        """
        pass
```

---

## Advanced Features

### Neural Engine Integration

```python
from mcp_server.neural_engine.neural_engine import NeuralEngine

class NeuralEnhancedProtocol(CognitiveProtocol):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.neural_engine = NeuralEngine()
        self.use_neural_fallback = config.get("neural_fallback", True)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Try rule-based approach first
            result = self._rule_based_processing(context["user_input"])
            confidence = self._calculate_confidence(result)
            
            # Use neural engine if confidence is low
            if confidence < 0.6 and self.use_neural_fallback:
                neural_result = self._neural_enhancement(context["user_input"])
                result = self._merge_results(result, neural_result)
                confidence = max(confidence, 0.8)  # Neural boost
            
            return {
                "result": result,
                "confidence": confidence,
                "method": "hybrid" if confidence > 0.6 else "neural_enhanced"
            }
            
        except Exception as e:
            return {"error": str(e), "partial_result": False}
    
    def _neural_enhancement(self, text: str) -> Dict[str, Any]:
        """Use neural engine for enhanced processing."""
        prompt = f"Analyze the following text: {text}"
        neural_response = self.neural_engine.generate_text(prompt)
        
        # Parse neural response into structured format
        return self._parse_neural_response(neural_response)
```

### Memory Integration

```python
from mcp_server.memory_manager.memory_manager import MemoryManager

class MemoryAwareProtocol(CognitiveProtocol):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.memory_manager = MemoryManager()
        self.use_memory = config.get("use_memory", True)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id")
        
        # Retrieve relevant memories
        relevant_memories = []
        if self.use_memory and session_id:
            relevant_memories = self.memory_manager.retrieve_memories(
                session_id, 
                context["user_input"]
            )
        
        # Process with memory context
        result = self._process_with_memory(
            context["user_input"], 
            relevant_memories
        )
        
        # Store important results as memories
        if self._is_memorable(result):
            self.memory_manager.store_memory(session_id, {
                "content": context["user_input"],
                "result": result,
                "protocol": self.__class__.__name__,
                "timestamp": time.time()
            })
        
        return result
    
    def _process_with_memory(self, text: str, memories: List[Dict]) -> Dict[str, Any]:
        """Process input considering relevant memories."""
        # Analyze input
        current_analysis = self._analyze_input(text)
        
        # Consider memory context
        memory_context = self._extract_memory_context(memories)
        
        # Merge current analysis with memory context
        enhanced_result = self._merge_with_memory(current_analysis, memory_context)
        
        return enhanced_result
```

### Multi-Protocol Dependencies

```python
class DependentProtocol(CognitiveProtocol):
    """Protocol that depends on outputs from other protocols."""
    
    def get_dependencies(self) -> List[str]:
        """Return list of required protocols."""
        return ["EmotionalStateLayerProtocol", "ReasoningAndExplanationProtocol"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Verify dependencies are present
        dependencies = self.get_dependencies()
        for dep in dependencies:
            if dep not in context:
                raise RuntimeError(f"Missing dependency: {dep}")
        
        # Extract dependency results
        emotional_state = context["EmotionalStateLayerProtocol"]
        reasoning_result = context["ReasoningAndExplanationProtocol"]
        
        # Process using dependency outputs
        result = self._process_with_dependencies(
            context["user_input"],
            emotional_state,
            reasoning_result
        )
        
        return result
    
    def _process_with_dependencies(
        self, 
        text: str, 
        emotions: Dict[str, Any], 
        reasoning: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process input using results from dependency protocols."""
        # Combine insights from multiple protocols
        combined_analysis = {
            "emotional_context": emotions.get("valence", "neutral"),
            "logical_context": reasoning.get("conclusion", ""),
            "confidence_boost": emotions.get("confidence", 0.5) * reasoning.get("confidence", 0.5)
        }
        
        return combined_analysis
```

---

## Support Resources

### Development Tools

- **Protocol Generator**: Command-line tool for creating protocol scaffolding
- **Testing Framework**: Automated testing utilities for protocols
- **Performance Profiler**: Resource usage analysis tools
- **Documentation Generator**: Auto-generate protocol documentation

### Community Resources

- **Protocol Registry**: Share and discover community protocols
- **Best Practices Guide**: Community-contributed guidelines
- **Example Protocols**: Reference implementations
- **Troubleshooting Wiki**: Common issues and solutions

### Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - System architecture details
- [API Documentation](./API_DOCUMENTATION.md) - API reference
- [Configuration Guide](./CONFIGURATION.md) - Environment setup
- [Security Guidelines](./SECURITY.md) - Security best practices

---

*Last updated: August 25, 2025*