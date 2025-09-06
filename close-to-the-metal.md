# SIM-ONE Framework: Complete Performance Implementation Plan

## Executive Summary

This document provides a detailed, step-by-step implementation plan for optimizing the SIM-ONE Framework through Rust extensions, GPU acceleration, and architectural improvements. The plan addresses Python GIL limitations, implements batch-first APIs, and creates a high-performance production system.

**Target Performance Improvements:**
- **Overall workflow:** 2-5x faster (25-36s → 8-15s)
- **Vector operations:** 10-50x faster (10-500ms → 1-10ms)
- **Memory consolidation:** 10x faster (10-30s → 1-5s)
- **Embedding generation:** 50x faster (1K/s → 50K/s)

**Implementation Timeline:** 12-16 weeks across 8 phases

---

## Phase 0: Baselines and Infrastructure (Week 1-2)

### **Objective:** Establish performance baselines and development infrastructure

### **0.1 Benchmarking Suite Setup**

**Create comprehensive benchmark framework:**

```python
# File: benchmarks/benchmark_suite.py
import time
import psutil
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class BenchmarkResult:
    operation: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    peak_memory_mb: float
    cpu_percent: float
    samples: int

class SIMONEBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_operation(self, name: str, operation_func, iterations: int = 100):
        """Benchmark any operation with comprehensive metrics"""
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # Memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Time the operation
            start_time = time.perf_counter()
            result = operation_func()
            end_time = time.perf_counter()
            
            # Memory after
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(mem_after - mem_before)
        
        # Calculate statistics
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        
        result = BenchmarkResult(
            operation=name,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            peak_memory_mb=max(memory_usage),
            cpu_percent=psutil.cpu_percent(),
            samples=iterations
        )
        
        self.results[name] = result
        return result

# Benchmark all current operations
def run_baseline_benchmarks():
    bench = SIMONEBenchmark()
    
    # Vector similarity benchmarks
    bench.benchmark_operation("vector_similarity_100", lambda: benchmark_vector_similarity(100))
    bench.benchmark_operation("vector_similarity_1000", lambda: benchmark_vector_similarity(1000))
    bench.benchmark_operation("vector_similarity_10000", lambda: benchmark_vector_similarity(10000))
    
    # Memory consolidation benchmarks
    bench.benchmark_operation("memory_consolidation_100", lambda: benchmark_memory_consolidation(100))
    bench.benchmark_operation("memory_consolidation_1000", lambda: benchmark_memory_consolidation(1000))
    
    # Five-agent workflow benchmark
    bench.benchmark_operation("five_agent_workflow", lambda: benchmark_full_workflow())
    
    # Embedding generation benchmarks
    bench.benchmark_operation("embedding_generation_10", lambda: benchmark_embedding_generation(10))
    bench.benchmark_operation("embedding_generation_100", lambda: benchmark_embedding_generation(100))
    
    return bench.results
```

**Benchmark target operations:**
```python
# File: benchmarks/target_operations.py
def benchmark_vector_similarity(vector_count: int):
    """Benchmark current vector similarity implementation"""
    from mcp_server.database.vector_search import VectorSimilarityEngine
    
    engine = VectorSimilarityEngine()
    query_vector = engine._generate_mock_embedding("test query")
    candidate_vectors = [engine._generate_mock_embedding(f"candidate {i}") 
                        for i in range(vector_count)]
    
    # Time the similarity calculations
    results = []
    for candidate in candidate_vectors:
        similarity = engine.calculate_cosine_similarity(query_vector, candidate)
        results.append(similarity)
    
    return results

def benchmark_memory_consolidation(memory_count: int):
    """Benchmark current memory consolidation implementation"""
    from mcp_server.memory_manager.memory_consolidation import MemoryConsolidationEngine
    
    engine = MemoryConsolidationEngine()
    
    # Create mock memories
    memories = [
        {
            'id': i,
            'content': f"This is memory content number {i} with some variation",
            'timestamp': f"2024-01-{i:02d}"
        }
        for i in range(memory_count)
    ]
    
    # Time the similarity finding
    groups = engine._find_similar_memories("test_session", similarity_threshold=0.85)
    return groups

def benchmark_full_workflow():
    """Benchmark complete five-agent workflow"""
    from mcp_server.orchestration_engine.orchestration_engine import OrchestrationEngine
    from mcp_server.protocol_manager.protocol_manager import ProtocolManager
    from mcp_server.resource_manager.resource_manager import ResourceManager
    from mcp_server.memory_manager.memory_manager import MemoryManager
    
    # Initialize components
    protocol_manager = ProtocolManager()
    resource_manager = ResourceManager()
    memory_manager = MemoryManager()
    orchestrator = OrchestrationEngine(protocol_manager, resource_manager, memory_manager)
    
    # Define five-agent workflow
    workflow = [
        {"step": "IdeatorProtocol"},
        {"step": "DrafterProtocol"},
        {"step": "CriticProtocol"},
        {"step": "RevisorProtocol"},
        {"step": "SummarizerProtocol"}
    ]
    
    context = {
        'session_id': 'benchmark_session',
        'user_input': 'Create a comprehensive analysis of renewable energy trends',
        'latency_info': {'budget_ms': 30000, 'start_time': time.time()}
    }
    
    # Time the complete workflow
    result = asyncio.run(orchestrator.execute_workflow(workflow, context))
    return result
```

### **0.2 Development Environment Setup**

**Rust development environment:**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
rustup component add clippy rustfmt

# Install Python development tools
pip install maturin pytest-benchmark scalene py-spy

# Install GPU development tools (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x  # For CuPy GPU acceleration
```

**Project structure setup:**
```
SIM-ONE/
├── code/
│   ├── mcp_server/           # Existing Python code
│   └── rust_extensions/      # New Rust modules
│       ├── Cargo.toml
│       ├── simone_similarity/
│       ├── simone_regex/
│       ├── simone_ast/
│       └── simone_hash/
├── benchmarks/               # Benchmark suite
├── profiling/               # Profiling results
└── docs/                   # Implementation documentation
```

### **0.3 SLO Definition**

**Performance targets:**
```python
# File: benchmarks/slo_targets.py
SLO_TARGETS = {
    # Workflow targets
    "five_agent_workflow_p95": 15000,  # 15 seconds
    "five_agent_workflow_p50": 10000,  # 10 seconds
    
    # Component targets
    "vector_similarity_1000_p95": 10,    # 10ms for 1K vectors
    "vector_similarity_10000_p95": 100,  # 100ms for 10K vectors
    "memory_consolidation_1000_p95": 5000,  # 5 seconds for 1K memories
    "embedding_generation_100_p95": 1000,   # 1 second for 100 embeddings
    
    # System targets
    "warm_start_p95": 300,     # 300ms warm start
    "cache_hit_ratio_min": 0.7,  # 70% cache hit rate
    "memory_growth_max_mb": 100,  # Max 100MB growth per hour
}
```

---

## Phase 1: Memory Layer and Caching (Week 3-4)

### **Objective:** Implement hierarchical caching with bounded growth and eviction policies

### **1.1 Cache Hierarchy Implementation**

**Base cache interface:**
```python
# File: code/mcp_server/caching/cache_interface.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass
import time

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    max_size_bytes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    def evict(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def get_metrics(self) -> CacheMetrics:
        pass
```

**Hot RAM cache implementation:**
```python
# File: code/mcp_server/caching/hot_ram_cache.py
import threading
import time
from collections import OrderedDict
from typing import Any, Optional
import sys

class HotRAMCache(CacheInterface):
    """High-performance in-memory cache with LRU eviction and size limits"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.data = OrderedDict()
        self.access_times = {}
        self.expiry_times = {}
        self.lock = threading.RLock()
        self.metrics = CacheMetrics(max_size_bytes=self.max_size_bytes)
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            current_time = time.time()
            
            # Check if key exists and not expired
            if key in self.data:
                if key in self.expiry_times and current_time > self.expiry_times[key]:
                    # Expired - remove and count as miss
                    self._remove_key(key)
                    self.metrics.misses += 1
                    return None
                
                # Hit - update access time and move to end (most recent)
                self.data.move_to_end(key)
                self.access_times[key] = current_time
                self.metrics.hits += 1
                return self.data[key]
            
            # Miss
            self.metrics.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        with self.lock:
            current_time = time.time()
            value_size = sys.getsizeof(value)
            
            # Check if adding this would exceed size limit
            if key not in self.data:
                while (self.metrics.size_bytes + value_size > self.max_size_bytes 
                       and len(self.data) > 0):
                    self._evict_lru()
            
            # Add/update the value
            if key in self.data:
                # Update existing - adjust size
                old_size = sys.getsizeof(self.data[key])
                self.metrics.size_bytes = self.metrics.size_bytes - old_size + value_size
            else:
                # New key
                self.metrics.size_bytes += value_size
            
            self.data[key] = value
            self.data.move_to_end(key)  # Mark as most recent
            self.access_times[key] = current_time
            
            # Set expiry time
            ttl = ttl_seconds or self.default_ttl
            self.expiry_times[key] = current_time + ttl
            
            return True
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.data:
            return
        
        # Get least recently used key (first in OrderedDict)
        lru_key = next(iter(self.data))
        self._remove_key(lru_key)
        self.metrics.evictions += 1
    
    def _remove_key(self, key: str):
        """Remove key and update metrics"""
        if key in self.data:
            value_size = sys.getsizeof(self.data[key])
            self.metrics.size_bytes -= value_size
            del self.data[key]
            self.access_times.pop(key, None)
            self.expiry_times.pop(key, None)
    
    def evict(self, key: str) -> bool:
        with self.lock:
            if key in self.data:
                self._remove_key(key)
                self.metrics.evictions += 1
                return True
            return False
    
    def clear(self) -> None:
        with self.lock:
            self.data.clear()
            self.access_times.clear()
            self.expiry_times.clear()
            self.metrics.size_bytes = 0
    
    def get_metrics(self) -> CacheMetrics:
        return self.metrics
```

**Memory-mapped archive implementation:**
```python
# File: code/mcp_server/caching/mmap_archive.py
import mmap
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

class MemoryMappedArchive:
    """Memory-mapped storage for large, infrequently accessed data"""
    
    def __init__(self, archive_path: str = "./data/archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.archive_path / "index.json"
        self.data_file = self.archive_path / "data.bin"
        self.index = self._load_index()
        self.mmap_file = None
        self.file_handle = None
    
    def _load_index(self) -> dict:
        """Load the index mapping keys to file positions"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save the index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)
    
    def _ensure_mmap(self):
        """Ensure memory mapping is available"""
        if self.mmap_file is None and self.data_file.exists():
            self.file_handle = open(self.data_file, 'r+b')
            self.mmap_file = mmap.mmap(self.file_handle.fileno(), 0)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from memory-mapped archive"""
        if key not in self.index:
            return None
        
        self._ensure_mmap()
        if self.mmap_file is None:
            return None
        
        # Get position and size from index
        position, size = self.index[key]
        
        # Read data from memory-mapped file
        self.mmap_file.seek(position)
        data_bytes = self.mmap_file.read(size)
        
        # Deserialize
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Store data in memory-mapped archive"""
        # Serialize data
        data_bytes = json.dumps(value).encode('utf-8')
        
        # Append to data file
        with open(self.data_file, 'ab') as f:
            position = f.tell()
            f.write(data_bytes)
            size = len(data_bytes)
        
        # Update index
        self.index[key] = (position, size)
        self._save_index()
        
        # Refresh memory mapping
        if self.mmap_file:
            self.mmap_file.close()
            self.file_handle.close()
            self.mmap_file = None
            self.file_handle = None
        
        return True
    
    def close(self):
        """Close memory mapping"""
        if self.mmap_file:
            self.mmap_file.close()
            self.file_handle.close()
```

### **1.2 Unified Cache Manager**

**Cache coordination layer:**
```python
# File: code/mcp_server/caching/cache_manager.py
from typing import Any, Optional, Dict
import hashlib
import json

class UnifiedCacheManager:
    """Coordinates multiple cache layers with intelligent routing"""
    
    def __init__(self):
        # Cache layers (ordered by speed)
        self.hot_cache = HotRAMCache(max_size_mb=100)  # Fastest
        self.embedding_cache = HotRAMCache(max_size_mb=500)  # GPU data
        self.session_cache = HotRAMCache(max_size_mb=200)  # Session data
        self.archive = MemoryMappedArchive()  # Slowest but largest
        
        # Cache routing rules
        self.cache_routes = {
            'vector_similarity': self.hot_cache,
            'embeddings': self.embedding_cache,
            'session_data': self.session_cache,
            'long_term': self.archive
        }
    
    def get(self, key: str, cache_type: str = 'auto') -> Optional[Any]:
        """Get value with intelligent cache routing"""
        if cache_type == 'auto':
            cache_type = self._determine_cache_type(key)
        
        # Try appropriate cache first
        primary_cache = self.cache_routes.get(cache_type, self.hot_cache)
        result = primary_cache.get(key)
        if result is not None:
            return result
        
        # Try other caches in order of speed
        for cache_name, cache in self.cache_routes.items():
            if cache != primary_cache:
                result = cache.get(key)
                if result is not None:
                    # Promote to primary cache
                    primary_cache.put(key, result)
                    return result
        
        return None
    
    def put(self, key: str, value: Any, cache_type: str = 'auto', ttl_seconds: Optional[int] = None) -> bool:
        """Put value in appropriate cache"""
        if cache_type == 'auto':
            cache_type = self._determine_cache_type(key)
        
        cache = self.cache_routes.get(cache_type, self.hot_cache)
        return cache.put(key, value, ttl_seconds)
    
    def _determine_cache_type(self, key: str) -> str:
        """Determine appropriate cache based on key pattern"""
        if key.startswith('embedding_'):
            return 'embeddings'
        elif key.startswith('session_'):
            return 'session_data'
        elif key.startswith('similarity_'):
            return 'vector_similarity'
        elif key.startswith('archive_'):
            return 'long_term'
        else:
            return 'vector_similarity'  # Default to hot cache
    
    def get_all_metrics(self) -> Dict[str, CacheMetrics]:
        """Get metrics from all cache layers"""
        return {
            'hot_cache': self.hot_cache.get_metrics(),
            'embedding_cache': self.embedding_cache.get_metrics(),
            'session_cache': self.session_cache.get_metrics(),
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.hot_cache.clear()
        self.embedding_cache.clear()
        self.session_cache.clear()
```

---

## Phase 2: Rust Extensions - Core Modules (Week 5-7)

### **Objective:** Implement high-performance Rust modules for CPU-intensive operations

### **2.1 Rust Project Setup**

**Cargo.toml for the workspace:**
```toml
# File: code/rust_extensions/Cargo.toml
[workspace]
members = [
    "simone_similarity",
    "simone_regex", 
    "simone_ast",
    "simone_hash"
]

[workspace.dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### **2.2 simone_similarity Module (High Priority)**

**Cargo.toml:**
```toml
# File: code/rust_extensions/simone_similarity/Cargo.toml
[package]
name = "simone_similarity"
version = "0.1.0"
edition = "2021"

[lib]
name = "simone_similarity"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { workspace = true, features = ["extension-module"] }
numpy = { workspace = true }
rayon = { workspace = true }
```

**Core similarity implementation:**
```rust
// File: code/rust_extensions/simone_similarity/src/lib.rs
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::arch::x86_64::*;

/// SIMD-optimized cosine similarity calculation
#[inline]
unsafe fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    
    if len < 8 {
        // Fallback for small vectors
        return cosine_similarity_scalar(a, b);
    }
    
    let mut dot_product = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();
    
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        dot_product = _mm256_fmadd_ps(va, vb, dot_product);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }
    
    // Horizontal sum for dot product
    let mut dot_sum = 0.0f32;
    let mut norm_a_sum = 0.0f32;
    let mut norm_b_sum = 0.0f32;
    
    let dot_array: [f32; 8] = std::mem::transmute(dot_product);
    let norm_a_array: [f32; 8] = std::mem::transmute(norm_a);
    let norm_b_array: [f32; 8] = std::mem::transmute(norm_b);
    
    for i in 0..8 {
        dot_sum += dot_array[i];
        norm_a_sum += norm_a_array[i];
        norm_b_sum += norm_b_array[i];
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }
    
    let norm_product = (norm_a_sum * norm_b_sum).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        dot_sum / norm_product
    }
}

/// Scalar fallback for cosine similarity
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Batch cosine similarity calculation with parallel processing
#[pyfunction]
fn cosine_similarity_batch(
    py: Python,
    query: PyReadonlyArray1<f32>,
    candidates: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let candidates_array = candidates.as_array();
    
    let results: Vec<f32> = candidates_array
        .axis_iter(numpy::Axis(0))
        .into_par_iter()
        .map(|candidate| {
            let candidate_slice = candidate.to_slice().unwrap();
            unsafe { cosine_similarity_simd(query_slice, candidate_slice) }
        })
        .collect();
    
    Ok(results.into_pyarray(py).to_owned())
}

/// Top-K similarity search with parallel processing
#[pyfunction]
fn top_k_similarity(
    py: Python,
    query: PyReadonlyArray1<f32>,
    candidates: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f32>>)> {
    let query_slice = query.as_slice()?;
    let candidates_array = candidates.as_array();
    
    let mut similarities: Vec<(usize, f32)> = candidates_array
        .axis_iter(numpy::Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let candidate_slice = candidate.to_slice().unwrap();
            let similarity = unsafe { cosine_similarity_simd(query_slice, candidate_slice) };
            (idx, similarity)
        })
        .collect();
    
    // Sort by similarity (descending) and take top-k
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    similarities.truncate(k);
    
    let indices: Vec<usize> = similarities.iter().map(|(idx, _)| *idx).collect();
    let scores: Vec<f32> = similarities.iter().map(|(_, score)| *score).collect();
    
    Ok((
        indices.into_pyarray(py).to_owned(),
        scores.into_pyarray(py).to_owned(),
    ))
}

/// Batch pairwise similarity matrix calculation
#[pyfunction]
fn similarity_matrix(
    py: Python,
    vectors: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let vectors_array = vectors.as_array();
    let n = vectors_array.shape()[0];
    
    let mut matrix = vec![vec![0.0f32; n]; n];
    
    // Parallel computation of upper triangle
    matrix
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            let vec_i = vectors_array.slice(s![i, ..]).to_slice().unwrap();
            for j in i..n {
                let vec_j = vectors_array.slice(s![j, ..]).to_slice().unwrap();
                let similarity = unsafe { cosine_similarity_simd(vec_i, vec_j) };
                row[j] = similarity;
            }
        });
    
    // Mirror to lower triangle
    for i in 0..n {
        for j in 0..i {
            matrix[i][j] = matrix[j][i];
        }
    }
    
    // Convert to numpy array
    let flat_matrix: Vec<f32> = matrix.into_iter().flatten().collect();
    let py_array = PyArray2::from_vec2(py, &vec![flat_matrix; 1])?;
    Ok(py_array.to_owned())
}

/// Python module definition
#[pymodule]
fn simone_similarity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity_matrix, m)?)?;
    Ok(())
}
```

**Python integration wrapper:**
```python
# File: code/mcp_server/rust_modules/similarity_wrapper.py
import numpy as np
from typing import List, Tuple, Optional
import logging

try:
    import simone_similarity
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust similarity module not available, falling back to Python")

class RustSimilarityEngine:
    """High-performance similarity engine using Rust + SIMD"""
    
    def __init__(self, fallback_to_python: bool = True):
        self.rust_available = RUST_AVAILABLE
        self.fallback_to_python = fallback_to_python
        
        if not self.rust_available and not fallback_to_python:
            raise RuntimeError("Rust similarity module not available and fallback disabled")
    
    def cosine_similarity_batch(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Batch cosine similarity calculation"""
        if self.rust_available:
            try:
                return simone_similarity.cosine_similarity_batch(
                    query.astype(np.float32),
                    candidates.astype(np.float32)
                )
            except Exception as e:
                if not self.fallback_to_python:
                    raise
                logging.warning(f"Rust similarity failed, falling back to Python: {e}")
        
        # Python fallback
        return self._python_cosine_similarity_batch(query, candidates)
    
    def top_k_similarity(self, query: np.ndarray, candidates: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-K similarity search"""
        if self.rust_available:
            try:
                return simone_similarity.top_k_similarity(
                    query.astype(np.float32),
                    candidates.astype(np.float32),
                    k
                )
            except Exception as e:
                if not self.fallback_to_python:
                    raise
                logging.warning(f"Rust top-k failed, falling back to Python: {e}")
        
        # Python fallback
        similarities = self._python_cosine_similarity_batch(query, candidates)
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_scores = similarities[top_indices]
        return top_indices, top_scores
    
    def similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Compute full similarity matrix"""
        if self.rust_available:
            try:
                return simone_similarity.similarity_matrix(vectors.astype(np.float32))
            except Exception as e:
                if not self.fallback_to_python:
                    raise
                logging.warning(f"Rust matrix failed, falling back to Python: {e}")
        
        # Python fallback
        return self._python_similarity_matrix(vectors)
    
    def _python_cosine_similarity_batch(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Python fallback for batch similarity"""
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(candidates_norm, query_norm)
        return similarities
    
    def _python_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Python fallback for similarity matrix"""
        # Normalize vectors
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Compute similarity matrix
        return np.dot(vectors_norm, vectors_norm.T)
```

### **2.3 simone_hash Module (High Priority)**

**Fast content hashing implementation:**
```rust
// File: code/rust_extensions/simone_hash/src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Fast content hashing using xxHash-like algorithm
fn fast_hash(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Batch content hashing with parallel processing
#[pyfunction]
fn hash_content_batch(contents: Vec<String>) -> PyResult<Vec<u64>> {
    let hashes: Vec<u64> = contents
        .par_iter()
        .map(|content| fast_hash(content.as_bytes()))
        .collect();
    
    Ok(hashes)
}

/// Content deduplication based on hashes
#[pyfunction]
fn deduplicate_by_hash(contents: Vec<String>) -> PyResult<(Vec<String>, Vec<usize>)> {
    let mut seen_hashes = std::collections::HashSet::new();
    let mut unique_contents = Vec::new();
    let mut original_indices = Vec::new();
    
    for (idx, content) in contents.iter().enumerate() {
        let hash = fast_hash(content.as_bytes());
        if seen_hashes.insert(hash) {
            unique_contents.push(content.clone());
            original_indices.push(idx);
        }
    }
    
    Ok((unique_contents, original_indices))
}

/// Generate content fingerprint for similarity detection
#[pyfunction]
fn content_fingerprint(content: String, shingle_size: usize) -> PyResult<Vec<u64>> {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() < shingle_size {
        return Ok(vec![fast_hash(content.as_bytes())]);
    }
    
    let fingerprints: Vec<u64> = chars
        .windows(shingle_size)
        .map(|window| {
            let shingle: String = window.iter().collect();
            fast_hash(shingle.as_bytes())
        })
        .collect();
    
    Ok(fingerprints)
}

#[pymodule]
fn simone_hash(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_content_batch, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_by_hash, m)?)?;
    m.add_function(wrap_pyfunction!(content_fingerprint, m)?)?;
    Ok(())
}
```

### **2.4 simone_regex Module (Medium Priority)**

**Compiled regex patterns with multi-pattern matching:**
```rust
// File: code/rust_extensions/simone_regex/src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;

/// Compiled regex pattern set for efficient matching
#[pyclass]
struct RegexSet {
    patterns: HashMap<String, Regex>,
}

#[pymethods]
impl RegexSet {
    #[new]
    fn new() -> Self {
        RegexSet {
            patterns: HashMap::new(),
        }
    }
    
    fn add_pattern(&mut self, name: String, pattern: String) -> PyResult<()> {
        match Regex::new(&pattern) {
            Ok(regex) => {
                self.patterns.insert(name, regex);
                Ok(())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid regex pattern: {}", e)
            )),
        }
    }
    
    fn match_text(&self, text: &str) -> PyResult<HashMap<String, bool>> {
        let mut results = HashMap::new();
        
        for (name, regex) in &self.patterns {
            results.insert(name.clone(), regex.is_match(text));
        }
        
        Ok(results)
    }
    
    fn match_batch(&self, texts: Vec<String>) -> PyResult<Vec<HashMap<String, bool>>> {
        let results: Vec<HashMap<String, bool>> = texts
            .par_iter()
            .map(|text| {
                let mut text_results = HashMap::new();
                for (name, regex) in &self.patterns {
                    text_results.insert(name.clone(), regex.is_match(text));
                }
                text_results
            })
            .collect();
        
        Ok(results)
    }
}

/// Extract structured data using regex patterns
#[pyfunction]
fn extract_structured_data(
    text: String,
    patterns: HashMap<String, String>,
) -> PyResult<HashMap<String, Vec<String>>> {
    let mut results = HashMap::new();
    
    for (field_name, pattern_str) in patterns {
        match Regex::new(&pattern_str) {
            Ok(regex) => {
                let matches: Vec<String> = regex
                    .find_iter(&text)
                    .map(|m| m.as_str().to_string())
                    .collect();
                results.insert(field_name, matches);
            }
            Err(_) => {
                results.insert(field_name, vec![]);
            }
        }
    }
    
    Ok(results)
}

#[pymodule]
fn simone_regex(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RegexSet>()?;
    m.add_function(wrap_pyfunction!(extract_structured_data, m)?)?;
    Ok(())
}
```

### **2.5 simone_ast Module (Medium Priority)**

**Syntax validation and AST operations:**
```rust
// File: code/rust_extensions/simone_ast/src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json;

/// Validate JSON syntax
#[pyfunction]
fn validate_json_batch(json_strings: Vec<String>) -> PyResult<Vec<bool>> {
    let results: Vec<bool> = json_strings
        .par_iter()
        .map(|json_str| serde_json::from_str::<serde_json::Value>(json_str).is_ok())
        .collect();
    
    Ok(results)
}

/// Extract JSON paths and values
#[pyfunction]
fn extract_json_paths(
    json_string: String,
    paths: Vec<String>,
) -> PyResult<HashMap<String, Option<String>>> {
    let mut results = HashMap::new();
    
    match serde_json::from_str::<serde_json::Value>(&json_string) {
        Ok(json_value) => {
            for path in paths {
                let value = extract_json_path(&json_value, &path);
                results.insert(path, value);
            }
        }
        Err(_) => {
            for path in paths {
                results.insert(path, None);
            }
        }
    }
    
    Ok(results)
}

fn extract_json_path(value: &serde_json::Value, path: &str) -> Option<String> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = value;
    
    for part in parts {
        match current {
            serde_json::Value::Object(obj) => {
                current = obj.get(part)?;
            }
            serde_json::Value::Array(arr) => {
                if let Ok(index) = part.parse::<usize>() {
                    current = arr.get(index)?;
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
    
    Some(current.to_string())
}

/// Validate code structure patterns
#[pyfunction]
fn validate_code_structure(
    code: String,
    language: String,
) -> PyResult<HashMap<String, bool>> {
    let mut results = HashMap::new();
    
    match language.as_str() {
        "python" => {
            results.insert("has_imports".to_string(), code.contains("import "));
            results.insert("has_functions".to_string(), code.contains("def "));
            results.insert("has_classes".to_string(), code.contains("class "));
            results.insert("proper_indentation".to_string(), validate_python_indentation(&code));
        }
        "json" => {
            results.insert("valid_json".to_string(), serde_json::from_str::<serde_json::Value>(&code).is_ok());
        }
        _ => {
            results.insert("unknown_language".to_string(), true);
        }
    }
    
    Ok(results)
}

fn validate_python_indentation(code: &str) -> bool {
    // Simple indentation validation
    let lines: Vec<&str> = code.lines().collect();
    let mut indent_stack = vec![0];
    
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        
        let indent_level = line.len() - line.trim_start().len();
        
        if line.trim().ends_with(':') {
            // Expecting increased indentation on next non-empty line
            indent_stack.push(indent_level);
        } else if indent_level < *indent_stack.last().unwrap() {
            // Dedent - pop from stack
            while let Some(&last_indent) = indent_stack.last() {
                if indent_level >= last_indent {
                    break;
                }
                indent_stack.pop();
            }
            
            if indent_stack.is_empty() {
                return false;
            }
        }
    }
    
    true
}

#[pymodule]
fn simone_ast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_json_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_json_paths, m)?)?;
    m.add_function(wrap_pyfunction!(validate_code_structure, m)?)?;
    Ok(())
}
```

### **2.6 Build System Integration**

**Maturin configuration:**
```toml
# File: code/rust_extensions/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "simone-rust-extensions"
version = "0.1.0"
description = "High-performance Rust extensions for SIM-ONE Framework"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "simone_rust_extensions"
```

**Build script:**
```bash
#!/bin/bash
# File: code/rust_extensions/build.sh

set -e

echo "Building Rust extensions for SIM-ONE..."

# Build each module
cd simone_similarity
maturin develop --release
cd ..

cd simone_hash
maturin develop --release
cd ..

cd simone_regex
maturin develop --release
cd ..

cd simone_ast
maturin develop --release
cd ..

echo "All Rust extensions built successfully!"

# Run tests
python -c "
import simone_similarity
import simone_hash
import simone_regex
import simone_ast
print('All modules imported successfully!')
"
```

---

## Phase 3: Concurrency Model - Multiprocessing (Week 8-9)

### **Objective:** Implement multiprocessing-based orchestration to avoid GIL limitations

### **3.1 Process Pool Orchestrator**

**Multiprocessing orchestration engine:**
```python
# File: code/mcp_server/orchestration_engine/process_orchestrator.py
import asyncio
import multiprocessing as mp
import concurrent.futures
import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import pickle
import os

@dataclass
class ProcessMetrics:
    process_id: int
    cpu_percent: float
    memory_mb: float
    execution_time_ms: float
    queue_depth: int

class ProcessPoolOrchestrator:
    """High-performance orchestrator using multiprocessing to avoid GIL"""
    
    def __init__(self, max_workers: Optional[int] = None, pin_cores: bool = True):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.pin_cores = pin_cores
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.process_metrics: Dict[int, ProcessMetrics] = {}
        self.task_queue_depth = 0
        
        # Initialize process pool
        self._initialize_pool()
        
        # Core pinning setup
        if self.pin_cores:
            self._setup_core_pinning()
    
    def _initialize_pool(self):
        """Initialize the process pool with optimized settings"""
        # Configure multiprocessing
        mp.set_start_method('spawn', force=True)  # Avoid fork issues
        
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=self._worker_initializer,
            initargs=(self.pin_cores,)
        )
        
        logging.info(f"Initialized process pool with {self.max_workers} workers")
    
    def _setup_core_pinning(self):
        """Setup core pinning for critical processes"""
        try:
            # Pin main process to core 0
            os.sched_setaffinity(0, {0})
            logging.info("Pinned main process to core 0")
        except (OSError, AttributeError):
            logging.warning("Core pinning not supported on this system")
    
    @staticmethod
    def _worker_initializer(pin_cores: bool):
        """Initialize worker process"""
        if pin_cores:
            try:
                # Pin worker to specific core (round-robin)
                worker_id = os.getpid() % os.cpu_count()
                os.sched_setaffinity(0, {worker_id})
            except (OSError, AttributeError):
                pass
        
        # Set process priority
        try:
            os.nice(-5)  # Higher priority for workers
        except (OSError, PermissionError):
            pass
    
    async def execute_protocol_parallel(
        self,
        protocol_tasks: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute multiple protocols in parallel using process pool"""
        
        if not self.process_pool:
            raise RuntimeError("Process pool not initialized")
        
        start_time = time.time()
        self.task_queue_depth = len(protocol_tasks)
        
        # Submit tasks to process pool
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in protocol_tasks:
            future = loop.run_in_executor(
                self.process_pool,
                execute_protocol_worker,
                task,
                context.copy()  # Each process gets its own context copy
            )
            futures.append(future)
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Protocol task {i} failed: {result}")
                    processed_results.append({
                        "error": str(result),
                        "task": protocol_tasks[i]
                    })
                else:
                    processed_results.append(result)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._update_process_metrics(execution_time)
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Parallel execution failed: {e}")
            raise
        finally:
            self.task_queue_depth = 0
    
    def _update_process_metrics(self, execution_time_ms: float):
        """Update process pool metrics"""
        try:
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                if child.pid not in self.process_metrics:
                    self.process_metrics[child.pid] = ProcessMetrics(
                        process_id=child.pid,
                        cpu_percent=0.0,
                        memory_mb=0.0,
                        execution_time_ms=0.0,
                        queue_depth=0
                    )
                
                # Update metrics
                metrics = self.process_metrics[child.pid]
                metrics.cpu_percent = child.cpu_percent()
                metrics.memory_mb = child.memory_info().rss / 1024 / 1024
                metrics.execution_time_ms = execution_time_ms
                metrics.queue_depth = self.task_queue_depth
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get comprehensive process pool metrics"""
        return {
            "max_workers": self.max_workers,
            "active_workers": len(self.process_metrics),
            "queue_depth": self.task_queue_depth,
            "process_metrics": list(self.process_metrics.values()),
            "total_memory_mb": sum(m.memory_mb for m in self.process_metrics.values()),
            "avg_cpu_percent": sum(m.cpu_percent for m in self.process_metrics.values()) / max(1, len(self.process_metrics))
        }
    
    def shutdown(self):
        """Gracefully shutdown the process pool"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
            logging.info("Process pool shut down")

def execute_protocol_worker(task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function that executes a protocol in a separate process"""
    try:
        # Import required modules in worker process
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        from mcp_server.protocol_manager.protocol_manager import ProtocolManager
        from mcp_server.resource_manager.resource_manager import ResourceManager
        
        # Initialize components in worker process
        protocol_manager = ProtocolManager()
        resource_manager = ResourceManager()
        
        # Get protocol
        protocol_name = task.get("protocol_name")
        if not protocol_name:
            raise ValueError("Protocol name not specified in task")
        
        protocol = protocol_manager.get_protocol(protocol_name)
        if not protocol:
            raise ValueError(f"Protocol '{protocol_name}' not found")
        
        # Execute protocol with resource monitoring
        start_time = time.time()
        
        with resource_manager.profile(protocol_name) as metrics:
            # Execute the protocol
            if hasattr(protocol, 'execute'):
                result = protocol.execute(context)
            else:
                raise ValueError(f"Protocol '{protocol_name}' has no execute method")
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "protocol_name": protocol_name,
            "result": result,
            "execution_time_ms": execution_time,
            "resource_usage": metrics,
            "process_id": os.getpid()
        }
        
    except Exception as e:
        return {
            "protocol_name": task.get("protocol_name", "unknown"),
            "error": str(e),
            "process_id": os.getpid()
        }
```

### **3.2 Enhanced Orchestration Engine**

**Updated orchestration engine with multiprocessing:**
```python
# File: code/mcp_server/orchestration_engine/enhanced_orchestrator.py
import asyncio
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from .process_orchestrator import ProcessPoolOrchestrator
from mcp_server.caching.cache_manager import UnifiedCacheManager
from mcp_server.rust_modules.similarity_wrapper import RustSimilarityEngine

class EnhancedOrchestrationEngine:
    """Enhanced orchestration engine with multiprocessing and caching"""
    
    def __init__(self):
        self.process_orchestrator = ProcessPoolOrchestrator()
        self.cache_manager = UnifiedCacheManager()
        self.rust_similarity = RustSimilarityEngine()
        
        # Performance tracking
        self.execution_history = []
        
    async def execute_workflow(
        self,
        workflow_def: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow with intelligent parallel processing"""
        
        # Batch memory pull with caching
        session_id = context.get('session_id')
        if session_id:
            cache_key = f"session_{session_id}_memories"
            batch_memory = self.cache_manager.get(cache_key, 'session_data')
            
            if batch_memory is None:
                # Load from database and cache
                from mcp_server.memory_manager.memory_manager import MemoryManager
                memory_manager = MemoryManager()
                batch_memory = memory_manager.get_all_memories(session_id)
                self.cache_manager.put(cache_key, batch_memory, 'session_data', ttl_seconds=1800)
            
            context['batch_memory'] = batch_memory
        
        # Execute workflow steps
        return await self._execute_steps_optimized(workflow_def, context)
    
    async def _execute_steps_optimized(
        self,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow steps with optimization"""
        
        for item in steps:
            if "step" in item:
                # Single protocol execution
                protocol_name = item["step"]
                result = await self._execute_single_protocol(protocol_name, context)
                context[protocol_name] = result
                
            elif "parallel" in item:
                # Parallel protocol execution
                parallel_steps = item.get("parallel", [])
                
                # Prepare tasks for parallel execution
                tasks = [
                    {
                        "protocol_name": step["step"],
                        "context_key": step["step"]
                    }
                    for step in parallel_steps
                ]
                
                # Execute in parallel using process pool
                results = await self.process_orchestrator.execute_protocol_parallel(
                    tasks, context
                )
                
                # Update context with results
                for i, result in enumerate(results):
                    protocol_name = parallel_steps[i]["step"]
                    context[protocol_name] = result
                    
            elif "loop" in item:
                # Loop execution with optimization
                loop_count = item["loop"]
                loop_steps = item.get("steps", [])
                
                for i in range(loop_count):
                    context = await self._execute_steps_optimized(loop_steps, context)
                    if "error" in context:
                        break
                    
                    # Handle revisor feedback loop
                    if "RevisorProtocol" in context:
                        revised_text = context.get("RevisorProtocol", {}).get("result", {}).get("revised_draft_text")
                        if revised_text:
                            if "DrafterProtocol" not in context:
                                context["DrafterProtocol"] = {}
                            context["DrafterProtocol"]["draft_text"] = revised_text
        
        return context
    
    async def _execute_single_protocol(
        self,
        protocol_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single protocol with caching and optimization"""
        
        # Check cache first
        cache_key = self._generate_cache_key(protocol_name, context)
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            logging.info(f"Cache hit for protocol {protocol_name}")
            return cached_result
        
        # Execute protocol
        task = {"protocol_name": protocol_name}
        results = await self.process_orchestrator.execute_protocol_parallel([task], context)
        
        if results and not isinstance(results[0], dict) or "error" not in results[0]:
            # Cache successful result
            self.cache_manager.put(cache_key, results[0], ttl_seconds=3600)
        
        return results[0] if results else {"error": "No result returned"}
    
    def _generate_cache_key(self, protocol_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for protocol execution"""
        # Create deterministic cache key based on protocol and relevant context
        import hashlib
        import json
        
        # Extract relevant context for caching
        cache_context = {
            "protocol": protocol_name,
            "user_input": context.get("user_input", ""),
            "session_id": context.get("session_id", ""),
        }
        
        # Add protocol-specific context
        if protocol_name == "IdeatorProtocol":
            cache_context["research_depth"] = context.get("research_depth", 1)
        elif protocol_name == "CriticProtocol":
            cache_context["draft_text"] = context.get("DrafterProtocol", {}).get("draft_text", "")[:100]  # First 100 chars
        
        # Generate hash
        context_str = json.dumps(cache_context, sort_keys=True)
        cache_hash = hashlib.md5(context_str.encode()).hexdigest()
        
        return f"protocol_{protocol_name}_{cache_hash}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "process_pool": self.process_orchestrator.get_pool_metrics(),
            "cache": self.cache_manager.get_all_metrics(),
            "rust_modules": {
                "similarity_available": self.rust_similarity.rust_available
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        self.process_orchestrator.shutdown()
        self.cache_manager.clear_all()
```

---

## Phase 4: GPU Integration and Batching (Week 10-11)

### **Objective:** Implement intelligent GPU acceleration with CPU fallbacks

### **4.1 GPU Batch Manager**

**Intelligent GPU batching with fixed memory budget:**
```python
# File: code/mcp_server/gpu/gpu_batch_manager.py
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import threading
from queue import Queue, Empty
import psutil

@dataclass
class GPUConfig:
    device: str = "cuda"
    max_memory_mb: int = 2048  # Fixed GPU memory budget
    batch_size_threshold: int = 32  # Minimum batch size for GPU
    max_batch_size: int = 1024  # Maximum batch size
    cpu_fallback_threshold: int = 16  # Use CPU for smaller batches
    workspace_size_mb: int = 512  # Preallocated workspace

class GPUBatchManager:
    """Manages GPU operations with batching and CPU fallbacks"""
    
    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.device = None
        self.gpu_available = False
        self.workspace_allocated = False
        self.workspace_tensors = {}
        
        # Batch queuing
        self.batch_queue = Queue()
        self.processing_thread = None
        self.shutdown_flag = threading.Event()
        
        # Performance tracking
        self.gpu_utilization_history = []
        self.batch_size_history = []
        
        self._initialize_gpu()
        self._allocate_workspace()
        self._start_batch_processor()
    
    def _initialize_gpu(self):
        """Initialize GPU with error handling"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device(self.config.device)
                torch.cuda.set_device(self.device)
                
                # Set memory fraction to respect budget
                memory_fraction = self.config.max_memory_mb / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
                memory_fraction = min(memory_fraction, 0.9)  # Cap at 90%
                
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                self.gpu_available = True
                logging.info(f"GPU initialized: {torch.cuda.get_device_name()}")
                logging.info(f"GPU memory budget: {self.config.max_memory_mb}MB")
            else:
                logging.warning("CUDA not available, using CPU fallback")
                self.device = torch.device("cpu")
                
        except Exception as e:
            logging.error(f"GPU initialization failed: {e}")
            self.device = torch.device("cpu")
            self.gpu_available = False
    
    def _allocate_workspace(self):
        """Preallocate GPU workspace to avoid fragmentation"""
        if not self.gpu_available:
            return
        
        try:
            workspace_bytes = self.config.workspace_size_mb * 1024 * 1024
            
            # Allocate common workspace tensors
            self.workspace_tensors = {
                'embedding_workspace': torch.empty(
                    (self.config.max_batch_size, 384), 
                    dtype=torch.float32, 
                    device=self.device
                ),
                'similarity_workspace': torch.empty(
                    (self.config.max_batch_size, self.config.max_batch_size),
                    dtype=torch.float32,
                    device=self.device
                ),
                'temp_workspace': torch.empty(
                    workspace_bytes // 4,  # float32 = 4 bytes
                    dtype=torch.float32,
                    device=self.device
                )
            }
            
            self.workspace_allocated = True
            logging.info("GPU workspace allocated successfully")
            
        except Exception as e:
            logging.error(f"GPU workspace allocation failed: {e}")
            self.workspace_allocated = False
    
    def _start_batch_processor(self):
        """Start background batch processing thread"""
        self.processing_thread = threading.Thread(
            target=self._batch_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def _batch_processing_loop(self):
        """Background loop for processing batched GPU operations"""
        while not self.shutdown_flag.is_set():
            try:
                # Wait for batch requests
                batch_request = self.batch_queue.get(timeout=1.0)
                if batch_request is None:  # Shutdown signal
                    break
                
                # Process the batch
                self._process_gpu_batch(batch_request)
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
    
    def should_use_gpu(self, batch_size: int, operation_type: str) -> bool:
        """Determine if GPU should be used for this operation"""
        if not self.gpu_available:
            return False
        
        # Size-based decision
        if batch_size < self.config.cpu_fallback_threshold:
            return False
        
        if batch_size < self.config.batch_size_threshold:
            return False
        
        # Operation-type based decision
        gpu_beneficial_ops = {
            'embedding_generation',
            'similarity_matrix',
            'large_vector_ops'
        }
        
        if operation_type not in gpu_beneficial_ops:
            return False
        
        # GPU utilization check
        if self.gpu_available:
            try:
                gpu_util = torch.cuda.utilization()
                if gpu_util > 90:  # GPU too busy
                    return False
            except:
                pass
        
        return True
    
    def batch_similarity_calculation(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray,
        operation_type: str = "similarity_matrix"
    ) -> np.ndarray:
        """Batch similarity calculation with GPU/CPU routing"""
        
        batch_size = len(candidate_vectors)
        
        if self.should_use_gpu(batch_size, operation_type):
            return self._gpu_similarity_calculation(query_vectors, candidate_vectors)
        else:
            return self._cpu_similarity_calculation(query_vectors, candidate_vectors)
    
    def _gpu_similarity_calculation(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated similarity calculation"""
        try:
            start_time = time.time()
            
            # Convert to GPU tensors
            query_tensor = torch.from_numpy(query_vectors).to(self.device, dtype=torch.float32)
            candidate_tensor = torch.from_numpy(candidate_vectors).to(self.device, dtype=torch.float32)
            
            # Normalize vectors
            query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=-1)
            candidate_norm = torch.nn.functional.normalize(candidate_tensor, p=2, dim=-1)
            
            # Compute similarities
            if query_vectors.ndim == 1:
                # Single query vs multiple candidates
                similarities = torch.mm(candidate_norm, query_norm.unsqueeze(-1)).squeeze()
            else:
                # Multiple queries vs multiple candidates
                similarities = torch.mm(candidate_norm, query_norm.t())
            
            # Convert back to numpy
            result = similarities.cpu().numpy()
            
            # Track performance
            execution_time = (time.time() - start_time) * 1000
            self._track_gpu_usage(len(candidate_vectors), execution_time)
            
            return result
            
        except Exception as e:
            logging.warning(f"GPU similarity calculation failed, falling back to CPU: {e}")
            return self._cpu_similarity_calculation(query_vectors, candidate_vectors)
    
    def _cpu_similarity_calculation(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray
    ) -> np.ndarray:
        """CPU fallback similarity calculation"""
        
        # Try Rust acceleration first
        try:
            from mcp_server.rust_modules.similarity_wrapper import RustSimilarityEngine
            rust_engine = RustSimilarityEngine()
            
            if query_vectors.ndim == 1:
                return rust_engine.cosine_similarity_batch(query_vectors, candidate_vectors)
            else:
                # Multiple queries - use similarity matrix
                return rust_engine.similarity_matrix(np.vstack([query_vectors, candidate_vectors]))
                
        except ImportError:
            logging.warning("Rust similarity not available, using numpy")
        
        # Pure numpy fallback
        query_norm = query_vectors / np.linalg.norm(query_vectors, axis=-1, keepdims=True)
        candidate_norm = candidate_vectors / np.linalg.norm(candidate_vectors, axis=-1, keepdims=True)
        
        if query_vectors.ndim == 1:
            return np.dot(candidate_norm, query_norm)
        else:
            return np.dot(candidate_norm, query_norm.T)
    
    def batch_embedding_generation(
        self,
        texts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> np.ndarray:
        """Batch embedding generation with GPU acceleration"""
        
        batch_size = len(texts)
        
        if self.should_use_gpu(batch_size, "embedding_generation"):
            return self._gpu_embedding_generation(texts, model_name)
        else:
            return self._cpu_embedding_generation(texts, model_name)
    
    def _gpu_embedding_generation(
        self,
        texts: List[str],
        model_name: str
    ) -> np.ndarray:
        """GPU-accelerated embedding generation"""
        try:
            # Use sentence-transformers with GPU
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(model_name, device=self.device)
            embeddings = model.encode(
                texts,
                batch_size=min(self.config.max_batch_size, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logging.warning(f"GPU embedding generation failed, falling back to CPU: {e}")
            return self._cpu_embedding_generation(texts, model_name)
    
    def _cpu_embedding_generation(
        self,
        texts: List[str],
        model_name: str
    ) -> np.ndarray:
        """CPU fallback embedding generation"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(model_name, device='cpu')
            embeddings = model.encode(
                texts,
                batch_size=32,  # Smaller batch size for CPU
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except ImportError:
            # Fallback to mock embeddings
            logging.warning("sentence-transformers not available, using mock embeddings")
            from mcp_server.database.vector_search import VectorSimilarityEngine
            
            engine = VectorSimilarityEngine()
            embeddings = []
            for text in texts:
                embedding = engine._generate_mock_embedding(text)
                embeddings.append(embedding)
            
            return np.array(embeddings)
    
    def _track_gpu_usage(self, batch_size: int, execution_time_ms: float):
        """Track GPU usage metrics"""
        try:
            if self.gpu_available:
                gpu_util = torch.cuda.utilization()
                memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
                self.gpu_utilization_history.append({
                    'timestamp': time.time(),
                    'batch_size': batch_size,
                    'execution_time_ms': execution_time_ms,
                    'gpu_utilization': gpu_util,
                    'memory_used_mb': memory_used
                })
                
                # Keep only recent history
                if len(self.gpu_utilization_history) > 1000:
                    self.gpu_utilization_history = self.gpu_utilization_history[-500:]
                    
        except Exception as e:
            logging.debug(f"GPU metrics tracking failed: {e}")
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics"""
        metrics = {
            'gpu_available': self.gpu_available,
            'device': str(self.device),
            'workspace_allocated': self.workspace_allocated,
            'config': {
                'max_memory_mb': self.config.max_memory_mb,
                'batch_size_threshold': self.config.batch_size_threshold,
                'cpu_fallback_threshold': self.config.cpu_fallback_threshold
            }
        }
        
        if self.gpu_available:
            try:
                metrics.update({
                    'gpu_name': torch.cuda.get_device_name(),
                    'total_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
                    'allocated_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'cached_memory_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                    'current_utilization': torch.cuda.utilization()
                })
            except Exception as e:
                logging.debug(f"Failed to get GPU metrics: {e}")
        
        # Recent performance statistics
        if self.gpu_utilization_history:
            recent_history = self.gpu_utilization_history[-100:]  # Last 100 operations
            metrics['performance'] = {
                'avg_batch_size': np.mean([h['batch_size'] for h in recent_history]),
                'avg_execution_time_ms': np.mean([h['execution_time_ms'] for h in recent_history]),
                'avg_gpu_utilization': np.mean([h['gpu_utilization'] for h in recent_history]),
                'operations_count': len(recent_history)
            }
        
        return metrics
    
    def shutdown(self):
        """Gracefully shutdown GPU manager"""
        self.shutdown_flag.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.batch_queue.put(None)  # Shutdown signal
            self.processing_thread.join(timeout=5.0)
        
        # Clear GPU memory
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared")
            except Exception as e:
                logging.error(f"Failed to clear GPU memory: {e}")
```

### **4.2 Integration with Vector Search Engine**

**GPU-accelerated vector search integration:**
```python
# File: code/mcp_server/database/gpu_vector_search.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from mcp_server.database.vector_search import VectorSimilarityEngine, VectorSearchResult
from mcp_server.gpu.gpu_batch_manager import GPUBatchManager, GPUConfig

class GPUVectorSimilarityEngine(VectorSimilarityEngine):
    """GPU-accelerated vector similarity engine"""
    
    def __init__(self, gpu_config: GPUConfig = None):
        super().__init__()
        self.gpu_manager = GPUBatchManager(gpu_config)
        
    async def similarity_search_batch(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray,
        k: int = 10
    ) -> List[List[VectorSearchResult]]:
        """Batch similarity search with GPU acceleration"""
        
        # Use GPU batch manager for similarity calculation
        similarity_matrix = self.gpu_manager.batch_similarity_calculation(
            query_vectors,
            candidate_vectors,
            operation_type="similarity_matrix"
        )
        
        results = []
        for i, query_similarities in enumerate(similarity_matrix):
            # Get top-k results for this query
            top_k_indices = np.argsort(query_similarities)[-k:][::-1]
            top_k_scores = query_similarities[top_k_indices]
            
            query_results = []
            for idx, score in zip(top_k_indices, top_k_scores):
                if score > self.similarity_threshold:
                    # Create result object (simplified for example)
                    result = VectorSearchResult(
                        memory_id=idx,
                        content=f"Content {idx}",  # Would be actual content
                        entity_name="",
                        similarity_score=float(score),
                        embedding_vector=candidate_vectors[idx],
                        metadata={},
                        emotional_salience=0.5,
                        session_id="",
                        timestamp=None
                    )
                    query_results.append(result)
            
            results.append(query_results)
        
        return results
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "mock"
    ) -> np.ndarray:
        """Generate embeddings in batch with GPU acceleration"""
        
        if model == "mock":
            # Use existing mock embedding generation
            embeddings = []
            for text in texts:
                embedding = self._generate_mock_embedding(text)
                embeddings.append(embedding)
            return np.array(embeddings)
        else:
            # Use GPU batch manager for neural embeddings
            return self.gpu_manager.batch_embedding_generation(texts, model)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = super().get_performance_metrics() if hasattr(super(), 'get_performance_metrics') else {}
        gpu_metrics = self.gpu_manager.get_gpu_metrics()
        
        return {
            **base_metrics,
            'gpu': gpu_metrics,
            'cache_size': len(self.embedding_cache),
            'vector_cache_size': len(self.memory_vectors)
        }
    
    def shutdown(self):
        """Shutdown GPU resources"""
        self.gpu_manager.shutdown()
```

---

## Phase 5: Retrieval and Routing Optimization (Week 12)

### **Objective:** Implement local-first retrieval with intelligent routing

### **5.1 Local-First RAG Manager**

**Optimized RAG manager with local prioritization:**
```python
# File: code/mcp_server/rag_manager/local_first_rag.py
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json

from mcp_server.rag_manager.rag_manager import RAGManager
from mcp_server.caching.cache_manager import UnifiedCacheManager
from mcp_server.database.gpu_vector_search import GPUVectorSimilarityEngine
from mcp_server.rust_modules.similarity_wrapper import RustSimilarityEngine

@dataclass
class CorpusMetadata:
    name: str
    language: str
    domain: str
    size: int
    last_updated: float
    hit_rate: float
    avg_response_time_ms: float

class LocalFirstRAGManager(RAGManager):
    """RAG manager with local-first retrieval and intelligent routing"""
    
    def __init__(self):
        super().__init__()
        self.cache_manager = UnifiedCacheManager()
        self.gpu_vector_engine = GPUVectorSimilarityEngine()
        self.rust_similarity = RustSimilarityEngine()
        
        # Local corpus management
        self.local_corpora: Dict[str, CorpusMetadata] = {}
        self.corpus_vectors: Dict[str, np.ndarray] = {}
        self.corpus_content: Dict[str, List[str]] = {}
        
        # Routing intelligence
        self.domain_router = DomainRouter()
        self.hit_rate_tracker = HitRateTracker()
        
        # Prewarming
        self.prewarmed_docs: Set[str] = set()
        
        self._initialize_local_corpora()
        self._prewarm_frequent_docs()
    
    def _initialize_local_corpora(self):
        """Initialize local document corpora"""
        corpus_dir = Path("./data/corpora")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing corpora metadata
        metadata_file = corpus_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                corpus_data = json.load(f)
                
            for name, data in corpus_data.items():
                self.local_corpora[name] = CorpusMetadata(**data)
        
        # Initialize default corpora
        default_corpora = [
            ("programming", "en", "technology"),
            ("science", "en", "research"),
            ("business", "en", "enterprise"),
            ("general", "en", "general")
        ]
        
        for name, lang, domain in default_corpora:
            if name not in self.local_corpora:
                self.local_corpora[name] = CorpusMetadata(
                    name=name,
                    language=lang,
                    domain=domain,
                    size=0,
                    last_updated=time.time(),
                    hit_rate=0.0,
                    avg_response_time_ms=0.0
                )
    
    def _prewarm_frequent_docs(self):
        """Prewarm frequently accessed documents"""
        # Load prewarming configuration
        prewarm_config = {
            "programming": [
                "python_best_practices.md",
                "rust_performance_guide.md",
                "async_programming_patterns.md"
            ],
            "science": [
                "ai_research_overview.md",
                "machine_learning_fundamentals.md"
            ]
        }
        
        for corpus_name, doc_names in prewarm_config.items():
            for doc_name in doc_names:
                doc_path = Path(f"./data/corpora/{corpus_name}/{doc_name}")
                if doc_path.exists():
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Cache the document
                        cache_key = f"prewarm_{corpus_name}_{doc_name}"
                        self.cache_manager.put(cache_key, content, 'long_term')
                        self.prewarmed_docs.add(cache_key)
                        
                    except Exception as e:
                        logging.warning(f"Failed to prewarm {doc_path}: {e}")
    
    async def perform_research(
        self,
        query: str,
        latency_info: Dict[str, Any],
        num_sources: int = 3
    ) -> str:
        """Perform research with local-first strategy"""
        
        start_time = time.time()
        
        # Step 1: Try local corpus first
        local_results = await self._search_local_corpora(query, num_sources)
        
        if local_results and len(local_results) >= num_sources:
            # Sufficient local results
            self.hit_rate_tracker.record_hit("local", time.time() - start_time)
            return self._format_research_results(local_results, "local")
        
        # Step 2: Check cache for similar queries
        cached_result = await self._check_cached_research(query)
        if cached_result:
            self.hit_rate_tracker.record_hit("cache", time.time() - start_time)
            return cached_result
        
        # Step 3: Selective external research
        remaining_sources = max(0, num_sources - len(local_results))
        external_results = await self._selective_external_research(
            query, remaining_sources, latency_info
        )
        
        # Combine results
        all_results = local_results + external_results
        final_result = self._format_research_results(all_results, "hybrid")
        
        # Cache the result
        cache_key = f"research_{hash(query)}"
        self.cache_manager.put(cache_key, final_result, 'session_data', ttl_seconds=3600)
        
        self.hit_rate_tracker.record_miss("external", time.time() - start_time)
        return final_result
    
    async def _search_local_corpora(
        self,
        query: str,
        num_sources: int
    ) -> List[Dict[str, Any]]:
        """Search local document corpora"""
        
        # Route query to appropriate corpora
        relevant_corpora = self.domain_router.route_query(query, self.local_corpora)
        
        if not relevant_corpora:
            return []
        
        results = []
        
        for corpus_name in relevant_corpora:
            if corpus_name not in self.corpus_vectors:
                continue
            
            try:
                # Generate query embedding
                query_embedding = await self.gpu_vector_engine.generate_embeddings_batch(
                    [query], model="mock"
                )
                
                # Search corpus vectors
                corpus_vectors = self.corpus_vectors[corpus_name]
                similarities = self.gpu_vector_engine.gpu_manager.batch_similarity_calculation(
                    query_embedding[0],
                    corpus_vectors,
                    operation_type="similarity_matrix"
                )
                
                # Get top results
                top_indices = np.argsort(similarities)[-num_sources:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.7:  # Similarity threshold
                        content = self.corpus_content[corpus_name][idx]
                        results.append({
                            "content": content,
                            "source": f"local:{corpus_name}",
                            "similarity": float(similarities[idx]),
                            "corpus": corpus_name
                        })
                
            except Exception as e:
                logging.error(f"Local corpus search failed for {corpus_name}: {e}")
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:num_sources]
    
    async def _check_cached_research(self, query: str) -> Optional[str]:
        """Check for cached research results"""
        
        # Try exact match first
        cache_key = f"research_{hash(query)}"
        cached = self.cache_manager.get(cache_key, 'session_data')
        if cached:
            return cached
        
        # Try semantic similarity with cached queries
        try:
            query_embedding = await self.gpu_vector_engine.generate_embeddings_batch(
                [query], model="mock"
            )
            
            # Get recent cached queries (simplified - would need proper implementation)
            # This is a placeholder for semantic cache lookup
            
        except Exception as e:
            logging.debug(f"Semantic cache lookup failed: {e}")
        
        return None
    
    async def _selective_external_research(
        self,
        query: str,
        num_sources: int,
        latency_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform selective external research with budget awareness"""
        
        if num_sources <= 0:
            return []
        
        # Check remaining latency budget
        elapsed_time = time.time() - latency_info.get("start_time", time.time())
        remaining_budget = latency_info.get("budget_ms", 10000) - (elapsed_time * 1000)
        
        if remaining_budget < 2000:  # Need at least 2 seconds for external research
            logging.info("Insufficient latency budget for external research")
            return []
        
        # Use parent class method with reduced scope
        try:
            async with self._sem:  # Respect concurrency limits
                external_result = await super().perform_research(
                    query, latency_info, num_sources=min(num_sources, 2)
                )
                
                return [{
                    "content": external_result,
                    "source": "external:web",
                    "similarity": 0.8,  # Assumed relevance
                    "corpus": "web"
                }]
                
        except Exception as e:
            logging.error(f"External research failed: {e}")
            return []
    
    def _format_research_results(
        self,
        results: List[Dict[str, Any]],
        source_type: str
    ) -> str:
        """Format research results into coherent response"""
        
        if not results:
            return "No relevant information found."
        
        formatted_parts = []
        
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
            content = result["content"][:500]  # Truncate long content
            source = result["source"]
            similarity = result.get("similarity", 0.0)
            
            formatted_parts.append(
                f"Source {i} ({source}, relevance: {similarity:.2f}):\n{content}\n"
            )
        
        research_summary = "\n".join(formatted_parts)
        
        # Add source attribution
        source_summary = f"\nResearch completed using {source_type} sources. "
        source_summary += f"Found {len(results)} relevant sources."
        
        return research_summary + source_summary

class DomainRouter:
    """Routes queries to appropriate document corpora"""
    
    def __init__(self):
        self.domain_keywords = {
            "programming": [
                "code", "python", "rust", "javascript", "programming", "software",
                "algorithm", "function", "class", "variable", "api", "framework"
            ],
            "science": [
                "research", "study", "analysis", "experiment", "data", "theory",
                "hypothesis", "methodology", "results", "conclusion", "scientific"
            ],
            "business": [
                "business", "market", "strategy", "revenue", "profit", "customer",
                "sales", "marketing", "management", "enterprise", "corporate"
            ],
            "general": []  # Fallback for everything else
        }
    
    def route_query(
        self,
        query: str,
        available_corpora: Dict[str, CorpusMetadata]
    ) -> List[str]:
        """Route query to most relevant corpora"""
        
        query_lower = query.lower()
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain, keywords in self.domain_keywords.items():
            if domain not in available_corpora:
                continue
            
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            # Boost score based on corpus hit rate
            corpus_meta = available_corpora[domain]
            score *= (1 + corpus_meta.hit_rate)
            
            domain_scores[domain] = score
        
        # Sort by score and return top domains
        sorted_domains = sorted(
            domain_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return domains with non-zero scores, or general as fallback
        relevant_domains = [domain for domain, score in sorted_domains if score > 0]
        
        if not relevant_domains and "general" in available_corpora:
            relevant_domains = ["general"]
        
        return relevant_domains[:3]  # Limit to top 3 domains

class HitRateTracker:
    """Tracks hit rates and performance metrics for different retrieval methods"""
    
    def __init__(self):
        self.metrics = {
            "local": {"hits": 0, "total_time": 0},
            "cache": {"hits": 0, "total_time": 0},
            "external": {"hits": 0, "total_time": 0}
        }
    
    def record_hit(self, source: str, response_time: float):
        """Record a cache/local hit"""
        if source in self.metrics:
            self.metrics[source]["hits"] += 1
            self.metrics[source]["total_time"] += response_time
    
    def record_miss(self, source: str, response_time: float):
        """Record a cache/local miss"""
        if source in self.metrics:
            self.metrics[source]["total_time"] += response_time
    
    def get_hit_rates(self) -> Dict[str, float]:
        """Get hit rates for each source"""
        hit_rates = {}
        
        total_requests = sum(
            self.metrics[source]["hits"] for source in self.metrics
        )
        
        if total_requests == 0:
            return {source: 0.0 for source in self.metrics}
        
        for source in self.metrics:
            hit_rates[source] = self.metrics[source]["hits"] / total_requests
        
        return hit_rates
    
    def get_avg_response_times(self) -> Dict[str, float]:
        """Get average response times for each source"""
        avg_times = {}
        
        for source, data in self.metrics.items():
            if data["hits"] > 0:
                avg_times[source] = data["total_time"] / data["hits"]
            else:
                avg_times[source] = 0.0
        
        return avg_times
```

---

## Phase 6: Governance Runtime Optimization (Week 13)

### **Objective:** Implement progressive reasoning and lazy evaluation for governance protocols

### **6.1 Progressive Governance Engine**

**Optimized governance with progressive reasoning:**
```python
# File: code/mcp_server/cognitive_governance_engine/progressive_governance.py
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from mcp_server.cognitive_governance_engine.cognitive_governance_engine import CognitiveGovernanceEngine
from mcp_server.protocols.governance.five_laws_validator.five_laws_validator import FiveLawsValidator
from mcp_server.caching.cache_manager import UnifiedCacheManager

class GovernanceLevel(Enum):
    COARSE = "coarse"      # Fast, basic checks
    MEDIUM = "medium"      # Moderate depth checks
    DEEP = "deep"         # Comprehensive analysis

@dataclass
class GovernanceResult:
    level: GovernanceLevel
    passed: bool
    confidence: float
    violations: List[str]
    execution_time_ms: float
    should_escalate: bool

class ProgressiveGovernanceEngine(CognitiveGovernanceEngine):
    """Governance engine with progressive reasoning and lazy evaluation"""
    
    def __init__(self):
        super().__init__()
        self.cache_manager = UnifiedCacheManager()
        self.five_laws_validator = FiveLawsValidator()
        
        # Progressive thresholds
        self.salience_thresholds = {
            GovernanceLevel.COARSE: 0.3,   # Low salience = coarse check
            GovernanceLevel.MEDIUM: 0.6,   # Medium salience = medium check
            GovernanceLevel.DEEP: 0.8      # High salience = deep check
        }
        
        # Performance tracking
        self.governance_metrics = {
            level: {"count": 0, "total_time": 0, "violations": 0}
            for level in GovernanceLevel
        }
    
    async def progressive_governance_check(
        self,
        content: str,
        context: Dict[str, Any],
        emotional_salience: float = 0.5
    ) -> GovernanceResult:
        """Perform progressive governance check based on salience"""
        
        # Determine governance level based on salience
        governance_level = self._determine_governance_level(emotional_salience, context)
        
        # Check cache first
        cache_key = self._generate_governance_cache_key(content, governance_level)
        cached_result = self.cache_manager.get(cache_key, 'vector_similarity')
        
        if cached_result:
            logging.debug(f"Governance cache hit for level {governance_level.value}")
            return cached_result
        
        # Perform governance check at appropriate level
        start_time = time.time()
        
        try:
            if governance_level == GovernanceLevel.COARSE:
                result = await self._coarse_governance_check(content, context)
            elif governance_level == GovernanceLevel.MEDIUM:
                result = await self._medium_governance_check(content, context)
            else:  # DEEP
                result = await self._deep_governance_check(content, context)
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            # Cache the result
            cache_ttl = 3600 if governance_level == GovernanceLevel.DEEP else 1800
            self.cache_manager.put(cache_key, result, 'vector_similarity', ttl_seconds=cache_ttl)
            
            # Update metrics
            self._update_governance_metrics(governance_level, execution_time, not result.passed)
            
            return result
            
        except Exception as e:
            logging.error(f"Governance check failed at level {governance_level.value}: {e}")
            return GovernanceResult(
                level=governance_level,
                passed=False,
                confidence=0.0,
                violations=[f"Governance check error: {str(e)}"],
                execution_time_ms=(time.time() - start_time) * 1000,
                should_escalate=True
            )
    
    def _determine_governance_level(
        self,
        emotional_salience: float,
        context: Dict[str, Any]
    ) -> GovernanceLevel:
        """Determine appropriate governance level"""
        
        # Base level on emotional salience
        if emotional_salience >= self.salience_thresholds[GovernanceLevel.DEEP]:
            base_level = GovernanceLevel.DEEP
        elif emotional_salience >= self.salience_thresholds[GovernanceLevel.MEDIUM]:
            base_level = GovernanceLevel.MEDIUM
        else:
            base_level = GovernanceLevel.COARSE
        
        # Escalate based on context factors
        escalation_factors = [
            context.get("is_public_facing", False),
            context.get("involves_sensitive_data", False),
            context.get("high_stakes_decision", False),
            context.get("user_explicitly_requested_verification", False)
        ]
        
        escalation_count = sum(escalation_factors)
        
        if escalation_count >= 2:
            # Escalate by one level
            if base_level == GovernanceLevel.COARSE:
                return GovernanceLevel.MEDIUM
            elif base_level == GovernanceLevel.MEDIUM:
                return GovernanceLevel.DEEP
        
        return base_level
    
    async def _coarse_governance_check(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> GovernanceResult:
        """Fast, basic governance checks"""
        
        violations = []
        
        # Basic content checks
        if len(content.strip()) == 0:
            violations.append("Empty content")
        
        # Simple keyword-based checks
        harmful_keywords = [
            "illegal", "dangerous", "harmful", "malicious",
            "exploit", "hack", "breach", "unauthorized"
        ]
        
        content_lower = content.lower()
        for keyword in harmful_keywords:
            if keyword in content_lower:
                violations.append(f"Potentially harmful keyword: {keyword}")
        
        # Basic Five Laws check (simplified)
        basic_five_laws_result = self._basic_five_laws_check(content)
        if not basic_five_laws_result["passed"]:
            violations.extend(basic_five_laws_result["violations"])
        
        passed = len(violations) == 0
        confidence = 0.7 if passed else 0.3  # Lower confidence for coarse checks
        
        return GovernanceResult(
            level=GovernanceLevel.COARSE,
            passed=passed,
            confidence=confidence,
            violations=violations,
            execution_time_ms=0,  # Will be set by caller
            should_escalate=len(violations) > 0  # Escalate if violations found
        )
    
    async def _medium_governance_check(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> GovernanceResult:
        """Moderate depth governance checks"""
        
        # Start with coarse check
        coarse_result = await self._coarse_governance_check(content, context)
        violations = coarse_result.violations.copy()
        
        # Additional medium-level checks
        
        # 1. Structural analysis
        structural_issues = self._analyze_content_structure(content)
        violations.extend(structural_issues)
        
        # 2. Partial Five Laws validation
        try:
            partial_five_laws = await self._partial_five_laws_check(content, context)
            if not partial_five_laws["passed"]:
                violations.extend(partial_five_laws["violations"])
        except Exception as e:
            violations.append(f"Five Laws partial check failed: {e}")
        
        # 3. Context consistency check
        consistency_issues = self._check_context_consistency(content, context)
        violations.extend(consistency_issues)
        
        passed = len(violations) == 0
        confidence = 0.85 if passed else 0.5
        
        return GovernanceResult(
            level=GovernanceLevel.MEDIUM,
            passed=passed,
            confidence=confidence,
            violations=violations,
            execution_time_ms=0,
            should_escalate=len(violations) > 2  # Escalate if multiple violations
        )
    
    async def _deep_governance_check(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> GovernanceResult:
        """Comprehensive governance analysis"""
        
        # Start with medium check
        medium_result = await self._medium_governance_check(content, context)
        violations = medium_result.violations.copy()
        
        # Deep analysis
        
        # 1. Full Five Laws validation
        try:
            full_five_laws = await self.five_laws_validator.validate_all_laws(content, context)
            if not full_five_laws["overall_compliance"]:
                for law_name, law_result in full_five_laws.items():
                    if isinstance(law_result, dict) and not law_result.get("compliant", True):
                        violations.append(f"Five Laws violation - {law_name}: {law_result.get('reason', 'Unknown')}")
        except Exception as e:
            violations.append(f"Five Laws full validation failed: {e}")
        
        # 2. Semantic analysis
        semantic_issues = await self._semantic_analysis(content, context)
        violations.extend(semantic_issues)
        
        # 3. Cross-reference validation
        cross_ref_issues = await self._cross_reference_validation(content, context)
        violations.extend(cross_ref_issues)
        
        # 4. Ethical implications analysis
        ethical_issues = await self._ethical_analysis(content, context)
        violations.extend(ethical_issues)
        
        passed = len(violations) == 0
        confidence = 0.95 if passed else 0.7
        
        return GovernanceResult(
            level=GovernanceLevel.DEEP,
            passed=passed,
            confidence=confidence,
            violations=violations,
            execution_time_ms=0,
            should_escalate=False  # Deep check is final
        )
    
    def _basic_five_laws_check(self, content: str) -> Dict[str, Any]:
        """Basic Five Laws check using simple heuristics"""
        violations = []
        
        # Law 1: Architectural Intelligence - Check for coherent structure
        if len(content.split('.')) < 2:  # Very basic coherence check
            violations.append("Law 1: Content lacks coherent structure")
        
        # Law 2: Cognitive Transparency - Check for clarity
        if len(content.split()) > 500 and content.count(',') < 5:  # Basic readability
            violations.append("Law 2: Content may lack clarity")
        
        # Law 3: Ethical Reasoning - Basic harmful content check
        harmful_patterns = ["harm", "damage", "destroy", "attack"]
        if any(pattern in content.lower() for pattern in harmful_patterns):
            violations.append("Law 3: Content may contain harmful elements")
        
        # Law 4: Energy Stewardship - Check for efficiency
        if len(content) > 2000:  # Basic length check
            violations.append("Law 4: Content may be inefficiently verbose")
        
        # Law 5: Adaptive Governance - Check for adaptability indicators
        rigid_patterns = ["always", "never", "must", "cannot"]
        rigid_count = sum(1 for pattern in rigid_patterns if pattern in content.lower())
        if rigid_count > 3:
            violations.append("Law 5: Content may be too rigid")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    async def _partial_five_laws_check(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Partial Five Laws check with moderate depth"""
        try:
            # Check most critical laws only
            critical_laws = ["law1", "law3"]  # Architectural Intelligence and Ethical Reasoning
            
            violations = []
            for law in critical_laws:
                law_result = await getattr(self.five_laws_validator, f"validate_{law}")(content, context)
                if not law_result.get("compliant", True):
                    violations.append(f"{law}: {law_result.get('reason', 'Violation detected')}")
            
            return {
                "passed": len(violations) == 0,
                "violations": violations
            }
        except Exception as e:
            return {
                "passed": False,
                "violations": [f"Partial Five Laws check error: {e}"]
            }
    
    def _analyze_content_structure(self, content: str) -> List[str]:
        """Analyze content structure for issues"""
        issues = []
        
        # Check for basic structure
        sentences = content.split('.')
        if len(sentences) < 2:
            issues.append("Content lacks sentence structure")
        
        # Check for paragraph structure
        paragraphs = content.split('\n\n')
        if len(content) > 500 and len(paragraphs) < 2:
            issues.append("Long content lacks paragraph structure")
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 50:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.1:  # More than 10% repetition
                issues.append("Content contains excessive repetition")
        
        return issues
    
    def _check_context_consistency(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Check consistency between content and context"""
        issues = []
        
        # Check if content matches expected domain
        expected_domain = context.get("domain", "")
        if expected_domain:
            domain_keywords = {
                "technical": ["code", "system", "algorithm", "implementation"],
                "business": ["market", "strategy", "revenue", "customer"],
                "academic": ["research", "study", "analysis", "methodology"]
            }
            
            if expected_domain in domain_keywords:
                keywords = domain_keywords[expected_domain]
                content_lower = content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
                
                if keyword_matches == 0:
                    issues.append(f"Content doesn't match expected domain: {expected_domain}")
        
        # Check content length vs expected scope
        expected_length = context.get("expected_length", "")
        content_length = len(content.split())
        
        if expected_length == "brief" and content_length > 200:
            issues.append("Content is too long for brief response")
        elif expected_length == "detailed" and content_length < 100:
            issues.append("Content is too short for detailed response")
        
        return issues
    
    async def _semantic_analysis(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Perform semantic analysis of content"""
        issues = []
        
        try:
            # Use GPU vector engine for semantic analysis
            from mcp_server.database.gpu_vector_search import GPUVectorSimilarityEngine
            
            gpu_engine = GPUVectorSimilarityEngine()
            
            # Generate embedding for content
            content_embedding = await gpu_engine.generate_embeddings_batch([content])
            
            # Compare with known problematic content patterns (simplified)
            # In practice, this would compare against a database of problematic content
            
            # For now, just check semantic coherence by analyzing sentence embeddings
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if len(sentences) > 3:
                sentence_embeddings = await gpu_engine.generate_embeddings_batch(sentences)
                
                # Check for semantic consistency between sentences
                similarities = []
                for i in range(len(sentence_embeddings) - 1):
                    sim = gpu_engine.gpu_manager.batch_similarity_calculation(
                        sentence_embeddings[i:i+1],
                        sentence_embeddings[i+1:i+2],
                        operation_type="similarity_matrix"
                    )[0]
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                if avg_similarity < 0.3:  # Very low semantic coherence
                    issues.append("Content lacks semantic coherence between sentences")
        
        except Exception as e:
            logging.debug(f"Semantic analysis failed: {e}")
            # Don't add this as a violation since it's a tool failure
        
        return issues
    
    async def _cross_reference_validation(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Cross-reference content with known facts"""
        issues = []
        
        try:
            # This would integrate with fact-checking databases
            # For now, implement basic consistency checks
            
            # Check for contradictory statements within the content
            contradictory_pairs = [
                ("always", "never"),
                ("all", "none"),
                ("impossible", "possible"),
                ("true", "false")
            ]
            
            content_lower = content.lower()
            for word1, word2 in contradictory_pairs:
                if word1 in content_lower and word2 in content_lower:
                    # Check if they're in close proximity (potential contradiction)
                    word1_pos = content_lower.find(word1)
                    word2_pos = content_lower.find(word2)
                    if abs(word1_pos - word2_pos) < 200:  # Within 200 characters
                        issues.append(f"Potential contradiction detected: '{word1}' and '{word2}' in close proximity")
        
        except Exception as e:
            logging.debug(f"Cross-reference validation failed: {e}")
        
        return issues
    
    async def _ethical_analysis(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Analyze content for ethical implications"""
        issues = []
        
        # Check for bias indicators
        bias_indicators = [
            "all [group] are",
            "every [group] is",
            "[group] always",
            "[group] never"
        ]
        
        # Check for potentially harmful advice
        harmful_advice_patterns = [
            "you should ignore",
            "don't tell anyone",
            "keep this secret",
            "bypass security",
            "avoid detection"
        ]
        
        content_lower = content.lower()
        
        for pattern in harmful_advice_patterns:
            if pattern in content_lower:
                issues.append(f"Potentially harmful advice detected: pattern '{pattern}'")
        
        # Check for privacy violations
        privacy_patterns = [
            "personal information",
            "private data",
            "confidential",
            "social security",
            "credit card"
        ]
        
        for pattern in privacy_patterns:
            if pattern in content_lower and "share" in content_lower:
                issues.append(f"Potential privacy violation: content mentions sharing {pattern}")
        
        return issues
    
    def _generate_governance_cache_key(
        self,
        content: str,
        level: GovernanceLevel
    ) -> str:
        """Generate cache key for governance results"""
        import hashlib
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"governance_{level.value}_{content_hash}"
    
    def _update_governance_metrics(
        self,
        level: GovernanceLevel,
        execution_time: float,
        had_violations: bool
    ):
        """Update governance performance metrics"""
        metrics = self.governance_metrics[level]
        metrics["count"] += 1
        metrics["total_time"] += execution_time
        if had_violations:
            metrics["violations"] += 1
    
    def get_governance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive governance metrics"""
        metrics_summary = {}
        
        for level, data in self.governance_metrics.items():
            if data["count"] > 0:
                metrics_summary[level.value] = {
                    "total_checks": data["count"],
                    "avg_execution_time_ms": data["total_time"] / data["count"],
                    "violation_rate": data["violations"] / data["count"],
                    "total_violations": data["violations"]
                }
            else:
                metrics_summary[level.value] = {
                    "total_checks": 0,
                    "avg_execution_time_ms": 0,
                    "violation_rate": 0,
                    "total_violations": 0
                }
        
        # Overall metrics
        total_checks = sum(data["count"] for data in self.governance_metrics.values())
        total_violations = sum(data["violations"] for data in self.governance_metrics.values())
        
        metrics_summary["overall"] = {
            "total_checks": total_checks,
            "overall_violation_rate": total_violations / total_checks if total_checks > 0 else 0,
            "cache_hit_rate": 0.0  # Would need to implement cache hit tracking
        }
        
        return metrics_summary
```

---

## Phase 7: Observability and Profiling (Week 14)

### **Objective:** Implement comprehensive monitoring and profiling infrastructure

### **7.1 Comprehensive Metrics System**

**Unified metrics collection and reporting:**
```python
# File: code/mcp_server/observability/metrics_system.py
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import psutil
import asyncio
from contextlib import contextmanager

@dataclass
class MetricPoint:
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class TimingMetric:
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Active timings
        self.active_timings: Dict[str, TimingMetric] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background collection
        self.collection_thread = None
        self.shutdown_flag = threading.Event()
        self._start_background_collection()
    
    def _start_background_collection(self):
        """Start background system metrics collection"""
        self.collection_thread = threading.Thread(
            target=self._background_collection_loop,
            daemon=True
        )
        self.collection_thread.start()
    
    def _background_collection_loop(self):
        """Background loop for collecting system metrics"""
        while not self.shutdown_flag.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Sleep for collection interval
                time.sleep(5.0)  # Collect every 5 seconds
                
            except Exception as e:
                logging.error(f"Background metrics collection failed: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            current_time = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_gauge("system.cpu.percent", cpu_percent, {"host": "local"})
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system.memory.percent", memory.percent, {"host": "local"})
            self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024, {"host": "local"})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge("system.disk.percent", disk.percent, {"host": "local"})
            
            # Process metrics
            process = psutil.Process()
            self.record_gauge("process.cpu.percent", process.cpu_percent(), {"process": "simone_mcp"})
            self.record_gauge("process.memory.mb", process.memory_info().rss / 1024 / 1024, {"process": "simone_mcp"})
            self.record_gauge("process.threads", process.num_threads(), {"process": "simone_mcp"})
            
            # GPU metrics (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    
                    self.record_gauge("gpu.memory.allocated_mb", gpu_memory, {"device": "cuda:0"})
                    self.record_gauge("gpu.utilization.percent", gpu_util, {"device": "cuda:0"})
            except ImportError:
                pass
            
        except Exception as e:
            logging.debug(f"System metrics collection failed: {e}")
    
    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            key = self._generate_metric_key(name, tags or {})
            self.counters[key] += value
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        with self.lock:
            key = self._generate_metric_key(name, tags or {})
            self.gauges[key] = value
            
            # Also store as time series
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.metrics[key].append(metric_point)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.lock:
            key = self._generate_metric_key(name, tags or {})
            self.histograms[key].append(value)
            
            # Keep histogram size manageable
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-500:]  # Keep last 500 values
    
    @contextmanager
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        timing_id = f"{name}_{time.time()}_{threading.get_ident()}"
        
        timing = TimingMetric(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        self.active_timings[timing_id] = timing
        
        try:
            yield timing
        finally:
            timing.end_time = time.time()
            
            # Record the timing
            self.record_histogram(f"{name}.duration_ms", timing.duration_ms, tags)
            
            # Clean up
            self.active_timings.pop(timing_id, None)
    
    def start_timing(self, name: str, tags: Dict[str, str] = None) -> str:
        """Start a timing operation and return timing ID"""
        timing_id = f"{name}_{time.time()}_{threading.get_ident()}"
        
        timing = TimingMetric(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self.lock:
            self.active_timings[timing_id] = timing
        
        return timing_id
    
    def end_timing(self, timing_id: str):
        """End a timing operation"""
        with self.lock:
            timing = self.active_timings.pop(timing_id, None)
            
        if timing:
            timing.end_time = time.time()
            self.record_histogram(f"{timing.name}.duration_ms", timing.duration_ms, timing.tags)
    
    def _generate_metric_key(self, name: str, tags: Dict[str, str]) -> str:
        """Generate a unique key for a metric with tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metric_summary(self, name: str, tags: Dict[str, str] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        key = self._generate_metric_key(name, tags or {})
        
        summary = {
            "name": name,
            "tags": tags or {},
            "counter": self.counters.get(key, 0),
            "gauge": self.gauges.get(key),
            "histogram": None
        }
        
        # Histogram statistics
        if key in self.histograms and self.histograms[key]:
            values = sorted(self.histograms[key])
            count = len(values)
            
            summary["histogram"] = {
                "count": count,
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / count,
                "p50": values[int(count * 0.5)],
                "p95": values[int(count * 0.95)],
                "p99": values[int(count * 0.99)] if count > 100 else values[-1]
            }
        
        return summary
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with summaries"""
        with self.lock:
            all_metric_names = set()
            all_metric_names.update(self.counters.keys())
            all_metric_names.update(self.gauges.keys())
            all_metric_names.update(self.histograms.keys())
            
            metrics_summary = {}
            
            for key in all_metric_names:
                # Parse metric name and tags from key
                if '[' in key:
                    name = key.split('[')[0]
                    tag_str = key.split('[')[1].rstrip(']')
                    tags = dict(tag.split('=') for tag in tag_str.split(',') if '=' in tag)
                else:
                    name = key
                    tags = {}
                
                metrics_summary[key] = self.get_metric_summary(name, tags)
            
            return {
                "metrics": metrics_summary,
                "active_timings": len(self.active_timings),
                "collection_timestamp": time.time()
            }


