"""
Comprehensive benchmark framework for SIM-ONE Framework performance optimization.
Provides detailed performance metrics with P50/P95/P99 latencies and resource utilization.
"""

import time
import psutil
import statistics
import asyncio
import logging
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    operation: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    peak_memory_mb: float
    avg_cpu_percent: float
    samples: int
    timestamp: str
    errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SystemMetrics:
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    python_version: str
    platform: str

class SIMONEBenchmark:
    """High-performance benchmarking suite for SIM-ONE operations"""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results = {}
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.system_metrics = self._get_system_metrics()
        
    def _get_system_metrics(self) -> SystemMetrics:
        """Collect system information for benchmark context"""
        import platform
        memory = psutil.virtual_memory()
        
        return SystemMetrics(
            cpu_count=psutil.cpu_count(),
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            python_version=sys.version,
            platform=platform.platform()
        )
    
    def benchmark_operation(self, name: str, operation_func: Callable, 
                          iterations: int = 100, warmup_iterations: int = 10) -> BenchmarkResult:
        """
        Benchmark any operation with comprehensive metrics
        
        Args:
            name: Operation name for identification
            operation_func: Function to benchmark (can be sync or async)
            iterations: Number of measurement iterations
            warmup_iterations: Number of warmup iterations (not measured)
        """
        logger.info(f"Benchmarking '{name}' with {iterations} iterations...")
        
        # Determine if operation is async
        is_async = asyncio.iscoroutinefunction(operation_func)
        
        # Warmup phase
        logger.debug(f"Warming up '{name}' with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            try:
                if is_async:
                    asyncio.run(operation_func())
                else:
                    operation_func()
            except Exception as e:
                logger.warning(f"Warmup iteration failed for '{name}': {e}")
        
        # Measurement phase
        times = []
        memory_deltas = []
        cpu_measurements = []
        errors = 0
        
        for i in range(iterations):
            # Pre-measurement state
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Time the operation
            start_time = time.perf_counter()
            try:
                if is_async:
                    asyncio.run(operation_func())
                else:
                    result = operation_func()
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed for '{name}': {e}")
                errors += 1
                continue
            end_time = time.perf_counter()
            
            # Post-measurement state
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Record measurements
            times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_deltas.append(mem_after - mem_before)
            cpu_measurements.append((cpu_before + cpu_after) / 2)
            
            # Progress logging
            if (i + 1) % max(1, iterations // 10) == 0:
                logger.debug(f"Completed {i + 1}/{iterations} iterations for '{name}'")
        
        if not times:
            logger.error(f"No successful iterations for '{name}'")
            return BenchmarkResult(
                operation=name,
                p50_ms=0, p95_ms=0, p99_ms=0, mean_ms=0, std_ms=0,
                peak_memory_mb=0, avg_cpu_percent=0,
                samples=0, timestamp=datetime.now().isoformat(),
                errors=errors
            )
        
        # Calculate statistics
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)] if len(times) > 20 else times[-1]
        p99 = times[int(len(times) * 0.99)] if len(times) > 100 else times[-1]
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        
        result = BenchmarkResult(
            operation=name,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            mean_ms=mean_time,
            std_ms=std_time,
            peak_memory_mb=max(memory_deltas) if memory_deltas else 0,
            avg_cpu_percent=statistics.mean(cpu_measurements) if cpu_measurements else 0,
            samples=len(times),
            timestamp=datetime.now().isoformat(),
            errors=errors
        )
        
        self.results[name] = result
        logger.info(f"'{name}' benchmark complete: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")
        return result
    
    def benchmark_async_operation(self, name: str, async_operation_func: Callable,
                                iterations: int = 100, warmup_iterations: int = 10) -> BenchmarkResult:
        """Specialized benchmark for async operations with proper event loop handling"""
        
        async def run_benchmark():
            logger.info(f"Benchmarking async '{name}' with {iterations} iterations...")
            
            # Warmup phase
            for _ in range(warmup_iterations):
                try:
                    await async_operation_func()
                except Exception as e:
                    logger.warning(f"Async warmup iteration failed for '{name}': {e}")
            
            # Measurement phase
            times = []
            errors = 0
            
            for i in range(iterations):
                start_time = time.perf_counter()
                try:
                    await async_operation_func()
                except Exception as e:
                    logger.error(f"Async benchmark iteration {i} failed for '{name}': {e}")
                    errors += 1
                    continue
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if not times:
                return BenchmarkResult(
                    operation=name,
                    p50_ms=0, p95_ms=0, p99_ms=0, mean_ms=0, std_ms=0,
                    peak_memory_mb=0, avg_cpu_percent=0,
                    samples=0, timestamp=datetime.now().isoformat(),
                    errors=errors
                )
            
            # Calculate statistics
            times.sort()
            p50 = times[len(times) // 2]
            p95 = times[int(len(times) * 0.95)] if len(times) > 20 else times[-1]
            p99 = times[int(len(times) * 0.99)] if len(times) > 100 else times[-1]
            
            return BenchmarkResult(
                operation=name,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                mean_ms=statistics.mean(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                peak_memory_mb=0,  # Memory tracking complex in async
                avg_cpu_percent=0,  # CPU tracking complex in async
                samples=len(times),
                timestamp=datetime.now().isoformat(),
                errors=errors
            )
        
        result = asyncio.run(run_benchmark())
        self.results[name] = result
        return result
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        output = {
            "system_metrics": asdict(self.system_metrics),
            "benchmark_results": {name: result.to_dict() for name, result in self.results.items()},
            "summary": self._generate_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return str(filepath)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks"""
        if not self.results:
            return {}
        
        total_operations = len(self.results)
        failed_operations = sum(1 for r in self.results.values() if r.errors > 0)
        
        all_p50s = [r.p50_ms for r in self.results.values() if r.samples > 0]
        all_p95s = [r.p95_ms for r in self.results.values() if r.samples > 0]
        
        return {
            "total_operations": total_operations,
            "failed_operations": failed_operations,
            "success_rate": (total_operations - failed_operations) / total_operations if total_operations > 0 else 0,
            "overall_p50_range": [min(all_p50s), max(all_p50s)] if all_p50s else [0, 0],
            "overall_p95_range": [min(all_p95s), max(all_p95s)] if all_p95s else [0, 0],
            "total_samples": sum(r.samples for r in self.results.values())
        }
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Dict[str, float]]:
        """Compare current results with baseline measurements"""
        baseline_path = self.results_dir / baseline_file
        
        if not baseline_path.exists():
            logger.error(f"Baseline file not found: {baseline_path}")
            return {}
        
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        baseline_results = baseline_data.get("benchmark_results", {})
        comparisons = {}
        
        for name, current_result in self.results.items():
            if name in baseline_results:
                baseline_result = baseline_results[name]
                
                comparisons[name] = {
                    "p50_ratio": current_result.p50_ms / baseline_result["p50_ms"] if baseline_result["p50_ms"] > 0 else float('inf'),
                    "p95_ratio": current_result.p95_ms / baseline_result["p95_ms"] if baseline_result["p95_ms"] > 0 else float('inf'),
                    "p99_ratio": current_result.p99_ms / baseline_result["p99_ms"] if baseline_result["p99_ms"] > 0 else float('inf'),
                    "improvement_p50": 1 - (current_result.p50_ms / baseline_result["p50_ms"]) if baseline_result["p50_ms"] > 0 else 0,
                    "improvement_p95": 1 - (current_result.p95_ms / baseline_result["p95_ms"]) if baseline_result["p95_ms"] > 0 else 0,
                }
        
        return comparisons
    
    def print_results_table(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            print("No benchmark results to display")
            return
        
        # Table header
        print("\n" + "="*120)
        print(f"{'Operation':<40} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Mean (ms)':<12} {'Samples':<10} {'Errors':<8}")
        print("="*120)
        
        # Sort results by P95 time for easier analysis
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].p95_ms, reverse=True)
        
        for name, result in sorted_results:
            print(f"{name:<40} {result.p50_ms:<12.2f} {result.p95_ms:<12.2f} {result.p99_ms:<12.2f} "
                  f"{result.mean_ms:<12.2f} {result.samples:<10} {result.errors:<8}")
        
        print("="*120)
        
        # Summary
        summary = self._generate_summary()
        print(f"\nSummary: {summary['total_operations']} operations, "
              f"{summary['success_rate']*100:.1f}% success rate, "
              f"{summary['total_samples']} total samples")
        print()