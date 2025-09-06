# SIM-ONE Benchmarking Suite Documentation

This directory contains the comprehensive benchmarking and validation infrastructure for the SIM-ONE Framework close-to-metal optimization project.

## 🎯 **Philosophy**

These benchmarks focus on **architectural intelligence** rather than raw computational performance, validating the core SIM-ONE principle: **"Intelligence is in the GOVERNANCE, not the LLM."**

## 📁 **Directory Structure**

```
benchmarks/
├── README.md                           # This documentation
├── __init__.py                         # Package initialization
├── benchmark_suite.py                  # Core benchmarking framework
├── cognitive_governance_benchmarks.py  # Governance-focused benchmarks
├── philosophical_validation.py         # Philosophy validation tests
├── run_baselines.py                   # Complete baseline measurement
├── setup_development.py               # Development environment setup
├── slo_targets.py                     # Performance targets & quality gates
├── target_operations.py               # Benchmark target functions
├── philosophical_answers.md           # Analysis of philosophical questions
├── results/                           # Benchmark results (JSON files)
└── docs/                              # Additional documentation
```

## 🚀 **Quick Start**

### **1. Run Complete Baseline Benchmarks**
```bash
# Run comprehensive architectural intelligence benchmarks
make benchmark

# Or directly:
PYTHONPATH=. python benchmarks/run_baselines.py
```

### **2. Run Quick Benchmark Subset**
```bash
# Run faster subset for development
make benchmark-fast

# Or directly:
PYTHONPATH=. python benchmarks/cognitive_governance_benchmarks.py
```

### **3. Run Philosophical Validation**
```bash
# Validate core SIM-ONE philosophy
PYTHONPATH=. python benchmarks/philosophical_validation.py
```

### **4. Verify Phase Completion**
```bash
# Check if Phase 0 is complete and ready for next phase
make check-phase0
```

## 📊 **Benchmark Categories**

### **1. Cognitive Governance Benchmarks**
*Location: `cognitive_governance_benchmarks.py`*

**Purpose**: Measure intelligence that emerges from governance coordination

**Key Metrics**:
- Protocol coordination efficiency
- Five Laws of Cognitive Governance compliance
- Multi-agent workflow performance
- MVLM stateless execution performance
- Truth validation and error prevention

**Usage**:
```python
from benchmarks.cognitive_governance_benchmarks import CognitiveGovernanceBenchmark

benchmark = CognitiveGovernanceBenchmark()
results = benchmark.run_comprehensive_governance_benchmark()
```

### **2. Philosophical Validation Tests**
*Location: `philosophical_validation.py`*

**Purpose**: Validate that intelligence comes from governance, not LLM scale

**Key Tests**:
- Intelligence attribution analysis (governance vs MVLM)
- Emergent capability detection
- System degradation without governance
- Quality vs performance measurement

**Usage**:
```python
from benchmarks.philosophical_validation import PhilosophicalValidator

validator = PhilosophicalValidator()
results = validator.run_comprehensive_philosophical_validation()
```

### **3. Performance Baseline Benchmarks**
*Location: `run_baselines.py`*

**Purpose**: Establish comprehensive performance baselines for optimization

**Key Measurements**:
- Architectural intelligence scores
- Governance efficiency metrics
- MVLM execution performance
- System-wide performance baselines
- Five Laws compliance validation

**Usage**:
```python
from benchmarks.run_baselines import run_architectural_intelligence_baseline

results = run_architectural_intelligence_baseline()
```

### **4. Target Operations Benchmarks**
*Location: `target_operations.py`*

**Purpose**: Benchmark specific operations that will be optimized

**Operations Covered**:
- Vector similarity calculations
- Memory consolidation processes
- Five-agent workflows
- Embedding generation
- Cache operations
- Database queries

**Usage**:
```python
from benchmarks.target_operations import benchmark_vector_similarity

results = benchmark_vector_similarity(vector_count=1000)
```

## 🎯 **Service Level Objectives (SLOs)**

### **Accessing SLO Targets**
```python
from benchmarks.slo_targets import SLO_TARGETS, get_slo_target

# Get target for specific metric
target = get_slo_target('protocol_coordination_p95')
# Returns: 100 (100ms target)

# Check Five Laws compliance
from benchmarks.slo_targets import calculate_five_laws_score, check_quality_gates

compliance = calculate_five_laws_score(your_results)
gates = check_quality_gates(your_results)
```

### **Key SLO Categories**
- **Cognitive Governance**: Protocol coordination, Five Laws compliance
- **MVLM Execution**: Stateless instruction execution performance
- **Multi-Agent Workflows**: Coordinated intelligence benchmarks
- **Memory System**: Recursive memory and semantic search
- **Energy Stewardship**: Architectural efficiency metrics
- **Deterministic Reliability**: Consistency and predictability

## 📈 **Understanding Benchmark Results**

### **Key Metrics Explained**

#### **Architectural Intelligence Score**
- **Range**: 0.0 - 2.0+
- **Meaning**: Intelligence multiplier through governance coordination
- **Target**: >1.0 (emergence through coordination)
- **Current Baseline**: 1.023

#### **Governance Efficiency**
- **Range**: 0.0 - 1.0
- **Meaning**: How efficiently protocols coordinate
- **Target**: >0.8
- **Current Baseline**: 0.89

#### **Intelligence Emergence Ratio**
- **Range**: 1.0+ 
- **Meaning**: Intelligence multiplier through coordination
- **Target**: >1.2
- **Current Baseline**: 1.23x

#### **Five Laws Compliance**
- **Range**: 0.0 - 1.0
- **Meaning**: Compliance with SIM-ONE foundational principles
- **Target**: >0.8
- **Current Baseline**: 0.963 (96.3%)

### **Performance Metrics**
- **P50**: 50th percentile (median) latency
- **P95**: 95th percentile latency (most operations complete within this time)
- **P99**: 99th percentile latency (nearly all operations complete within this time)

## 🧪 **Writing Custom Benchmarks**

### **Basic Benchmark Example**
```python
from benchmarks.benchmark_suite import SIMONEBenchmark

def my_custom_operation():
    # Your operation here
    import time
    time.sleep(0.01)  # 10ms operation
    return "result"

benchmark = SIMONEBenchmark()
result = benchmark.benchmark_operation(
    "my_custom_test",
    my_custom_operation,
    iterations=100
)

print(f"P95 latency: {result.p95_ms}ms")
```

### **Async Benchmark Example**
```python
async def my_async_operation():
    import asyncio
    await asyncio.sleep(0.01)
    return "async_result"

result = benchmark.benchmark_async_operation(
    "my_async_test", 
    my_async_operation,
    iterations=100
)
```

### **Governance Intelligence Benchmark Template**
```python
def benchmark_my_governance_feature():
    """Template for governance-focused benchmarks"""
    
    def test_with_governance():
        # Test with full governance enabled
        governance_active = True
        quality_score = 0.95  # High quality with governance
        processing_time = 0.02  # Slower but higher quality
        
        return {
            'quality': quality_score,
            'time': processing_time,
            'governance': governance_active
        }
    
    def test_without_governance():
        # Test without governance (MVLM only)
        governance_active = False
        quality_score = 0.40  # Lower quality without governance
        processing_time = 0.01  # Faster but lower quality
        
        return {
            'quality': quality_score,
            'time': processing_time,
            'governance': governance_active
        }
    
    # Benchmark both scenarios
    with_gov = benchmark.benchmark_operation("with_governance", test_with_governance)
    without_gov = benchmark.benchmark_operation("without_governance", test_without_governance)
    
    # Calculate intelligence attribution
    quality_improvement = test_with_governance()['quality'] / test_without_governance()['quality']
    speed_cost = test_with_governance()['time'] / test_without_governance()['time']
    
    return {
        'with_governance': with_gov,
        'without_governance': without_gov,
        'quality_improvement': quality_improvement,
        'speed_cost': speed_cost,
        'intelligence_ratio': quality_improvement / speed_cost  # Intelligence per computational cost
    }
```

## 📊 **Result Analysis**

### **Loading Results**
```python
import json
from pathlib import Path

# Load latest results
results_dir = Path("benchmarks/results")
latest_results = sorted(results_dir.glob("simone_baseline_*.json"))[-1]

with open(latest_results) as f:
    data = json.load(f)

print(f"Five Laws Compliance: {data['summary']['five_laws_compliance']:.1%}")
```

### **Comparing Results**
```python
baseline_results = benchmark.compare_with_baseline("simone_baseline_20250906_161045.json")

for metric, comparison in baseline_results.items():
    improvement = comparison['improvement_p95']
    print(f"{metric}: {improvement:+.1%} improvement")
```

## 🔧 **Development Workflow**

### **1. Before Making Changes**
```bash
# Establish baseline
make benchmark
cp benchmarks/results/simone_baseline_*.json benchmarks/results/baseline_before_changes.json
```

### **2. After Making Changes**
```bash
# Test changes
make benchmark-fast

# Compare with baseline
make benchmark
# Results will include comparison with previous baseline
```

### **3. Validate Philosophy Compliance**
```bash
# Ensure changes don't break SIM-ONE principles
python benchmarks/philosophical_validation.py

# Check Five Laws compliance
make check-phase0
```

## 📋 **Quality Gates**

All benchmarks must pass these quality gates before proceeding to the next phase:

### **Five Laws Compliance** (>80% each)
- Law 1 (Architectural Intelligence): Intelligence through coordination
- Law 2 (Cognitive Governance): Specialized protocol governance
- Law 3 (Truth Foundation): Grounded reasoning validation  
- Law 4 (Energy Stewardship): Efficiency through architecture
- Law 5 (Deterministic Reliability): Consistent behavior

### **Performance Gates**
- Architectural Intelligence Score: >0.8
- Governance Efficiency: >0.8
- Intelligence Emergence Ratio: >1.2
- Overall Five Laws Compliance: >0.8

### **Philosophy Gates**
- Intelligence attribution validation: >60% confidence
- Emergent capabilities evidence: >60% strength
- Governance criticality confirmed: TRUE

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/workspaces/SIM-ONE:$PYTHONPATH

# Or use make commands which set it automatically
make benchmark
```

#### **Missing Dependencies**
```bash
# Install development requirements
pip install -r requirements-dev.txt
```

#### **Benchmark Failures**
```python
# Check system resources
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")

# Run with fewer iterations for testing
result = benchmark.benchmark_operation("test", operation_func, iterations=10)
```

#### **Low Performance**
- Check for other processes consuming resources
- Ensure sufficient memory available
- Consider running benchmarks on dedicated hardware

### **Debugging Benchmarks**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run single iteration to debug
benchmark.benchmark_operation("debug_test", operation_func, iterations=1)
```

## 📚 **Additional Resources**

- **Phase 0 Complete Documentation**: `docs/optimization/PHASE0_COMPLETE.md`
- **Philosophical Analysis**: `benchmarks/philosophical_answers.md`
- **SLO Targets Reference**: `benchmarks/slo_targets.py`
- **Development Setup**: `benchmarks/setup_development.py`
- **Make Commands**: `Makefile` (run `make help` for full list)

## 🤝 **Contributing**

When adding new benchmarks:

1. **Follow the philosophy**: Focus on architectural intelligence, not raw performance
2. **Include governance comparison**: Test with/without governance where applicable
3. **Add to documentation**: Update this README with new benchmark descriptions
4. **Validate philosophy**: Ensure new benchmarks support SIM-ONE principles
5. **Update SLO targets**: Add appropriate targets for new metrics

## 📞 **Support**

For questions about benchmarks:
1. Check this documentation first
2. Review existing benchmark code for examples
3. Run `make help` for available commands
4. Check `benchmarks/philosophical_answers.md` for philosophy questions

---

**Remember**: These benchmarks measure **architectural intelligence**, not computational performance. The goal is to validate that intelligence emerges from governance coordination, not LLM scale.