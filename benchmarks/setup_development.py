"""
Setup development environment for SIM-ONE close-to-metal optimizations.
Prepares Rust toolchain and creates project structure for Phase 1-8 implementations.
"""

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rust_environment():
    """Setup Rust development environment for close-to-metal optimizations"""
    
    logger.info("Setting up Rust development environment...")
    
    # Check if Rust is installed
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        logger.info(f"Rust already installed: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.info("Rust not found. Please install Rust manually:")
        logger.info("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        return False
    
    # Install required Rust components
    rust_components = ['clippy', 'rustfmt']
    for component in rust_components:
        try:
            subprocess.run(['rustup', 'component', 'add', component], check=True)
            logger.info(f"✅ Installed Rust component: {component}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {component}: {e}")
    
    return True

def create_project_structure():
    """Create project structure for close-to-metal optimizations"""
    
    logger.info("Creating project structure for optimization phases...")
    
    directories = [
        # Rust extensions
        "code/rust_extensions",
        "code/rust_extensions/simone_similarity/src",
        "code/rust_extensions/simone_hash/src",
        "code/rust_extensions/simone_regex/src",
        "code/rust_extensions/simone_ast/src",
        
        # Caching system
        "code/mcp_server/caching",
        
        # Concurrency modules
        "code/mcp_server/concurrency",
        
        # GPU acceleration
        "code/mcp_server/gpu_acceleration",
        
        # Profiling and observability
        "profiling/results",
        "profiling/flame_graphs",
        
        # Documentation
        "docs/optimization",
        "docs/benchmarks",
        "docs/deployment"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def create_rust_workspace():
    """Create Rust workspace configuration for SIM-ONE extensions"""
    
    logger.info("Creating Rust workspace configuration...")
    
    # Main Cargo.toml for workspace
    cargo_workspace = """[workspace]
members = [
    "simone_similarity",
    "simone_hash", 
    "simone_regex",
    "simone_ast"
]

[workspace.dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
rayon = "1.7"
regex = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
"""
    
    workspace_file = Path("code/rust_extensions/Cargo.toml")
    with open(workspace_file, 'w') as f:
        f.write(cargo_workspace)
    logger.info(f"✅ Created Rust workspace: {workspace_file}")
    
    # Individual Cargo.toml files for each extension
    extensions = {
        "simone_similarity": "High-performance SIMD vector similarity operations",
        "simone_hash": "Fast content hashing and deduplication",
        "simone_regex": "Compiled regex pattern matching",
        "simone_ast": "Fast AST parsing and traversal"
    }
    
    for extension, description in extensions.items():
        cargo_toml = f"""[package]
name = "{extension}"
version = "0.1.0"
edition = "2021"
description = "{description}"

[lib]
name = "{extension}"
crate-type = ["cdylib"]

[dependencies]
pyo3.workspace = true
numpy.workspace = true
rayon.workspace = true

[dependencies.regex]
workspace = true
optional = true

[features]
default = []
regex = ["dep:regex"]
"""
        
        extension_file = Path(f"code/rust_extensions/{extension}/Cargo.toml")
        extension_file.parent.mkdir(parents=True, exist_ok=True)
        with open(extension_file, 'w') as f:
            f.write(cargo_toml)
        logger.info(f"✅ Created {extension} Cargo.toml")
        
        # Create basic lib.rs stub
        lib_rs = f"""//! {description}
//! 
//! Part of the SIM-ONE Framework close-to-metal optimizations.
//! This module provides {description.lower()} for the cognitive governance system.

use pyo3::prelude::*;

/// Python module for {extension}
#[pymodule]
fn {extension}(_py: Python, m: &PyModule) -> PyResult<()> {{
    // Module functions will be implemented in Phase 2
    Ok(())
}}
"""
        
        lib_file = Path(f"code/rust_extensions/{extension}/src/lib.rs")
        with open(lib_file, 'w') as f:
            f.write(lib_rs)
        logger.info(f"✅ Created {extension} lib.rs stub")

def create_phase_documentation():
    """Create documentation templates for each optimization phase"""
    
    logger.info("Creating phase documentation templates...")
    
    phases = {
        "phase1_caching": "Hierarchical Caching System",
        "phase2_rust_extensions": "Rust Extensions - Core Modules", 
        "phase3_concurrency": "Concurrency Model - Multiprocessing",
        "phase4_gpu": "GPU Integration and Batching",
        "phase5_retrieval": "Retrieval and Routing Optimization",
        "phase6_governance": "Governance Runtime Optimization",
        "phase7_observability": "Observability and Profiling",
        "phase8_integration": "Integration Testing and Deployment"
    }
    
    for phase, title in phases.items():
        doc_content = f"""# {title}

## Overview
Implementation documentation for {title} in the SIM-ONE Framework close-to-metal optimization plan.

## Objectives
- Preserve architectural intelligence while optimizing execution performance
- Maintain Five Laws of Cognitive Governance compliance
- Focus on coordination efficiency rather than raw computational power

## Key Metrics
- Intelligence emergence through governance
- Protocol coordination efficiency  
- MVLM execution performance (stateless CPU-like behavior)
- Energy stewardship through architectural design

## Implementation Status
- [ ] Planning complete
- [ ] Implementation started
- [ ] Benchmarks integrated
- [ ] Five Laws compliance verified
- [ ] Performance targets met
- [ ] Ready for next phase

## Results
*Results will be documented here during implementation*

## Next Steps
*Next steps will be identified during implementation*
"""
        
        doc_file = Path(f"docs/optimization/{phase}.md")
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        logger.info(f"✅ Created documentation: {doc_file}")

def create_requirements_files():
    """Create requirements files for development dependencies"""
    
    logger.info("Creating requirements files...")
    
    # Python development requirements
    dev_requirements = """# SIM-ONE Framework Development Requirements
# Phase 0: Baseline Infrastructure and Development Environment

# Benchmarking and profiling
psutil>=5.9.0
numpy>=1.24.0
pytest-benchmark>=4.0.0
py-spy>=0.3.14
scalene>=1.5.26
memory-profiler>=0.60.0

# Rust integration
maturin>=1.3.0
setuptools-rust>=1.7.0

# Data analysis
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Async support
asyncio-throttle>=1.0.2
aiofiles>=23.2.1

# Optional GPU acceleration (install manually if needed)
# cupy-cuda12x>=12.2.0
# torch>=2.1.0

# Testing and quality
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.2.0
"""
    
    with open("requirements-dev.txt", 'w') as f:
        f.write(dev_requirements)
    logger.info("✅ Created requirements-dev.txt")
    
    # Production requirements (minimal)
    prod_requirements = """# SIM-ONE Framework Production Requirements
# Core dependencies for deployed system

numpy>=1.24.0
psutil>=5.9.0

# Add production-specific requirements here
"""
    
    with open("requirements.txt", 'w') as f:
        f.write(prod_requirements)
    logger.info("✅ Created requirements.txt")

def create_makefile():
    """Create Makefile for common development tasks"""
    
    logger.info("Creating development Makefile...")
    
    makefile_content = """# SIM-ONE Framework Development Makefile
# Close-to-Metal Optimization Development Tasks

.PHONY: help setup benchmark rust clean test

help: ## Show this help message
	@echo "SIM-ONE Framework Development Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

setup: ## Setup development environment
	@echo "Setting up SIM-ONE development environment..."
	pip install -r requirements-dev.txt
	python benchmarks/setup_development.py

benchmark: ## Run baseline benchmarks
	@echo "Running SIM-ONE baseline benchmarks..."
	PYTHONPATH=. python benchmarks/run_baselines.py

benchmark-fast: ## Run quick benchmark subset
	@echo "Running quick benchmark subset..."
	PYTHONPATH=. python benchmarks/cognitive_governance_benchmarks.py

rust: ## Build Rust extensions (Phase 2+)
	@echo "Building Rust extensions..."
	cd code/rust_extensions && cargo build --release

rust-test: ## Test Rust extensions
	@echo "Testing Rust extensions..."
	cd code/rust_extensions && cargo test

rust-install: ## Install Rust extensions as Python modules
	@echo "Installing Rust extensions..."
	cd code/rust_extensions && maturin develop --release

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	cd code/rust_extensions && cargo clean

test: ## Run Python tests
	@echo "Running Python tests..."
	PYTHONPATH=. pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	PYTHONPATH=. pytest tests/ --cov=code/mcp_server --cov-report=html

lint: ## Run code linting
	@echo "Running code linting..."
	black --check code/ benchmarks/
	isort --check-only code/ benchmarks/
	flake8 code/ benchmarks/

format: ## Format code
	@echo "Formatting code..."
	black code/ benchmarks/
	isort code/ benchmarks/

profile: ## Profile application performance  
	@echo "Profiling application performance..."
	PYTHONPATH=. py-spy record -o profile.svg -- python benchmarks/run_baselines.py

phase1: ## Start Phase 1 implementation (Hierarchical Caching)
	@echo "Starting Phase 1: Hierarchical Caching System"
	@echo "Ensure Phase 0 baselines are established first"
	
phase2: ## Start Phase 2 implementation (Rust Extensions)
	@echo "Starting Phase 2: Rust Extensions - Core Modules"
	make rust

docs: ## Build documentation
	@echo "Building documentation..."
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	mkdocs serve

install: ## Install SIM-ONE framework
	@echo "Installing SIM-ONE framework..."
	pip install -e .

# Phase-specific targets
.PHONY: phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8

check-phase0: ## Verify Phase 0 completion
	@echo "Checking Phase 0 completion..."
	PYTHONPATH=. python -c "from benchmarks.run_baselines import run_architectural_intelligence_baseline; r=run_architectural_intelligence_baseline(); exit(0 if r['summary']['phase_0_complete'] else 1)"
"""
    
    with open("Makefile", 'w') as f:
        f.write(makefile_content)
    logger.info("✅ Created development Makefile")

def main():
    """Setup complete development environment for SIM-ONE optimizations"""
    
    print("="*80)
    print("   SIM-ONE FRAMEWORK: PHASE 0 DEVELOPMENT ENVIRONMENT SETUP")
    print("   Close-to-Metal Optimization Development Environment")
    print("="*80)
    
    # 1. Setup Rust environment
    rust_ready = setup_rust_environment()
    if not rust_ready:
        logger.warning("⚠️  Rust not available - Phases 2+ will require manual Rust installation")
    
    # 2. Create project structure
    create_project_structure()
    
    # 3. Create Rust workspace
    if rust_ready:
        create_rust_workspace()
    
    # 4. Create documentation templates
    create_phase_documentation()
    
    # 5. Create requirements files
    create_requirements_files()
    
    # 6. Create development Makefile
    create_makefile()
    
    print("\n" + "="*80)
    print("   ✅ PHASE 0 DEVELOPMENT ENVIRONMENT SETUP COMPLETE")
    print("="*80)
    print()
    print("🚀 NEXT STEPS:")
    print("   1. Install development dependencies: pip install -r requirements-dev.txt")
    print("   2. Run baseline benchmarks: make benchmark")
    print("   3. Verify Five Laws compliance: make check-phase0")
    print("   4. Get approval for Phase 1: Hierarchical Caching System")
    print()
    print("📋 AVAILABLE COMMANDS:")
    print("   make help          - Show all available commands")
    print("   make benchmark     - Run comprehensive baselines")
    print("   make benchmark-fast - Run quick benchmark subset")
    print("   make phase1        - Start Phase 1 when approved")
    print()
    print("🎯 PHILOSOPHY REMINDER:")
    print("   Intelligence is in the GOVERNANCE, not the LLM")
    print("   Focus on architectural coordination, not computational scale")
    print("="*80)

if __name__ == "__main__":
    main()