# SIM-ONE Framework Development Makefile
# Close-to-Metal Optimization Development Tasks

.PHONY: help setup benchmark rust clean test

help: ## Show this help message
	@echo "SIM-ONE Framework Development Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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
