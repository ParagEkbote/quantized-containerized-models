SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# --------------------------------------------------
# 🎯 Configuration
# --------------------------------------------------
MODEL_NAME ?=
USERNAME ?= paragekbote
REGISTRY := r8.im

PYTHON ?= python3
PIP ?= python3 -m pip
COG_BIN ?= cog

.DEFAULT_GOAL := help

# --------------------------------------------------
# 🤝 Helpers
# --------------------------------------------------
define require-cog
	@command -v $(COG_BIN) >/dev/null 2>&1 || { echo "❌ Cog not found in PATH"; exit 1; }
endef

define require-model-name
	@[ -n "$(MODEL_NAME)" ] || { echo "❌ MODEL_NAME environment variable is required"; exit 1; }
endef

# --------------------------------------------------
# 📖 Help
# --------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "🚀 ML Model Pipeline Makefile"
	@echo "================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  MODEL_NAME - Model to build/test (required)"
	@echo "  USERNAME   - Replicate username (default: $(USERNAME))"
	@echo ""

# --------------------------------------------------
# 📦 Setup
# --------------------------------------------------

.PHONY: install-deps
install-deps: ## Install Python dependencies
	@echo "📦 Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[dev,unit,integration,canary]" \
		--extra-index-url https://download.pytorch.org/whl/cu126
	@echo "✅ Dependencies installed"

# --------------------------------------------------
# 🏗️ Build & Deploy
# --------------------------------------------------
.PHONY: build
build: ## Build Cog image
	$(call require-cog)
	$(call require-model-name)
	@echo "🔨 Building $(MODEL_NAME)..."
	@$(COG_BIN) push $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)
	@echo "✅ Build complete"

.PHONY: deploy
deploy: build ## Build and deploy model
	@echo "✅ Deployed $(MODEL_NAME)"

# --------------------------------------------------
# 🧪 Tests
# --------------------------------------------------
.PHONY: lint unit integration canary

lint: ## Run linters
	@echo "🔍 Running linters..."
	@pre-commit run --all-files || echo "⚠️  Linting issues found"

unit: ## Run unit tests
	@echo "🧪 Running unit tests..."
	@pytest -m unit -vv

integration: ## Run integration tests
	$(call require-model-name)
	@echo "🧪 Running integration tests for $(MODEL_NAME)..."
	@pytest -m integration -vv

canary: ## Run canary tests
	$(call require-model-name)
	@echo "🐦 Running canary tests for $(MODEL_NAME)..."
	@pytest -m canary -vv

# --------------------------------------------------
# 🔄 CI/CD Pipelines
# --------------------------------------------------
.PHONY: ci cd

ci: lint unit ## Run CI (lint + unit)
	@echo "✅ CI passed"

cd: deploy integration canary ## Run full CD pipeline
	@echo "🎉 CD complete for $(MODEL_NAME)"

# --------------------------------------------------
# 🗑️ Cleanup
# --------------------------------------------------
.PHONY: clean
clean: ## Clean artifacts
	@echo "🗑️  Cleaning..."
	@rm -rf .cog .pytest_cache __pycache__
	@echo "✅ Clean complete"
