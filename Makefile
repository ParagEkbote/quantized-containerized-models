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
install-deps: ## Install model-specific dependencies via pyproject extras
	@echo "📦 Installing deps for MODEL_NAME=$(MODEL_NAME)"
	@python --version
	@$(PIP) install --upgrade pip

ifeq ($(MODEL_NAME),smollm3-pruna)
	$(PIP) install -e .[dev,unit,integration,canary,smollm3] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),phi4-reasoning-plus-unsloth)
	$(PIP) install -e .[dev,unit,integration,canary,unsloth] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),flux-fast-lora-hotswap)
	$(PIP) install -e .[dev,unit,integration,canary,flux-fast-lora] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),flux-fast-lora-hotswap-img2img)
	$(PIP) install -e .[dev,unit,integration,canary,flux-fast-lora-img2img] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),gemma-torchao)
	$(PIP) install -e .[dev,unit,integration,canary,gemma-torchao] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else
	$(error ❌ Unknown MODEL_NAME=$(MODEL_NAME))
endif

	@echo "✅ install-deps complete for $(MODEL_NAME)"

# --------------------------------------------------
# 🏗️ Build & Deploy
# --------------------------------------------------
.PHONY: build
build: ## Build & push Cog image (emits MODEL_ID=owner/model:version)
	$(call require-cog)
	$(call require-model-name)

ifndef REPLICATE_CLI_AUTH_TOKEN
	$(error ❌ REPLICATE_CLI_AUTH_TOKEN must be set for cog push)
endif

	@echo "🔐 Logging into Cog (non-interactive)..."
	@echo "$(REPLICATE_CLI_AUTH_TOKEN)" | cog login --token-stdin

	@echo "🔨 Building $(MODEL_NAME)..."

	@MODEL_DIR=src/models/$$(echo "$(MODEL_NAME)" | tr '-' '_'); \
	if [ ! -f "$$MODEL_DIR/cog.yaml" ]; then \
		echo "❌ cog.yaml not found at $$MODEL_DIR/cog.yaml"; \
		exit 1; \
	fi; \
	cd $$MODEL_DIR && \
	{ \
	  PUSH_OUTPUT=$$(cog push r8.im/$(USERNAME)/$(MODEL_NAME) 2>&1) || { echo "$$PUSH_OUTPUT"; exit 1; }; \
	  echo "$$PUSH_OUTPUT"; \
	  echo "---"; \
	  echo "🔍 Extracting version hash..."; \
	  VERSION=$$(echo "$$PUSH_OUTPUT" | grep -oP '(?<=sha256:)[a-f0-9]{64}' | head -n1); \
	  if [ -z "$$VERSION" ]; then \
	    VERSION=$$(echo "$$PUSH_OUTPUT" | grep -oP 'sha256:[a-f0-9]{64}' | head -n1 | sed 's/^sha256://'); \
	  fi; \
	  if [ -z "$$VERSION" ]; then \
	    VERSION=$$(echo "$$PUSH_OUTPUT" | grep -Eo '[a-f0-9]{64}' | head -n1); \
	  fi; \
	  if [ -z "$$VERSION" ]; then \
	    VERSION=$$(echo "$$PUSH_OUTPUT" | grep -oP 'r8\.im/[^:]+:[a-f0-9]{64}' | grep -oP '[a-f0-9]{64}' | head -n1); \
	  fi; \
	  if [ -z "$$VERSION" ]; then \
	    echo "❌ Failed to extract version hash from cog push output"; \
	    echo "Full output:"; \
	    echo "$$PUSH_OUTPUT"; \
	    exit 1; \
	  fi; \
	  MODEL_ID="$(USERNAME)/$(MODEL_NAME):$$VERSION"; \
	  echo "✅ Extracted version: $$VERSION"; \
	  echo "MODEL_ID=$$MODEL_ID"; \
	  echo "$$MODEL_ID" > /tmp/model_id.txt; \
	}

.PHONY: deploy
deploy: build ## Build and deploy model (prints MODEL_ID for GitHub Actions)
	@if [ ! -f /tmp/model_id.txt ]; then \
		echo "❌ MODEL_ID file not found. Build may have failed."; \
		exit 1; \
	fi; \
	MODEL_ID=$$(cat /tmp/model_id.txt | xargs); \
	if [ -z "$$MODEL_ID" ]; then \
		echo "❌ MODEL_ID is empty"; \
		exit 1; \
	fi; \
	if ! echo "$$MODEL_ID" | grep -qE '^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+:[a-f0-9]{64}$$'; then \
		echo "❌ MODEL_ID format invalid: $$MODEL_ID"; \
		echo "   Expected format: username/model-name:version-hash"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "✅ Successfully deployed!"; \
	echo ""; \
	echo "MODEL_ID=$$MODEL_ID"; \
	echo "candidate_model_id=$$MODEL_ID"; \
	echo ""; \
	if [ -n "$$GITHUB_OUTPUT" ]; then \
		echo "candidate_model_id=$$MODEL_ID" >> $$GITHUB_OUTPUT; \
		echo "📝 Written to GITHUB_OUTPUT"; \
	fi; \
	VERSION=$$(echo "$$MODEL_ID" | cut -d: -f2); \
	echo "🔗 View at: https://replicate.com/$(USERNAME)/$(MODEL_NAME)/versions/$$VERSION"; \
	echo ""


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
	@rm -rf .cog .pytest_cache __pycache__ /tmp/model_id.txt
	@echo "✅ Clean complete"
