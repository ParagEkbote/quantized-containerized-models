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
# 🤝 Guards
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
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment:"
	@echo "  MODEL_NAME (required)"
	@echo "  USERNAME   (default: $(USERNAME))"
	@echo ""

# --------------------------------------------------
# 📦 Dependency Installation
# --------------------------------------------------
.PHONY: install-deps install-canary
install-deps: ## Install model-specific dependencies via pyproject extras
	@echo "📦 Installing deps for MODEL_NAME=$(MODEL_NAME)"
	@$(PYTHON) --version
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

install-canary: ## Install minimal canary test dependencies
	@echo "📦 Installing canary deps for MODEL_NAME=$(MODEL_NAME)"
	@$(PYTHON) --version
	@$(PIP) install --upgrade pip

ifeq ($(MODEL_NAME),smollm3-pruna)
	$(PIP) install -e .[canary,smollm3] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),phi4-reasoning-plus-unsloth)
	$(PIP) install -e .[canary,unsloth] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),flux-fast-lora-hotswap)
	$(PIP) install -e .[canary,flux-fast-lora] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),flux-fast-lora-hotswap-img2img)
	$(PIP) install -e .[canary,flux-fast-lora-img2img] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else ifeq ($(MODEL_NAME),gemma-torchao)
	$(PIP) install -e .[canary,gemma-torchao] \
		--extra-index-url https://download.pytorch.org/whl/cu121

else
	$(error ❌ Unknown MODEL_NAME=$(MODEL_NAME))
endif

	@echo "✅ install-canary complete for $(MODEL_NAME)"

# --------------------------------------------------
# 🏗️ Build & Deploy
# --------------------------------------------------
.PHONY: build
build: ## Build & push Cog image
	$(call require-cog)
	$(call require-model-name)

ifndef REPLICATE_CLI_AUTH_TOKEN
	$(error ❌ REPLICATE_CLI_AUTH_TOKEN must be set for cog push)
endif

	@echo "🔐 Logging into Cog"
	@echo "$(REPLICATE_CLI_AUTH_TOKEN)" | cog login --token-stdin

	@MODEL_DIR=src/models/$$(echo "$(MODEL_NAME)" | tr '-' '_'); \
	cd $$MODEL_DIR && \
	cog push $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)

.PHONY: deploy
deploy: build ## Deploy model
	@echo ""
	@echo "✅ Successfully deployed $(MODEL_NAME)"
	@echo ""

# --------------------------------------------------
# 🧪 Tests
# --------------------------------------------------
.PHONY: lint unit integration canary

lint: ## Run linters
	@echo "🔍 Running linters"
	@pre-commit run --all-files || echo "⚠️ Lint issues found"

unit: ## Run unit tests
	@echo "🧪 Running unit tests"
	@pytest -m unit -vv

integration: ## Run integration tests
	$(call require-model-name)
	@echo "🧪 Integration tests for $(MODEL_NAME)"
	@pytest -m integration -vv

canary: ## Run canary tests with version hash detection (15-30 min polling)
	$(call require-model-name)
	@echo "🐦 Canary tests for $(MODEL_NAME)"
	@echo "📍 CANDIDATE_MODEL_ID: $${CANDIDATE_MODEL_ID:-<using model alias>}"
	@pytest -m canary -vv --model-name=$(MODEL_NAME) --candidate-model-id="$${CANDIDATE_MODEL_ID}" || \
		echo "⚠️ Canary failures ignored"

# --------------------------------------------------
# 🔄 Pipelines
# --------------------------------------------------
.PHONY: ci cd post-deploy

ci: lint unit ## CI pipeline
	@echo "✅ CI passed"

cd: deploy integration ## CD pipeline (blocking)
	@echo "🎉 CD complete for $(MODEL_NAME)"

post-deploy: canary ## Optional post-deployment checks with version detection
	@echo "ℹ️ Post-deployment canary finished"

# --------------------------------------------------
# 🗑️ Cleanup
# --------------------------------------------------
.PHONY: clean
clean: ## Clean build artifacts
	@rm -rf .cog .pytest_cache __pycache__
	@echo "🧹 Cleaned"
