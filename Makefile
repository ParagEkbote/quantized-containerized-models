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
.PHONY: install-deps
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

# --------------------------------------------------
# 🏗️ Build & Deploy
# --------------------------------------------------
.PHONY: build
build: ## Build & push Cog image (best-effort MODEL_ID)
	$(call require-cog)
	$(call require-model-name)

ifndef REPLICATE_CLI_AUTH_TOKEN
	$(error ❌ REPLICATE_CLI_AUTH_TOKEN must be set for cog push)
endif

	@echo "🔐 Logging into Cog"
	@echo "$(REPLICATE_CLI_AUTH_TOKEN)" | cog login --token-stdin

	@MODEL_DIR=src/models/$$(echo "$(MODEL_NAME)" | tr '-' '_'); \
	cd $$MODEL_DIR && \
	{ \
	  PUSH_OUTPUT=$$(cog push $(REGISTRY)/$(USERNAME)/$(MODEL_NAME) 2>&1) || { echo "$$PUSH_OUTPUT"; exit 1; }; \
	  echo "$$PUSH_OUTPUT"; \
	  echo "---"; \
	  echo "🔍 Attempting to extract version hash (best-effort)"; \
	  VERSION=$$(echo "$$PUSH_OUTPUT" | grep -m1 -oP '(?<=sha256:)[a-f0-9]{64}' || true); \
	  if [ -n "$$VERSION" ]; then \
	    MODEL_ID="$(USERNAME)/$(MODEL_NAME):$$VERSION"; \
	    echo "ℹ️ Extracted version: $$VERSION"; \
	    echo "$$MODEL_ID" > /tmp/model_id.txt; \
	  else \
	    echo "ℹ️ Version not extracted (allowed)"; \
	  fi; \
	}

.PHONY: deploy
deploy: build ## Deploy model (MODEL_ID is advisory only)
	@MODEL_ID=""; \
	if [ -f /tmp/model_id.txt ]; then \
	  MODEL_ID=$$(cat /tmp/model_id.txt | xargs); \
	fi; \
	echo ""; \
	echo "✅ Successfully deployed"; \
	if [ -n "$$MODEL_ID" ]; then \
	  echo "ℹ️ Candidate model ID: $$MODEL_ID"; \
	  if [ -n "$$GITHUB_OUTPUT" ]; then \
	    echo "candidate_model_id=$$MODEL_ID" >> $$GITHUB_OUTPUT; \
	  fi; \
	  VERSION=$$(echo "$$MODEL_ID" | cut -d: -f2); \
	  echo "🔗 https://replicate.com/$(USERNAME)/$(MODEL_NAME)/versions/$$VERSION"; \
	else \
	  echo "ℹ️ No model ID emitted (allowed)"; \
	fi; \
	echo ""

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

canary: ## Run canary tests (non-blocking)
	$(call require-model-name)
	@echo "🐦 Canary tests for $(MODEL_NAME)"
	@pytest -m canary -vv || echo "⚠️ Canary failures ignored"

# --------------------------------------------------
# 🔄 Pipelines
# --------------------------------------------------
.PHONY: ci cd post-deploy

ci: lint unit ## CI pipeline
	@echo "✅ CI passed"

cd: deploy integration ## CD pipeline (blocking)
	@echo "🎉 CD complete for $(MODEL_NAME)"

post-deploy: canary ## Optional post-deployment checks
	@echo "ℹ️ Post-deployment checks finished"

# --------------------------------------------------
# 🗑️ Cleanup
# --------------------------------------------------
.PHONY: clean
clean: ## Clean build artifacts
	@rm -rf .cog .pytest_cache __pycache__ /tmp/model_id.txt
	@echo "🧹 Cleaned"
