SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# --------------------------------------------------
# Paths / Identity
# --------------------------------------------------
MODEL_DIR ?= $(CURDIR)
MODEL_DIR := $(abspath $(MODEL_DIR))
MODEL_NAME := $(notdir $(patsubst %/,%,$(MODEL_DIR)))

USERNAME ?= paragekbote
REGISTRY := r8.im
IMAGE_TAG := $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)

# --------------------------------------------------
# Tooling
# --------------------------------------------------
COG_BIN ?= cog
COG_CMD := $(shell command -v $(COG_BIN) 2>/dev/null)

PYTHON ?= python3
PIP ?= python3 -m pip

.DEFAULT_GOAL := help

# --------------------------------------------------
# Helpers
# --------------------------------------------------
define require-cog
	@if [ -z "$(COG_CMD)" ]; then \
		echo "❌ Cog not found in PATH"; exit 1; \
	fi
endef

# --------------------------------------------------
# Help
# --------------------------------------------------
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | sed 's/:.*##/:  /'

# --------------------------------------------------
# Environment / Tooling
# --------------------------------------------------
.PHONY: install-cog
install-cog: ## Install Cog CLI
	curl -fsSL https://cog.run/install.sh | sh

.PHONY: install-deps
install-deps: ## Install Python deps (dev + tests)
	$(PIP) install --upgrade pip
	$(PIP) install \
		".[dev,unit,integration,canary,gemma-torchao,flux-fast-lora,flux-fast-lora-img2img,unsloth,smollm3]" \
		--extra-index-url https://download.pytorch.org/whl/cu126

# --------------------------------------------------
# Build & Deploy
# --------------------------------------------------
.PHONY: build
build: ## Build Cog image locally
	$(call require-cog)
	test -f cog.yaml
	cd "$(MODEL_DIR)" && $(COG_CMD) build -t "$(IMAGE_TAG)"

.PHONY: push
push: ## Push image to Replicate
	$(call require-cog)
	cd "$(MODEL_DIR)" && $(COG_CMD) push "$(IMAGE_TAG)"

.PHONY: deploy
deploy: build push ## Build + push (CD entrypoint)

# --------------------------------------------------
# Tests
# --------------------------------------------------
.PHONY: lint unit integration canary deployment

lint: ## Lint
	pre-commit run --all-files || echo "⚠️ Pre-commit reported issues (non-blocking)"

unit: ## Unit tests
	pytest -m unit

integration: ## Integration tests (candidate only)
	pytest -m integration

canary: ## Canary tests (candidate vs stable)
	pytest -m canary


# --------------------------------------------------
# CI / CD meta targets
# --------------------------------------------------
.PHONY: ci cd

ci: lint unit ## CI = lint + unit

cd: install-deps deploy integration canary ## CD = deploy → integration → canary

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
.PHONY: remove-image clean-local

remove-image:
	@command -v docker >/dev/null 2>&1 || exit 0
	docker rmi -f "$(IMAGE_TAG)" 2>/dev/null || true

clean-local:
	rm -rf .cog
	$(MAKE) -s remove-image
