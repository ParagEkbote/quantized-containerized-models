MODEL_DIR ?= $(CURDIR)
MODEL_DIR := $(abspath $(MODEL_DIR))
MODEL_NAME := $(notdir $(MODEL_DIR))

COG_BIN ?= /usr/local/bin/cog
LOCAL_COG := $(MODEL_DIR)/.cog/bin/cog
COG_CMD := $(if $(wildcard $(LOCAL_COG)),$(LOCAL_COG),$(COG_BIN))

USERNAME ?= paragekbote
REGISTRY := r8.im
IMAGE_TAG := $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)

MKDOCS = mkdocs
CONFIG_FILE = /workspaces/ParagEkbote.github.io/mkdocs.yml

# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | sed 's/:.*##/:  /'


# ----------------------------------------
# Login
# ----------------------------------------
.PHONY: login
login: ## Login to Replicate using Cog
	@if [ ! -x "$(COG_CMD)" ]; then echo "Cog not found"; exit 1; fi
	"$(COG_CMD)" login


# ----------------------------------------
# Lint
# ----------------------------------------
.PHONY: lint
lint: ## Run ruff linting and pytest collection checks
	pre-commit run --all-files || true


# ----------------------------------------
# Build
# ----------------------------------------
.PHONY: build
build: ## Build Cog image
	@if [ ! -x "$(COG_CMD)" ]; then echo "Cog not found"; exit 1; fi
	@if [ ! -f "$(MODEL_DIR)/cog.yaml" ]; then echo "cog.yaml missing"; exit 1; fi
	cd "$(MODEL_DIR)" && "$(COG_CMD)" build -t "$(MODEL_NAME)"


# ----------------------------------------
# Push
# ----------------------------------------
.PHONY: push
push: ## Push image to Replicate
	cd "$(MODEL_DIR)" && "$(COG_CMD)" push "$(IMAGE_TAG)"


# ----------------------------------------
# Deploy
# ----------------------------------------
.PHONY: deploy
deploy: ## Auto-login and push
	@if ! "$(COG_CMD)" whoami >/dev/null 2>&1; then "$(COG_CMD)" login; fi
	cd "$(MODEL_DIR)" && "$(COG_CMD)" push "$(IMAGE_TAG)"


# ----------------------------------------
# Remove local Docker images
# ----------------------------------------
.PHONY: remove-image
remove-image: ## Remove local Docker images
	@if command -v docker >/dev/null 2>&1; then \
		docker rmi -f "$(MODEL_NAME)" 2>/dev/null || true; \
		docker rmi -f "$(IMAGE_TAG)" 2>/dev/null || true; \
	fi


# ----------------------------------------
# Delete local Cog + images
# ----------------------------------------
.PHONY: delete-local
delete-local: ## Delete .cog and local images
	@if [ -f "$(LOCAL_COG)" ]; then rm -f "$(LOCAL_COG)"; fi
	$(MAKE) -s remove-image


# ----------------------------------------
# Clean pycache
# ----------------------------------------
.PHONY: clean
clean: ## Clean __pycache__ and pyc files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


# ----------------------------------------
# Testing (unit / integration / deployment)
# ----------------------------------------
.PHONY: unit
unit: ## Run unit tests
	pytest -m "unit"

.PHONY: integration
integration: ## Run integration tests
	pytest -m "integration"

.PHONY: deployment
deployment: ## Run deployment tests
	pytest -m "deployment"

# ----------------------------------------
# CI → Unit+Linting tests only
# ----------------------------------------
.PHONY: ci
ci: lint unit

# ----------------------------------------
# CD → All tests+Linting
# ----------------------------------------
.PHONY: cd
cd: lint unit integration deployment ## Run all tests for CD pipeline

# ----------------------------------------
# Mkdocs Commands
# ----------------------------------------
.PHONY: serve docs
serve docs:
	$(MKDOCS) serve -f $(CONFIG_FILE)

.PHONY:build docs
build docs:
	$(MKDOCS) build -f $(CONFIG_FILE)

.PHONY:clean docs
clean docs:
	rm -rf site/

.DEFAULT_GOAL := help
