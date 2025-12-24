SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

MODEL_DIR ?= $(CURDIR)
MODEL_DIR := $(abspath $(MODEL_DIR))
MODEL_NAME := $(notdir $(patsubst %/,%,$(MODEL_DIR)))

USERNAME ?= paragekbote
REGISTRY := r8.im
IMAGE_TAG := $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)

COG_BIN ?= cog
COG_CMD := $(shell command -v $(COG_BIN) 2>/dev/null)

MKDOCS ?= mkdocs
CONFIG_FILE ?= mkdocs.yml

.DEFAULT_GOAL := help


# ----------------------------------------
# Helpers
# ----------------------------------------
define require-cog
	@if [ -z "$(COG_CMD)" ]; then \
		echo "❌ Cog not found in PATH"; exit 1; \
	fi
endef


# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | sed 's/:.*##/:  /'


# ----------------------------------------
# Login
# ----------------------------------------
.PHONY: login
login: ## Login to Replicate using Cog
	$(call require-cog)
	$(COG_CMD) login

.PHONY:install-cog
install-cog: 
	sh <(curl -fsSL https://cog.run/install.sh)
# ----------------------------------------
# Build
# ----------------------------------------
.PHONY: build
build: ## Build Cog image
	$(call require-cog)
	@test -f cog.yaml || (echo "❌ cog.yaml missing" && exit 1)
	cd "$(MODEL_DIR)" && $(COG_CMD) build -t "$(IMAGE_TAG)"


# ----------------------------------------
# Push
# ----------------------------------------
.PHONY: push
push: ## Push image to Replicate
	$(call require-cog)
	cd "$(MODEL_DIR)" && $(COG_CMD) push "$(IMAGE_TAG)"


# ----------------------------------------
# Deploy
# ----------------------------------------
.PHONY: deploy
deploy: ## Push image (requires prior login)
	$(call require-cog)
	@$(COG_CMD) whoami >/dev/null 2>&1 || \
		(echo "❌ Not logged in. Run make login" && exit 1)
	cd "$(MODEL_DIR)" && $(COG_CMD) push "$(IMAGE_TAG)"


# ----------------------------------------
# Remove local Docker images
# ----------------------------------------
.PHONY: remove-image
remove-image:
	@command -v docker >/dev/null 2>&1 || exit 0
	docker rmi -f "$(IMAGE_TAG)" 2>/dev/null || true


# ----------------------------------------
# Delete local Cog + images
# ----------------------------------------
.PHONY: delete-local
delete-local:
	rm -rf .cog
	$(MAKE) -s remove-image


# ----------------------------------------
# Lint / Tests
# ----------------------------------------
.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: all-tests
unit:
	pytest -m unit

integration:
	pytest -m integration

deployment:
	pytest -m deployment


# ----------------------------------------
# CI / CD
# ----------------------------------------
.PHONY: ci cd
ci: lint unit
cd: lint unit integration deployment


# ----------------------------------------
# MkDocs
# ----------------------------------------
.PHONY: docs-serve docs-build docs-clean docs-deploy

docs-serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

docs-build:
	$(MKDOCS) build -f $(CONFIG_FILE)

docs-clean:
	rm -rf site/

docs-deploy:
	$(MKDOCS) gh-deploy --force
