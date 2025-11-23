# ==============================================================
# Minimal Cog Makefile: Build, Push, Deploy, Clean, Delete
# ==============================================================

MODEL_DIR ?= $(CURDIR)
MODEL_DIR := $(abspath $(MODEL_DIR))
MODEL_NAME := $(notdir $(MODEL_DIR))

COG_BIN ?= /usr/local/bin/cog
LOCAL_COG := $(MODEL_DIR)/.cog/bin/cog
COG_CMD := $(if $(wildcard $(LOCAL_COG)),$(LOCAL_COG),$(COG_BIN))

USERNAME ?= paragekbote
REGISTRY := r8.im
IMAGE_TAG := $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)


# --------------------------------------------------------------
# Skinny help
# --------------------------------------------------------------
.PHONY: help
help: ## Show available commands
	@echo "Minimal Cog Makefile Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | sed 's/:.*##/:  /'


# --------------------------------------------------------------
# Build image
# --------------------------------------------------------------
.PHONY: build
build: ## Build Cog image for MODEL_DIR
	@if [ ! -x "$(COG_CMD)" ]; then echo "Cog not found at $(COG_CMD)"; exit 1; fi
	@if [ ! -f "$(MODEL_DIR)/cog.yaml" ]; then echo "cog.yaml missing in $(MODEL_DIR)"; exit 1; fi
	cd "$(MODEL_DIR)" && "$(COG_CMD)" build -t "$(MODEL_NAME)"
	echo "Build complete: $(MODEL_NAME)"


# --------------------------------------------------------------
# Push image
# --------------------------------------------------------------
.PHONY: push
push: ## Push model to Replicate (no auto-login)
	cd "$(MODEL_DIR)" && "$(COG_CMD)" push "$(IMAGE_TAG)"
	echo "Pushed: $(IMAGE_TAG)"


# --------------------------------------------------------------
# Deploy with auto-login
# --------------------------------------------------------------
.PHONY: deploy
deploy: ## Deploy model to Replicate (auto-login, prefers model-local cog)
	@if [ ! -x "$(COG_CMD)" ]; then echo "Cog not found at $(COG_CMD)"; exit 1; fi
	@if [ ! -f "$(MODEL_DIR)/cog.yaml" ]; then echo "cog.yaml missing in $(MODEL_DIR)"; exit 1; fi
	@if ! "$(COG_CMD)" whoami >/dev/null 2>&1; then \
		echo "Not logged in → running cog login..."; \
		"$(COG_CMD)" login || exit 1; \
	fi
	cd "$(MODEL_DIR)" && "$(COG_CMD)" push "$(IMAGE_TAG)"
	echo "Deployed: $(IMAGE_TAG)"


# --------------------------------------------------------------
# Remove local Docker images
# --------------------------------------------------------------
.PHONY: remove-image
remove-image: ## Remove local Docker images for MODEL_DIR
	@if command -v docker >/dev/null 2>&1; then \
		docker rmi -f "$(MODEL_NAME)" 2>/dev/null || true; \
		docker rmi -f "$(IMAGE_TAG)" 2>/dev/null || true; \
		echo "Local images removed."; \
	else \
		echo "Docker not installed; skipping."; \
	fi


# --------------------------------------------------------------
# Delete model-local Cog + images
# --------------------------------------------------------------
.PHONY: delete-local
delete-local: ## Remove model-local Cog and Docker artifacts
	@if [ -f "$(LOCAL_COG)" ]; then rm -f "$(LOCAL_COG)"; echo "Removed model-local Cog."; fi
	$(MAKE) -s remove-image MODEL_DIR="$(MODEL_DIR)"
	echo "delete-local complete."


# --------------------------------------------------------------
# Clean Python cache
# --------------------------------------------------------------
.PHONY: clean
clean: ## Remove pycache and *.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	echo "Clean complete."


.DEFAULT_GOAL := help
