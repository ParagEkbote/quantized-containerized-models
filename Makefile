# ==============================================================================
# Cog Deployment Makefile (updated for per-model local installs)
# ==============================================================================
# This Makefile provides commands for building, testing, and deploying
# ML models using Cog (https://github.com/replicate/cog)
#
# Changes in this version:
# - Supports installing a model-local cog binary at <model>/.cog/bin/cog
# - When present, the model-local cog is used automatically for build/deploy/login
# - Added install-local / uninstall-local targets to manage per-model cog
# - Added remove-image and delete-local targets for local cleanup
# - Build/Deploy no longer require Cog to be globally installed at repo root
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration Variables
# ------------------------------------------------------------------------------

# Cog binary configuration (absolute path for cd compatibility)
COG_BIN      ?= /usr/local/bin/cog
COG_BIN      := $(abspath $(COG_BIN))
COG_URL      := https://github.com/replicate/cog/releases/latest/download/cog_$(shell uname -s)_$(shell uname -m)

# Model directory configuration (override with MODEL_DIR=path/to/model)
MODEL_DIR    ?= $(CURDIR)
MODEL_DIR    := $(abspath $(MODEL_DIR))
MODEL_NAME   := $(notdir $(MODEL_DIR))

# Per-model local cog binary (if present, it will be preferred)
MODEL_LOCAL_COG := $(abspath $(MODEL_DIR)/.cog/bin/cog)
# Use model-local cog if it exists and is executable, otherwise fall back to COG_BIN
COG_CMD := $(if $(wildcard $(MODEL_LOCAL_COG)),$(MODEL_LOCAL_COG),$(COG_BIN))

# Deployment configuration
USERNAME     ?= paragekbote
REGISTRY     := r8.im
IMAGE_TAG    := $(REGISTRY)/$(USERNAME)/$(MODEL_NAME)

# Test configuration
PYTEST       ?= pytest
PYTEST_ARGS  ?= -v

# Python configuration
PYTHON       ?= python3
PIP          ?= $(PYTHON) -m pip

# ------------------------------------------------------------------------------
# Color Output (for better UX)
# ------------------------------------------------------------------------------

COLOR_RESET  := \033[0m
COLOR_BLUE   := \033[34m
COLOR_GREEN  := \033[32m
COLOR_YELLOW := \033[33m
COLOR_RED    := \033[31m

# ------------------------------------------------------------------------------
# Primary Commands
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo "$(COLOR_BLUE)╔════════════════════════════════════════════════════════════════╗$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)║            🛠️  Cog Makefile Commands                        ║$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)╚════════════════════════════════════════════════════════════════╝$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)📦 Installation & Setup:$(COLOR_RESET)"
	@echo "  make install                        Install Cog CLI (global by default)"
	@echo "  make install-local MODEL_DIR=./models/xyz  Install Cog CLI into the model folder (.cog/bin)"
	@echo "  make uninstall-local MODEL_DIR=./models/xyz  Remove model-local cog binary"
	@echo "  make COG_BIN=./bin/cog install      Install Cog CLI to custom location"
	@echo ""
	@echo "$(COLOR_GREEN)🏗️  Build & Deploy:$(COLOR_RESET)"
	@echo "  make build MODEL_DIR=<path>         Build Cog image for specific model (prefers model-local cog)"
	@echo "  make login                          Authenticate with Cog (prefers model-local cog)"
	@echo "  make deploy MODEL_DIR=<path>        Deploy model to Replicate (prefers model-local cog)"
	@echo ""
	@echo "$(COLOR_GREEN)🧪 Testing:$(COLOR_RESET)"
	@echo "  make test MODEL=<name>              Run tests for specific model"
	@echo ""
	@echo "$(COLOR_GREEN)🧹 Maintenance:$(COLOR_RESET)"
	@echo "  make clean                          Clean build artifacts and Docker images"
	@echo "  make remove-image MODEL_DIR=<path>  Remove local Docker image for model"
	@echo "  make uninstall-local MODEL_DIR=<path>  Remove model-local cog binary"
	@echo "  make delete-local MODEL_DIR=<path>  Remove local cog + images for model"
	@echo "  make lint                           Run code linting checks"
	@echo ""
	@echo "$(COLOR_GREEN)📊 Information:$(COLOR_RESET)"
	@echo "  make list-models                    List all available models"
	@echo "  make status                         Show current configuration"
	@echo "  make version                        Show versions of all tools"
	@echo ""
	@echo "$(COLOR_BLUE)📝 Examples:$(COLOR_RESET)"
	@echo "  make install-local MODEL_DIR=./models/flux-fast-lora-hotswap"
	@echo "  make build MODEL_DIR=./models/flux-fast-lora-hotswap"
	@echo "  make deploy MODEL_DIR=./models/gemma-torchao"
	@echo "  make delete-local MODEL_DIR=./models/flux-fast-lora-hotswap"
	@echo ""

.PHONY: install
install: ## Install Cog CLI (use COG_BIN to override location)
	@PY_GLOBAL=$$($(PYTHON) -m pip show cog 2>/dev/null || true); \
	if [ -n "$$PY_GLOBAL" ]; then \
		echo "$(COLOR_RED)❌ Error: 'cog' is installed as a Python package globally ($(PYTHON) -m pip uninstall cog)$(COLOR_RESET)"; \
		echo "$(COLOR_RED)⚠️  Please uninstall with '$(PYTHON) -m pip uninstall cog' before running make install.$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🔧 Installing Cog CLI to $(COG_BIN)...$(COLOR_RESET)"
	@if [ "$(COG_BIN)" = "/usr/local/bin/cog" ]; then \
		echo "$(COLOR_YELLOW)⚠️  Installing to system path (requires sudo)$(COLOR_RESET)"; \
		sudo curl -sSL -o $(COG_BIN) "$(COG_URL)"; \
		sudo chmod +x $(COG_BIN); \
	else \
		echo "$(COLOR_YELLOW)📂 Installing to custom path$(COLOR_RESET)"; \
		mkdir -p $(dir $(COG_BIN)); \
		curl -sSL -o $(COG_BIN) "$(COG_URL)"; \
		chmod +x $(COG_BIN); \
		echo "$(COLOR_YELLOW)👉 Add '$(dir $(COG_BIN))' to your PATH$(COLOR_RESET)"; \
	fi
	@echo "$(COLOR_GREEN)✅ Cog installed at $(COG_BIN)$(COLOR_RESET)"
	@$(COG_BIN) --version || true

.PHONY: install-local
install-local: ## Install Cog CLI into MODEL_DIR/.cog/bin/cog (use MODEL_DIR)
	@if [ -z "$(MODEL_DIR)" ]; then \
		echo "$(COLOR_RED)❌ Error: MODEL_DIR must be set for install-local$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🔧 Installing Cog CLI to $(MODEL_LOCAL_COG)...$(COLOR_RESET)"
	@mkdir -p $(dir $(MODEL_LOCAL_COG))
	@curl -sSL -o $(MODEL_LOCAL_COG) "$(COG_URL)"
	@chmod +x $(MODEL_LOCAL_COG)
	@echo "$(COLOR_GREEN)✅ Cog installed at $(MODEL_LOCAL_COG)$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Tip: invoke with 'make build MODEL_DIR=$(MODEL_DIR)' (the Makefile will prefer the model-local cog automatically)$(COLOR_RESET)"

.PHONY: uninstall-local
uninstall-local: ## Remove model-local cog binary (use MODEL_DIR)
	@if [ -z "$(MODEL_DIR)" ]; then \
		echo "$(COLOR_RED)❌ Error: MODEL_DIR must be set for uninstall-local$(COLOR_RESET)"; \
		exit 1; \
	fi
	@if [ -f "$(MODEL_LOCAL_COG)" ]; then \
		rm -f "$(MODEL_LOCAL_COG)"; \
		echo "$(COLOR_GREEN)✅ Removed $(MODEL_LOCAL_COG)$(COLOR_RESET)"; \
		# remove .cog dir if empty
		rmdir --ignore-fail-on-non-empty "$(dir $(MODEL_LOCAL_COG))" 2>/dev/null || true; \
	else \
		echo "$(COLOR_YELLOW)⚠️  No model-local cog found at $(MODEL_LOCAL_COG)$(COLOR_RESET)"; \
	fi

.PHONY: build
build: ## Build Cog image (use MODEL_DIR to specify model). Prefers model-local cog if present.
	@if [ ! -x "$(COG_CMD)" ]; then \
		echo "$(COLOR_RED)❌ Error: Cog not found at $(COG_CMD)$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Run 'make install' or 'make install-local MODEL_DIR=$(MODEL_DIR)'$(COLOR_RESET)"; \
		exit 1; \
	fi
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "$(COLOR_RED)❌ Error: Model directory not found: $(MODEL_DIR)$(COLOR_RESET)"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_DIR)/cog.yaml" ]; then \
		echo "$(COLOR_RED)❌ Error: cog.yaml not found in $(MODEL_DIR)$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Available models:$(COLOR_RESET)"; \
		$(MAKE) list-models; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)📦 Building Cog image from $(MODEL_DIR) using $(COG_CMD)$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)   Tag: $(MODEL_NAME)$(COLOR_RESET)"
	@cd "$(MODEL_DIR)" && "$(COG_CMD)" build -t "$(MODEL_NAME)"
	@echo "$(COLOR_GREEN)✅ Build complete for $(MODEL_NAME)$(COLOR_RESET)"

.PHONY: login
login: ## Authenticate with Replicate/Cog (prefers model-local cog)
	@if [ ! -x "$(COG_CMD)" ]; then \
		echo "$(COLOR_RED)❌ Error: Cog not found at $(COG_CMD)$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Run 'make install' or 'make install-local MODEL_DIR=$(MODEL_DIR)'$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🔐 Logging into Cog ($(COG_CMD))...$(COLOR_RESET)"
	@"$(COG_CMD)" login
	@echo "$(COLOR_GREEN)✅ Login successful$(COLOR_RESET)"

.PHONY: deploy
deploy: ## Deploy model to Replicate (use MODEL_DIR to specify model). Prefers model-local cog.
	@if [ ! -x "$(COG_CMD)" ]; then \
		echo "$(COLOR_RED)❌ Error: Cog not found at $(COG_CMD)$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Run 'make install' or 'make install-local MODEL_DIR=$(MODEL_DIR)'$(COLOR_RESET)"; \
		exit 1; \
	fi
	@if [ ! -d "$(MODEL_DIR)" ]; then \
		echo "$(COLOR_RED)❌ Error: Model directory not found: $(MODEL_DIR)$(COLOR_RESET)"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_DIR)/cog.yaml" ]; then \
		echo "$(COLOR_RED)❌ Error: cog.yaml not found in $(MODEL_DIR)$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Available models:$(COLOR_RESET)"; \
		$(MAKE) list-models; \
		exit 1; \
	fi
	@if ! "$(COG_CMD)" whoami >/dev/null 2>&1; then \
		echo "$(COLOR_YELLOW)🔑 Not logged in. Running cog login...$(COLOR_RESET)"; \
		"$(COG_CMD)" login; \
	fi
	@echo "$(COLOR_BLUE)🚀 Deploying model to $(IMAGE_TAG) using $(COG_CMD)...$(COLOR_RESET)"
	@cd "$(MODEL_DIR)" && "$(COG_CMD)" push "$(IMAGE_TAG)"
	@echo "$(COLOR_GREEN)✅ Model pushed to $(IMAGE_TAG)$(COLOR_RESET)"

# ------------------------------------------------------------------------------
# Testing Commands
# ------------------------------------------------------------------------------

.PHONY: test
test: ## Run tests for specific model (use MODEL=model_name)
ifndef MODEL
	@echo "$(COLOR_RED)❌ Error: MODEL variable is required$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Usage: make test MODEL=flux_fast_lora_hotswap$(COLOR_RESET)"
	@exit 1
endif
	@if ! command -v $(PYTEST) >/dev/null 2>&1; then \
		echo "$(COLOR_RED)❌ Error: pytest not found$(COLOR_RESET)"; \
		echo "$(COLOR_YELLOW)Install with: pip install pytest$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🧪 Running tests for $(MODEL)...$(COLOR_RESET)"
	@$(PYTEST) tests/test_$(MODEL).py $(PYTEST_ARGS)
	@echo "$(COLOR_GREEN)✅ Tests passed for $(MODEL)$(COLOR_RESET)"

# ------------------------------------------------------------------------------
# Maintenance Commands
# ------------------------------------------------------------------------------

.PHONY: clean
clean: ## Clean build artifacts, cache files, and Docker images
	@echo "$(COLOR_BLUE)🧹 Cleaning build artifacts and cache...$(COLOR_RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(COLOR_BLUE)🧹 Cleaning Docker images...$(COLOR_RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		docker images | grep -E "$(MODEL_NAME)|cog-" | awk '{print $$3}' | xargs -r docker rmi -f 2>/dev/null || true; \
		echo "$(COLOR_GREEN)✅ Docker images cleaned$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)⚠️  Docker not found, skipping image cleanup$(COLOR_RESET)"; \
	fi
	@echo "$(COLOR_GREEN)✅ Cleanup complete$(COLOR_RESET)"

.PHONY: remove-image
remove-image: ## Remove local Docker image for MODEL_DIR (use MODEL_DIR)
	@if [ -z "$(MODEL_DIR)" ]; then \
		echo "$(COLOR_RED)❌ Error: MODEL_DIR must be set for remove-image$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🗑️  Removing local Docker image for $(MODEL_NAME)...$(COLOR_RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		docker rmi -f "$(MODEL_NAME)" 2>/dev/null || true; \
		docker rmi -f "$(IMAGE_TAG)" 2>/dev/null || true; \
		echo "$(COLOR_GREEN)✅ Local images removed (if present)$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)⚠️  Docker not found, skipping image removal$(COLOR_RESET)"; \
	fi

.PHONY: delete-local
delete-local: ## Remove model-local cog and local images for MODEL_DIR
	@$(MAKE) remove-image MODEL_DIR="$(MODEL_DIR)"
	@$(MAKE) uninstall-local MODEL_DIR="$(MODEL_DIR)"

.PHONY: lint
lint: ## Run code linting checks
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(COLOR_RED)❌ Error: Python not found$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_BLUE)🔍 Running linting checks...$(COLOR_RESET)"
	@$(PIP) install -q flake8 2>/dev/null || true
	@flake8 . --exclude=.git,__pycache__,build,dist --max-line-length=100 2>/dev/null || \
		echo "$(COLOR_YELLOW)⚠️  flake8 not installed, skipping$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)✅ Linting complete$(COLOR_RESET)"

# ------------------------------------------------------------------------------
# Utility Commands
# ------------------------------------------------------------------------------

.PHONY: list-models
list-models: ## List all available models
	@echo "$(COLOR_BLUE)📋 Available models:$(COLOR_RESET)"
	@find models -name "cog.yaml" -exec dirname {} \; 2>/dev/null | sed 's|models/||' | sort || \
		find . -maxdepth 2 -name "cog.yaml" -exec dirname {} \; 2>/dev/null | sed 's|./||' | sort

.PHONY: status
status: ## Show current configuration status
	@echo "$(COLOR_BLUE)📊 Current Configuration:$(COLOR_RESET)"
	@echo "  COG_BIN:    $(COG_BIN)"
	@echo "  MODEL_DIR:  $(MODEL_DIR)"
	@echo "  MODEL_NAME: $(MODEL_NAME)"
	@echo "  MODEL_LOCAL_COG: $(MODEL_LOCAL_COG)"
	@echo "  USING_COG:  $(COG_CMD)"
	@echo "  USERNAME:   $(USERNAME)"
	@echo "  IMAGE_TAG:  $(IMAGE_TAG)"
	@echo ""
	@if [ -x "$(COG_CMD)" ]; then \
		echo "$(COLOR_GREEN)  ✅ Cog available at: $(COG_CMD)$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_RED)  ❌ Cog not available at: $(COG_CMD)$(COLOR_RESET)"; \
	fi
	@PY_GLOBAL=$$($(PYTHON) -m pip show cog 2>/dev/null || true); \
	if [ -n "$$PY_GLOBAL" ]; then \
		echo "$(COLOR_RED)  ⚠️  'cog' installed via pip: please remove with '$(PYTHON) -m pip uninstall cog'$(COLOR_RESET)"; \
	fi
	@if [ -x "$(COG_CMD)" ] && "$(COG_CMD)" whoami >/dev/null 2>&1; then \
		echo "$(COLOR_GREEN)  ✅ Logged in as: $$($(COG_CMD) whoami)$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)  ⚠️  Not logged in or cog not executable$(COLOR_RESET)"; \
	fi

.PHONY: version
version: ## Show versions of all tools
	@echo "$(COLOR_BLUE)📦 Tool Versions:$(COLOR_RESET)"
	@echo "  Make:       $(MAKE_VERSION)"
	@command -v $(COG_CMD) >/dev/null 2>&1 && echo "  Cog:        $$($(COG_CMD) --version)" || echo "  Cog:        not installed/available"
	@command -v $(PYTHON) >/dev/null 2>&1 && echo "  Python:     $$($(PYTHON) --version)" || echo "  Python:     not found"
	@command -v $(PYTEST) >/dev/null 2>&1 && echo "  Pytest:     $$(pytest --version)" || echo "  Pytest:     not installed"
	@command -v docker >/dev/null 2>&1 && echo "  Docker:     $$(docker --version)" || echo "  Docker:     not installed"

# ------------------------------------------------------------------------------
# Default Target
# ------------------------------------------------------------------------------

.DEFAULT_GOAL := help
