# Makefile for Cog CLI setup, model build, and deployment

# Default global install path, can override with `make COG_BIN=./bin/cog install`
COG_BIN       ?= /usr/local/bin/cog
COG_URL       := https://github.com/replicate/cog/releases/latest/download/cog_$(shell uname -s)_$(shell uname -m)
MODEL_DIR     := $(CURDIR)
MODEL_NAME    := $(notdir $(MODEL_DIR))
USERNAME      ?= paragekbote

.PHONY: install build login deploy help

install:
	@echo "🔧 Installing Cog CLI to $(COG_BIN)..."
	@if [ "$(COG_BIN)" = "/usr/local/bin/cog" ]; then \
		sudo curl -sSL -o $(COG_BIN) "$(COG_URL)"; \
		sudo chmod +x $(COG_BIN); \
	else \
		mkdir -p $(dir $(COG_BIN)); \
		curl -sSL -o $(COG_BIN) "$(COG_URL)"; \
		chmod +x $(COG_BIN); \
		echo "👉 Add '$(dir $(COG_BIN))' to your PATH"; \
	fi
	@echo "✅ Cog installed at $(COG_BIN)"

build:
	@echo "📦 Building Cog image for model: $(MODEL_NAME)..."
	cd $(MODEL_DIR) && $(COG_BIN) build -t $(MODEL_NAME)
	@echo "✅ Build complete for $(MODEL_NAME)"

login:
	@echo "🔐 Logging into Cog..."
	$(COG_BIN) login
	@echo "✅ Login successful."

deploy:
	@if ! $(COG_BIN) whoami >/dev/null 2>&1; then \
		echo "🔑 Not logged in. Running cog login..."; \
		$(COG_BIN) login; \
	fi
	@echo "🚀 Deploying model to r8.im/$(USERNAME)/$(MODEL_NAME)..."
	cd $(MODEL_DIR) && $(COG_BIN) push r8.im/$(USERNAME)/$(MODEL_NAME)
	@echo "✅ Model pushed to r8.im/$(USERNAME)/$(MODEL_NAME)"

help:
	@echo "🛠️  Cog Makefile Commands:"
	@echo "  make install           # Install Cog CLI (global by default)"
	@echo "  make COG_BIN=./bin/cog install   # Install Cog CLI locally"
	@echo "  make build             # Build Cog image from current folder"
	@echo "  make login             # Authenticate with Cog"
	@echo "  make deploy            # Push model to Replicate"
	@echo "  MODEL_DIR=path/to/model-folder make build/deploy  # Use specific folder"
