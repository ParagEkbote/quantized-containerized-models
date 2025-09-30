# Default Cog binary location (absolute so it works after cd)
COG_BIN      ?= /usr/local/bin/cog
COG_BIN      := $(abspath $(COG_BIN))
COG_URL      := https://github.com/replicate/cog/releases/latest/download/cog_$(shell uname -s)_$(shell uname -m)

# Directory containing cog.yaml (override on CLI)
MODEL_DIR    ?= $(CURDIR)
MODEL_DIR    := $(abspath $(MODEL_DIR))
MODEL_NAME   := $(notdir $(MODEL_DIR))
USERNAME     ?= paragekbote

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
	@echo "📦 Building Cog image from $(MODEL_DIR) (tag: $(MODEL_NAME))..."
	$(COG_BIN) build -t $(MODEL_NAME) $(MODEL_DIR)
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
	$(COG_BIN) push r8.im/$(USERNAME)/$(MODEL_NAME)
	@echo "✅ Model pushed to r8.im/$(USERNAME)/$(MODEL_NAME)"

help:
	@echo "🛠️  Cog Makefile Commands:"
	@echo "  make install                        # Install Cog CLI (global by default)"
	@echo "  make COG_BIN=./bin/cog install      # Install Cog CLI locally"
	@echo "  make build MODEL_DIR=path/to/model  # Build Cog image from a folder containing cog.yaml"
	@echo "  make login                          # Authenticate with Cog"
	@echo "  make deploy MODEL_DIR=path/to/model # Push model to Replicate"
