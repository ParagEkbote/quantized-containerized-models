# ⚡ Quick Start

This guide helps you run your **first quantized, containerized model** in minutes.
You can run models **locally with Docker or Cog** or optionally deploy them to Replicate.

No vendor lock-in is required.

---

## Prerequisites

Ensure the following are available:

* **Git**
* **Docker** (with NVIDIA Container Toolkit)
* **NVIDIA GPU + CUDA drivers** (strongly recommended)
* **Python 3.11+** (only required for `cog predict`)

> All deployments are Cog-based and can run **fully offline on Docker**, without Replicate.

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

This repository contains:

* Cog configurations
* Quantized model weights
* Production-ready containers

---

## Step 2: Choose a Deployment

Pick a deployment based on your task:

* **Text-to-Image** → Flux Fast Lora Hotswap
* **Image-to-Image** → Flux Fast Lora Hotswap Img2Img
* **Multimodal Model** → Gemma Torchao
* **Reasoning Model** → Phi4 Reasoning Plus Unsloth
* **Lightweight Model** → SmolLM3 Pruna

See the full list in the [Deployment Overview](deployment.md).

---

## Step 3: Run with Docker Only (No Replicate)

You can run any Cog-based model **directly via Docker**, avoiding hosted platforms and vendor lock-in.

### Start the Container

```bash
docker run -d \
  -p 5000:5000 \
  --gpus=all \
  r8.im/paragekbote/flux-fast-lora-hotswap
```

This:

* Launches the model server locally
* Exposes an HTTP API on port `5000`
* Uses your local GPU via CUDA

---

## Step 4: Make an HTTP Inference Request

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Skyscrapers hover above clouds, bathed in golden sunrise.",
      "trigger_word": "Photographic"
    }
  }' \
  http://localhost:5000/predictions
```

The response will contain generated image URLs or streamed outputs, depending on the model.

---

## Step 5: Run with Cog CLI (Local Inference)

If you prefer direct CLI-based inference:

```bash
pip install cog
```

Run inference for the model:

```bash
cog predict \
  -i prompt="dreamy lake" \
  -i trigger_word="GHIBSKY"
```

This:

* Builds the container if needed
* Runs inference locally
* Guarantees the same environment as production

---

## Optional: Deploy to Replicate

If you want a managed, hosted endpoint on [Replicate](https://replicate.com/):

```bash
cog login
cog push r8.im/<username>/<model-name>
```

Deployment is optional.
**Local Docker and Cog usage are fully supported.**

---

## Why This Matters (No Vendor Lock-In)

* Models run identically on **local Docker, on-prem GPUs or cloud**
* No dependency on proprietary inference services
* Containers are portable across environments
* Easy migration between self-hosted and hosted deployments

This design ensures long-term maintainability and operational freedom.

---

## Next Steps

* 🧱 Understand internals via [Architecture](architecture.md)
* 🔌 Review inputs and outputs in the [API Reference](deployment.md)
* 🧪 Explore workflows in [Usage Examples](examples.md)

---

## Troubleshooting

* **Docker can’t see GPU**: Verify `nvidia-smi` and NVIDIA Container Toolkit
* **Slow inference**: Confirm quantized weights are being used
* **Schema errors**: Check deployment-specific input fields

For unresolved issues, open a GitHub issue with logs and hardware details.

---