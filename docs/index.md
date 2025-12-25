# Home

![Project Hero](assets/project_hero_img.webp)

# Deploy AI Models with Quantization & Containerization

Production-ready, optimized AI model deployments with quantization, compilation and containerization. Run locally, on-prem, or hosted without vendor lock-in.

## Quick Start

Get up and running in under 5 minutes:

1. **Clone the repository** and install dependencies
2. **Choose a deployment** based on your task (text, image, reasoning, or multimodal)
3. **Run locally with Docker** or push to Replicate for managed hosting

[Start with the Quick Start Guide →](quickstart.md)

---

## Available Deployments

Five production-ready deployments covering text generation, reasoning, multimodal understanding and image generation:

| Deployment | Task | Optimization | VRAM |
|---|---|---|---|
| **Flux Fast LoRA Hotswap** | Text → Image | torch.compile + BitsAndBytes + LoRA | ~16 GB |
| **Flux Fast LoRA Hotswap Img2Img** | Image → Image | torch.compile + BitsAndBytes + LoRA | ~18 GB |
| **SmolLM3 Pruna** | Text Generation | Pruna + HQQ + torch.compile | ~5 GB |
| **Phi-4 Reasoning Plus** | Reasoning & Explanation | Unsloth kernels + quantization | ~12 GB |
| **Gemma Torchao** | Multimodal (Vision + Text) | INT8 quantization + sparsity + torch.compile | ~6 GB |

[Explore all deployments →](deployment.md)

---

## Features

### Performance Improvements

- **70%+ model size reduction** through INT8/INT4 quantization
- **2–3× faster inference** compared to FP32 baselines
- **95–98% accuracy retention** across benchmarks
- **60–75% cost savings** on inference expenses

### Reproducible Deployments

- **Cog-based containers** ensure identical behavior across environments
- **GPU optimizations** with CUDA support
- **Portable execution** across local Docker, on-prem GPUs and cloud platforms
- **One-command deployment** to Replicate

---

### The Result

- **Hardware efficiency** — Run larger models on smaller GPUs
- **Inference speed** — Achieve 2–3× latency improvements
- **Quality preservation** — Maintain 95–98% output quality
- **Operational simplicity** — Deploy identically across environments

## Documentation Overview

### Getting Started

[**Quick Start**](quickstart.md)
: Set up and run your first model in under 5 minutes using Docker or Cog CLI.

### Understanding the System

[**System Architecture**](architecture.md)
: Deep dive into quantization strategies, compilation techniques, pruning approaches and design decisions for each deployment.

[**Deployment Reference**](deployment.md)
: Complete specifications for all five deployments including input/output schemas, performance metrics and configuration options.

### Building with Examples

[**Usage Examples**](examples.md)
: Practical code examples for text generation, reasoning, image generation and multimodal understanding across Python SDK, Docker and Cog CLI.

---

## Core Design Principles

All deployments follow these principles:

- **Post-training optimization** — No retraining required; works with existing models
- **Inference-first** — Optimized for latency, throughput and memory efficiency
- **Selective risk** — Preserve fragile model components while aggressively optimizing compute-heavy layers
- **Portability** — Containerized with Cog and Docker; run anywhere without vendor lock-in
- **Reproducibility** — Deterministic builds and fixed schemas for consistent behavior

---

## Next Steps

1. **[Quick Start](quick-start.md)** — Get a model running in 5 minutes
2. **[Deployment Reference](deployment.md)** — Explore available models and APIs
3. **[Usage Examples](examples.md)** — See practical code for your use case
4. **[System Architecture](architecture.md)** — Understand the technical implementation

---

**Ready to deploy?** [Start the Quick Start Guide →](quick-start.md)
