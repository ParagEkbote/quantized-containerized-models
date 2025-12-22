# 🏠 Home

![Project Hero](assets/project_hero_img.webp)

**Deploy AI models with an API through quantization and containerization.**

---

## 🚀 Quick Navigation

### Getting Started
- [Quick Start](quickstart.md) - Run your first model

### Architecture & Design
- [System Architecture](architecture.md) - High-level system design

### Deployments
- [Deployment Overview](deployment.md) - All available model deployments
- [API Reference](deployment.md) - Complete endpoint documentation

### Examples & Tutorials
- [Usage Examples](examples.md) - Common use cases

---

## 📦 Available Deployments

We provide **5 production-ready deployments** showcasing different model types and quantization strategies:

| Deployment                     | Modality                  | Optimization Stack                        | Status |
| ------------------------------ | ------------------------- | ----------------------------------------- | ------ |
| [Flux Fast Lora Hotswap](deployment.md)         | Text → Image              | torch.compile + BitsAndBytes + PEFT LoRA  | ✅ Live   |
| [Flux Fast Lora Hotswap Img2Img](deployment.md) | Image → Image             | torch.compile + BitsAndBytes + PEFT LoRA  | ✅ Live   |
| [SmolLM3 Pruna](deployment.md)   | Text (LLM)| Pruna + HQQ + torch.compile               |✅ Live   |
|[Phi4 Reasoning Plus Unsloth](deployment.md)  | Reasoning LLM| Unsloth kernels + quantization|✅ Live   |
|  [Gemma Torchao](deployment.md)                  | Multimodal (Text + Image) | INT8 (torchao) + sparsity + torch.compile | ✅ Live   |

[View all deployment details →](deployment.md)

---

## 🎯 Key Features

### Significant Model Improvements
- **70%+ size reduction** through INT8/INT4 quantization
- **2-3x faster inference** compared to FP32 models
- **95-98% accuracy retention** across benchmarks
- **Cost savings** of 60-75% on inference costs

### Reproducible Containerization
- **Cog-based containers** for consistent environments
- **GPU deployment** with CUDA support
- **One-command deployment** to Replicate

---

## 💡 Why This Project?

### The Problem
Deploying AI models in production is challenging:
- Large model sizes require expensive hardware
- Slow inference times impact user experience
- Inconsistent environments cause deployment failures
- High costs limit accessibility and scale

### Solution
Combine quantization and containerization for:
- **Hardware efficiency** - Run larger models on smaller GPUs
- **Speed** - Faster inference without sacrificing quality
- **Reproducibility** - Identical behavior across environments

---

## 📚 Documentation Structure

This documentation is organized into focused sections:

**For New Users:**
Start with [Quick Start](quickstart.md) to run your first model in under 5 minutes.

**For Developers and Builders:**
Read the [Architecture Overview](architecture.md) to understand the system design and application. Check [API Reference](deployment.md) for endpoint specifications and [Examples](examples.md) for code examples.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.html) for details.

---

**Ready to get started?** → [Quick Start Guide](quickstart.md)
