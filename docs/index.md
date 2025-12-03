# Home

![Project Hero](assets/project_hero_img.webp)

**Deploy AI models with an API through quantization and containerization.**

---

## 🚀 Quick Navigation

### Getting Started
- [Core Concepts](docs/concepts.md) - Understand quantization and containerization
- [Quick Start](docs/quickstart.md) - Run your first model

### Architecture & Design
- [System Architecture](docs/architecture.md) - High-level system design

### Deployments
- [Deployment Overview](docs/deployment.md) - All available model deployments
- [API Reference](docs/deployment.md) - Complete endpoint documentation

### Examples & Tutorials
- [Usage Examples](docs/examples.md) - Common use cases

---

## 📦 Available Deployments

We provide **5 production-ready deployments** showcasing different model types and quantization strategies:

| Deployment | Model Type | Quantization | Use Case | Status |
|-----------|------------|--------------|----------|---------|
| [Text Generation](deployments/text-generation.md) | LLM (7B params) | INT8 | Content creation, chatbots | ✅ Live |
| [Image Classification](deployments/image-classification.md) | Vision Model | INT8 | Image tagging, recognition | ✅ Live |
| [Object Detection](deployments/object-detection.md) | YOLO-based | INT4 | Real-time detection | ✅ Live |
| [Sentiment Analysis](deployments/sentiment-analysis.md) | BERT-based | INT8 | Text classification | ✅ Live |
| [Image Generation](deployments/image-generation.md) | Diffusion Model | Mixed (INT8/FP16) | AI art, creative tools | ✅ Live |

[View all deployment details →](docs/deployment.md)

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
Start with [Quick Start](docs/quickstart.md) to run your first model in under 5 minutes.

**For Developers and Builders:**
Read the [Architecture Overview](docs/architecture.md) to understand the system design and application. Check [API Reference](docs/deployment.md) for endpoint specifications and [Examples](docs/examples.md) for code examples.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](../LICENSE) file for details.

---

**Ready to get started?** → [Quick Start Guide](docs/quickstart.md)