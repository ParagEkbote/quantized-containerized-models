# quantized-containerized-models

**quantized-containerized-models** is a collection of experiments and best practices for deploying **optimized AI models** in efficient, containerized environments. The goal is to showcase how modern techniques—quantization, containerization and continuous integration/deployment (CI/CD) can work together to deliver fast, lightweight, and production-ready model deployments.

---

## Features

* **Quantization** – Reduce model size and accelerate inference using techniques like `nf4`, int8, and sparsity.
* **Containerization** – Package models with [Cog](https://cog.run/), ensuring reproducible builds and smooth deployments.
* **CI/CD Integration** – Automated pipelines for linting, testing, building and deployment directly to [Replicate](https://replicate.com).
* **Deployment Tracking** – Status Page for visibility into workflow health and deployment status.(TODO)
* **Open Source** – Fully licensed under [Apache 2.0](LICENSE).

---

## 🚀 Active Deployments

* [flux-fast-lora-hotswap](https://replicate.com/paragekbote/flux-fast-lora-hotswap):
  Built on the [LoRA fast blog post](https://huggingface.co/blog/lora-fast), this deployment uses `flux.1-dev` models with **two LoRAs** that can be hot-swapped to reduce generation time and avoid graph breaks.

  * Optimized with `nf4` quantization and `torch.compile` for speedups.
  * Includes an [Img2Img variant](https://replicate.com/paragekbote/flux-fast-lora-hotswap-img2img).
  * Featured in the official [Hugging Face blogpost](https://huggingface.co/blog/lora-fast#resources).
  * [Source code](https://github.com/ParagEkbote/quantized-containerized-models/tree/58e64c6e652b2f82f10cab42f25c4093a1252974/flux.1-dev).

* [smollm3-3b-smashed](https://replicate.com/paragekbote/smollm3-3b-smashed):
  Uses [Pruna](https://github.com/PrunaAI/pruna) to quantize and `torch.compile` the smollm3-3b model, enabling **lower VRAM usage** and **faster generation**.

  * Supports **16k token context windows** and hybrid reasoning.
  * [Source code](https://github.com/ParagEkbote/quantized-containerized-models/blob/9b914464ffe521506c68146f7109572ffffaa520/smollm3-3b-pruna/predict.py).

* [phi-4-reasoning-plus-unsloth](https://replicate.com/paragekbote/phi-4-reasoning-plus-unsloth):
  Accelerates Microsoft’s Phi-4 reasoning model with [Unsloth](https://docs.unsloth.ai/), achieving **faster inference** and a **smaller memory footprint**.

* [gemma3-torchao-quant-sparse](https://replicate.com/paragekbote/gemma3-torchao-quant-sparse):
  Improves inference performance for Gemma-3-4B-IT using **torchao int8 quantization** combined with sparsity techniques such as granular and magnitude pruning.

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
