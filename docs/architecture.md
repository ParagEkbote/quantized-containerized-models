# 🧱 System Architecture

This project provides a set of **performance-optimized, production-ready AI model deployments** built around a common architectural philosophy:

> **Reduce memory footprint and inference cost without sacrificing output quality**, while maintaining reproducibility and deployment portability.

The system combines **quantization, compilation, sparsity, and containerization** to achieve this across multiple model classes (LLMs, multimodal models, and diffusion models).

---

## High-Level Design Principles

Across all deployments, the architecture adheres to the following principles:

* **Post-training optimization** (no retraining required)
* **Inference-first design** (latency, throughput, VRAM efficiency)
* **Selective risk** (optimize compute-heavy layers, preserve fragile ones)
* **Portability** (Cog + Docker, no vendor lock-in)
* **Reproducibility** (deterministic builds, fixed schemas)

---

## Architecture Overview by Model Class

### 1. SmolLM3-3B (Pruna + HQQ)

![alt text](assets/smollm3-3b-smashed.webp)

This deployment packages **SmolLM3-3B** using **Pruna**, **HQQ quantization**, and **`torch.compile`** to enable fast, memory-efficient text generation.

#### Core Components

* **Pruna**
  Orchestrates model loading, quantization, compilation, and output handling.

* **HQQ Quantization**

  * Reduces memory footprint and compute cost
  * Preserves output quality through high-fidelity post-training quantization
  * Particularly effective for mid-sized decoder-only LLMs

* **`torch.compile`**

  * Fuses operations and optimizes execution graphs
  * Reduces Python overhead and improves kernel efficiency
  * Applied after quantization for maximal benefit

#### Features

* Dynamic text generation with:

  * `"think"` (reasoning-enabled)
  * `"no_think"` (concise output)
* Configurable inference:

  * `max_new_tokens` (up to 16k)
  * `seed` for reproducibility
  * `mode` selection
* Automatic output persistence:

  * Generated responses saved as `.txt` artifacts

This setup achieves **reduced VRAM usage**, **faster inference**, and **stable output quality**, making it suitable for long-form generation workloads.

---

### 2. Flux.1 [dev] (Text-to-Image with LoRA Hot-Swapping)

![alt text](assets/flux-fast-lora-hotswap.webp)

This deployment packages **black-forest-labs/FLUX.1-dev** for high-performance text-to-image generation with **dynamic LoRA switching**.

> This work builds on techniques featured in the Hugging Face blog.

#### Core Components

* **PyTorch 2.x + `torch.compile`**

  * Accelerates diffusion inference
  * Improves kernel fusion and scheduling

* **BitsAndBytes Quantization**

  * Reduces VRAM usage significantly
  * Enables larger models to run on smaller GPUs

* **PEFT LoRA**

  * Lightweight adapters injected at runtime
  * Enables instant style switching without reloading the base model

#### Dynamic LoRA Hot-Swapping

LoRAs are activated via **trigger words** in the prompt.

##### Available Styles

**Enhanced Image Preferences**

* Trigger words:
  `["Cinematic", "Photographic", "Anime", "Manga", "Digital art", "Pixel art", "Fantasy art", "Neonpunk", "3D Model", "Painting", "Animation", "Illustration"]`
* LoRA: `data-is-better-together/open-image-preferences-v1-flux-dev-lora`
* Purpose: Preference-aligned, high-quality image generation

**Ghibsky Illustration**

* Trigger: `GHIBSKY`
* LoRA: `aleksa-codes/flux-ghibsky-illustration`
* Purpose: Studio Ghibli–inspired skies and landscapes

#### Performance Characteristics

* **Speed:** Up to 2× faster generation via `torch.compile`
* **Memory:** ~40% VRAM reduction through quantization
* **Quality:** Full FLUX.1-dev image quality preserved
* **Flexibility:** Instant style switching with no model reload

---

### 3. Gemma 3 4B IT (Multimodal, Quantized & Sparse)

![alt text](assets/gemma3-torchao-quant-sparse.webp)

This deployment is a **performance-optimized adaptation of `google/gemma-3-4b-it`**, targeting efficient **image + text generation**.

---

## Technical Details

### INT8 Weight-Only Quantization

**Why**

* Reduces parameter size and memory bandwidth
* Enables real inference speedups on modern hardware (NVIDIA, Intel)

**How**

* Applied post-training using:

  ```python
  torchao.quantization.quantize_(model, Int8WeightOnlyConfig())
  ```
* Weights are sanitized (contiguous, non-meta) prior to quantization

**Caveats**

* Requires `torchao` at load and inference time
* Quantized checkpoints should always be smoke-tested

---

### Pruning Strategies

Pruning targets **`nn.Linear` layers only**, focusing on compute-dominant regions.

#### Motivation

MLP layers and QKV projections account for the majority of FLOPs per transformer block.

Reducing parameters here yields significant gains with minimal quality loss.

---

#### Implemented Strategies

1. **Magnitude-Based Pruning**

   * Retains top-K weights by absolute value
   * Simple and predictable
   * Implementation:

     * Flatten → `torch.topk` → threshold → mask → `weight.mul_(mask)`

2. **Gradual Magnitude Pruning**

   * Sparsity increases progressively over multiple steps
   * Helps avoid sudden performance degradation
   * Suitable for prune → fine-tune loops

3. **Structured Sparsity (Safe Mode)**

   * Removes entire output channels (rows)
   * Importance measured via L2 norm
   * Preserves tensor shapes and avoids downstream breakage

---

### Filter Map (`gemma_filter_fn`)

A critical safety mechanism that defines **where pruning is allowed**.

* **Whitelisted layers**

  * QKV projections
  * MLP up / down projections

* **Blacklisted layers**

  * Embeddings
  * `lm_head`
  * LayerNorm
  * Output projections

**Why it matters**

The filter map encodes pruning policy and prevents semantic breakage by:

* Targeting compute-heavy layers
* Avoiding structurally fragile components

---

### Selective `torch.compile`

Rather than compiling the entire model, compilation is applied **selectively**.

* Applied to:

  * Stable, compute-heavy submodules
* Avoided for:

  * Embeddings
  * LM heads
  * Fragile or version-sensitive components

**Rationale**

* Full-model compilation is costly and brittle
* Selective compilation captures most performance gains with lower risk
* Improves portability across PyTorch versions and hardware

---

### 4. Phi-4 Reasoning Plus (Unsloth)

![alt text](assets/phi-4-reasoning-plus-unsloth.webp)

This deployment packages **`unsloth/phi-4-reasoning-plus`**, an optimized variant of Microsoft’s Phi-4 reasoning model. The architecture focuses on **reasoning-centric inference**, optimized through **Unsloth kernel techniques** to reduce memory usage and accelerate execution.

#### Core Components

* **Unsloth Optimization**

  * Custom kernel fusion and memory-efficient execution paths
  * Reduces VRAM pressure, enabling smooth inference on smaller GPUs
  * Improves throughput without altering model semantics

* **Reasoning-Oriented Model Design**

  * Tuned for structured, step-by-step logical reasoning
  * Strong performance on explanation, problem solving, and analytical tasks
  * Maintains conversational fluency without requiring prompt engineering

#### Features

* **Reasoning-first LLM**

  * Produces explicit, logical explanations
  * Well-suited for educational, analytical, and problem-solving use cases

* **Memory-efficient inference**

  * Unsloth optimizations allow deployment on constrained GPUs

* **Natural conversational behavior**

  * Performs well with short, natural prompts
  * No need for chain-of-thought prompting tricks

* **Flexible decoding controls**

  * Supports:

    * `temperature`
    * `top_p`
    * `max_new_tokens`
  * Enables tuning between creativity, determinism, and verbosity

#### Usage Guidance

* Prefer **short, natural prompts**
  Example:
  *“Explain why the sky is blue in simple steps.”*

* Control output length via `max_new_tokens`:

  * Lower values for concise explanations
  * Higher values for detailed reasoning traces

This deployment is ideal when **reasoning quality and interpretability** are higher priorities than raw generation speed.

---

### 5. FLUX.1-dev LoRA Hotswap (Image-to-Image)

![alt text](assets/flux-fast-lora-hotswap-img2img.webp)

This deployment extends the FLUX architecture to **image-to-image generation**, using **LoRA hot-swapping** to apply stylistic transformations to an input image while preserving content structure.

#### Core Components

* **Base Model**

  * `black-forest-labs/FLUX.1-dev`

* **PyTorch 2.x + `torch.compile`**

  * Accelerates diffusion inference
  * Improves kernel fusion and execution efficiency

* **BitsAndBytes Quantization**

  * Reduces VRAM usage significantly
  * Enables image-to-image workflows on smaller GPUs

* **PEFT LoRA Hot-Swapping**

  * LoRA adapters dynamically injected at runtime
  * Style changes applied via trigger words
  * No base-model reload required

---

#### Features

* **Optimized performance**

  * `torch.compile` enables faster inference paths

* **Dynamic LoRA switching**

  * Swap styles instantly using prompt-level triggers

* **Memory efficiency**

  * Quantization reduces GPU memory footprint

* **Multi-style image transformations**

  * Two LoRAs preloaded for immediate use

---

#### Available Styles

**Enhanced Image Preferences**

* **Trigger words:**
  `["Cinematic", "Photographic", "Anime", "Manga", "Digital art", "Pixel art", "Fantasy art", "Neonpunk", "3D Model", "Painting", "Animation", "Illustration"]`
* **LoRA:**
  `data-is-better-together/open-image-preferences-v1-flux-dev-lora`
* **Description:**
  Applies refined stylistic preferences learned from curated human preference data to the input image.

**Ghibsky Illustration**

* **Trigger:** `GHIBSKY`
* **LoRA:** `aleksa-codes/flux-ghibsky-illustration`
* **Description:**
  Transforms the input image into Studio Ghibli–inspired skies and landscapes.

---

#### Performance Characteristics

* **Speed:** Up to 2× faster processing with `torch.compile`
* **Memory:** ~40% VRAM reduction via quantization
* **Quality:** Preserves FLUX.1-dev fidelity while restyling inputs
* **Flexibility:** On-the-fly LoRA switching without model reloads

This architecture is particularly effective for **creative image transformation pipelines** where style experimentation and rapid iteration are required.

---

## Summary

This architecture demonstrates that **carefully applied post-training optimizations**—quantization, sparsity, selective compilation, and LoRA modularity—can deliver:

* Substantial memory savings
* Faster inference
* Stable, high-quality outputs
* Portable, vendor-neutral deployments

All deployments share a common design : **optimize where it matters, preserve where it breaks**.