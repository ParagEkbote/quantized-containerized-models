# Deployments & API Reference

All deployments in this repository are **production-ready**, **quantization-aware** and **containerized using Cog**, enabling seamless execution across local Docker, on-prem GPUs and hosted platforms such as Replicate.

## Available Deployments

The repository provides five optimized deployments covering text generation, reasoning, multimodal inference and diffusion-based image generation.

| Deployment | Task | Input | Output |
|---|---|---|---|
| **Flux Fast LoRA Hotswap** | Text-to-Image | Text prompt + style trigger | PNG image |
| **Flux Fast LoRA Hotswap Img2Img** | Image-to-Image | Image + text prompt + style trigger | PNG image |
| **SmolLM3 Pruna** | Text Generation | Text prompt | Text output |
| **Phi-4 Reasoning Plus (Unsloth)** | Reasoning & Explanation | Text prompt | Structured text output |
| **Gemma Torchao** | Multimodal QA | Image + text prompt | Text output |

Each deployment exposes a **stable input schema**, supports deterministic inference and can be executed without vendor lock-in.

---

## Flux Fast LoRA Hotswap

### Input Schema

```json
{
  "prompt": "string (required)",
  "trigger_word": "string (optional)",
  "height": "integer (default: 768)",
  "width": "integer (default: 768)",
  "num_inference_steps": "integer (default: 30)",
  "guidance_scale": "float (default: 3.5)",
  "seed": "integer (optional)"
}
```

### Output

- **Type:** PNG image (base64-encoded or URL)
- **Resolution:** 768×768 (configurable)
- **Format:** RGB

### Available Styles

**Enhanced Image Preferences** — `data-is-better-together/open-image-preferences-v1-flux-dev-lora`

Trigger words: `Cinematic`, `Photographic`, `Anime`, `Manga`, `Digital art`, `Pixel art`, `Fantasy art`, `Neonpunk`, `3D Model`, `Painting`, `Animation`, `Illustration`

**Ghibsky Illustration** — `aleksa-codes/flux-ghibsky-illustration`

Trigger word: `GHIBSKY`

### Performance

- **Inference latency:** 8–15 seconds (with torch.compile)
- **VRAM usage:** ~18 GB (optimized from 24 GB baseline)
- **Quality:** Full FLUX.1-dev fidelity

### Example Request

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Golden sunset over mountain peaks, cinematic lighting",
      "trigger_word": "Cinematic",
      "num_inference_steps": 28
    }
  }'
```

---

## Flux Fast LoRA Hotswap Img2Img

### Input Schema

```json
{
  "image": "file or URL (required)",
  "prompt": "string (required)",
  "trigger_word": "string (optional)",
  "strength": "float (0.0–1.0, default: 0.7)",
  "num_inference_steps": "integer (default: 30)",
  "guidance_scale": "float (default: 3.5)",
  "seed": "integer (optional)"
}
```

### Output

- **Type:** PNG image (base64-encoded or URL)
- **Resolution:** Matches input image dimensions
- **Format:** RGB

### Parameters

**strength**
: Controls how much the original image is modified. Lower values (0.3–0.5) preserve structure; higher values (0.7–0.9) allow more creative transformation.

**trigger_word**
: Same style triggers as text-to-image deployment. Omit for content-preserving edits.

### Performance

- **Inference latency:** 6–12 seconds (optimized)
- **VRAM usage:** ~18 GB
- **Quality:** Preserves FLUX.1-dev fidelity while restyling

### Example Request

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/photo.jpg",
      "prompt": "Transform into Studio Ghibli style",
      "trigger_word": "GHIBSKY",
      "strength": 0.8
    }
  }'
```

---

## SmolLM3 Pruna

### Input Schema

```json
{
  "prompt": "string (required)",
  "max_new_tokens": "integer (default: 512, max: 16384)",
  "temperature": "float (default: 0.7, range: 0.0–2.0)",
  "top_p": "float (default: 0.9, range: 0.0–1.0)",
  "mode": "string (default: 'no_think', options: 'think', 'no_think')",
  "seed": "integer (optional)"
}
```

### Output

- **Type:** Plain text
- **Persistence:** Automatically saved as `.txt` artifact
- **Format:** UTF-8

### Modes

**think**
: Enables extended reasoning. Model produces step-by-step logical chains before final output. Useful for complex analysis or problem-solving.

**no_think**
: Direct, concise response without intermediate reasoning steps. Faster and more suitable for simple queries.

### Performance

- **Inference latency:** 1–3 seconds per 256 tokens (optimized)
- **VRAM usage:** ~8 GB (reduced from 12 GB baseline)
- **Throughput:** ~80–120 tokens/second

### Example Request

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Explain quantum entanglement in simple terms",
      "max_new_tokens": 256,
      "mode": "no_think",
      "temperature": 0.6
    }
  }'
```

---

## Phi-4 Reasoning Plus (Unsloth)

### Input Schema

```json
{
  "prompt": "string (required)",
  "max_new_tokens": "integer (default: 2048)",
  "temperature": "float (default: 0.7, range: 0.0–2.0)",
  "top_p": "float (default: 0.95, range: 0.0–1.0)",
  "seed": "integer (optional)"
}
```

### Output

- **Type:** Plain text with reasoning annotations
- **Format:** UTF-8
- **Structure:** Explicit logical steps and explanations

### Characteristics

- **Reasoning-first design:** Naturally produces structured explanations
- **No prompt engineering required:** Works well with natural, conversational prompts
- **Explanation-focused:** Ideal for educational and analytical tasks

### Performance

- **Inference latency:** 2–5 seconds per 256 tokens
- **VRAM usage:** ~12 GB (optimized via Unsloth)
- **Throughput:** ~50–80 tokens/second

### Example Request

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Why do planets orbit the sun? Explain step-by-step.",
      "max_new_tokens": 1024,
      "temperature": 0.5
    }
  }'
```

---

## Gemma Torchao

### Input Schema

```json
{
  "image": "file or URL (required)",
  "prompt": "string (required)",
  "max_new_tokens": "integer (default: 256)",
  "temperature": "float (default: 0.7, range: 0.0–2.0)",
  "top_p": "float (default: 0.9, range: 0.0–1.0)",
  "seed": "integer (optional)"
}
```

### Output

- **Type:** Plain text
- **Format:** UTF-8
- **Content:** Image description, answer to query, or analysis

### Capabilities

- **Visual understanding:** Analyzes image content with high accuracy
- **Question answering:** Responds to specific queries about image regions
- **Description generation:** Produces natural-language descriptions

### Performance

- **Inference latency:** 1–3 seconds (vision encoding + text generation)
- **VRAM usage:** ~6 GB (INT8 quantization + sparsity)
- **Quality:** Retains full model fidelity despite optimizations

### Example Request

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/photo.jpg",
      "prompt": "What objects are visible in this image?",
      "max_new_tokens": 128
    }
  }'
```

---

## Testing & Validation

![alt text](assets/hero_img_testing.webp)

The repository follows a structured approach to ensure robustness across rapid upstream changes.

### Tier 1: Unit Tests

**Purpose:** Validate individual components in isolation

**Coverage:**
- Quantization routines
- Pruning filters
- Input schema validation
- Utility functions

**Characteristics:**
- CPU-first execution
- Fast runtime (seconds)
- Minimal resource requirements

### Tier 2: Integration Tests

**Purpose:** Validate end-to-end model behavior with optimizations

**Coverage:**
- Quantized checkpoint loading
- Single inference step validation
- Output shape verification
- Schema compatibility checks

**Characteristics:**
- GPU-enabled (optional)
- Small input batches
- Exponential retry logic
- Focus on correctness over performance

### Tier 3: Canary Release Tests

**Purpose:**
Detect functional, semantic, and performance regressions by comparing a newly deployed candidate against a pinned, known-good stable baseline before promotion.

---

### Coverage
- Live inference via Replicate deployments (stable vs candidate)
- Schema-correct inputs (text, multimodal, or image as applicable)
- Output sanity checks (length, format, degeneration)
- Semantic equivalence checks (e.g., embedding similarity for text)
- Latency and throughput regression detection

---

### Characteristics
- Executed **against deployed models**, not local containers
- GPU execution provided by the production inference platform (e.g., Replicate)
- CPU-only CI runners are sufficient for test execution
- Deterministic inputs and seeds where supported
- Pinned stable baseline per deployment to avoid cross-model collisions
- Lightweight enough to run on every deployment event

---

### What This Tier Explicitly Avoids
- Local Docker-in-Docker execution
- Building or pushing containers
- Full offline benchmark suites
- Synthetic load or stress testing

These concerns are handled earlier (build-time) or separately (benchmarking).

### Benchmarking Metrics

![alt text](assets/hero_img_benchmark.webp)

Each deployment is evaluated across:

- **Latency:** End-to-end inference time (including I/O)
- **VRAM usage:** Peak GPU memory during inference
- **Throughput:** Tokens/second or images/second
- **Quality:** Task-specific metrics (visual fidelity, semantic correctness)

---

## Observability & Debugging

### Logging

All deployments emit structured logs at key points:

- **Model initialization:** Weights loaded, quantization applied, compilation status
- **Inference start:** Input schema validation, processing details
- **Inference complete:** Output shape, timing, file paths (if applicable)

### Error Handling

Clear error messages for common issues:

- **Schema violations:** Detailed message listing required/optional fields and types
- **GPU out-of-memory:** Suggestions for reducing batch size or resolution
- **Missing dependencies:** Installation instructions for required libraries

### Output Inspection

- **Text outputs:** Plain UTF-8 files for easy inspection
- **Image outputs:** PNG files with metadata preserved
- **Artifacts:** Automatically persisted for auditing and reproducibility

---
