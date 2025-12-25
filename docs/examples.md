# Usage Examples

This section provides concrete examples for running inference across all available deployments. Examples use **Replicate's Python SDK**, but the same inputs work seamlessly with Docker-only and Cog CLI deployments.

## Text Generation (SmolLM3 Pruna)

### Basic Long-Form Generation

Generate a long-form response with configurable length and reproducibility:

```python
import replicate

input = {
    "seed": 18,
    "prompt": "What are the advantages of Hugging Face for model hosting?",
    "max_new_tokens": 1420
}

output = replicate.run(
    "paragekbote/smollm3-3b-smashed:232b6f87dac025cb54803cfbc52135ab8366c21bbe8737e11cd1aee4bf3a2423",
    input=input
)

# Access the generated file URL
print(output.url)
# Example: https://replicate.delivery/.../output.txt

# Save the output to disk
with open("output.txt", "wb") as file:
    file.write(output.read())
```

This model returns output as a **file artifact**, making it ideal for long-form generation, offline analysis and evaluation pipelines.

### Controlling Output Length

Use `max_new_tokens` to adjust response size based on your latency and content requirements:

```python
input = {
    "prompt": "Summarize the benefits of model quantization.",
    "max_new_tokens": 256,
    "temperature": 0.7
}

output = replicate.run(
    "paragekbote/smollm3-3b-smashed:...",
    input=input
)
```

Shorter token limits are useful for latency-sensitive applications and cost-controlled inference.

### Deterministic Outputs with Seed

For reproducible outputs across runs, explicitly set the seed:

```python
input = {
    "prompt": "Explain INT8 vs INT4 quantization.",
    "seed": 42,
    "max_new_tokens": 512
}

output = replicate.run(
    "paragekbote/smollm3-3b-smashed:...",
    input=input
)

# Same seed + same prompt = same output every time
```

---

## Reasoning (Phi-4 Reasoning Plus Unsloth)

### Structured Reasoning with Step-by-Step Output

Generate explicit, logical reasoning chains:

```python
import replicate

input = {
    "prompt": "Why do planets orbit the sun? Explain step-by-step.",
    "max_new_tokens": 1024,
    "temperature": 0.5,
    "seed": 123
}

output = replicate.run(
    "paragekbote/phi-4-reasoning-plus-unsloth:...",
    input=input
)

print(output)
```

This model excels at:

- Educational explanations with clear logical steps
- Problem-solving with intermediate reasoning
- Analytical tasks requiring interpretability

### Controlling Reasoning Depth

Adjust output length to balance reasoning detail and inference speed:

```python
# Concise reasoning
input = {
    "prompt": "Why is water essential for life?",
    "max_new_tokens": 256
}

# Detailed reasoning
input = {
    "prompt": "Why is water essential for life?",
    "max_new_tokens": 2048
}
```

---

## Text-to-Image (Flux Fast LoRA Hotswap)

### Basic Image Generation

Generate an image from a text prompt:

```python
import replicate

input = {
    "prompt": "Golden sunset over mountain peaks, cinematic lighting",
    "trigger_word": "Cinematic",
    "height": 768,
    "width": 768,
    "num_inference_steps": 28,
    "seed": 42
}

output = replicate.run(
    "paragekbote/flux-fast-lora-hotswap:...",
    input=input
)

print(output)
# Returns: https://replicate.delivery/.../image.png
```

### Style Switching with Trigger Words

Apply different artistic styles using built-in LoRA adapters:

```python
# Photographic style
input = {
    "prompt": "A serene lake reflecting snow-capped mountains",
    "trigger_word": "Photographic"
}

# Anime style
input = {
    "prompt": "A serene lake reflecting snow-capped mountains",
    "trigger_word": "Anime"
}

# Studio Ghibli style
input = {
    "prompt": "A serene lake reflecting snow-capped mountains",
    "trigger_word": "GHIBSKY"
}

output = replicate.run(
    "paragekbote/flux-fast-lora-hotswap:...",
    input=input
)
```

Available styles: `Cinematic`, `Photographic`, `Anime`, `Manga`, `Digital art`, `Pixel art`, `Fantasy art`, `Neonpunk`, `3D Model`, `Painting`, `Animation`, `Illustration`, `GHIBSKY`

### Fine-Tuning Generation Parameters

Control inference quality and speed:

```python
input = {
    "prompt": "A cyberpunk city at night",
    "height": 768,
    "width": 768,
    "num_inference_steps": 50,  # More steps = higher quality, slower
    "guidance_scale": 3.5,       # Higher = stronger prompt adherence
    "seed": 999
}

output = replicate.run(
    "paragekbote/flux-fast-lora-hotswap:...",
    input=input
)
```

---

## Image-to-Image (Flux Fast LoRA Hotswap Img2Img)

### Stylizing an Existing Image

Transform an image while preserving content structure:

```python
import replicate

input = {
    "image": "https://example.com/photo.jpg",
    "prompt": "Transform into Studio Ghibli style",
    "trigger_word": "GHIBSKY",
    "strength": 0.8,
    "num_inference_steps": 28,
    "seed": 42
}

output = replicate.run(
    "paragekbote/flux-fast-lora-hotswap-img2img:...",
    input=input
)

print(output)
```

### Controlling Transformation Strength

The `strength` parameter controls how much the image is modified:

```python
# Subtle style transfer (preserve original look)
input = {
    "image": "url_to_image",
    "prompt": "Make it anime",
    "strength": 0.3
}

# Moderate transformation
input = {
    "image": "url_to_image",
    "prompt": "Make it anime",
    "strength": 0.6
}

# Creative transformation (allow major changes)
input = {
    "image": "url_to_image",
    "prompt": "Make it anime",
    "strength": 0.9
}
```

---

## Multimodal Vision-Language (Gemma Torchao)

### Image Analysis and Question Answering

Ask questions about image content:

```python
import replicate

input = {
    "image": "https://example.com/photo.jpg",
    "prompt": "What objects are visible in this image?",
    "max_new_tokens": 256,
    "temperature": 0.7
}

output = replicate.run(
    "paragekbote/gemma-torchao:...",
    input=input
)

print(output)
# Example output: "The image shows a mountain landscape with..."
```

### Detailed Image Description

Generate comprehensive image descriptions:

```python
input = {
    "image": "https://example.com/photo.jpg",
    "prompt": "Describe this image in detail, including colors, composition and mood.",
    "max_new_tokens": 512,
    "temperature": 0.6
}

output = replicate.run(
    "paragekbote/gemma-torchao:...",
    input=input
)
```

### Visual Reasoning Tasks

Ask analytical questions about images:

```python
input = {
    "image": "https://example.com/chart.png",
    "prompt": "What trend does this chart show? Analyze the data.",
    "max_new_tokens": 1024,
    "temperature": 0.5
}

output = replicate.run(
    "paragekbote/gemma-torchao:...",
    input=input
)
```

---

## Advanced Patterns

### Batch Processing with Multiple Prompts

Generate multiple outputs sequentially:

```python
import replicate

prompts = [
    "A futuristic city at sunset",
    "A cozy bookshelf in warm lighting",
    "An underwater coral reef"
]

outputs = []

for prompt in prompts:
    input = {
        "prompt": prompt,
        "trigger_word": "Photographic"
    }

    output = replicate.run(
        "paragekbote/flux-fast-lora-hotswap:...",
        input=input
    )

    outputs.append(output)

print(f"Generated {len(outputs)} images")
```

### Conditional Logic Based on Model Output

Use text generation outputs to trigger downstream tasks:

```python
import replicate

# Generate analysis
analysis_input = {
    "prompt": "Analyze the impact of quantization on model performance.",
    "max_new_tokens": 512
}

analysis = replicate.run(
    "paragekbote/smollm3-3b-smashed:...",
    input=analysis_input
)

# Parse analysis and trigger image generation
if "memory" in analysis.lower():
    image_input = {
        "prompt": "Visualization of GPU memory optimization",
        "trigger_word": "Digital art"
    }

    image = replicate.run(
        "paragekbote/flux-fast-lora-hotswap:...",
        input=image_input
    )
```

### Error Handling and Retries

Gracefully handle failures with exponential backoff:

```python
import replicate
import time

def run_with_retry(model_uri, input, max_retries=3):
    for attempt in range(max_retries):
        try:
            output = replicate.run(model_uri, input=input)
            return output
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

output = run_with_retry(
    "paragekbote/smollm3-3b-smashed:...",
    input={"prompt": "Test", "max_new_tokens": 256}
)
```

---

## Deployment Portability

The same input structures work across all execution environments without modification:

=== "Replicate SDK"

    ```python
    import replicate

    output = replicate.run(
        "paragekbote/model:version",
        input=input
    )
    ```

=== "Docker"

    ```bash
    curl -X POST http://localhost:5000/predictions \
      -H "Content-Type: application/json" \
      -d '{"input": {...}}'
    ```

=== "Cog CLI"

    ```bash
    cog predict -i prompt="..." -i max_new_tokens=256
    ```

This ensures **portable inference logic** across self-hosted and managed platforms without code changes.

---

## Next Steps

- [API Reference](deployment.md) — Detailed input/output schemas
- [Architecture](architecture.md) — Technical implementation details
- [Quick Start](quickstart.md) — Get up and running in minutes
