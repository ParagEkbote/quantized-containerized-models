# 🧪 Usage Examples

This section provides concrete examples for running inference using the available deployments.
All examples below use **Replicate’s Python SDK**, but the same inputs can apply to Docker-only and Cog 
cli deployments.

---

## Example 1: Basic Text Generation (SmolLM3 Pruna)

This example runs the **smollm3-3b-smashed** deployment to generate a long-form response.

### Python Example

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

This model returns its output as a **file artifact**, making it suitable for:

* Long-form generation
* Offline analysis
* Logging and evaluation pipelines

---

## Example 2: Controlling Output Length

Use `max_new_tokens` to control response size.

```python
input = {
    "prompt": "Summarize the benefits of model quantization.",
    "max_new_tokens": 256
}
```

This is useful for:

* Latency-sensitive applications
* Short answers or summaries
* Cost-controlled inference

---

## Example 3: Deterministic Outputs with Seed

For reproducible outputs, explicitly set the seed.

```python
input = {
    "prompt": "Explain INT8 vs INT4 quantization.",
    "seed": 42,
    "max_new_tokens": 512
}
```

Setting the same seed with the same inputs yields deterministic results across runs.

---

## Example 4: Reasoning vs No-Reasoning Mode

Some deployments expose a `mode` flag to control reasoning behavior.

```python
input = {
    "prompt": "Derive the time complexity of binary search.",
    "mode": "think",
    "max_new_tokens": 512
}
```

Available modes:

* `"think"` – Enables explicit reasoning steps
* `"no_think"` (default) – Produces concise final answers

---

## Notes on Deployment Flexibility

The same inputs work across:

* Replicate-hosted endpoints
* Local Docker deployments
* Cog-based CLI inference

This ensures **portable inference logic** without rewriting client code.

---

## Next Examples

* Image generation and Img2Img deployments → see model-specific docs
* Integration with evaluation pipelines

For endpoint-level details, refer to the [API Reference](deployment.md).
