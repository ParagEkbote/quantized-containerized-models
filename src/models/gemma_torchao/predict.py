import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import requests
import torch
import torch.nn as nn
from cog import BasePredictor, Input
from PIL import Image
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from transformers import AutoModelForImageTextToText, AutoProcessor


def login_with_env_token(env_var: str = "HF_TOKEN") -> None:
    """
    Load the Hugging Face token from the environment and log in.

    Args:
        env_var (str): The environment variable name holding the token.

    Raises:
        ValueError: If the token is not found in the environment.
    """
    load_dotenv()  # loads variables from .env file into environment
    hf_token: str | None = os.getenv(env_var)

    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError(f"{env_var} not found in .env file or environment")


login_with_env_token()

# ------------------------
# Save output utility
# ------------------------
def save_output_to_file(
    text: str,
    output_folder: Path = Path("outputs"),
    seed: int | None = None,
    index: int | str | None = None,
    filename: str | None = None,
) -> Path:
    """
    Save text to a file inside the given output folder.
    - If filename is provided, use that.
    - Else, generate one with seed and index if provided.
    - If neither provided, fall back to a timestamp-based filename.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    if filename is not None:
        output_path = output_folder / filename
    elif seed is not None and index is not None:
        output_path = output_folder / f"output_{seed}_{index}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_folder / f"output_{timestamp}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


# ------------------------
# Safe sparsity utilities
# ------------------------
def gemma_filter_fn(module: nn.Module, full_name: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    name = full_name.lower()
    if any(skip in name for skip in ["embed", "lm_head", "output", "norm", "layernorm"]):
        return False
    target_layers = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    return any(target in name for target in target_layers)


def torchao_filter_fn(module: nn.Module, full_name: str) -> bool:
    """
    Return True if this module should be quantized.
    Gemma-3 requires skipping embeddings, lm_head, and norms.
    """
    name = full_name.lower()

    if not isinstance(module, nn.Linear):
        return False

    if any(
        skip in name
        for skip in [
            "embed_tokens",
            "embedding",
            "lm_head",
            "norm",
            "layernorm",
        ]
    ):
        return False

    return True


def magnitude_based_pruning(model, sparsity_ratio=0.5, filter_fn=None):
    print(f"\n[Magnitude Pruning] Target sparsity: {sparsity_ratio:.2%}")
    with torch.no_grad():
        for name, module in model.named_modules():
            if filter_fn and not filter_fn(module, name):
                continue
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                weight = module.weight.data
                flat_weights = weight.abs().flatten().float()
                try:
                    k = int(flat_weights.numel() * (1 - sparsity_ratio))
                    if k == 0:
                        continue
                    topk_vals, _ = torch.topk(flat_weights, k)
                    threshold = topk_vals[-1]
                    mask = weight.abs() >= threshold
                    weight.mul_(mask)
                    module.register_buffer("sparse_mask", mask)
                    sparsity_achieved = (weight == 0).float().mean().item()
                    print(f"Layer {name}: {sparsity_achieved:.2%} sparsity achieved")
                except RuntimeError as e:
                    print(f"Layer {name}: Skipping sparsity due to large tensor ({e})")


def gradual_magnitude_pruning(
    model,
    target_sparsity: float = 0.5,
    current_step: int = 10,
    total_steps: int = 1000,
    filter_fn=None,
):
    current_sparsity = target_sparsity * min(current_step / total_steps, 1.0)
    progress_percent = min(current_step / total_steps * 100, 100)
    print(f"\n[Gradual Pruning] Step {current_step}/{total_steps} ({progress_percent:.1f}%) - Current sparsity: {current_sparsity:.2%}")

    total_pruned = 0
    total_weights = 0

    for name, module in model.named_modules():
        if filter_fn and not filter_fn(module, name):
            continue
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            total_weights += weight.numel()

            k = int(weight.numel() * current_sparsity)
            if k > 0:
                threshold, _ = torch.topk(weight.abs().view(-1), k, largest=False)
                mask = weight.abs() >= threshold[-1]
                num_pruned = weight.numel() - mask.sum().item()
                weight *= mask.view_as(weight)
                total_pruned += num_pruned
                print(f"  Layer '{name}': pruned {num_pruned}/{weight.numel()} weights ({num_pruned / weight.numel():.2%})")

    actual_sparsity = total_pruned / total_weights if total_weights > 0 else 0
    print(f"  Total progress: {total_pruned}/{total_weights} weights pruned ({actual_sparsity:.2%})")
    return current_sparsity


def layer_norm_pruning(model, sparsity_ratio: float = 0.3, filter_fn=None):
    """
    Fast layer-norm based pruning: prune weights with lowest L2 norms per layer
    Very fast, no forward pass needed, preserves output quality well
    """
    print(f"\n[Layer Norm Pruning] Target sparsity: {sparsity_ratio:.2%}")
    print("Analyzing weight norms (fast method)...")

    # Calculate importance scores for all layers
    layer_importance = {}

    for name, module in model.named_modules():
        if filter_fn and not filter_fn(module, name):
            continue
        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            weight = module.weight.data
            # Use mean L2 norm as layer importance
            layer_importance[name] = weight.norm(p=2).item() / weight.numel()

    print(f"Analyzed {len(layer_importance)} layers")

    # Prune uniformly across all layers based on magnitude within each layer
    total_pruned = 0
    total_params = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if filter_fn and not filter_fn(module, name):
                continue
            if isinstance(module, nn.Linear) and hasattr(module, "weight"):
                weight = module.weight.data
                total_params += weight.numel()

                # Prune lowest magnitude weights in this layer
                k = int(weight.numel() * sparsity_ratio)
                if k > 0:
                    flat_weights = weight.abs().flatten()
                    threshold, _ = torch.topk(flat_weights, k, largest=False)
                    mask = weight.abs() >= threshold[-1]
                    weight *= mask.view_as(weight)

                    pruned_count = (weight == 0).sum().item()
                    total_pruned += pruned_count

                    layer_norm = layer_importance.get(name, 0)
                    print(f"  Layer '{name}' (norm: {layer_norm:.6f}): pruned {pruned_count}/{weight.numel()} weights")

    actual_sparsity = total_pruned / total_params if total_params > 0 else 0
    print(f"  Total pruned: {total_pruned}/{total_params} weights ({actual_sparsity:.2%})")
    return actual_sparsity


def apply_safe_sparsity(model, sparsity_type="magnitude", sparsity_ratio=0.3):
    print(f"\nApplying {sparsity_type} sparsity with ratio {sparsity_ratio}")

    if sparsity_type == "magnitude":
        magnitude_based_pruning(model, sparsity_ratio, filter_fn=gemma_filter_fn)
    elif sparsity_type == "gradual":
        gradual_magnitude_pruning(
            model,
            target_sparsity=sparsity_ratio,
            current_step=500,  # Simulate middle of training
            total_steps=1000,
            filter_fn=gemma_filter_fn,
        )
    elif sparsity_type == "layer_norm":
        layer_norm_pruning(model, sparsity_ratio=sparsity_ratio, filter_fn=gemma_filter_fn)
    else:
        print(f"Unknown sparsity type: {sparsity_type}")
        return 0.0

    # Calculate overall sparsity
    total_params = sum(p.numel() for p in model.parameters())
    sparse_params = sum((p == 0).sum().item() for p in model.parameters())
    overall_sparsity = sparse_params / total_params
    print(f"Overall model sparsity: {overall_sparsity:.2%}")
    return overall_sparsity


# ------------------------
# Quantization helper
# ------------------------
def sanitize_weights_for_quantization(model: torch.nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight
            if w.is_meta:
                continue
            if hasattr(w, "__sparse_coo_tensor_unsafe__") or "SparseSemiStructured" in str(type(w)):
                continue
            try:
                new_w = w.detach().contiguous()
                new_w.requires_grad = w.requires_grad
                module._parameters["weight"] = nn.Parameter(new_w)
            except Exception as e:
                print(f"Warning: Could not sanitize weight for {name}: {e}")


# ------------------------
# Chat formatting
# ------------------------
def format_chat_messages(prompt: str, image_url: str | None = None):
    system_message = "You are a helpful assistant."
    messages = [{"role": "system", "content": [{"type": "text", "text": system_message}]}]

    user_content = []
    if image_url:
        user_content.append({"type": "image", "url": image_url})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


# ------------------------
# Predictor
# ------------------------
class Predictor(BasePredictor):
    def setup(self):
        hf_token = os.environ.get("HF_TOKEN")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

        MODEL_ID = "google/gemma-3-4b-it"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            token=hf_token,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def add_sparsity(self, sparsity_type="magnitude", sparsity_ratio=0.3):
        if sparsity_ratio > 0:
            try:
                overall_sparsity = apply_safe_sparsity(self.model, sparsity_type, sparsity_ratio)
                print(f"Successfully applied {sparsity_type} sparsity: {overall_sparsity:.2%}")
            except Exception as e:
                print(f"Warning: Sparsity application failed: {e}")
                print("Continuing without sparsity...")

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        image_url: str | None = Input(description="Optional image URL", default=None),
        max_new_tokens: int = Input(default=128, ge=1, le=2500, description="Maximum number of new tokens"),
        temperature: float = Input(default=0.7, description="Sampling temperature", ge=0.0, le=2.0),
        top_p: float = Input(default=0.9, description="Top-p nucleus sampling", ge=0.0, le=1.0),
        seed: int = Input(default=42, description="Seed for reproducibility"),
        use_quantization: str = Input(default="true", description="Enable INT8 quantization using torchao"),
        use_sparsity: str = Input(default="false", description="Enable sparsity optimization"),
        sparsity_type: str = Input(default="magnitude", description="Type of sparsity: magnitude, gradual, layer_norm"),
        sparsity_ratio: float = Input(default=0.3, ge=0.0, le=0.4),
    ) -> str:
        torch.manual_seed(seed)

        use_quantization_flag = str(use_quantization).strip().lower() in {"true", "1", "yes", "y"}
        use_sparsity_flag = str(use_sparsity).strip().lower() in {"true", "1", "yes", "y"}

        # Apply optimizations
        if use_sparsity_flag and sparsity_ratio > 0:
            self.add_sparsity(sparsity_type, sparsity_ratio)

        if use_quantization_flag:
            try:
                sanitize_weights_for_quantization(self.model)
                quantize_(
                    self.model,
                    Int8WeightOnlyConfig(),
                    filter_fn=torchao_filter_fn,
                )
                print("Quantization applied successfully")
            except Exception as e:
                print(f"Warning: Quantization failed: {e}")

        # Load image if provided
        image = None
        if image_url:
            try:
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                print(f"Image loaded successfully from {image_url}")
            except Exception as e:
                print(f"Warning: Failed to load image from {image_url}: {e}")

        # Format messages
        messages = format_chat_messages(prompt, image_url if image else None)
        try:
            formatted_prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Warning: Chat template failed, using raw prompt: {e}")
            formatted_prompt = prompt

        # Process inputs
        try:
            inputs = self.processor(
                text=formatted_prompt,
                images=image if image else None,
                return_tensors="pt",
            ).to(self.model.device)
        except Exception as e:
            print(f"Error processing inputs: {e}")
            inputs = self.processor(
                text=prompt,
                images=image if image else None,
                return_tensors="pt",
            ).to(self.model.device)

        do_sample = temperature > 0.0
        # Generate response
        start = time.time()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
        except Exception as e:
            print(f"Generation failed: {e}")
            raise

        elapsed = time.time() - start
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"VRAM used: {vram:.2f} GB | Time: {elapsed:.2f}s")

        # Decode output
        try:
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            final_output = generated_text.strip() if generated_text.strip() else decoded
        except Exception as e:
            print(f"Decoding error: {e}")
            final_output = str(outputs)

        # Save output to file
        try:
            file_path = save_output_to_file(final_output, seed=seed, index=0)
            print(f"Saved output persistently to {file_path}")
        except Exception as e:
            print(f"Warning: Could not save output to file: {e}")

        return final_output
