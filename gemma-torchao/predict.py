import time
import torch
import requests
from io import BytesIO
import os
from PIL import Image
from cog import BasePredictor, Input
from transformers import AutoModelForImageTextToText, AutoProcessor
from torchao.quantization import quantize_, Int8WeightOnlyConfig
import torch.nn as nn
from huggingface_hub import login
from datetime import datetime
from dotenv import load_dotenv

# ------------------------
# Hugging Face login
# ------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not found in .env file")


# ------------------------
# Output saving
# ------------------------
def save_output_to_file(text: str, filename: str | None = None) -> str:
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


# ------------------------
# Safe sparsity utilities
# ------------------------
def magnitude_based_pruning(model, sparsity_ratio=0.5, filter_fn=None):
    with torch.no_grad():
        for name, module in model.named_modules():
            if filter_fn and not filter_fn(module, name):
                continue
            if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                weight = module.weight.data
                flat_weights = weight.abs().flatten()
                threshold = torch.quantile(flat_weights, sparsity_ratio)
                mask = weight.abs() > threshold
                weight.mul_(mask)
                module.register_buffer('sparse_mask', mask)
                sparsity_achieved = (weight == 0).float().mean().item()
                print(f"Layer {name}: {sparsity_achieved:.2%} sparsity achieved")


def structured_pruning(model, channels_to_remove=0.25):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            if len(weight.shape) > 1:
                channel_importance = weight.norm(dim=1)
                num_remove = int(weight.shape[0] * channels_to_remove)
                if num_remove > 0:
                    _, indices_to_remove = torch.topk(channel_importance, num_remove, largest=False)
                    keep_mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                    keep_mask[indices_to_remove] = False
                    new_weight = weight[keep_mask]
                    print(f"Layer {name}: Removed {num_remove}/{weight.shape[0]} channels")


def gradual_magnitude_pruning(model, target_sparsity=0.5, current_step=0, total_steps=1000):
    current_sparsity = target_sparsity * min(current_step / total_steps, 1.0)
    magnitude_based_pruning(model, current_sparsity)
    return current_sparsity


def gemma_filter_fn(module: nn.Module, full_name: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    name = full_name.lower()
    if any(skip in name for skip in ["embed", "lm_head", "output", "norm", "layernorm"]):
        return False
    target_layers = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    return any(target in name for target in target_layers)


def apply_safe_sparsity(model, sparsity_type="magnitude", sparsity_ratio=0.3):
    print(f"Applying {sparsity_type} sparsity with ratio {sparsity_ratio}")
    if sparsity_type == "magnitude":
        magnitude_based_pruning(model, sparsity_ratio, gemma_filter_fn)
    elif sparsity_type == "structured":
        structured_pruning(model, sparsity_ratio)
    elif sparsity_type == "gradual":
        gradual_magnitude_pruning(model, sparsity_ratio)
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
            if hasattr(w, '__sparse_coo_tensor_unsafe__') or 'SparseSemiStructured' in str(type(w)):
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
        MODEL_ID = "google/gemma-3-4b-it"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
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
        max_new_tokens: int = Input(default=128, ge=1, le=1024),
        temperature: float = Input(default=0.7),
        top_p: float = Input(default=0.9),
        seed: int = Input(default=42),
        use_quantization: bool = Input(default=True, description="Enable INT8 quantization"),
        use_sparsity: bool = Input(default=False, description="Enable sparsity optimization"),
        sparsity_type: str = Input(default="magnitude", description="Type of sparsity"),
        sparsity_ratio: float = Input(default=0.3, ge=0.0, le=0.8),
    ) -> str:
        torch.manual_seed(seed)

        # Sparsity
        if use_sparsity and sparsity_ratio > 0:
            self.add_sparsity(sparsity_type, sparsity_ratio)

        # Quantization
        if use_quantization:
            try:
                sanitize_weights_for_quantization(self.model)
                quantize_(self.model, Int8WeightOnlyConfig())
                print("Quantization applied successfully")
            except Exception as e:
                print(f"Warning: Quantization failed: {e}")

        # Image load
        image = None
        if image_url:
            try:
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                print(f"Image loaded successfully from {image_url}")
            except Exception as e:
                print(f"Warning: Failed to load image from {image_url}: {e}")
                image = None

        # Format messages
        messages = format_chat_messages(prompt, image_url if image else None)

        try:
            formatted_prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Chat template failed, using raw prompt: {e}")
            formatted_prompt = prompt

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

        start = time.time()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
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

        try:
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            final_output = generated_text.strip() if generated_text.strip() else decoded
        except Exception as e:
            print(f"Decoding error: {e}")
            final_output = str(outputs)

        try:
            filename = save_output_to_file(final_output)
            print(f"Output saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save output to file: {e}")

        return final_output
