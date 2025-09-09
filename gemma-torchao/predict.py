import time
import torch
import requests
from io import BytesIO
import os
from PIL import Image
from cog import BasePredictor, Input
from transformers import AutoModelForImageTextToText, AutoProcessor
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torchao.sparsity import sparsify_, semi_sparse_weight
import torch.nn as nn
from huggingface_hub import login
from datetime import datetime
from dotenv import load_dotenv

# Load HF_TOKEN from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not found in .env file")

def filter_fn(module: nn.Module, full_name: str) -> bool:
    """
    Decide whether to sparsify a module.

    Rules:
    - Only sparsify nn.Linear layers.
    - Sparsify QKV projections and MLP feed-forward layers.
    - Skip embeddings and output heads to preserve accuracy.
    - Skip LayerNorms and other sensitive layers.
    """
    if not isinstance(module, nn.Linear):
        return False

    name = full_name.lower()

    if "embed" in name or "lm_head" in name or "output_projection" in name:
        return False
    if "norm" in name or "layernorm" in name:
        return False
    if "self_attn" in name or "attn" in name:
        return True
    if "mlp" in name or "fc" in name:
        return True

    return False

def save_output_to_file(text: str, filename: str = None) -> str:
    """Save model output to a text file. Defaults to timestamped filename if none provided."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

def sanitize_weights_for_quantization(model: torch.nn.Module):
    """
    Prepare weights so that quantization doesn't hit view/aliasing errors.
    Special-cases sparse semi-structured tensors that don't support .clone().
    """
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
            w = m.weight
            if w.is_meta:
                continue

            # Handle sparse semi-structured weights
            if w.__class__.__name__ == "SparseSemiStructuredTensorCUSPARSELT":
                new_w = w.detach()  # no clone, keep sparse structure
            else:
                new_w = w.detach().clone().contiguous()

            new_w.requires_grad = w.requires_grad
            m._parameters["weight"] = torch.nn.Parameter(new_w)


class Predictor(BasePredictor):
    def setup(self):
        MODEL_ID = "google/gemma-3-4b-it"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        image_url: str = Input(description="Optional image URL"),
        max_new_tokens: int = Input(default=128, ge=1, le=1024),
        temperature: float = Input(default=0.7, description="Sampling temperature"),
        top_p: float = Input(default=0.9, description="Top-p nucleus sampling"),
        seed: int = Input(default=42, description="Random seed for reproducibility"),
        use_sparsity: str = Input(default="false", description="Enable 2:4 semi-sparsity on Linear layers ('true'/'false')"),
    ) -> str:
        torch.manual_seed(seed)

        # Convert string to boolean
        use_sparsity_flag = str(use_sparsity).strip().lower() in {"true", "1", "yes", "y"}

        # Apply sparsity first if requested
        if use_sparsity_flag:
            sparsify_(self.model, semi_sparse_weight(), filter_fn)

        # Apply quantization after sparsity
        sanitize_weights_for_quantization(self.model)
        quantize_(self.model, Int8WeightOnlyConfig())

        # Load image if provided
        image = None
        if image_url:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image if image else None,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.time()
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
        elapsed = time.time() - start

        try:
            vram = torch.cuda.memory_allocated() / 1e9
        except Exception:
            vram = 0.0
        print(f"VRAM used: {vram:.2f} GB | Time: {elapsed:.2f}s")

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        filename = save_output_to_file(decoded)
        print(f"Output saved to {filename}")

        return decoded
