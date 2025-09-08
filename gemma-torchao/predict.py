import time
import torch
import requests
from io import BytesIO
from PIL import Image
from cog import BasePredictor, Input
from transformers import AutoModelForImageTextToText, AutoProcessor
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torchao.sparsity import sparsify_
from torchao.sparse_api import semi_sparse_weight
import torch.nn as nn

MODEL_ID = "google/gemma-3-12b-it"

def filter_fn(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)

def save_output_to_file(text: str, filename: str = None) -> str:
    """Save model output to a text file. Default: timestamped filename."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

class Predictor(BasePredictor):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        quantize_(model, Int8WeightOnlyConfig())
        self.model = model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        image_url: str = Input(description="Optional image URL", default=None),
        max_new_tokens: int = Input(default=128, ge=1, le=1024),
        temperature: float = Input(default=0.7,description="Sampling temperature"),
        top_p: float = Input(default=0.9, description="Top-p nucleus sampling"),
        seed: int = Input(default=42,description="Random seed for reproducibility"),
        use_sparsity: bool = Input(default=False, description="Enable 2:4 semi-sparsity on Linear layers"),
    ) -> str:
        torch.manual_seed(seed)

        if use_sparsity:
            self.model = sparsify_(self.model, semi_sparse_weight(), filter_fn).eval()

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

        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Time: {elapsed:.2f}s")

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        filename = save_output_to_file(decoded)
        print(f"Output saved to {filename}")

        return decoded
