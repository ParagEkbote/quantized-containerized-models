import uuid
from pathlib import Path

from cog import BasePredictor, Input
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from PIL import Image
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Step 1: Load token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Step 2: Login to HF
if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in the .env file.")
login(token=hf_token)


def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            quantization_config=PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )).to("cuda")

        self.pipe.enable_lora_hotswap(target_rank=8)

        # Load default adapter
        self.pipe.load_lora_weights(
            "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="open-image-preferences",
            hotswap=True
        )

        self.second_lora_loaded = False

        self.lora1_triggers = [
            "Cinematic", "Photographic", "Anime", "Manga", "Digital art",
            "Pixel art", "Fantasy art", "Neonpunk", "3D Model",
            "Painting", "Animation", "Illustration"
        ]
        self.lora2_triggers = ["GHIBSKY"]

    def predict(
        self,
        prompt: str = Input(description="The text prompt to generate the image from."),
        trigger_word: str = Input(description="Style keyword that triggers a specific LoRA.")
    ) -> Path:

        # Adapter switching
        if trigger_word in self.lora2_triggers and not self.second_lora_loaded:
            self.pipe.load_lora_weights(
                "aleksa-codes/flux-ghibsky-illustration",
                weight_name="lora_v2.safetensors",
                adapter_name="flux-ghibsky",
                hotswap=True
            )
            self.second_lora_loaded = True

        elif trigger_word in self.lora1_triggers and self.second_lora_loaded:
            self.pipe.load_lora_weights(
                "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="open-image-preferences",
                hotswap=True
            )
            self.second_lora_loaded = False

        pipe_kwargs = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "max_sequence_length": 512,
        }

        image = self.pipe(**pipe_kwargs).images[0]
        return save_image(image)