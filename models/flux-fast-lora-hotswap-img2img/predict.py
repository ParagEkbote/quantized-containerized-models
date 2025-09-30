import os
import time
import uuid
from io import BytesIO
from pathlib import Path

import requests
import torch
from cog import BasePredictor, Input
from diffusers import FluxImg2ImgPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not found in .env file")


def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


def load_image(init_image: str) -> Image.Image:
    """Load an image from a URL."""
    if init_image.startswith("http://") or init_image.startswith("https://"):
        resp = requests.get(init_image, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(init_image).convert("RGB")


class Predictor(BasePredictor):
    def setup(self):
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            dtype=torch.bfloat16,
            quantization_config=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            ),
        ).to("cuda")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe.enable_lora_hotswap(target_rank=8)

        self.pipe.load_lora_weights(
            "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="open-image-preferences",
        )

        self.pipe.load_lora_weights(
            "aleksa-codes/flux-ghibsky-illustration",
            weight_name="lora_v2.safetensors",
            adapter_name="flux-ghibsky",
        )

        self.current_adapter = "open-image-preferences"

        self.lora1_triggers = {
            "Cinematic",
            "Photographic",
            "Anime",
            "Manga",
            "Digital art",
            "Pixel art",
            "Fantasy art",
            "Neonpunk",
            "3D Model",
            "Painting",
            "Animation",
            "Illustration",
        }
        self.lora2_triggers = {"GHIBSKY"}

        # Always compile key components
        self.pipe.text_encoder = torch.compile(
            self.pipe.text_encoder, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.text_encoder_2 = torch.compile(
            self.pipe.text_encoder_2, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.vae = torch.compile(self.pipe.vae, fullgraph=False, mode="reduce-overhead")

    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation."),
        trigger_word: str = Input(description="Trigger word to select LoRA."),
        init_image: str = Input(description="Initial image (URL)."),
        strength: float = Input(
            description="Strength of transformation.", default=0.6, ge=0, le=1
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale follows the text prompt.",
            default=7.5,
            ge=0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps for image generation.", default=28, ge=1
        ),
        seed: int = Input(description="Random seed.", default=42),
    ) -> Path:
        # Pick adapter
        if trigger_word in self.lora2_triggers and self.current_adapter != "flux-ghibsky":
            self.pipe.set_adapters(["flux-ghibsky"], adapter_weights=[0.8])
            self.current_adapter = "flux-ghibsky"
        elif (
            trigger_word in self.lora1_triggers
            and self.current_adapter != "open-image-preferences"
        ):
            self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])
            self.current_adapter = "open-image-preferences"

        # Load init image
        image = load_image(init_image)

        # Generation
        pipe_kwargs = dict(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.manual_seed(seed) if seed is not None else None,
        )

        start_time = time.time()
        with torch.no_grad():
            output = self.pipe(**pipe_kwargs).images[0]
        elapsed = time.time() - start_time

        print(f"[Prompt]: {prompt} | [Trigger Word]: {trigger_word}")
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Time: {elapsed:.2f}s")

        return save_image(output)
