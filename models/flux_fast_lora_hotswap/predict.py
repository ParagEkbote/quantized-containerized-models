import os
import uuid
from pathlib import Path

import torch
from cog import BasePredictor, Input
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image


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


def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        self.pipe = DiffusionPipeline.from_pretrained(
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

        self.pipe.enable_lora_hotswap(target_rank=8)

        # Load both adapters initially(no hotswap for first load)
        self.pipe.load_lora_weights(
            "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="open-image-preferences",
        )

        self.pipe.load_lora_weights(
            "aleksa-codes/flux-ghibsky-illustration",
            weight_name="lora_v2.safetensors",
            adapter_name="flux-ghibsky",  # No hotswap=True here either - this is also first load
        )

        self.current_adapter = "open-image-preferences"

        self.lora1_triggers = [
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
        ]
        self.lora2_triggers = ["GHIBSKY"]

    def predict(
        self,
        prompt: str = Input(description="The text prompt to generate the image from."),
        trigger_word: str = Input(description="Style keyword that triggers a specific LoRA."),
    ) -> Path:

        # Switch adapters based on trigger word
        if trigger_word in self.lora2_triggers and self.current_adapter != "flux-ghibsky":
            self.pipe.set_adapters(["flux-ghibsky"], adapter_weights=[0.8])
            self.current_adapter = "flux-ghibsky"

        elif (
            trigger_word in self.lora1_triggers
            and self.current_adapter != "open-image-preferences"
        ):
            self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])
            self.current_adapter = "open-image-preferences"

        # Compile only once if needed — here assumed every time
        self.pipe.text_encoder = torch.compile(
            self.pipe.text_encoder, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.text_encoder_2 = torch.compile(
            self.pipe.text_encoder_2, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.vae = torch.compile(self.pipe.vae, fullgraph=False, mode="reduce-overhead")

        pipe_kwargs = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "max_sequence_length": 512,
        }

        with torch.no_grad():
            image = self.pipe(**pipe_kwargs).images[0]
            print(f"\n[Prompt]: {prompt} | [Trigger Word]: {trigger_word}")
            print(f"Used memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            return save_image(image)
