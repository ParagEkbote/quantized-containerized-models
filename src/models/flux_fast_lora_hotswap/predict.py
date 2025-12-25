import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from cog import BasePredictor, Input
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
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


login_with_env_token()

def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    """
    Function to save the generated image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            ),
        )

        self.pipe.transformer.set_attention_backend("flash_hub")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Enable LoRA hotswap for efficient adapter switching
        self.pipe.enable_lora_hotswap(target_rank=8)

        # Load both adapters with hotswap enabled
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
        self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])

        # Trigger word mappings as sets for O(1) lookup
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

        # Compile modules with dynamic=True to handle variable shapes
        self._compile_modules()

        self._compiled = False

    def _compile_modules(self) -> None:
        """Compile key pipeline modules with dynamic shape support."""
        try:
            self.pipe.text_encoder = torch.compile(
                self.pipe.text_encoder,
                fullgraph=False,
                mode="reduce-overhead",
                dynamic=True,
            )
            self.pipe.text_encoder_2 = torch.compile(
                self.pipe.text_encoder_2,
                fullgraph=False,
                mode="reduce-overhead",
                dynamic=True,
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae,
                fullgraph=False,
                mode="reduce-overhead",
                dynamic=True,
            )
            self._compiled = True
            print("[Setup] torch.compile() applied with dynamic=True")
        except Exception as e:
            print(f"[Warning] torch.compile() failed: {e}. Continuing without compilation.")
            self._compiled = False

    def _switch_adapter(self, trigger_word: str) -> None:
        """
        Switch LoRA adapter based on trigger word.
        Only switches if different from current adapter (efficient).
        """
        target_adapter = None
        adapter_weight = 1.0

        if trigger_word in self.lora2_triggers:
            target_adapter = "flux-ghibsky"
            adapter_weight = 0.8
        elif trigger_word in self.lora1_triggers:
            target_adapter = "open-image-preferences"
            adapter_weight = 1.0

        # Only switch if target adapter differs from current
        if target_adapter and self.current_adapter != target_adapter:
            self.pipe.set_adapters([target_adapter], adapter_weights=[adapter_weight])
            self.current_adapter = target_adapter
            print(f"[Adapter] Switched to: {target_adapter}")

    def predict(
        self,
        prompt: str = Input(description="The text prompt to generate the image from."),
        trigger_word: str = Input(description="Style keyword that triggers a specific LoRA."),
        height: int = Input(description="Image height.", default=1024, ge=512, le=2048),
        width: int = Input(description="Image width.", default=1024, ge=512, le=2048),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale, higher leads to creative images.",
            default=3.5,
            ge=0,
            le=20,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps.",
            default=40,
            ge=1,
            le=250,
        ),
    ) -> Path:
        # Switch LoRA adapter efficiently (only if needed)
        self._switch_adapter(trigger_word)

        # Build generation kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": 512,
        }

        # Generate image
        with torch.no_grad():
            image = self.pipe(**pipe_kwargs).images[0]

        # Log metrics
        print(f"[Prompt]: {prompt}")
        print(f"[Trigger Word]: {trigger_word}")
        print(f"[Adapter]: {self.current_adapter}")
        print(f"[VRAM Used]: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return save_image(image)
