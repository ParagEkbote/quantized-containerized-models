import tempfile
import logging
from pathlib import Path

import torch
from cog import BasePredictor, Input
from diffusers import SanaPipeline
from pruna import SmashConfig, smash

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --- Utility Classes ---

class ImageSaver:
    """Utility class to handle saving images."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ImageSaver initialized at directory: {self.output_dir}")

    def save(self, image, seed: int, index: int | str) -> Path:
        output_path = self.output_dir / f"output_{seed!s}_{index}.png"
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        return output_path


class SmashedPipelineLoader:
    """Handles loading and smashing the diffusion pipeline."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def load_pipeline(self):
        logger.info("Loading Sana pipeline...")
        pipeline = SanaPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="balanced",
        )
        logger.debug("Sana pipeline loaded.")
        return pipeline

    def smash_pipeline(self, pipeline):
        logger.info("Applying Pruna smashing optimizations...")
        smash_config = SmashConfig()
        smash_config["quantizer"] = "torchao"
        smash_config._prepare_saving = False

        smashed_pipeline = smash(pipeline, smash_config)
        logger.info("Pipeline successfully smashed.")
        return smashed_pipeline


# --- Main Predictor Class ---

class Predictor(BasePredictor):
    """
    The main predictor class for Cog.
    It sets up the model and defines the prediction logic.
    """
    def setup(self) -> None:
        """Initializes the model and pipeline."""
        logger.info("Setting up the ImageGenerationPredictor...")
        model_path = "Efficient-Large-Model/Sana_600M_512px_diffusers"
        loader = SmashedPipelineLoader(model_path)
        base_pipeline = loader.load_pipeline()
        self.pipeline = loader.smash_pipeline(base_pipeline)
        logger.info("Predictor setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation"),
        negative_prompt: str = Input(description="Things to avoid", default=""),
        seed: int = Input(description="Seed for reproducibility", default=-1),
        guidance_scale: float = Input(
            description="Guidance scale (higher values mean stricter prompt adherence)",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        num_inference_steps: int = Input(description="Number of denoising steps", default=25, ge=1, le=100),
        width: int = Input(description="Image width in pixels", default=512, ge=128, le=2048),
        height: int = Input(description="Image height in pixels", default=512, ge=128, le=2048),
        num_images: int = Input(description="Number of images to generate", default=1, ge=1, le=4),
        output_format: str = Input(description="Output format for the generated image", default="pil", choices=["pil", "np_array"]),
    ) -> Path:
        """Runs a single prediction on the model."""
        logger.info(f"Generating image for prompt: '{prompt}'")


        # Set up the random seed generator for reproducibility
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

        try:
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    output_format=output_format,
                )
            logger.info("Image generation successful.")
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            raise

        # Save and return the path to the first generated image
        image = result.images[0]
        saver = ImageSaver()
        output_seed = seed if seed != -1 else torch.randint(0, 2**32 - 1, (1,)).item()
        return saver.save(image, seed=output_seed, index=0)