import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input
from diffusers import SanaPipeline
from pruna import SmashConfig, smash


def save_image(output_folder: Path, seed: int, index: int | str, image) -> Path:
    """Save the generated image to disk as a .png file."""
    output_path = output_folder / f"output_{seed!s}_{index}.png"
    image.save(output_path)
    return Path(output_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the text-to-image generation model into memory."""
        model_path = "Efficient-Large-Model/Sana_600M_512px"

        print("Loading text-to-image generation model pipeline")

        # Load base Stable Diffusion pipeline
        base_pipeline = SanaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        # Configure smashing with more explicit settings
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq_diffusers"
        smash_config["factorizer"] = "qkv_diffusers"
        smash_config["cacher"] = "deepcache"
        smash_config._prepare_saving = False

        # Smash the pipeline and store it
        print("Smashing text-to-image model...")
        self.smashed_image_pipeline = smash(
            model=base_pipeline,
            smash_config=smash_config,
        )

        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation"),
        seed: int = Input(
            description="Seed for reproducibility",
            default=-1
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale",
            default=7.5,
            ge=1.0,
            le=20.0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=25,
            ge=1,
            le=100,
        ),
        negative_prompt: str = Input(
            description="What to avoid in the image (e.g., 'blurry, dark, low quality')",
            default="",
        ),
        width: int = Input(
            description="Width of the generated image (in pixels)",
            default=512,
            ge=128,
            le=2048,
        ),
        height: int = Input(
            description="Height of the generated image (in pixels)",
            default=512,
            ge=128,
            le=2048,
        ),
        num_images: int = Input(
            description="How many images to generate from the same prompt",
            default=1,
            ge=1,
            le=4,
        ),
        output_format: str = Input(
            description="Choose image output type",
            default="pil",
            choices=["pil", "np_array"],
        ),
    ) -> Path:
        """Run a single prediction on the text-to-image model."""

        # Set seed for reproducibility
        if seed != -1:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        # Generate image
        with torch.no_grad():
            image = self.smashed_image_pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images=num_images,
                output_format=output_format
            ).images[0]

        # Create output directory and save the image
        output_dir = Path(tempfile.mkdtemp())
        image_path = save_image(
            output_folder=output_dir,
            seed=seed if seed != -1 else 0,
            index=0,
            image=image,
        )

        return image_path
