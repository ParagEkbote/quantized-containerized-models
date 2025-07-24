import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pruna import SmashConfig, smash


def save_text(output_folder: Path, seed: int, index: int | str, text: str) -> Path:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return Path(output_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the text generation model into memory and prepare pipeline."""
        model_path = "HuggingFaceTB/SmolLM3-3B"

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("Loading text generation model pipeline")

        # Configure smashing
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        # Smash the model
        print("Smashing text generation model...")
        smashed_model = smash(
            model=base_model,
            smash_config=smash_config,
        )
        print("Model smashing complete.")

        # Create text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=smashed_model,
            tokenizer=self.tokenizer,
            device=0,
        )
        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for text generation"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate", default=128
        ),
        temperature: float = Input(
            description="Sampling temperature", default=1.0
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling", default=0.95
        ),
        seed: int = Input(description="Seed for reproducibility", default=-1),
    ) -> Path:
        """Run a single prediction on the text generation model."""

        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)

        # Generate text using the pipeline
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        generated_text = outputs[0]["generated_text"]

        # Save output text to file
        output_dir = Path(tempfile.mkdtemp())
        text_path = save_text(
            output_folder=output_dir,
            seed=seed,
            index=0,
            text=generated_text,
        )

        return text_path
