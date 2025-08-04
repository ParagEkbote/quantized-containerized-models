import tempfile
from pathlib import Path as StdPath
from typing import Optional

import torch
from cog import BasePredictor, Input, Path
from pruna import SmashConfig, smash
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_text(output_folder: StdPath, seed: int, index: int | str, text: str) -> StdPath:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load and smash the text generation model."""
        model_path = "HuggingFaceTB/SmolLM3-3B"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Smash config
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        print("Smashing text generation model...")
        self.smashed_text_model = smash(base_model, smash_config)

        self.cache_length = 2048
        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for text generation"),
        max_new_tokens: str = Input(description="Maximum number of new tokens to generate", default=128),
        temperature: str = Input(description="Sampling temperature", default=1.0),
        top_p: str = Input(description="Top-p (nucleus) sampling", default=0.95),
        seed: str = Input(description="Random seed (-1 for no seed)", default=-1),
    ) -> Path:
        """Generate text and save the result to a .txt file."""

        max_new_tokens = int(max_new_tokens)
        temperature = float(temperature)
        top_p = float(top_p)
        seed = int(seed)

        # Set seed if applicable
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cache_length - max_new_tokens,
        ).to("cuda")

        # Generate tokens
        with torch.no_grad():
            output_ids = self.smashed_text_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )

        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        full_output = prompt + generated_text
        print(full_output)

        # Save output to file
        output_dir = StdPath(tempfile.mkdtemp())
        text_path = save_text(output_folder=output_dir, seed=seed if seed != -1 else 0, index=0, text=full_output)

        return Path(text_path)
