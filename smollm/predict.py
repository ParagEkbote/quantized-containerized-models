import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
from pruna import SmashConfig, smash


def save_text(output_folder: Path, seed: int, index: int | str, text: str) -> Path:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return Path(output_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the text generation model into memory."""
        model_path = "HuggingFaceTB/SmolLM3-3B"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        print("Loading text generation model pipeline")

        # Configure smashing
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        # Smash the model and store it
        print("Smashing text generation model...")
        self.smashed_text_model = smash(
            model=base_model,
            smash_config=smash_config,
        )
        
        # Cache length setup
        self.cache_length = 2048
        
        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for text generation"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate", 
            default=128,
            ge=1,
            le=1024
        ),
        temperature: float = Input(
            description="Sampling temperature", 
            default=1.0,
            ge=0.1,
            le=2.0
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling", 
            default=0.95,
            ge=0.1,
            le=1.0
        ),
        seed: int = Input(
            description="Seed for reproducibility", 
            default=-1
        ),
    ) -> str:
        """Run a single prediction on the text generation model."""

        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.cache_length - max_new_tokens
        ).to("cuda")

        # Generate with minimal compatible kwargs
        with torch.no_grad():
            try:
                output_ids = self.smashed_text_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"Generation failed: {e}")
                output_ids = self.smashed_text_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Combine with original prompt
        full_output = prompt + generated_text
        print(full_output)

        # Save to disk
        output_dir = Path(tempfile.mkdtemp())
        text_path = save_text(
            output_folder=output_dir,
            seed=seed if seed != -1 else 0,
            index=0,
            text=full_output,
        )

        # Return as string for JSON serialization compatibility
        return str(text_path)
