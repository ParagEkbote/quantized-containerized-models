import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input  # DO NOT import Path from cog (shadowing)

from pruna import SmashConfig, smash   # or pruna, whichever is correct in your env
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_text(output_folder: Path, seed: int, index: int | str, text: str) -> Path:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    # ensure parent exists (should already, since we create tmpdir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the text generation model into memory."""
        model_path = "HuggingFaceTB/SmolLM3-3B"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use device_map="auto" unless you know the exact mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        print("Loading text generation model pipeline")

        # Configure smashing (adjust to actual API of pruna_pro/pruna)
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        self.smashed_text_model = smash(model=base_model, smash_config=smash_config)

        # Cache length setup
        self.cache_length = 2048
        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for text generation"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate", default=512, ge=1, le=16384
        ),
        mode: str = Input(description="Reasoning mode: 'think' or 'no_think'", default="no_think"),
        seed: int = Input(description="Seed for reproducibility", default=-1),
    ) -> Path:
        """Run a single prediction on the text generation model."""


        # Validate lengths
        max_input_length = max(1, self.cache_length - max_new_tokens)

        messages = [
        {"role": "system", "content": f"/{mode}"},
        {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        )

        # Tokenize and move tensors to CUDA
        inputs = self.tokenizer(
            text,
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        device = next(self.smashed_text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
                output_ids = self.smashed_text_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, 
                )

        # Decode only the newly generated tokens
        start_idx = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            output_ids[0][start_idx:], skip_special_tokens=True
        )

        full_output = prompt + generated_text
        print(full_output)

        # Save to disk
        output_dir = Path(tempfile.mkdtemp())
        text_path = save_text(
            output_folder=output_dir,
            seed=(seed if seed != -1 else 0),
            index=0,
            text=full_output,
        )

        return text_path
