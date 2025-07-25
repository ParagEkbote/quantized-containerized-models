import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from pruna import SmashConfig, smash


def save_text(output_folder: Path, seed: int, index: int | str, text: str) -> Path:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return Path(output_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the audio-text-to-text generation model into memory."""
        model_path = "mistralai/Voxtral-Mini-3B-2507"

        print("Loading audio-text-to-text model pipeline")

        # Load processor (for both audio and text inputs) and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Configure smashing with explicit settings
        smash_config = SmashConfig()
        smash_config["quantizer"] = "hqq"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False

        # Smash the model and store it
        print("Smashing audio-text-to-text model...")
        self.smashed_model = smash(
            model=base_model,
            smash_config=smash_config,
        )

        print("Setup complete.")

    def predict(
        self,
        audio_path: Path = Input(description="Path to input audio file (.wav)"),
        prompt: str = Input(description="Instruction or prompt for generation"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=128,
            ge=1,
            le=1024,
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=1.0,
            ge=0.1,
            le=2.0,
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling",
            default=0.95,
            ge=0.1,
            le=1.0,
        ),
        seed: int = Input(
            description="Seed for reproducibility",
            default=-1,
        ),
    ) -> Path:
        """Run a single prediction on the audio-text-to-text model."""

        # Load and process audio
        audio_inputs = self.processor(
            audio_path,
            return_tensors="pt",
            sampling_rate=16000,  # adjust if model requires specific rate
        )

        # Process text prompt as input_ids if model uses both
        text_inputs = self.processor(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Merge inputs based on model requirement (example concatenation)
        inputs = {**audio_inputs, **text_inputs}
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        # Generate output text
        with torch.no_grad():
            output_ids = self.smashed_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                generator=generator,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode generated tokens
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        # Create output directory and save the text
        output_dir = Path(tempfile.mkdtemp())
        text_path = save_text(
            output_folder=output_dir,
            seed=seed if seed != -1 else 0,
            index=0,
            text=generated_text,
        )

        return text_path
