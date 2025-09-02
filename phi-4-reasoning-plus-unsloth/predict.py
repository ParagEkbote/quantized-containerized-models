from unsloth import FastLanguageModel #noqa
from pathlib import Path as SysPath

import torch
from cog import BasePredictor, Input, Path
from unsloth import FastLanguageModel


def save_text(output_folder: SysPath, seed: int, index: str, text: str) -> SysPath:
    output_path = output_folder / f"output_{seed}_{index}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        model_id = "unsloth/Phi-4-reasoning-plus"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_id} (device={self.device})")

        # Set appropriate dtype based on device capabilities
        if self.device.type == "cuda":
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            self.dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self.dtype = torch.float32

        # Load model and tokenizer with static cache disabled
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=131072,
            dtype=self.dtype,
            load_in_4bit=True,
            trust_remote_code=True,
            use_cache=True,
        )

        self.cache_length = 2048
        print("Setup completed successfully")

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens", default=3000, ge=1, le=25000
        ),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.1, le=1),
        top_p: float = Input(description="Top-p nucleus sampling", default=0.95, ge=0.1, le=1),
        seed: int = Input(description="Random seed", default=42),
    ) -> Path:
        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Tokenize input and move to model's device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)

        # Move input tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate text with cache configuration disabled
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid static cache issues
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log results
        print(f"\n[Prompt]: {prompt}")
        print(f"[Generated Output]: {generated_text[:500]}...")

        if torch.cuda.is_available():
            print(f"Used memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Save and return output
        output_path = save_text(SysPath("/tmp"), seed, "pred", generated_text)
        print(f"Saved output to {output_path}")
        return Path(str(output_path))
