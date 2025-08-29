from unsloth import FastLanguageModel
import torch
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer
from pathlib import Path as SysPath


def save_text(output_folder: SysPath, seed: int, index: int | str, text: str) -> SysPath:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed!s}_{index}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        """
        Setup runs once when the container starts.
        Loads the Phi-4 reasoning-plus model with Unsloth in 4-bit.
        """

        # Target model
        model_id = "unsloth/Phi-4-reasoning-plus"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {model_id} ...")
        print(f"Using device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=131072,       # Phi-4 supports very long context
            dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
            use_static_cache=False, 
        )

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_new_tokens: int = Input(description="Maximum number of new tokens", default=1024),
        temperature: float = Input(description="Sampling temperature", default=0.7),
        top_p: float = Input(description="Top-p nucleus sampling", default=0.95),
        seed: int = Input(
            description="Random seed used in the output filename",
            default=42,
        ),
    ) -> Path:
        """
        Run inference on the given prompt and always save the output as .txt file.
        """

        # Prepare chat-style input with reasoning effort
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"\n[Prompt]: {prompt}")
        print(f"[Generated Output]: {result[:500]}...\n")  # shorten print
        if torch.cuda.is_available():
            print(f"Used memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Always save output into /tmp for Cog
        output_path = save_text(SysPath("/tmp"), seed, "pred", result)
        print(f"Saved output to {output_path}")

        return Path(str(output_path))
