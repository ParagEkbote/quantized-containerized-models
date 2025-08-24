# predictor.py

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
        Loads the Unsloth GPT-OSS-20B model in 4-bit quantization.
        """

        # Recommended: pre-quantized 4-bit version for Cog
        model_id = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

        print(f"Loading model: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=4096,       # Supports long context
            dtype=torch.float16,       # FP16 for efficiency
            load_in_4bit=True,         # 4-bit quantization
            trust_remote_code=True,
        )
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_new_tokens: int = Input(description="Maximum number of new tokens", default=512),
        temperature: float = Input(description="Sampling temperature", default=0.7),
        top_p: float = Input(description="Top-p nucleus sampling", default=0.95),
        reasoning_effort: str = Input(
            description="Reasoning effort level: low, medium, or high",
            choices=["low", "medium", "high"],
            default="medium",
        ),
        seed: int = Input(
            description="Random seed used in the output filename",
            default=42,
        ),
    ) -> Path:
        """
        Run inference on the given prompt and always save the output as .txt file.
        """

        # Prepare chat-style input
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,
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
        print(f"[Generated Output]: {result}\n")
        print(f"Used memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Always save output into /tmp for Cog
        output_path = save_text(SysPath("/tmp"), seed, "pred", result)
        print(f"Saved output to {output_path}")

        return Path(str(output_path))
