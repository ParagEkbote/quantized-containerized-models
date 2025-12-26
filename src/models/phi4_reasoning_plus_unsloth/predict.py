from unsloth import FastLanguageModel  # noqa  # isort: skip
import tempfile
from pathlib import Path as SysPath
from typing import Any

import torch
from cog import BasePredictor, Input, Path


def save_text(output_folder: SysPath, seed: int, index: str, text: str) -> SysPath:
    """Save the generated text to disk as a .txt file."""
    output_path = output_folder / f"output_{seed}_{index}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        model_id = "unsloth/Phi-4-reasoning-plus-unsloth-bnb-4bit"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_id} (device={self.device})")

        if self.device.type == "cuda":
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            self.dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self.dtype = torch.float32

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=131072,
            dtype=self.dtype,
            load_in_4bit=True,
            trust_remote_code=True,
            use_cache=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Setup completed successfully")

    def predict(
        self,
        prompt: str = Input(description="User prompt"),
        max_new_tokens: int = Input(default=512, ge=1, le=40000),
        temperature: float = Input(default=0.8, ge=0.0, le=1.0),
        top_p: float = Input(default=0.95, ge=0.0, le=1.0),
        top_k: int = Input(default=50, ge=1, le=100),
        seed: int = Input(default=42),
    ) -> Path:
        if not prompt or not prompt.strip():
            raise ValueError("User prompt must be non-empty")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        FastLanguageModel.for_inference(self.model)

        # ---- MANUAL PHI-4 FORMATTING (bypassing broken chat template) ----
        # Phi-4 uses this format: <|system|>...<|end|><|user|>...<|end|><|assistant|>
        system_msg = "You are Phi, a helpful assistant trained by Microsoft. Answer the user's question directly and clearly."

        formatted_prompt = f"<|system|>\n{system_msg}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        print(f"[Formatted Prompt]:\n{formatted_prompt}")
        print(f"[User Question]: {prompt}")

        # Tokenize the manually formatted prompt
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,  # We already added Phi-4 special tokens
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        input_len = input_ids.shape[-1]

        # ---- Correct sampling contract ----
        do_sample = temperature > 0.0

        gen_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True,
        )

        if do_sample:
            gen_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        # ---- Decode only the NEW tokens (assistant's response) ----
        generated_ids = outputs[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        # Clean up any remaining special tokens
        cleanup_patterns = ["<|end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"]
        for pattern in cleanup_patterns:
            generated_text = generated_text.split(pattern)[0].strip()

        print(f"[Assistant Response]: {generated_text[:500]}")

        tmpdir = Path(tempfile.mkdtemp(prefix="phi4_"))
        output_path = save_text(tmpdir, seed, "pred", generated_text)
        return output_path
