import torch
from cog import BasePredictor, Input
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

class Predictor(BasePredictor):
    def setup(self):
        model_id = "unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def predict(self, prompt: str = Input(description="Input prompt")) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
