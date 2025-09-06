import torch
import pytest
from pathlib import Path as SysPath

import Predictor


def test_save_text_creates_file_and_writes(tmp_path):
    """Ensure save_text writes correct content to file."""
    output_folder = tmp_path / "outdir"
    seed = 123
    index = "abc"
    text = "hello world\nthis is a test"

    result_path = predictor.save_text(output_folder, seed, index, text)

    assert isinstance(result_path, SysPath)
    assert result_path.exists()
    assert result_path.parent == output_folder
    assert result_path.read_text(encoding="utf-8") == text


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 99

    def __call__(self, text, return_tensors="pt", truncation=False):
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded generated text"


class DummyModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, device=device))

    def generate(self, *args, **kwargs):
        # Default dummy implementation (can be overridden in RecordingModel)
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


def _fake_from_pretrained(**kwargs):
    return DummyModel(torch.device("cpu")), DummyTokenizer()


def test_predictor_setup_and_predict_cpu(monkeypatch, tmp_path):
    """Basic CPU test for setup() + predict()."""
    monkeypatch.setattr(predictor.FastLanguageModel, "from_pretrained", staticmethod(_fake_from_pretrained))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_save_text(output_folder, seed, index, text):
        out = tmp_path / f"output_{seed}_{index}.txt"
        out.write_text(text, encoding="utf-8")
        return out

    monkeypatch.setattr(predictor, "save_text", fake_save_text)

    pred = predictor.Predictor()
    pred.setup()

    assert pred.dtype == torch.float32  # CPU path
    out_path_obj = pred.predict(prompt="test", max_new_tokens=5, temperature=0.5, top_p=0.9, seed=999)

    out_path = SysPath(str(out_path_obj))
    assert out_path.exists()
    assert "decoded generated text" in out_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("cuda_available,bf16_supported,expected_dtype", [
    (False, False, torch.float32),
    (True, True, torch.bfloat16),
    (True, False, torch.float16),
])
def test_predictor_dtype_selection(monkeypatch, cuda_available, bf16_supported, expected_dtype):
    """Check dtype selection logic depending on CUDA + bf16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported, raising=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        dtype = torch.float32

    assert dtype == expected_dtype


def test_predictor_generate_params(monkeypatch, tmp_path):
    """Verify all params passed to model.generate(), esp. use_cache=False."""

    captured_kwargs = {}

    class RecordingModel(DummyModel):
        def generate(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def fake_from_pretrained(**kwargs):
        return RecordingModel(torch.device("cpu")), DummyTokenizer()

    monkeypatch.setattr(predictor.FastLanguageModel, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(predictor, "save_text", lambda folder, seed, index, text: tmp_path / "out.txt")

    pred = predictor.Predictor()
    pred.setup()
    pred.predict(prompt="test", max_new_tokens=42, temperature=0.33, top_p=0.77, seed=123)

    # Assert all key kwargs passed to generate
    assert captured_kwargs["max_new_tokens"] == 42
    assert captured_kwargs["do_sample"] is True
    assert captured_kwargs["temperature"] == 0.33
    assert captured_kwargs["top_p"] == 0.77
    assert captured_kwargs["pad_token_id"] == pred.tokenizer.eos_token_id
    assert captured_kwargs["use_cache"] is False
    assert "input_ids" in captured_kwargs
    assert "attention_mask" in captured_kwargs
