# tests/test_predictor.py
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.phi_4_reasoning_plus_unsloth.predict import Predictor, save_text


# ---------------------------
# save_text test
# ---------------------------
def test_save_text(tmp_path):
    out = save_text(tmp_path, seed=123, index="x", text="hello")
    assert out.exists()
    assert out.read_text() == "hello"
    assert out.name == "output_123_x.txt"


# ---------------------------
# helper: mock FastLanguageModel.from_pretrained
# ---------------------------
def mock_fast_model():
    model = MagicMock()
    model.parameters.return_value = [torch.zeros(1)]
    model.generate.return_value = torch.tensor([[1, 2, 3]])

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.decode.return_value = "decoded text"

    return model, tokenizer


# ---------------------------
# Predictor.setup test
# ---------------------------
@patch("torch.cuda.is_available", lambda: False)
@patch("models.unsloth.predict.FastLanguageModel.from_pretrained")
def test_setup(mock_from_pretrained):
    model, tok = mock_fast_model()
    mock_from_pretrained.return_value = (model, tok)

    pred = Predictor()
    pred.setup()

    assert pred.model is model
    assert pred.tokenizer is tok
    assert pred.dtype == torch.float32  # CPU fallback


# ---------------------------
# Predictor.predict test
# ---------------------------
@patch("torch.cuda.is_available", lambda: False)
@patch("models.unsloth.predict.FastLanguageModel.from_pretrained")
@patch("torch.no_grad")
def test_predict(mock_no_grad, mock_from_pretrained, tmp_path, monkeypatch):
    model, tok = mock_fast_model()
    mock_from_pretrained.return_value = (model, tok)

    # force save_text to tmp_path/result.txt
    monkeypatch.setattr(
        "models.unsloth.predict.save_text",
        lambda folder, seed, idx, txt: tmp_path / "result.txt",
    )

    pred = Predictor()
    pred.setup()
    out = pred.predict(prompt="hello", max_new_tokens=10)

    assert isinstance(out, Path)
    assert out.name == "result.txt"

    tok.assert_called_once()                 # tokenization
    model.generate.assert_called_once()      # text generation
    tok.decode.assert_called_once()          # decoding
    mock_no_grad.return_value.__enter__.assert_called_once()
