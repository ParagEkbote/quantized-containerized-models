import torch
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys

# Mock unsloth BEFORE importing anything that uses it
sys.modules['unsloth'] = MagicMock()
sys.modules['unsloth.models'] = MagicMock()

# Now safe to import
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
    model.to.return_value = model
    model.eval.return_value = model

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    tokenizer.decode.return_value = "decoded text"

    return model, tokenizer


# ---------------------------
# Predictor.setup test
# ---------------------------
@patch("models.phi_4_reasoning_plus_unsloth.predict.FastLanguageModel")
@patch("torch.cuda.is_available", return_value=False)
def test_setup(mock_cuda, mock_fast_language_model):
    model, tok = mock_fast_model()
    mock_fast_language_model.from_pretrained.return_value = (model, tok)

    pred = Predictor()
    pred.setup()

    assert pred.model is model
    assert pred.tokenizer is tok
    assert pred.dtype == torch.float32  # CPU fallback
    mock_fast_language_model.from_pretrained.assert_called_once()


# ---------------------------
# Predictor.predict test
# ---------------------------
@patch("models.phi_4_reasoning_plus_unsloth.predict.FastLanguageModel")
@patch("models.phi_4_reasoning_plus_unsloth.predict.save_text")
@patch("torch.cuda.is_available", return_value=False)
def test_predict(mock_cuda, mock_save_text, mock_fast_language_model, tmp_path):
    model, tok = mock_fast_model()
    mock_fast_language_model.from_pretrained.return_value = (model, tok)
    
    result_path = tmp_path / "result.txt"
    mock_save_text.return_value = result_path

    pred = Predictor()
    pred.setup()
    out = pred.predict(prompt="hello", max_new_tokens=10)

    assert isinstance(out, Path)
    assert out == result_path

    tok.assert_called()
    model.generate.assert_called_once()
    tok.decode.assert_called_once()
    mock_save_text.assert_called_once()