import io
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import pytest
from unittest.mock import patch, MagicMock

# Import directly from the package (no dynamic import)
from models.gemma_torchao.predict import (
    login_with_env_token,
    save_output_to_file,
    gemma_filter_fn,
    magnitude_based_pruning,
    format_chat_messages,
    sanitize_weights_for_quantization,
    Predictor,
)

@pytest.fixture(autouse=True)
def _fix_cog_fieldinfo(monkeypatch):
    """
    Convert Cog Input(FieldInfo) attributes into real Python values
    inside models.gemma_torchao.predict.Predictor.
    """

    import models.gemma_torchao.predict as mod
    Predictor = mod.Predictor

    # List all Input-based attributes inside your Predictor
    # ◀️ EDIT this list to match your file
    fields_to_patch = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42,
        "guidance_scale": 1.2,
        "image_strength": 0.5,   # if exists
        "height": 1024,          # if exists
        "width": 1024,           # if exists
    }

    # Apply the patches
    for attr, fixed_value in fields_to_patch.items():
        if hasattr(Predictor, attr):
            monkeypatch.setattr(Predictor, attr, fixed_value, raising=False)
# --------------------------------------------------------------------
# login_with_env_token
# --------------------------------------------------------------------
@patch("models.gemma_torchao.predict.login")
@patch("models.gemma_torchao.predict.load_dotenv")
def test_login_with_env_token_success(mock_load, mock_login, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "abc123")
    login_with_env_token()
    mock_load.assert_called_once()
    mock_login.assert_called_once_with(token="abc123")


@patch("models.gemma_torchao.predict.login")
@patch("models.gemma_torchao.predict.load_dotenv")
def test_login_with_env_token_missing(mock_load, mock_login, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError):
        login_with_env_token()
    mock_login.assert_not_called()


# --------------------------------------------------------------------
# save_output_to_file
# --------------------------------------------------------------------
def test_save_output_to_file_creates_file(tmp_path):
    text = "Hello, world!"
    file_path = save_output_to_file(text, output_folder=tmp_path, seed=42, index=1)
    assert file_path.exists()
    assert "42_1" in file_path.name
    assert file_path.read_text(encoding="utf-8") == text


# --------------------------------------------------------------------
# gemma_filter_fn
# --------------------------------------------------------------------
def test_gemma_filter_fn_positive():
    layer = nn.Linear(10, 10)
    assert gemma_filter_fn(layer, "transformer.self_attn.q_proj")


def test_gemma_filter_fn_negative():
    layer = nn.Linear(10, 10)
    assert not gemma_filter_fn(layer, "embed_tokens")
    assert not gemma_filter_fn(layer, "layernorm1")


# --------------------------------------------------------------------
# magnitude_based_pruning
# --------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)


def test_magnitude_based_pruning_reduces_nonzero_weights():
    model = TinyModel()
    before = (model.l1.weight != 0).sum().item()
    magnitude_based_pruning(model, sparsity_ratio=0.5)
    after = (model.l1.weight != 0).sum().item()
    assert after <= before


# --------------------------------------------------------------------
# format_chat_messages
# --------------------------------------------------------------------
def test_format_chat_messages_with_image():
    messages = format_chat_messages("hello", "http://example.com/img.png")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][1]["type"] == "text"
    assert messages[1]["content"][1]["text"] == "hello"


def test_format_chat_messages_without_image():
    messages = format_chat_messages("hi")
    assert len(messages[1]["content"]) == 1
    assert messages[1]["content"][0]["text"] == "hi"


# --------------------------------------------------------------------
# sanitize_weights_for_quantization
# --------------------------------------------------------------------
def test_sanitize_weights_makes_contiguous():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.lin.weight = nn.Parameter(self.lin.weight.T)  # make non-contiguous

    model = Dummy()
    assert not model.lin.weight.is_contiguous()
    sanitize_weights_for_quantization(model)
    assert model.lin.weight.is_contiguous()


# --------------------------------------------------------------------
# Predictor.setup + Predictor.predict (mocked)
# --------------------------------------------------------------------
@patch("models.gemma_torchao.predict.AutoProcessor")
@patch("models.gemma_torchao.predict.AutoModelForImageTextToText")
def test_predictor_predict_text_only(mock_model_cls, mock_proc_cls, tmp_path):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_processor.tokenizer = MagicMock()
    mock_processor.tokenizer.eos_token_id = 0
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.batch_decode.return_value = ["Hello world"]
    mock_processor.tokenizer.decode.return_value = "Hello world"
    mock_proc_instance = MagicMock()
    mock_proc_instance.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_processor.return_value = mock_proc_instance

    predictor = Predictor()
    predictor.setup()

    with patch("models.gemma_torchao.predict.save_output_to_file") as mock_save:
        mock_save.return_value = tmp_path / "fake.txt"
        result = predictor.predict(prompt="hi", image_url=None)

    assert isinstance(result, str)
    assert "Hello world" in result


# --------------------------------------------------------------------
# Predictor.predict with image (mocked requests)
# --------------------------------------------------------------------
@patch("models.gemma_torchao.predict.requests.get")
@patch("models.gemma_torchao.predict.AutoProcessor")
@patch("models.gemma_torchao.predict.AutoModelForImageTextToText")
def test_predictor_predict_with_image(mock_model_cls, mock_proc_cls, mock_get, tmp_path):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_model_cls.from_pretrained.return_value = mock_model

    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_image.getvalue()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_processor.tokenizer = MagicMock()
    mock_processor.tokenizer.eos_token_id = 0
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.batch_decode.return_value = ["A picture of a cat"]
    mock_processor.tokenizer.decode.return_value = "A picture of a cat"
    mock_proc_instance = MagicMock()
    mock_proc_instance.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_processor.return_value = mock_proc_instance

    predictor = Predictor()
    predictor.setup()

    with patch("models.gemma_torchao.predict.save_output_to_file") as mock_save:
        mock_save.return_value = tmp_path / "fake.txt"
        result = predictor.predict(
            prompt="describe this image",
            image_url="http://fake-url.com/image.png",
            use_quantization="false",
        )

    assert isinstance(result, str)
    assert "cat" in result.lower()
    mock_get.assert_called_once()
