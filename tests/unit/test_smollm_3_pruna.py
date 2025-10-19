import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from models.smollm3_pruna.predict import (
    Predictor,
    save_text,
)


class TestSaveText:
    """Test cases for the save_text function."""

    def test_save_text_creates_file(self, tmp_data_dir):
        """Test that save_text creates a file with correct content."""
        seed = 42
        index = 0
        text = "This is a test output"

        result_path = save_text(tmp_data_dir, seed, index, text)

        assert result_path.exists()
        assert result_path.name == f"output_{seed}_{index}.txt"
        assert result_path.read_text(encoding="utf-8") == text

    def test_save_text_creates_parent_directories(self, tmp_data_dir):
        """Test that save_text creates parent directories if needed."""
        nested_path = tmp_data_dir / "nested" / "dir"
        seed = 123
        index = "test"
        text = "Nested output"

        result_path = save_text(nested_path, seed, index, text)

        assert result_path.exists()
        assert result_path.parent == nested_path

    def test_save_text_with_string_index(self, tmp_data_dir):
        """Test save_text with string index."""
        seed = 99
        index = "custom_name"
        text = "Custom indexed output"

        result_path = save_text(tmp_data_dir, seed, index, text)

        assert result_path.name == f"output_{seed}_{index}.txt"
        assert result_path.read_text(encoding="utf-8") == text


class TestPredictor:
    """Test cases for the Predictor class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 2
        tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        tokenizer.decode = Mock(return_value=" generated text here")
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.parameters = Mock(return_value=[torch.tensor([1.0]).cuda()])
        model.generate = Mock(
            return_value=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        )
        return model

    @pytest.fixture
    def predictor(self, mock_tokenizer, mock_model):
        """Create a Predictor instance with mocked dependencies."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.AutoTokenizer") as mock_auto_tokenizer, \
             patch("models.flux_fast_lora_hotswap_img2img.predict.AutoModelForCausalLM") as mock_auto_model, \
             patch("models.flux_fast_lora_hotswap_img2img.predict.smash") as mock_smash:
            
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_auto_model.from_pretrained.return_value = Mock()
            mock_smash.return_value = mock_model

            predictor = Predictor()
            predictor.setup()
            
            return predictor

    def test_setup_loads_model_and_tokenizer(self):
        """Test that setup correctly loads the model and tokenizer."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.AutoTokenizer") as mock_auto_tokenizer, \
             patch("models.flux_fast_lora_hotswap_img2img.predict.AutoModelForCausalLM") as mock_auto_model, \
             patch("models.flux_fast_lora_hotswap_img2img.predict.smash") as mock_smash:
            
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            mock_model = Mock()
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_smash.return_value = Mock()

            predictor = Predictor()
            predictor.setup()

            # Verify tokenizer was loaded
            mock_auto_tokenizer.from_pretrained.assert_called_once_with(
                "HuggingFaceTB/SmolLM3-3B"
            )
            assert predictor.tokenizer.pad_token == "<eos>"

            # Verify model was loaded
            mock_auto_model.from_pretrained.assert_called_once()
            
            # Verify smash was called
            mock_smash.assert_called_once()

            # Verify cache_length is set
            assert predictor.cache_length == 2048

    def test_predict_returns_file_path(self, predictor, tmp_data_dir):
        """Test that predict returns a valid file path."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            result = predictor.predict(
                prompt="Test prompt",
                max_new_tokens=100,
                mode="no_think",
                seed=42
            )

            assert isinstance(result, Path)
            assert result.exists()
            assert result.suffix == ".txt"

    def test_predict_uses_correct_chat_template(self, predictor, tmp_data_dir):
        """Test that predict applies the correct chat template."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            predictor.predict(
                prompt="What is AI?",
                max_new_tokens=50,
                mode="think",
                seed=10
            )

            predictor.tokenizer.apply_chat_template.assert_called_once()
            call_args = predictor.tokenizer.apply_chat_template.call_args
            
            messages = call_args[0][0]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "/think"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What is AI?"

    def test_predict_tokenizes_input(self, predictor, tmp_data_dir):
        """Test that predict tokenizes the input correctly."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            predictor.predict(
                prompt="Hello world",
                max_new_tokens=200,
                mode="no_think",
                seed=5
            )

            predictor.tokenizer.assert_called_once()
            call_kwargs = predictor.tokenizer.call_args[1]
            
            assert call_kwargs["return_tensors"] == "pt"
            assert call_kwargs["padding"] is True
            assert call_kwargs["truncation"] is True

    def test_predict_generates_tokens(self, predictor, tmp_data_dir):
        """Test that predict calls the model's generate method."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            predictor.predict(
                prompt="Generate text",
                max_new_tokens=128,
                mode="no_think",
                seed=7
            )

            predictor.smashed_text_model.generate.assert_called_once()
            call_kwargs = predictor.smashed_text_model.generate.call_args[1]
            
            assert call_kwargs["max_new_tokens"] == 128
            assert call_kwargs["pad_token_id"] == predictor.tokenizer.eos_token_id
            assert call_kwargs["eos_token_id"] == predictor.tokenizer.eos_token_id

    def test_predict_decodes_output(self, predictor, tmp_data_dir):
        """Test that predict decodes the generated tokens."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            predictor.predict(
                prompt="Test decoding",
                max_new_tokens=64,
                mode="no_think",
                seed=3
            )

            predictor.tokenizer.decode.assert_called_once()
            call_kwargs = predictor.tokenizer.decode.call_args[1]
            assert call_kwargs["skip_special_tokens"] is True

    def test_predict_saves_full_output(self, predictor, tmp_data_dir):
        """Test that predict saves the full output (prompt + generation)."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            prompt = "Original prompt"
            result_path = predictor.predict(
                prompt=prompt,
                max_new_tokens=100,
                mode="no_think",
                seed=42
            )

            content = result_path.read_text(encoding="utf-8")
            # Should contain both prompt and generated text
            assert prompt in content

    def test_predict_with_different_modes(self, predictor, tmp_data_dir):
        """Test predict with different reasoning modes."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            # Test 'think' mode
            predictor.predict(
                prompt="Think about this",
                max_new_tokens=50,
                mode="think",
                seed=1
            )
            
            messages_think = predictor.tokenizer.apply_chat_template.call_args[0][0]
            assert messages_think[0]["content"] == "/think"

            # Reset mock
            predictor.tokenizer.apply_chat_template.reset_mock()

            # Test 'no_think' mode
            predictor.predict(
                prompt="Don't think",
                max_new_tokens=50,
                mode="no_think",
                seed=2
            )
            
            messages_no_think = predictor.tokenizer.apply_chat_template.call_args[0][0]
            assert messages_no_think[0]["content"] == "/no_think"

    def test_predict_respects_max_input_length(self, predictor, tmp_data_dir):
        """Test that predict calculates max_input_length correctly."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            max_new_tokens = 512
            predictor.predict(
                prompt="Test max length",
                max_new_tokens=max_new_tokens,
                mode="no_think",
                seed=8
            )

            expected_max_length = predictor.cache_length - max_new_tokens
            call_kwargs = predictor.tokenizer.call_args[1]
            assert call_kwargs["max_length"] == expected_max_length

    def test_predict_with_negative_seed(self, predictor, tmp_data_dir):
        """Test that negative seed defaults to 0 in filename."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            result_path = predictor.predict(
                prompt="Test seed",
                max_new_tokens=50,
                mode="no_think",
                seed=-1
            )

            # Filename should use 0 instead of -1
            assert "output_0_0.txt" in str(result_path)

    def test_predict_uses_no_grad_context(self, predictor, tmp_data_dir):
        """Test that predict uses torch.no_grad() for inference."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)), \
             patch("models.flux_fast_lora_hotswap_img2img.predict.torch.no_grad") as mock_no_grad:
            
            predictor.predict(
                prompt="Test no grad",
                max_new_tokens=50,
                mode="no_think",
                seed=1
            )

            mock_no_grad.assert_called_once()

    def test_predict_moves_inputs_to_device(self, predictor, tmp_data_dir):
        """Test that predict moves input tensors to the correct device."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            predictor.predict(
                prompt="Test device placement",
                max_new_tokens=50,
                mode="no_think",
                seed=1
            )

            # Verify tokenizer was called and returned tensors
            assert predictor.tokenizer.called

    def test_predict_with_edge_case_max_new_tokens(self, predictor, tmp_data_dir):
        """Test predict with edge case token limits."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            # Test with minimum tokens (1)
            predictor.predict(
                prompt="Minimal output",
                max_new_tokens=1,
                mode="no_think",
                seed=1
            )

            call_kwargs = predictor.tokenizer.call_args[1]
            expected_max_length = max(1, predictor.cache_length - 1)
            assert call_kwargs["max_length"] == expected_max_length

    def test_predict_output_filename_format(self, predictor, tmp_data_dir):
        """Test that output filename follows the correct format."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            seed = 999
            result_path = predictor.predict(
                prompt="Test filename",
                max_new_tokens=50,
                mode="no_think",
                seed=seed
            )

            # Verify filename format: output_{seed}_{index}.txt
            expected_pattern = f"output_{seed}_0.txt"
            assert result_path.name == expected_pattern

    def test_predict_with_hf_token_set(self, predictor, tmp_data_dir, ensure_hf_token):
        """Test that predict works when HF_TOKEN is set (via conftest autouse)."""
        with patch("models.flux_fast_lora_hotswap_img2img.predict.tempfile.mkdtemp", return_value=str(tmp_data_dir)):
            result = predictor.predict(
                prompt="Test with token",
                max_new_tokens=50,
                mode="no_think",
                seed=1
            )

            assert result.exists()
            assert result.suffix == ".txt"