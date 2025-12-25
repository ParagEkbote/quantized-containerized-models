import pytest
import torch
import torch.nn as nn
from PIL import Image

from models.flux_fast_lora_hotswap.predict import (
    save_image,
)
from models.flux_fast_lora_hotswap_img2img.predict import (
    load_image,
    save_image,
)
from models.gemma_torchao.predict import (
    apply_safe_sparsity,
    format_chat_messages,
    gemma_filter_fn,
    gradual_magnitude_pruning,
    layer_norm_pruning,
    magnitude_based_pruning,
    sanitize_weights_for_quantization,
    save_output_to_file,
)
from models.smollm3_pruna.predict import (
    save_text,
)


# =======================================================
#  SUPPORT TINY MODEL FOR SPARSITY TESTS
# =======================================================
class TinyLinearModel(nn.Module):
    """Small model for pruning tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# =======================================================
# 2. save_image
# =======================================================
@pytest.mark.unit
def test_save_image_creates_file(tmp_path):
    img = Image.new("RGB", (32, 32), color="red")
    out = save_image(img, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


# =======================================================
# 3. load_image (local only – avoids network)
# =======================================================
@pytest.mark.unit
def test_load_image_local(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), "blue").save(img_path)
    loaded = load_image(str(img_path))
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (16, 16)


# =======================================================
# 4. save_output_to_file
# =======================================================
@pytest.mark.unit
def test_save_output_to_file_filename(tmp_path):
    out = save_output_to_file("hello", output_folder=tmp_path, filename="x.txt")
    assert out.exists()
    assert out.read_text() == "hello"


@pytest.mark.unit
def test_save_output_to_file_seed_index(tmp_path):
    out = save_output_to_file("text", output_folder=tmp_path, seed=7, index=9)
    assert out.exists()
    assert out.name == "output_7_9.txt"


# =======================================================
# 5. gemma_filter_fn
# =======================================================
@pytest.mark.unit
def test_gemma_filter_fn_accepts_target_layers():
    lin = nn.Linear(4, 4)
    assert gemma_filter_fn(lin, "self_attn.q_proj")
    assert gemma_filter_fn(lin, "mlp.up_proj")


@pytest.mark.unit
def test_gemma_filter_fn_rejects_wrong_layers():
    conv = nn.Conv2d(1, 1, 3)
    assert not gemma_filter_fn(conv, "self_attn.q_proj")
    assert not gemma_filter_fn(nn.Linear(4, 4), "embed_tokens")


# =======================================================
# 6. magnitude_based_pruning
# =======================================================
@pytest.mark.unit
def test_magnitude_based_pruning_runs():
    model = TinyLinearModel()
    magnitude_based_pruning(model, sparsity_ratio=0.3, filter_fn=gemma_filter_fn)


# =======================================================
# 7. gradual_magnitude_pruning
# =======================================================
@pytest.mark.unit
def test_gradual_magnitude_pruning_runs():
    model = TinyLinearModel()
    s = gradual_magnitude_pruning(model, target_sparsity=0.5, current_step=20, total_steps=100)
    assert 0 <= s <= 0.5


# =======================================================
# 8. layer_norm_pruning
# =======================================================
@pytest.mark.unit
def test_layer_norm_pruning_runs():
    model = TinyLinearModel()
    sparsity = layer_norm_pruning(model, sparsity_ratio=0.2, filter_fn=gemma_filter_fn)
    assert 0 <= sparsity <= 1


# =======================================================
# 9. apply_safe_sparsity
# =======================================================
@pytest.mark.unit
def test_apply_safe_sparsity_runs():
    model = TinyLinearModel()
    s = apply_safe_sparsity(model, sparsity_type="magnitude", sparsity_ratio=0.25)
    assert 0 <= s <= 1


# =======================================================
# 10. sanitize_weights_for_quantization
# =======================================================
@pytest.mark.unit
def test_sanitize_weights_for_quantization_runs():
    model = TinyLinearModel()
    sanitize_weights_for_quantization(model)


# =======================================================
# 11. format_chat_messages
# =======================================================
@pytest.mark.unit
def test_format_chat_messages_no_image():
    msgs = format_chat_messages("hello")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][-1]["text"] == "hello"


@pytest.mark.unit
def test_format_chat_messages_with_image():
    msgs = format_chat_messages("hello", "http://example.com/img.png")
    assert msgs[1]["content"][0]["url"] == "http://example.com/img.png"


# =======================================================
# 12. save_text
# =======================================================
@pytest.mark.unit
def test_save_text(tmp_path):
    out = save_text(tmp_path, seed=10, index="a", text="xyz")
    assert out.exists()
    assert out.read_text() == "xyz"
