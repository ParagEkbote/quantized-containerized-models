import inspect
from pathlib import Path

import pytest
from pydantic.fields import FieldInfo

from models.flux_fast_lora_hotswap.predict import (
    Predictor,
)


# ---------------------------------------------------------------------------
# 1. Contract: predict() signature is stable
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_signature_stable():
    sig = inspect.signature(Predictor.predict)

    params = [p.name for p in sig.parameters.values()]
    # Remove self
    if params[0] == "self":
        params = params[1:]

    expected = ["prompt", "trigger_word"]
    assert params == expected, f"Signature changed. Expected {expected}, got {params}"


# ---------------------------------------------------------------------------
# 2. Contract: Required fields are required Input() fields
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_required_fields_are_required():
    sig = inspect.signature(Predictor.predict)
    params = sig.parameters

    required = ["prompt", "trigger_word"]

    for name in required:
        assert name in params, f"{name} missing in predict signature"

        default = params[name].default
        assert isinstance(default, FieldInfo), f"{name} must be FieldInfo"

        # Verify the FieldInfo marks the field as required
        assert default.default is None or str(default.default) == "PydanticUndefined", f"{name} must be required; got default={default.default}"

    # Test that calling unbound method without self raises TypeError
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'self'"):
        Predictor.predict()


# ---------------------------------------------------------------------------
# 3. Contract: Input metadata must match schema
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_input_metadata_matches_schema():
    pred = Predictor()
    sig = inspect.signature(pred.predict)

    prompt_meta = sig.parameters["prompt"].default
    trigger_meta = sig.parameters["trigger_word"].default

    assert "prompt" in prompt_meta.description.lower()
    assert "trigger" in trigger_meta.description.lower()


# ---------------------------------------------------------------------------
# 4. Contract: predict() returns a Path without running GPU code
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_returns_path(monkeypatch, tmp_path):
    pred = Predictor()

    # ------------------------
    # Minimal mock pipeline
    # ------------------------
    class DummyImg:
        def save(self, path):
            Path(path).write_text("IMAGE")

    class DummyResp:
        images = [DummyImg()]

    class DummyPipe:
        def __call__(self, **kwargs):
            return DummyResp()

        def set_adapters(self, *args, **kwargs):
            pass

        # Attributes needed because predict() compiles them
        text_encoder = lambda self, *a, **k: None
        text_encoder_2 = lambda self, *a, **k: None
        vae = lambda self, *a, **k: None

    pred.pipe = DummyPipe()
    pred.lora1_triggers = ["A"]
    pred.lora2_triggers = ["B"]
    pred.current_adapter = "open-image-preferences"

    # ------------------------
    # Monkeypatch save_image
    # ------------------------
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap.predict.save_image",
        lambda img, output_dir=tmp_path: tmp_path / "generated.png",
    )

    out = pred.predict(prompt="hello", trigger_word="A")

    assert isinstance(out, Path)
    assert out.name == "generated.png"


# ---------------------------------------------------------------------------
# 5. Contract: LoRA switching logic works without model backend
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_lora_switching_logic():
    pred = Predictor()

    # Dummy mock pipe for adapter switching
    class DummyPipe:
        def set_adapters(self, adapters, adapter_weights):
            pred.adapter_called = (adapters, adapter_weights)

        # required attrs to avoid compile() errors
        text_encoder = lambda self, *a, **k: None
        text_encoder_2 = lambda self, *a, **k: None
        vae = lambda self, *a, **k: None

        def __call__(self, **kw):
            class R:
                images = [type("Img", (), {"save": lambda self, p: None})()]

            return R()

    pred.pipe = DummyPipe()

    pred.lora1_triggers = ["A"]
    pred.lora2_triggers = ["B"]

    pred.current_adapter = "open-image-preferences"

    # Trigger LoRA2
    pred.predict(prompt="x", trigger_word="B")
    assert pred.current_adapter == "flux-ghibsky"

    # Trigger LoRA1
    pred.predict(prompt="x", trigger_word="A")
    assert pred.current_adapter == "open-image-preferences"
