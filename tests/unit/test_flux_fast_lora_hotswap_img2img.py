import inspect
from pathlib import Path

import pytest
from cog import Input
from pydantic.fields import FieldInfo, PydanticUndefined

from models.flux_fast_lora_hotswap_img2img.predict import (
    Predictor,
)


def extract_constraints(field: FieldInfo):
    ge = None
    le = None
    for meta in field.metadata:
        cls_name = meta.__class__.__name__.lower()
        if cls_name == "ge":
            ge = meta.ge
        elif cls_name == "le":
            le = meta.le
    return ge, le


@pytest.mark.unit
def test_numeric_constraints_strength_guidance_steps():
    """
    Contract test: Ensure strength, guidance_scale, num_inference_steps
    preserve ge/le constraints exactly as defined in the schema.
    """

    sig = inspect.signature(Predictor.predict)
    params = sig.parameters

    # --- Strength ---
    strength_field: FieldInfo = params["strength"].default
    assert isinstance(strength_field, FieldInfo)
    ge, le = extract_constraints(strength_field)

    assert ge == 0, "strength.ge must be 0"
    assert le == 1, "strength.le must be 1"
    assert strength_field.default == 0.6

    # --- Guidance Scale ---
    gs_field: FieldInfo = params["guidance_scale"].default
    assert isinstance(gs_field, FieldInfo)
    ge, le = extract_constraints(gs_field)

    assert ge == 0, "guidance_scale.ge must be 0"
    assert le is None, "guidance_scale.le must be None"
    assert gs_field.default == 7.5

    # --- Num Inference Steps ---
    steps_field: FieldInfo = params["num_inference_steps"].default
    assert isinstance(steps_field, FieldInfo)
    ge, le = extract_constraints(steps_field)

    assert ge == 1, "num_inference_steps.ge must be 1"
    assert le is None, "num_inference_steps.le must be None"
    assert steps_field.default == 28


# -----------------------------------------------------------------------------
# Contract 1 — Function signature must remain stable
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_signature_stable():
    pred = Predictor()
    sig = inspect.signature(pred.predict)
    params = list(sig.parameters.values())

    expected = [
        "prompt",
        "trigger_word",
        "init_image",
        "strength",
        "guidance_scale",
        "num_inference_steps",
        "seed",
    ]

    assert [p.name for p in params] == expected, (
        "Predict signature changed unexpectedly. "
        f"Expected {expected}, got {[p.name for p in params]}"
    )


# -----------------------------------------------------------------------------
# Contract 2 — Required fields must be enforced
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_missing_required_fields_raise():
    sig = inspect.signature(Predictor.predict)
    params = sig.parameters

    required_fields = ["prompt", "trigger_word", "init_image"]

    for name in required_fields:
        assert name in params, f"{name} missing from predict signature"

        field: FieldInfo = params[name].default
        assert isinstance(field, FieldInfo), f"{name} must be a FieldInfo (Cog Input)"

        # Cog marks required fields by the absence of a *default attribute*
        assert field.default is PydanticUndefined, (
            f"{name} must be required (no default), found default={field.default}"
        )

    # Calling unbound method without required args must raise TypeError
    with pytest.raises(TypeError):
        Predictor.predict()


# -----------------------------------------------------------------------------
# Contract 3 — Input() defaults and constraints must remain identical to schema
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_input_constraints_intact():
    pred = Predictor()
    sig = inspect.signature(pred.predict)
    params = sig.parameters

    # Get FieldInfo objects from parameters
    strength_meta: Input = params["strength"].default
    guidance_meta: Input = params["guidance_scale"].default
    steps_meta: Input = params["num_inference_steps"].default
    seed_meta: Input = params["seed"].default

    # Helper to extract constraints from metadata
    def get_constraints(field_info):
        constraints = {}
        for item in field_info.metadata:
            if hasattr(item, "ge"):
                constraints["ge"] = item.ge
            if hasattr(item, "le"):
                constraints["le"] = item.le
        return constraints

    # strength: 0..1
    strength_constraints = get_constraints(strength_meta)
    assert strength_constraints.get("ge") == 0
    assert strength_constraints.get("le") == 1

    # guidance_scale: ge=0
    guidance_constraints = get_constraints(guidance_meta)
    assert guidance_constraints.get("ge") == 0

    # num_inference_steps: ge=1
    steps_constraints = get_constraints(steps_meta)
    assert steps_constraints.get("ge") == 1

    # Verify FieldInfo objects exist
    assert strength_meta is not None
    assert guidance_meta is not None
    assert steps_meta is not None
    assert seed_meta is not None


# -----------------------------------------------------------------------------
# Contract 4 — Invalid numerical values must raise BEFORE heavy compute
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_numeric_arguments_raise():
    pred = Predictor()
    pred.setup = lambda: None  # disable loading huge pipeline
    pred.lora2_triggers = {}  # Initialize required attribute
    pred.pipe = None

    # strength out of bounds
    with pytest.raises(Exception):
        pred.predict(prompt="hello", trigger_word="Cinematic", init_image="x.png", strength=-0.1)

    with pytest.raises(Exception):
        pred.predict(prompt="hello", trigger_word="Cinematic", init_image="x.png", strength=2.0)

    # num_inference_steps invalid
    with pytest.raises(Exception):
        pred.predict(
            prompt="hello", trigger_word="Cinematic", init_image="x.png", num_inference_steps=0
        )


# -----------------------------------------------------------------------------
# Minimal mocks needed for running predict() without GPU / Internet
# -----------------------------------------------------------------------------
class DummyPipeline:
    def __init__(self):
        self._images = [ImagePlaceholder()]

    def __call__(self, **kwargs):
        return self

    @property
    def images(self):
        return self._images


class ImagePlaceholder:
    """Fake PIL Image-like object with save()."""

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"fake-image")


# -----------------------------------------------------------------------------
# Contract 5 — predict() must return a Path object
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_returns_path(tmp_path, monkeypatch):
    pred = Predictor()

    # Initialize required attributes
    pred.lora2_triggers = {"Cinematic": "cinematic-adapter", "GHIBSKY": "flux-ghibsky"}
    pred.current_adapter = None

    # mock everything heavy:
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.load_image", lambda url: ImagePlaceholder()
    )

    class DummyPipe:
        def set_adapters(self, names, adapter_weights):
            pass

        def __call__(self, **kwargs):
            return DummyPipeline()

    pred.pipe = DummyPipe()

    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.save_image",
        lambda img, output_dir=tmp_path: tmp_path / "result.png",
    )

    # Explicitly pass all parameters
    out = pred.predict(
        prompt="hello",
        trigger_word="Cinematic",
        init_image="http://fake",
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=28,
        seed=42,
    )

    assert isinstance(out, Path)
    assert out.name == "result.png"


# -----------------------------------------------------------------------------
# Contract 6 — Trigger word must select correct adapter
# (No real LoRA loading; just check state transition)
# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_trigger_word_switching(monkeypatch):
    pred = Predictor()

    # Initialize required attributes
    pred.lora2_triggers = {"Cinematic": "cinematic-adapter", "GHIBSKY": "flux-ghibsky"}
    pred.current_adapter = "open-image-preferences"

    # Mock image
    monkeypatch.setattr(
        "models.flux_fast_lora_hotswap_img2img.predict.load_image", lambda url: ImagePlaceholder()
    )

    class DummyPipe:
        def __init__(self):
            self.adapter_history = []

        def set_adapters(self, names, adapter_weights):
            self.adapter_history.append(names[0])

        def __call__(self, **k):
            return DummyPipeline()

    pred.pipe = DummyPipe()

    # use GHIBSKY → should switch to flux-ghibsky
    pred.predict(
        prompt="hello",
        trigger_word="GHIBSKY",
        init_image="http://fake",
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=28,
        seed=42,
    )

    assert pred.current_adapter == "flux-ghibsky"
    assert pred.pipe.adapter_history[-1] == "flux-ghibsky"
