import inspect
import pytest
from models.smollm3_pruna.predict import (
    Predictor,
    save_text,
)

import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Contract Test 1: Function signature should not change unexpectedly
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_function_signature_stable():
    pred = Predictor()

    sig = inspect.signature(pred.predict)
    params = list(sig.parameters.values())

    # Expected signature order
    expected = ["prompt", "max_new_tokens", "mode", "seed"]
    assert [p.name for p in params] == expected, (
        "Predict function signature has changed. "
        f"Expected {expected}, got {[p.name for p in params]}"
    )


# ---------------------------------------------------------------------------
# Contract Test 2: Required fields must remain required
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_required_fields_are_enforced():
    pred = Predictor()
    pred.cache_length = 2048

    with pytest.raises(TypeError):
        pred.predict()   # Missing required parameter "prompt"


# ---------------------------------------------------------------------------
# Contract Test 3: Validate Input() constraints are present
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_input_constraints_intact():
    pred = Predictor()
    
    # Get the actual signature to inspect defaults properly
    sig = inspect.signature(pred.predict)
    params = sig.parameters
    
    # Check max_new_tokens
    max_new_tokens_param = params['max_new_tokens']
    max_new_tokens_meta = max_new_tokens_param.default
    
    # Verify it's a FieldInfo object (Cog Input)
    assert hasattr(max_new_tokens_meta, 'metadata'), "max_new_tokens should be a Cog Input"
    
    # Check for constraints in metadata
    constraints = {}
    for item in max_new_tokens_meta.metadata:
        if hasattr(item, 'ge'):
            constraints['ge'] = item.ge
        if hasattr(item, 'le'):
            constraints['le'] = item.le
    
    assert constraints.get('ge') == 1, "max_new_tokens minimum should be 1"
    assert constraints.get('le') == 16384, "max_new_tokens maximum should be 16384"
    
    # Check mode parameter
    mode_param = params['mode']
    mode_meta = mode_param.default
    assert "think" in mode_meta.description.lower()
    assert "no_think" in mode_meta.description.lower()
    
    # Check seed parameter
    seed_param = params['seed']
    seed_meta = seed_param.default
    assert "seed" in seed_meta.description.lower()
    
    # Verify descriptions exist
    assert max_new_tokens_meta.description is not None
    assert mode_meta.description is not None
    assert seed_meta.description is not None


# ---------------------------------------------------------------------------
# Contract Test 4: Invalid numerical arguments must raise errors
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_max_new_tokens_raises():
    pred = Predictor()
    pred.setup = lambda: None  # disable heavy setup
    pred.cache_length = 2048

    with pytest.raises(Exception):
        pred.predict(prompt="hi", max_new_tokens=0)

    with pytest.raises(Exception):
        pred.predict(prompt="hi", max_new_tokens=999999)


# ---------------------------------------------------------------------------
# Contract Test 5: Invalid mode must raise errors
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_mode_raises():
    pred = Predictor()
    pred.setup = lambda: None
    pred.cache_length = 2048

    with pytest.raises(Exception):
        pred.predict(prompt="hello", mode="invalid_mode")


# ---------------------------------------------------------------------------
# Contract Test 6: predict() must return a valid Path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_returns_path(tmp_path, monkeypatch):
    pred = Predictor()
    pred.cache_length = 2048

    # Mock tokenizer
    class MockTok:
        pad_token = "x"
        eos_token = "x"
        eos_token_id = 2  

        def apply_chat_template(self, *args, **kwargs):
            return "text"

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, **kwargs):
            return "GENERATED"

    pred.tokenizer = MockTok()

    # Mock model
    class MockModel:
        def parameters(self):
            yield torch.zeros(1)

        def generate(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])

    pred.smashed_text_model = MockModel()
    pred.cache_length = 2048

    # Mock save_text
    monkeypatch.setattr(
        "models.smollm3_pruna.predict.save_text",
        lambda *args, **kwargs: tmp_path / "out.txt"
    )

    result = pred.predict(prompt="hello",
        max_new_tokens=512,
        mode="no_think",
        seed=-1)

    assert isinstance(result, Path)
    assert result.name == "out.txt"

