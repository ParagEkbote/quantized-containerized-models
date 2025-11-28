import pytest
import inspect
import torch
from pathlib import Path
from unittest.mock import Mock, patch

# ---------------------------------------------------------------------------
# FIX: Patch BEFORE any imports that use Unsloth
# ---------------------------------------------------------------------------
# Create mock objects
class DummyTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def __call__(self, prompt, *a, **k):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, **k):
        return "MOCK_GENERATED_TEXT"

class DummyModel:
    def parameters(self):
        yield torch.zeros(1)

    def generate(self, *a, **k):
        return torch.tensor([[1, 2, 3, 4]])

def fake_from_pretrained(*a, **k):
    return DummyModel(), DummyTokenizer()

# Patch at module level BEFORE importing Predictor
with patch('torch.cuda.is_available', return_value=False), \
     patch('torch.cuda.is_bf16_supported', return_value=False):
    
    # Mock unsloth before import
    import sys
    from unittest.mock import MagicMock
    
    mock_unsloth = MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained = fake_from_pretrained
    sys.modules['unsloth'] = mock_unsloth
    
    # Now import predictor
    from models.phi4_reasoning_plus_unsloth.predict import Predictor
    from cog import Input


# ---------------------------------------------------------------------------
# 1. CONTRACT: Function signature must not change
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_signature_stable():
    pred = Predictor()
    sig = inspect.signature(pred.predict)
    params = [p.name for p in sig.parameters.values()]
    expected = ["prompt", "max_new_tokens", "temperature", "top_p", "seed"]
    assert params == expected


# ---------------------------------------------------------------------------
# 2. CONTRACT: Required fields must remain required
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_required_prompt_enforced():
    pred = Predictor()
    with pytest.raises(TypeError):
        pred.predict()  # no prompt


# ---------------------------------------------------------------------------
# 3. CONTRACT: Input() defaults & constraints must match schema
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_input_constraints_intact():
    pred = Predictor()
    sig = inspect.signature(pred.predict)
    params = sig.parameters

    max_new_tokens_meta: Input = params['max_new_tokens'].default
    temp_meta: Input = params['temperature'].default
    topp_meta: Input = params['top_p'].default
    seed_meta: Input = params['seed'].default

    # Extract constraints from metadata
    def get_constraints(field_info):
        constraints = {}
        for item in field_info.metadata:
            if hasattr(item, 'ge'):
                constraints['ge'] = item.ge
            if hasattr(item, 'le'):
                constraints['le'] = item.le
            if hasattr(item, 'gt'):
                constraints['gt'] = item.gt
            if hasattr(item, 'lt'):
                constraints['lt'] = item.lt
        return constraints

    # max_new_tokens
    max_tok_constraints = get_constraints(max_new_tokens_meta)
    assert max_tok_constraints.get('ge') == 1
    assert max_tok_constraints.get('le') == 25000

    # temperature
    temp_constraints = get_constraints(temp_meta)
    assert temp_constraints.get('ge') == 0.1
    assert temp_constraints.get('le') == 1

    # top_p
    topp_constraints = get_constraints(topp_meta)
    assert topp_constraints.get('ge') == 0.1
    assert topp_constraints.get('le') == 1

    # Verify descriptions exist
    assert max_new_tokens_meta.description is not None
    assert temp_meta.description is not None
    assert topp_meta.description is not None
    assert seed_meta.description is not None


# ---------------------------------------------------------------------------
# 4. CONTRACT: Invalid numeric inputs must raise
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_numeric_inputs_raise():
    pred = Predictor()
    pred.setup()  # initializes dummy model

    with pytest.raises(Exception):
        pred.predict(prompt="hi", max_new_tokens=0)

    with pytest.raises(Exception):
        pred.predict(prompt="hi", temperature=5)

    with pytest.raises(Exception):
        pred.predict(prompt="hi", top_p=-1)


# ---------------------------------------------------------------------------
# 5. CONTRACT: predict() must return a Path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_returns_path(tmp_path, monkeypatch):
    pred = Predictor()
    pred.setup()

    # Patch save_text to deterministic file
    monkeypatch.setattr(
        "models.phi4_reasoning_plus_unsloth.predict.save_text",
        lambda *args, **kwargs: tmp_path / "out.txt",
    )

    # Explicitly pass all parameter values
    result = pred.predict(
        prompt="hello",
        max_new_tokens=3000,
        temperature=0.7,
        top_p=0.95,
        seed=42
    )

    assert isinstance(result, Path)
    assert result.name == "out.txt"