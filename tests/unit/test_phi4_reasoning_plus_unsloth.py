import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Dummy test doubles (must match real interface)
# ---------------------------------------------------------------------------
class DummyTokenizer:
    eos_token = "</s>"
    eos_token_id = 1
    pad_token = "</s>"
    pad_token_id = 1

    def __call__(self, prompt, *a, **k):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, ids, **k):
        return "MOCK_GENERATED_TEXT"


class DummyModel:
    device = torch.device("cpu")

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, *a, **k):
        return torch.tensor([[1, 2, 3, 4, 5]])


def fake_from_pretrained(*a, **k):
    return DummyModel(), DummyTokenizer()


# ---------------------------------------------------------------------------
# Patch environment BEFORE importing Predictor
# ---------------------------------------------------------------------------
with (
    patch("torch.cuda.is_available", return_value=False),
    patch("torch.cuda.is_bf16_supported", return_value=False),
):
    import sys
    from unittest.mock import MagicMock

    mock_unsloth = MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained = fake_from_pretrained
    mock_unsloth.FastLanguageModel.for_inference = lambda *a, **k: None
    sys.modules["unsloth"] = mock_unsloth

    from cog import Input

    from models.phi4_reasoning_plus_unsloth.predict import Predictor


# ---------------------------------------------------------------------------
# 1. CONTRACT: Function signature must not change
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_signature_stable():
    pred = Predictor()
    sig = inspect.signature(pred.predict)
    params = [p.name for p in sig.parameters.values()]

    expected = [
        "prompt",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    ]
    assert params == expected


# ---------------------------------------------------------------------------
# 2. CONTRACT: Required prompt enforced via invalid values
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_required_prompt_enforced():
    pred = Predictor()

    with pytest.raises(ValueError):
        pred.predict(prompt="")

    with pytest.raises(ValueError):
        pred.predict(prompt="   ")


# ---------------------------------------------------------------------------
# 3. CONTRACT: Input() defaults & constraints must match schema
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_input_constraints_intact():
    sig = inspect.signature(Predictor.predict)
    params = sig.parameters

    max_new_tokens_meta: Input = params["max_new_tokens"].default
    temp_meta: Input = params["temperature"].default
    topp_meta: Input = params["top_p"].default
    seed_meta: Input = params["seed"].default

    def get_constraints(field_info):
        constraints = {}
        for item in field_info.metadata:
            for k in ("ge", "le", "gt", "lt"):
                if hasattr(item, k):
                    constraints[k] = getattr(item, k)
        return constraints

    # max_new_tokens
    max_tok = get_constraints(max_new_tokens_meta)
    assert max_tok["ge"] == 1
    assert max_tok["le"] == 40000

    # temperature
    temp = get_constraints(temp_meta)
    assert temp["ge"] == 0.0
    assert temp["le"] == 1.0

    # top_p
    topp = get_constraints(topp_meta)
    assert topp["ge"] == 0.0
    assert topp["le"] == 1.0

    # Descriptions must exist
    assert max_new_tokens_meta.description
    assert temp_meta.description
    assert topp_meta.description
    assert seed_meta.description


# ---------------------------------------------------------------------------
# 4. CONTRACT: Invalid numeric inputs must raise at runtime
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_numeric_inputs_raise():
    pred = Predictor()
    pred.setup()

    with pytest.raises(Exception):
        pred.predict(prompt="hi", max_new_tokens=0)

    with pytest.raises(Exception):
        pred.predict(prompt="hi", temperature=5.0)

    with pytest.raises(Exception):
        pred.predict(prompt="hi", top_p=-1.0)


# ---------------------------------------------------------------------------
# 5. CONTRACT: predict() must return a Path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_predict_returns_path():
    pred = Predictor()
    pred.setup()

    result = pred.predict(
        prompt="hello",
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        seed=42,
    )

    assert isinstance(result, Path)
    assert result.name.startswith("output_")
    assert result.suffix == ".txt"
    assert result.exists()
