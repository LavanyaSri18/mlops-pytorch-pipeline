import pytest
import torch

from src.model import get_model


def test_resnet18_returns_one_logit_vector_per_image():
    model = get_model("resnet18", num_classes=10)
    result = model(torch.randn(2, 3, 32, 32))
    assert result.shape == (2, 10)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model("unknown-model")
