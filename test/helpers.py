"""Various test helpers that don't work as fixtures."""
import torch
import pytest


GPU_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU_SKIP)]
parametrize_device = pytest.mark.parametrize("device", DEVICES)
