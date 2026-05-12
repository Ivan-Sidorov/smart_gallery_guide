"""Unit tests for VisionEncoder output normalization."""

from types import SimpleNamespace

import torch

from core.encoders.vision import VisionEncoder


def test_ensure_tensor_accepts_plain_tensor() -> None:
    raw = torch.ones(2, 4)
    out = VisionEncoder._ensure_tensor(raw)
    assert out.shape == (2, 4)


def test_ensure_tensor_uses_pooler_output() -> None:
    raw = SimpleNamespace(pooler_output=torch.ones(1, 8))
    out = VisionEncoder._ensure_tensor(raw)
    assert out.shape == (1, 8)


def test_ensure_tensor_pools_last_hidden_state() -> None:
    raw = SimpleNamespace(last_hidden_state=torch.ones(1, 3, 6))
    out = VisionEncoder._ensure_tensor(raw)
    assert out.shape == (1, 6)
