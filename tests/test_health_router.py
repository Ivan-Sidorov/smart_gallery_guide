"""Unit tests for health router helpers."""

from api.routers.health import _vllm_models_url


def test_vllm_models_url_when_base_ends_with_v1() -> None:
    assert _vllm_models_url("http://vllm:8000/v1") == "http://vllm:8000/v1/models"


def test_vllm_models_url_when_base_without_v1() -> None:
    assert _vllm_models_url("http://vllm:8000") == "http://vllm:8000/v1/models"
