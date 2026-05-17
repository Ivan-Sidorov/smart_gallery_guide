"""Torch device selection for local HuggingFace pipelines."""

import torch


def resolve_pipeline_device(explicit: str | None = None) -> int | str:
    """Pick a device for `transformers.pipeline` (CUDA index, `mps`, or `-1` for CPU).

    Args:
        explicit: `None`/`auto` - cuda>mps>cpu;
            `mps`/`apple`/`metal` - Apple Silicon GPU;
            `cuda`/`gpu` - NVIDIA GPU;
            `cpu` - CPU;
            `0`, `1`, etc. - CUDA device index.

    Raises:
        RuntimeError: if the requested device is not available.
    """
    if explicit and explicit.strip().lower() not in ("", "auto"):
        key = explicit.strip().lower()
        if key == "cpu":
            return -1
        if key in ("mps", "apple", "metal"):
            _require_mps()
            return "mps"
        if key in ("cuda", "gpu"):
            _require_cuda()
            return 0
        if key.isdigit():
            idx = int(key)
            _require_cuda()
            return idx
        if key.startswith("cuda:"):
            _require_cuda()
            return key
        return explicit

    if torch.cuda.is_available():
        return 0
    if _mps_ready():
        return "mps"
    return -1


def empty_device_cache(device: int | str) -> None:
    """Release cached memory after a model run."""
    if isinstance(device, str) and device.startswith("mps"):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return
    if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mps_ready() -> bool:
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _require_mps() -> None:
    if not getattr(torch.backends, "mps", None):
        raise RuntimeError("MPS backend is not present in this PyTorch build")
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was built without MPS (Apple Silicon GPU) support")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available (macOS 12.3+ and Apple Silicon or AMD GPU required)"
        )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
