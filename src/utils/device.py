from __future__ import annotations

import os
from typing import Any, Dict


def configure_runtime_env() -> None:
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")


def torch_available() -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    return True


def get_torch_device() -> Any:
    configure_runtime_env()
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for device selection.") from exc
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_summary() -> Dict[str, Any]:
    configure_runtime_env()
    summary: Dict[str, Any] = {"torch_available": False, "device": "cpu"}
    try:
        import torch  # type: ignore
    except ImportError:
        summary["reason"] = "torch not installed"
        return summary

    summary["torch_available"] = True
    summary["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    summary["gpu_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        summary["gpu_name"] = torch.cuda.get_device_name(0)
    return summary
