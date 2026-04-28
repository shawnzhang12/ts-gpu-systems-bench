from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class DeviceMetadata:
    device: str
    cuda_available: bool
    gpu_name: str
    capability: str
    torch_version: str
    cuda_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_device() -> DeviceMetadata:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        capability = f"{props.major}.{props.minor}"
        return DeviceMetadata(
            device=f"cuda:{idx}",
            cuda_available=True,
            gpu_name=props.name,
            capability=capability,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "unknown",
        )

    return DeviceMetadata(
        device="cpu",
        cuda_available=False,
        gpu_name="none",
        capability="n/a",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "none",
    )
