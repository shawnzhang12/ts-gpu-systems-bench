from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def export_preprocess_trace(trace_path: str | Path, fn, warmup: int = 3, steps: int = 10) -> Path:
    out = Path(trace_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        for _ in range(steps):
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(out))
    return out
