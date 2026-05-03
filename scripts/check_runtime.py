from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from typing import Any


def _try_import(name: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return {
            "ok": True,
            "version": str(version),
            "import_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "import_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate runtime dependencies and CUDA visibility.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero if CUDA is unavailable.")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="Exit non-zero if flash-attn import fails.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {},
    }

    modules = [
        "torch",
        "triton",
        "tilelang",
        "mamba_ssm",
        "flash_attn",
        "mlflow",
        "optuna",
    ]

    for name in modules:
        report["modules"][name] = _try_import(name)

    try:
        from src.models.transformer_model import flash_impl_name

        report["transformer_flash_impl"] = flash_impl_name()
    except Exception as exc:
        report["transformer_flash_impl"] = f"unavailable: {exc}"

    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch_cuda": str(torch.version.cuda),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cuda["device_name"] = torch.cuda.get_device_name(0)
            cuda["capability"] = list(torch.cuda.get_device_capability(0))
        report["cuda"] = cuda
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}

    print(json.dumps(report, indent=2))

    # hard fail only when core stack is broken
    core = ["torch", "triton", "mlflow", "optuna"]
    failed = [name for name in core if not report["modules"][name]["ok"]]
    if args.require_cuda and not report.get("cuda", {}).get("available", False):
        failed.append("cuda")
    if args.require_flash_attn and not report["modules"]["flash_attn"]["ok"]:
        failed.append("flash_attn")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
