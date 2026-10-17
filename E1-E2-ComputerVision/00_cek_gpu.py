"""Verify this Python environment actually performs a CUDA operation."""
from __future__ import annotations
import json
import sys


def main():
    try:
        import torch
        import ultralytics
        from common import require_gpu
        info = require_gpu(0)
        x = torch.ones((256, 256), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        assert y[0, 0].item() == 256
        info["ultralytics"] = ultralytics.__version__
        print(json.dumps(info, indent=2))
        print("\nLULUS: CUDA tersedia dan operasi GPU berhasil. Siap menyiapkan dataset.")
        return 0
    except Exception as exc:
        print(f"BELUM SIAP: {exc}")
        print("Periksa nvidia-smi, driver NVIDIA, dan pemasangan PyTorch CUDA di .venv yang benar.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
