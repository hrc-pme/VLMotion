#!/usr/bin/env python3
"""Fail early when the training container cannot access CUDA."""

import sys

import torch


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable. Start the container with NVIDIA GPU access "
            "and verify that /dev/nvidia0 and /dev/nvidiactl exist.",
            file=sys.stderr,
        )
        return 1
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
