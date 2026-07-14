from __future__ import annotations

import platform
import sys
import time

import torch


def to_gib(value: int) -> float:
    return value / (1024 ** 3)


def main() -> None:
    print("=" * 68)
    print("HSI PROJECT - GPU ENVIRONMENT VERIFICATION")
    print("=" * 68)

    print(f"Python:            {platform.python_version()}")
    print(f"Executable:        {sys.executable}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA runtime:      {torch.version.cuda}")
    print(f"CUDA available:    {torch.cuda.is_available()}")
    print(f"cuDNN version:     {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        raise SystemExit("GPU TEST FAILED: CUDA is unavailable.")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print(f"Selected GPU:      {properties.name}")
    print(f"Total GPU memory:  {to_gib(properties.total_memory):.2f} GiB")
    print(
        f"Compute capability:{properties.major}.{properties.minor}"
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nRunning RTX 4070 matrix test...")

    size = 4096
    matrix_a = torch.randn(
        size,
        size,
        dtype=torch.float32,
        device=device,
    )
    matrix_b = torch.randn(
        size,
        size,
        dtype=torch.float32,
        device=device,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    result = matrix_a @ matrix_b

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if result.device.type != "cuda":
        raise RuntimeError("Calculation did not execute on CUDA.")

    if not torch.isfinite(result).all():
        raise RuntimeError("GPU calculation produced invalid values.")

    print(f"Result shape:      {tuple(result.shape)}")
    print(f"Execution time:    {elapsed:.4f} seconds")
    print(
        "Peak GPU memory:  "
        f"{to_gib(torch.cuda.max_memory_allocated()):.3f} GiB"
    )
    print(f"Result checksum:   {result.mean().item():.8f}")

    del matrix_a, matrix_b, result
    torch.cuda.empty_cache()

    print("\nGPU TEST PASSED")
    print("RTX 4070 is ready for HSI model training.")
    print("=" * 68)


if __name__ == "__main__":
    main()

