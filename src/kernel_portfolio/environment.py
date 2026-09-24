"""Small, serializable environment report for reproducible measurements."""

import importlib.metadata
import json
import platform
import subprocess

import torch


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def describe():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        # Native Windows uses the community "triton-windows" distribution of the same module.
        "triton": _version("triton") or _version("triton-windows"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            gpu=props.name,
            device_index=index,
            compute_capability=list(torch.cuda.get_device_capability(index)),
            total_memory_bytes=props.total_memory,
            l2_cache_bytes=getattr(props, "L2_cache_size", None),
            multiprocessor_count=props.multi_processor_count,
        )
    return info


def main():
    print(json.dumps(describe(), indent=2))


if __name__ == "__main__":
    main()
