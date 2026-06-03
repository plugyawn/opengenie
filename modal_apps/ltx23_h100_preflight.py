from __future__ import annotations

import json
import subprocess
import time

import modal


APP_NAME = "ltx23-h100-preflight"
GPU = "H100"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "numpy<2",
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
)

app = modal.App(APP_NAME)


@app.function(gpu=GPU, image=image, timeout=5 * 60)
def h100_preflight() -> dict:
    import torch

    started_at = time.time()
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    )
    torch_cuda = torch.cuda.is_available()
    payload = {
        "started_at": started_at,
        "gpu": smi.strip(),
        "torch": str(torch.__version__),
        "cuda_available": torch_cuda,
        "device_name": torch.cuda.get_device_name(0) if torch_cuda else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch_cuda else None,
    }
    print(json.dumps(payload, indent=2))
    return payload


@app.local_entrypoint()
def main() -> None:
    print(h100_preflight.remote())
