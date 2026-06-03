from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "ltx23-h100-bench"
GPU = "H100"
REMOTE_REPO = "/workspace/video_gen"
FASTVIDEO_DIR = "/opt/FastVideo"

cache_volume = modal.Volume.from_name("ltx23-model-cache", create_if_missing=True)

preflight_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
)

fastvideo_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate",
        "diffusers",
        "einops",
        "fastapi",
        "hf_transfer",
        "httpx",
        "imageio[ffmpeg]",
        "numpy<2",
        "opencv-python-headless",
        "pydantic>=2.8.0",
        "safetensors",
        "sentencepiece",
        "setuptools-scm",
        "transformers<5",
        "uvicorn[standard]",
        "wheel",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/hao-ai-lab/FastVideo.git /opt/FastVideo",
        "cd /opt/FastVideo && pip install -e . --no-build-isolation",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/huggingface",
            "PYTHONPATH": f"{REMOTE_REPO}:{FASTVIDEO_DIR}",
            "FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA",
            "FASTVIDEO_STAGE_LOGGING": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        }
    )
    .workdir(REMOTE_REPO)
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO,
        ignore=[
            ".git",
            ".venv",
            ".playwright-cli",
            "__pycache__",
            "output",
            "outputs",
        ],
    )
)

app = modal.App(APP_NAME, volumes={"/cache": cache_volume})


@app.function(gpu=GPU, image=preflight_image, timeout=5 * 60)
def h100_preflight() -> dict:
    import torch

    started_at = time.time()
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
    torch_cuda = torch.cuda.is_available()
    payload = {
        "started_at": started_at,
        "gpu": smi.strip(),
        "torch": torch.__version__,
        "cuda_available": torch_cuda,
        "device_name": torch.cuda.get_device_name(0) if torch_cuda else None,
        "capability": torch.cuda.get_device_capability(0) if torch_cuda else None,
    }
    print(json.dumps(payload, indent=2))
    return payload


@app.function(gpu=GPU, image=fastvideo_image, timeout=60 * 60)
def run_fastvideo_benchmark(
    bucket: str = "fast_5s_v1",
    runs: int = 1,
    warmup: int = 0,
    steps: int = 5,
    quant: str = "none",
    attention_backend: str = "TORCH_SDPA",
    compile_model: bool = False,
    save_video: bool = True,
) -> dict:
    output_dir = Path("/cache/benchmarks") / f"{bucket}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "benchmarks.fastvideo_ltx23_benchmark",
        "--bucket",
        bucket,
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--quant",
        quant,
        "--attention-backend",
        attention_backend,
        "--output-dir",
        str(output_dir),
    ]
    if compile_model:
        cmd.append("--compile")
    if save_video:
        cmd.append("--save-video")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    result = {
        "returncode": proc.returncode,
        "output_dir": str(output_dir),
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
    }
    summary_files = sorted(output_dir.glob("*_summary.json"))
    if summary_files:
        result["summary"] = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    cache_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    bucket: str = "fast_5s_v1",
    runs: int = 1,
    warmup: int = 0,
    steps: int = 5,
    quant: str = "none",
    attention_backend: str = "TORCH_SDPA",
    compile_model: bool = False,
    save_video: bool = True,
) -> None:
    if mode == "preflight":
        print(h100_preflight.remote())
        return
    if mode == "fastvideo":
        print(
            run_fastvideo_benchmark.remote(
                bucket=bucket,
                runs=runs,
                warmup=warmup,
                steps=steps,
                quant=quant,
                attention_backend=attention_backend,
                compile_model=compile_model,
                save_video=save_video,
            )
        )
        return
    raise ValueError(f"Unknown mode: {mode}")
