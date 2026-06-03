from __future__ import annotations

import os
from pathlib import Path

import modal

from modal_apps.ltx23_official_warm_experiments import (
    GPU,
    REMOTE_REPO,
    cache_volume,
    official_image,
    _ensure_assets,
    _worker_env,
)


APP_NAME = "ltx23-live-h100-worker"
LIVE_BUCKET = "hd_16x9_15fps_5s_overlap22_v1"
live_image = official_image.add_local_dir(
    "modal_apps",
    remote_path=f"{REMOTE_REPO}/modal_apps",
    ignore=["__pycache__"],
)

app = modal.App(APP_NAME, volumes={"/cache": cache_volume})


@app.function(
    gpu=GPU,
    image=live_image,
    min_containers=1,
    max_containers=1,
    buffer_containers=0,
    timeout=60 * 60,
    scaledown_window=60 * 60,
    secrets=[modal.Secret.from_dotenv(filename="env.local")],
)
@modal.asgi_app()
def live_worker():
    assets = _ensure_assets()
    output_root = Path("/cache/live_h100_outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    _worker_env(
        output_root=output_root,
        bucket=LIVE_BUCKET,
        checkpoint_path=assets["checkpoint_path"],
        gemma_root=assets["gemma_root"],
        attention="sdpa-cudnn",
        emit_overlap_frames=False,
    )
    os.environ.update(
        {
            "LTX_WORKER_LOAD_ON_STARTUP": "1",
            "LTX_WORKER_ENCODE_DETACH_AFTER_FIRST_BYTE": "1",
            "LTX_WORKER_ENCODE_CPU_MATERIALIZE_BEFORE_DETACH": "0",
            "LTX_WORKER_ENCODE_AUDIO_PIPE": "1",
            "LTX_WORKER_LATENT_CONTINUATION_EMIT_TO_TAIL": "1",
        }
    )

    from ltx_serve.remote_official_worker import app as fastapi_app
    from ltx_serve.remote_official_worker import get_engine

    get_engine().load()
    return fastapi_app
