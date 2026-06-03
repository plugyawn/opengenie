from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "ltx23-official-warm-experiments"
GPU = "H100"
REMOTE_REPO = "/workspace/video_gen"
LTX2_DIR = "/opt/LTX-2"
LTX2_COMMIT = "d6053703e00195bc668cbd1d5eda9dc0b2e7b74a"

cache_volume = modal.Volume.from_name("ltx23-model-cache", create_if_missing=True)

official_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.0.1",
        "av==17.0.1",
        "diffusers==0.38.0",
        "einops==0.8.2",
        "fastapi==0.124.2",
        "hf_transfer==0.1.9",
        "huggingface_hub==0.36.0",
        "numpy<2",
        "opencv-python-headless==4.11.0.86",
        "openimageio==3.0.19.0",
        "pillow",
        "pydantic==2.13.4",
        "safetensors==0.8.0rc1",
        "scipy==1.17.1",
        "sentencepiece==0.2.1",
        "tqdm==4.67.3",
        "transformers==4.57.3",
        "uvicorn[standard]==0.38.0",
    )
    .run_commands(
        f"git clone https://github.com/Lightricks/LTX-2.git {LTX2_DIR}",
        f"cd {LTX2_DIR} && git checkout {LTX2_COMMIT}",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/huggingface",
            "PYTHONPATH": (
                f"{REMOTE_REPO}:"
                f"{LTX2_DIR}/packages/ltx-core/src:"
                f"{LTX2_DIR}/packages/ltx-pipelines/src"
            ),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        }
    )
    .workdir(REMOTE_REPO)
    .add_local_dir(
        "benchmarks",
        remote_path=f"{REMOTE_REPO}/benchmarks",
        ignore=["__pycache__"],
    )
    .add_local_dir(
        "ltx_serve",
        remote_path=f"{REMOTE_REPO}/ltx_serve",
        ignore=["__pycache__"],
    )
    .add_local_dir(
        "frame_gallery",
        remote_path=f"{REMOTE_REPO}/frame_gallery",
        ignore=["__pycache__"],
    )
    .add_local_file("pyproject.toml", remote_path=f"{REMOTE_REPO}/pyproject.toml")
)

app = modal.App(APP_NAME, volumes={"/cache": cache_volume})


HAMSTER_PROMPTS = {
    "hamster_wheel": (
        "A realistic close-up documentary video of a small golden hamster running inside a transparent "
        "exercise wheel in a cozy living room, natural daylight, shallow depth of field, detailed fur, "
        "stable camera, real-world lighting, subtle wheel motion, no text, no captions, generated subtle "
        "room audio and soft wheel squeak."
    ),
    "hamster_water": (
        "A realistic close-up documentary video of a small golden hamster inside a transparent exercise "
        "wheel in a cozy living room as a gentle stream of clear water falls around it and lightly splashes "
        "over its fur, playful safe surreal scene, natural daylight, detailed wet fur, stable camera, no "
        "text, no captions, generated soft water and wheel audio."
    ),
    "hamster_purple": (
        "A realistic close-up documentary video of a small golden hamster running inside a transparent "
        "exercise wheel in a cozy living room, its fur gradually turns vivid purple while it keeps running, "
        "natural daylight, detailed fur, stable camera, real-world lighting, no text, no captions, generated "
        "subtle room audio and soft wheel squeak."
    ),
}

STEERING_PLANS = {
    "water_purple_run": [
        {
            "name": "base",
            "prompt": HAMSTER_PROMPTS["hamster_wheel"],
        },
        {
            "name": "water_falls",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "A gentle stream of clear water starts falling around the hamster and lightly splashes over its "
                "fur while it keeps running, playful safe surreal scene, detailed wet fur, no text, no captions, "
                "generated soft water and wheel audio."
            ),
        },
        {
            "name": "turns_purple",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "The hamster keeps running as its fur gradually turns vivid purple, realistic fur detail, stable "
                "camera, no text, no captions, generated subtle room audio and soft wheel squeak."
            ),
        },
        {
            "name": "purple_continues",
            "prompt": (
                "Continue the exact same purple hamster, transparent exercise wheel, room, lighting, and camera "
                "angle. The hamster shakes off a few water droplets and continues running steadily in the wheel, "
                "realistic documentary style, no text, no captions, generated soft wheel and room audio."
            ),
        },
    ],
    "jump_turn_face": [
        {
            "name": "base",
            "prompt": HAMSTER_PROMPTS["hamster_wheel"],
        },
        {
            "name": "jumps",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "The hamster makes a small playful jump inside the wheel and lands safely, then keeps moving, "
                "natural fur motion, no text, no captions, generated soft wheel and room audio."
            ),
        },
        {
            "name": "turns",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "The hamster slows down, turns its body, and briefly faces the camera while staying inside the "
                "wheel, realistic close-up, no text, no captions, generated soft room audio."
            ),
        },
        {
            "name": "runs_again",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "The hamster turns forward again and resumes running smoothly in the wheel, stable documentary "
                "camera, no text, no captions, generated soft wheel squeak."
            ),
        },
    ],
    "continue_then_color": [
        {
            "name": "base",
            "prompt": HAMSTER_PROMPTS["hamster_wheel"],
        },
        {
            "name": "no_new_action",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "No new action is introduced; the hamster simply keeps running naturally in the wheel, stable "
                "camera, realistic fur, no text, no captions, generated soft wheel and room audio."
            ),
        },
        {
            "name": "blue_light",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "A soft blue room light gradually turns on while the hamster keeps running, realistic close-up, "
                "no text, no captions, generated subtle room audio."
            ),
        },
        {
            "name": "purple_hamster",
            "prompt": (
                "Continue the exact same hamster, transparent exercise wheel, room, lighting, and camera angle. "
                "The hamster gradually turns vivid purple while the soft blue room light remains on and it keeps "
                "running, detailed fur, no text, no captions, generated soft wheel audio."
            ),
        },
    ],
}

STEERING_SEGMENTS = STEERING_PLANS["water_purple_run"]


RABBIT_IMAGE_ACTION_PLAN = [
    {
        "name": "forward",
        "text": "rabbit goes forward",
        "prompt": (
            "The reference image comes to life as a realistic wildlife video of the same brown rabbit in the same "
            "green grass. The rabbit begins moving forward through the grass in the direction it is already facing, "
            "small natural hops, ears and fur moving subtly, shallow depth of field, stable low camera, natural "
            "outdoor light, generated soft grass rustle and ambience, no subtitles, no captions, no added text."
        ),
    },
    {
        "name": "left",
        "text": "rabbit turns left",
        "prompt": (
            "Continue the exact same rabbit, grassy field, low camera angle, lens, lighting, and audio bed. The "
            "rabbit naturally turns left while hopping forward, keeping the same body markings and realistic fur "
            "detail, grass bends under its paws, no subtitles, no captions, no added text."
        ),
    },
    {
        "name": "right",
        "text": "rabbit turns right",
        "prompt": (
            "Continue the exact same rabbit and grassy field from the previous shot. The rabbit smoothly turns right "
            "and keeps moving through the grass, same camera distance and natural daylight, detailed fur and ears, "
            "soft grass rustle audio, no subtitles, no captions, no added text."
        ),
    },
    {
        "name": "grass_fire",
        "text": "grass catches on fire",
        "prompt": (
            "Continue the exact same rabbit and camera setup. Small cinematic patches of orange flame begin catching "
            "in the grass behind and around the rabbit at a safe distance while the rabbit remains unharmed and keeps "
            "moving, realistic smoke wisps, warm firelight flicker on the grass, natural outdoor ambience with soft "
            "crackling fire audio, no subtitles, no captions, no added text."
        ),
    },
    {
        "name": "pink",
        "text": "rabbit becomes pink",
        "prompt": (
            "Continue the exact same scene with the same rabbit and grassy field. The rabbit remains safe as its fur "
            "gradually changes from brown to bright pink while it keeps moving through the grass, small flames stay "
            "in the background, realistic fur texture, stable low camera, soft crackling and grass rustle audio, no "
            "subtitles, no captions, no added text."
        ),
    },
]


FOREST_WORLD_MODEL_PLAN = [
    {
        "name": "follow_hold",
        "text": "camera follows character",
        "prompt": "Walk forward slowly, camera follows from behind. No text, no captions.",
    },
    {
        "name": "move_forward",
        "text": "character moves forward",
        "prompt": "Keep moving forward, camera follows from behind. No text, no captions.",
    },
    {
        "name": "turn_right",
        "text": "character follows right turn",
        "prompt": (
            "The trail curves clearly to the right; the character follows the right turn, "
            "camera follows from behind. No text, no captions."
        ),
    },
    {
        "name": "enter_forest",
        "text": "character continues right",
        "prompt": (
            "Continue along the right-curving forest trail, camera follows from behind. "
            "No text, no captions."
        ),
    },
    {
        "name": "continue_deeper",
        "text": "character continues deeper",
        "prompt": (
            "Keep walking deeper on the same right-curving trail, camera follows from behind. "
            "No text, no captions."
        ),
    },
]


FOREST_HARD_RIGHT_PLAN = [
    {
        "name": "rear_tracking_start",
        "text": "rear tracking start",
        "prompt": (
            "Continue from the image. The hooded character is seen from behind and walks forward slowly. "
            "The camera stays directly behind the character at waist height, a smooth rear tracking shot, "
            "the character remains centered. No text, no captions."
        ),
    },
    {
        "name": "turn_body_right",
        "text": "turn right begins",
        "prompt": (
            "The character in front clearly turns to the viewer's right: their shoulders, feet, and cloak rotate "
            "right, then they take several steps toward the right side of the frame. The camera follows from behind, "
            "panning and tracking right so the character's back stays centered. No text, no captions."
        ),
    },
    {
        "name": "move_screen_right",
        "text": "moves screen-right",
        "prompt": (
            "Continue the same shot. The character keeps walking to the viewer's right along the forest path, "
            "moving rightward across the frame. The rear camera follows from behind and slightly turns right with "
            "the character, keeping their back centered. No text, no captions."
        ),
    },
    {
        "name": "camera_follows_right",
        "text": "camera follows right",
        "prompt": (
            "The right turn continues in one unbroken rear tracking shot. The character is still in front of the "
            "camera, walking on the rightward path; the camera smoothly swings right and follows from behind. "
            "No cut, no new scene, no text, no captions."
        ),
    },
    {
        "name": "travel_right_path",
        "text": "travels right path",
        "prompt": (
            "Continue forward after the right turn. The character now travels deeper along the path that bends to "
            "the viewer's right, with the camera behind the character, following their back at the same distance. "
            "No text, no captions."
        ),
    },
]


# Each chunk prompt is self-contained. The worker only encodes the current prompt;
# temporal continuity comes from latent conditioning, not from previous text.
FOREST_REALWORLD_PROMPT_RIGHT_PLAN = [
    {
        "name": "approach_fork",
        "text": "approaches right fork",
        "prompt": (
            "A lone adult hiker in a dark rain jacket walks away from the camera on a damp real forest trail. "
            "The shot is a continuous handheld rear tracking shot at waist height, camera directly behind the hiker, "
            "soft overcast daylight, wet leaves, tree trunks, natural forest ambience. A clear fork in the trail is "
            "visible ahead, with the right-hand branch curving into the woods."
        ),
    },
    {
        "name": "turn_right_at_fork",
        "text": "turns right at fork",
        "prompt": (
            "A lone adult hiker in a dark rain jacket is seen from behind on a damp real forest trail at a visible "
            "fork in the path. The hiker turns to the viewer's right onto the right-hand branch: shoulders, feet, "
            "and backpack rotate right, then the hiker walks several steps down the right branch. The handheld "
            "camera stays behind the hiker at waist height, pans right, and follows the hiker's back, wet leaves "
            "and tree trunks around them, natural forest ambience."
        ),
    },
    {
        "name": "follow_right_branch",
        "text": "camera follows right branch",
        "prompt": (
            "A lone adult hiker in a dark rain jacket walks away from the camera along the right-hand branch of a "
            "damp real forest trail. The path curves to the viewer's right and the hiker follows that right curve, "
            "still seen from behind. The handheld rear tracking camera stays directly behind at waist height and "
            "pans right with the trail, realistic documentary footage, wet leaves, tree trunks, natural forest ambience."
        ),
    },
    {
        "name": "deeper_after_turn",
        "text": "deeper after right turn",
        "prompt": (
            "A lone adult hiker in a dark rain jacket is already on the right-curving branch of a damp real forest "
            "trail, walking away from the camera. The hiker continues deeper around the right bend, back facing the "
            "camera. The handheld rear tracking camera follows smoothly from behind, tree trunks sliding past on "
            "both sides, damp leaves underfoot, natural forest ambience."
        ),
    },
    {
        "name": "hold_rear_tracking",
        "text": "hold rear tracking",
        "prompt": (
            "A lone adult hiker in a dark rain jacket walks away from the camera on a right-curving damp forest "
            "trail. The camera remains behind the hiker in a steady handheld rear tracking composition as the path "
            "continues into the woods, overcast daylight, wet leaves, subtle footsteps and forest ambience."
        ),
    },
]


FOREST_WORLD_MODEL_PLANS = {
    "right_curve": FOREST_WORLD_MODEL_PLAN,
    "hard_right": FOREST_HARD_RIGHT_PLAN,
    "realworld_prompt_right": FOREST_REALWORLD_PROMPT_RIGHT_PLAN,
}


def _ensure_assets() -> dict[str, str]:
    from huggingface_hub import hf_hub_download, snapshot_download

    checkpoint_dir = Path("/cache/models/LTX-2.3-fp8")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ltx-2.3-22b-distilled-fp8.safetensors"
    if not checkpoint_path.exists():
        print(f"Downloading official LTX-2.3 FP8 checkpoint to {checkpoint_path}", flush=True)
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3-fp8",
            filename="ltx-2.3-22b-distilled-fp8.safetensors",
            local_dir=str(checkpoint_dir),
        )
    else:
        print(f"Using cached checkpoint {checkpoint_path}", flush=True)

    gemma_dir = Path("/cache/models/FastVideo-LTX2-Distilled-Diffusers")
    gemma_root = gemma_dir / "text_encoder" / "gemma"
    if not gemma_root.exists():
        print(f"Downloading Gemma text encoder to {gemma_root}", flush=True)
        snapshot_download(
            repo_id="FastVideo/LTX2-Distilled-Diffusers",
            allow_patterns=["text_encoder/gemma/**"],
            local_dir=str(gemma_dir),
        )
    else:
        print(f"Using cached Gemma text encoder {gemma_root}", flush=True)
    if not gemma_root.exists():
        raise FileNotFoundError(f"Gemma text encoder root was not downloaded: {gemma_root}")

    cache_volume.commit()
    return {
        "checkpoint_path": str(checkpoint_path),
        "gemma_root": str(gemma_root),
    }


def _worker_env(
    *,
    output_root: Path,
    bucket: str,
    checkpoint_path: str,
    gemma_root: str,
    attention: str,
    emit_overlap_frames: bool = False,
) -> None:
    os.environ.update(
        {
            "LTX_WORKER_OUTPUT_DIR": str(output_root),
            "LTX_WORKER_LOAD_ON_STARTUP": "0",
            "LTX_WORKER_BUCKET": bucket,
            "LTX23_CHECKPOINT_PATH": checkpoint_path,
            "LTX23_GEMMA_ROOT": gemma_root,
            "LTX_WORKER_QUANTIZATION": "fp8-scaled-mm",
            "LTX_WORKER_ATTENTION": attention,
            "LTX_WORKER_STEPS": "8",
            "LTX_WORKER_ENCODE_CODEC": "libx264",
            "LTX_WORKER_ENCODE_CONTAINER": "frag-mp4",
            "LTX_WORKER_ENCODE_FEED_MODE": "frame-stream",
            "LTX_WORKER_ENCODE_FRAME_BATCH": "2",
            "LTX_WORKER_ENCODE_X264_PRESET": "ultrafast",
            "LTX_WORKER_ENCODE_X264_CRF": "19",
            "LTX_WORKER_ENCODE_PRESTART": "1",
            "LTX_WORKER_ENCODE_DETACH_AFTER_FIRST_BYTE": "0",
            "LTX_WORKER_DECODE_AUDIO": "1",
            "LTX_WORKER_ENCODE_MUX_AUDIO": "1",
            "LTX_WORKER_EMIT_OVERLAP_FRAMES": "1" if emit_overlap_frames else "0",
            "LTX_WORKER_LATENT_CONTINUATION_EMIT_TO_TAIL": "1",
            "LTX_WORKER_CACHE_ROPE_EMBEDDINGS": "1",
            "LTX_WORKER_SPD_SCALE": "0.5",
            "LTX_WORKER_SPD_TRANSITION_STEP": "5",
            "LTX_WORKER_SPD_TAPER": "8",
            "LTX_WORKER_SPD_INITIAL_DCT_DOWNSCALE": "1",
            "LTX_WORKER_SPD_HIGHFREQ_NOISE": "1",
            "LTX_WORKER_ALLOW_QUALITY_RISK_RESIDUAL_CACHE": "1",
            "LTX_WORKER_RESIDUAL_CACHE_THRESHOLD": "0.02",
            "LTX_WORKER_RESIDUAL_CACHE_MAX_SKIPS": "1",
            "LTX_WORKER_RESIDUAL_CACHE_RETENTION_RATIO": "0.25",
            "LTX_WORKER_RESIDUAL_CACHE_METRIC_ELEMENT_STRIDE": "64",
        }
    )


def _set_variant(args: Any, *, variant: str, mag_ratios_path: Path | None = None) -> None:
    args.spd = variant.startswith("spd_")
    args.residual_cache_mode = "off"
    args.residual_cache_mag_ratios = ""
    args.residual_cache_force_skip_steps = ""
    args.residual_cache_metric_element_stride = 64
    args.residual_cache_threshold = 0.02
    args.residual_cache_max_skips = 1
    args.residual_cache_retention_ratio = 0.25
    args.allow_quality_risk_residual_cache = True

    if variant == "dense":
        return
    if variant in {"dense_calibrate", "spd_calibrate"}:
        args.residual_cache_mode = "calibrate"
        return
    if variant in {"dense_magcache_002", "spd_magcache_002"}:
        if mag_ratios_path is None:
            raise ValueError(f"{variant} requires mag_ratios_path")
        args.residual_cache_mode = "magcache"
        args.residual_cache_mag_ratios = str(mag_ratios_path)
        return
    raise ValueError(f"unknown variant: {variant}")


def _ratios_from_calibration(response_payload: dict[str, Any]) -> list[float]:
    records = ((response_payload.get("residual_cache") or {}).get("records") or [])
    ratios = [
        float(record["mag_calibration_ratio"])
        for record in records
        if record.get("mag_calibration_ratio") is not None
    ]
    if not ratios:
        ratios = [1.0] * 8
    return ratios


def _response_payload(response: Any, *, output_root: Path, job_id: str, prompt_name: str, variant: str) -> dict[str, Any]:
    payload = response.model_dump()
    payload["prompt_name"] = prompt_name
    payload["variant"] = variant
    payload["volume_output_path"] = str(
        output_root.relative_to("/cache") / job_id / Path(payload["output_path"]).name
    )
    return payload


def _print_row(payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "plan_name": payload.get("plan_name"),
                "segment_index": payload.get("segment_index"),
                "prompt_name": payload["prompt_name"],
                "variant": payload["variant"],
                "time_to_first_video_byte_s": payload["time_to_first_video_byte_s"],
                "runtime_s": payload["runtime_s"],
                "denoise_s": payload["denoise_s"],
                "output": payload["volume_output_path"],
                "skipped_steps": (payload.get("residual_cache") or {}).get("skipped_steps"),
                "requested_continuation_frames": payload.get("stage_times", {}).get("requested_continuation_frames"),
                "continuation_frames": payload.get("stage_times", {}).get("continuation_frames"),
                "latent_video_tail_frames": payload.get("stage_times", {}).get("latent_video_tail_frames"),
                "latent_effective_continuation_frames": payload.get("stage_times", {}).get(
                    "latent_effective_continuation_frames"
                ),
                "latent_emit_to_tail": payload.get("stage_times", {}).get("latent_emit_to_tail"),
                "emitted_start_frame": payload.get("stage_times", {}).get("emitted_start_frame"),
                "emitted_frames": payload.get("stage_times", {}).get("emitted_frames"),
            }
        ),
        flush=True,
    )


@app.function(
    gpu=GPU,
    image=official_image,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_dotenv(filename="env.local")],
)
def run_hamster_suite(
    *,
    bucket: str = "hd_16x9_15fps_5s_v1",
    attention: str = "sdpa-cudnn",
    seed: int = 20260602,
    include_dense: bool = True,
) -> dict[str, Any]:
    run_id = time.strftime("modal_hamster_%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path("/cache/modal_official_outputs") / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assets = _ensure_assets()
    _worker_env(
        output_root=output_root,
        bucket=bucket,
        checkpoint_path=assets["checkpoint_path"],
        gemma_root=assets["gemma_root"],
        attention=attention,
    )

    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    from ltx_serve.remote_official_worker import OfficialEngine, build_engine_args
    from ltx_serve.schemas import GenerateRequest

    args = build_engine_args()
    args.seed = seed
    engine = OfficialEngine(args)
    engine.load()

    rows: list[dict[str, Any]] = []
    prompt_file = output_root / "prompts.json"
    prompt_file.write_text(json.dumps(HAMSTER_PROMPTS, indent=2), encoding="utf-8")

    variants = ["dense", "spd_calibrate", "spd_magcache_002"] if include_dense else ["spd_calibrate", "spd_magcache_002"]
    for prompt_name, prompt in HAMSTER_PROMPTS.items():
        mag_ratios_path: Path | None = None
        calibration_payload: dict[str, Any] | None = None
        for variant in variants:
            if variant == "spd_magcache_002":
                if calibration_payload is None:
                    raise RuntimeError(f"missing calibration payload for {prompt_name}")
                ratios = _ratios_from_calibration(calibration_payload)
                mag_ratios_path = output_root / prompt_name / "mag_ratios.json"
                mag_ratios_path.parent.mkdir(parents=True, exist_ok=True)
                mag_ratios_path.write_text(json.dumps({"mag_ratios": ratios}, indent=2), encoding="utf-8")
            _set_variant(engine.args, variant=variant, mag_ratios_path=mag_ratios_path)
            job_id = f"{prompt_name}_{variant}"
            job_dir = output_root / job_id
            response = engine.generate(
                GenerateRequest(
                    prompt=prompt,
                    job_id=job_id,
                    duration_s=5,
                    tier="realtime",
                    bucket=bucket,
                    seed=seed,
                    audio=True,
                ),
                job_dir,
            )
            payload = _response_payload(
                response,
                output_root=output_root,
                job_id=job_id,
                prompt_name=prompt_name,
                variant=variant,
            )
            rows.append(payload)
            _print_row(payload)
            if variant == "spd_calibrate":
                calibration_payload = payload

    summary = {
        "run_id": run_id,
        "bucket": bucket,
        "attention": attention,
        "quantization": "fp8-scaled-mm",
        "steps": 8,
        "seed": seed,
        "assets": assets,
        "trtllm_scaled_mm_usable": trtllm_scaled_mm_usable(),
        "ltx2_commit": LTX2_COMMIT,
        "output_root": str(output_root),
        "volume_output_root": str(output_root.relative_to("/cache")),
        "prompts": HAMSTER_PROMPTS,
        "rows": rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache_volume.commit()
    engine.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    gpu=GPU,
    image=official_image,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_dotenv(filename="env.local")],
)
def run_prompt_steering_suite(
    *,
    bucket: str = "hd_16x9_15fps_5s_overlap22_v1",
    attention: str = "sdpa-cudnn",
    seed: int = 20260602,
    continuation_frames: int = 22,
    plan_limit: int = 3,
    steering_variants: str = "quality_spd_calibrate,fast_spd_magcache",
    emit_overlap_frames: bool = False,
) -> dict[str, Any]:
    run_id = time.strftime("modal_prompt_steer_%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path("/cache/modal_official_outputs") / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assets = _ensure_assets()
    _worker_env(
        output_root=output_root,
        bucket=bucket,
        checkpoint_path=assets["checkpoint_path"],
        gemma_root=assets["gemma_root"],
        attention=attention,
        emit_overlap_frames=emit_overlap_frames,
    )

    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    from ltx_serve.remote_official_worker import OfficialEngine, build_engine_args
    from ltx_serve.schemas import GenerateRequest

    args = build_engine_args()
    args.seed = seed
    engine = OfficialEngine(args)
    engine.load()

    rows: list[dict[str, Any]] = []
    selected_plans = dict(list(STEERING_PLANS.items())[: max(1, int(plan_limit))])
    prompts_path = output_root / "steering_plans.json"
    prompts_path.write_text(json.dumps(selected_plans, indent=2), encoding="utf-8")

    variant_map = {
        "dense": ("dense", "dense"),
        "dense_calibrate": ("dense_calibrate", "dense_calibrate"),
        "dense_magcache": ("dense_magcache", "dense_magcache_002"),
        "quality_spd_calibrate": ("quality_spd_calibrate", "spd_calibrate"),
        "fast_spd_magcache": ("fast_spd_magcache", "spd_magcache_002"),
    }
    requested_variants = [item.strip() for item in steering_variants.split(",") if item.strip()]
    if not requested_variants:
        raise ValueError("at least one steering variant is required")
    variant_specs = []
    for name in requested_variants:
        if name not in variant_map:
            raise ValueError(f"unknown steering variant {name!r}; expected one of {sorted(variant_map)}")
        variant_specs.append(variant_map[name])
    if "dense_magcache" in requested_variants and "dense_calibrate" not in requested_variants:
        raise ValueError("dense_magcache requires dense_calibrate in the same run")
    if "fast_spd_magcache" in requested_variants and "quality_spd_calibrate" not in requested_variants:
        raise ValueError("fast_spd_magcache requires quality_spd_calibrate in the same run")

    for plan_name, segments in selected_plans.items():
        calibration_by_segment: dict[str, dict[str, Any]] = {}
        for sequence_name, variant in variant_specs:
            previous_output_path: str | None = None
            for index, segment in enumerate(segments):
                segment_name = str(segment["name"])
                segment_key = f"{plan_name}:{segment_name}"
                mag_ratios_path = None
                if variant in {"dense_magcache_002", "spd_magcache_002"}:
                    calibration_family = "spd" if variant.startswith("spd_") else "dense"
                    calibration_payload = calibration_by_segment[f"{calibration_family}:{segment_key}"]
                    ratios = _ratios_from_calibration(calibration_payload)
                    mag_ratios_path = output_root / plan_name / sequence_name / f"{index:02d}_{segment_name}_mag_ratios.json"
                    mag_ratios_path.parent.mkdir(parents=True, exist_ok=True)
                    mag_ratios_path.write_text(json.dumps({"mag_ratios": ratios}, indent=2), encoding="utf-8")

                _set_variant(engine.args, variant=variant, mag_ratios_path=mag_ratios_path)
                job_id = f"{plan_name}_{sequence_name}_{index:02d}_{segment_name}"
                response = engine.generate(
                    GenerateRequest(
                        prompt=str(segment["prompt"]),
                        job_id=job_id,
                        duration_s=5,
                        tier="realtime",
                        bucket=bucket,
                        seed=seed + index,
                        audio=True,
                        continuation_mode="latent",
                        continuation_video_path=previous_output_path,
                        continuation_frames=continuation_frames if previous_output_path is not None else 0,
                        continuation_strength=1.0,
                    ),
                    output_root / job_id,
                )
                payload = _response_payload(
                    response,
                    output_root=output_root,
                    job_id=job_id,
                    prompt_name=f"{plan_name}/{segment_name}",
                    variant=sequence_name,
                )
                payload["plan_name"] = plan_name
                payload["segment_index"] = index
                payload["segment_name"] = segment_name
                payload["prompt"] = str(segment["prompt"])
                rows.append(payload)
                _print_row(payload)
                previous_output_path = payload["output_path"]
                if variant in {"dense_calibrate", "spd_calibrate"}:
                    calibration_family = "spd" if variant.startswith("spd_") else "dense"
                    calibration_by_segment[f"{calibration_family}:{segment_key}"] = payload

    summary = {
        "run_id": run_id,
        "bucket": bucket,
        "attention": attention,
        "quantization": "fp8-scaled-mm",
        "steps": 8,
        "seed": seed,
        "continuation_frames": continuation_frames,
        "emit_overlap_frames": emit_overlap_frames,
        "steering_variants": requested_variants,
        "assets": assets,
        "trtllm_scaled_mm_usable": trtllm_scaled_mm_usable(),
        "ltx2_commit": LTX2_COMMIT,
        "output_root": str(output_root),
        "volume_output_root": str(output_root.relative_to("/cache")),
        "plans": selected_plans,
        "rows": rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache_volume.commit()
    engine.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _prepare_bucket_image(input_path: Path, output_path: Path, *, width: int, height: int) -> dict[str, Any]:
    from PIL import Image, ImageOps

    image = Image.open(input_path).convert("RGB")
    prepared = ImageOps.fit(
        image,
        (int(width), int(height)),
        method=Image.Resampling.LANCZOS,
        centering=(0.50, 0.58),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(output_path, quality=95)
    return {
        "input_path": str(input_path),
        "input_size": list(image.size),
        "output_path": str(output_path),
        "output_size": [int(width), int(height)],
    }


@app.function(
    gpu=GPU,
    image=official_image,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_dotenv(filename="env.local")],
)
def run_rabbit_image_action_sequence(
    *,
    bucket: str = "hd_16x9_15fps_5s_overlap22_v1",
    attention: str = "sdpa-cudnn",
    seed: int = 20260603,
    continuation_frames: int = 22,
    image_strength: float = 1.0,
) -> dict[str, Any]:
    run_id = time.strftime("modal_rabbit_%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path("/cache/modal_official_outputs") / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assets = _ensure_assets()
    _worker_env(
        output_root=output_root,
        bucket=bucket,
        checkpoint_path=assets["checkpoint_path"],
        gemma_root=assets["gemma_root"],
        attention=attention,
        emit_overlap_frames=False,
    )

    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    from ltx_serve.buckets import BUCKETS
    from ltx_serve.remote_official_worker import OfficialEngine, build_engine_args
    from ltx_serve.schemas import GenerateRequest

    bucket_spec = BUCKETS[bucket]
    prepared_image = output_root / "rabbit_init_1280x736.jpg"
    image_info = _prepare_bucket_image(
        Path(f"{REMOTE_REPO}/frame_gallery/rabbit.jpg"),
        prepared_image,
        width=int(bucket_spec.width),
        height=int(bucket_spec.height),
    )

    args = build_engine_args()
    args.seed = seed
    engine = OfficialEngine(args)
    engine.load()

    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    previous_output_path: str | None = None
    for index, segment in enumerate(RABBIT_IMAGE_ACTION_PLAN):
        _set_variant(engine.args, variant="dense")
        job_id = f"rabbit_25s_dense_{index:02d}_{segment['name']}"
        response = engine.generate(
            GenerateRequest(
                prompt=str(segment["prompt"]),
                job_id=job_id,
                duration_s=5,
                tier="standard",
                bucket=bucket,
                seed=seed + index,
                audio=True,
                image_conditioning_path=str(prepared_image) if index == 0 else None,
                image_conditioning_frame_idx=0,
                image_conditioning_strength=image_strength,
                continuation_mode="latent",
                continuation_video_path=previous_output_path,
                continuation_frames=continuation_frames if previous_output_path is not None else 0,
                continuation_strength=1.0,
            ),
            output_root / job_id,
        )
        payload = _response_payload(
            response,
            output_root=output_root,
            job_id=job_id,
            prompt_name=f"rabbit/{segment['name']}",
            variant="dense",
        )
        payload["segment_index"] = index
        payload["segment_name"] = segment["name"]
        payload["action_text"] = segment["text"]
        payload["prompt"] = str(segment["prompt"])
        rows.append(payload)
        actions.append(
            {
                "segment_index": index,
                "text": segment["text"] if index > 0 else "initial image; rabbit goes forward",
                "prompt": segment["prompt"],
                "state": "ready",
                "job_id": job_id,
                "bucket": bucket,
                "output_url": payload["output_url"],
                "output_path": payload["output_path"],
                "media_type": payload["media_type"],
                "runtime_s": payload["runtime_s"],
                "time_to_first_video_byte_s": payload["time_to_first_video_byte_s"],
                "stage_times": payload["stage_times"],
                "residual_cache": payload.get("residual_cache"),
            }
        )
        _print_row(payload)
        previous_output_path = payload["output_path"]

    session = {
        "session_id": run_id,
        "base_prompt": RABBIT_IMAGE_ACTION_PLAN[0]["prompt"],
        "tier": "standard",
        "duration_s": 5,
        "target_duration_s": float(bucket_spec.output_duration_s),
        "audio": True,
        "seed": seed,
        "initial_image_path": str(prepared_image),
        "initial_image_strength": image_strength,
        "continuity_frames": continuation_frames,
        "continuity_strength": 1.0,
        "bucket": bucket,
        "actions": actions,
    }
    (output_root / "session_final.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    (output_root / "prompts.json").write_text(json.dumps(RABBIT_IMAGE_ACTION_PLAN, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "bucket": bucket,
        "attention": attention,
        "quantization": "fp8-scaled-mm",
        "steps": 8,
        "variant": "dense",
        "seed": seed,
        "continuation_frames": continuation_frames,
        "image_strength": image_strength,
        "image_info": image_info,
        "assets": assets,
        "trtllm_scaled_mm_usable": trtllm_scaled_mm_usable(),
        "ltx2_commit": LTX2_COMMIT,
        "output_root": str(output_root),
        "volume_output_root": str(output_root.relative_to("/cache")),
        "rows": rows,
        "session": session,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache_volume.commit()
    engine.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    gpu=GPU,
    image=official_image,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_dotenv(filename="env.local")],
)
def run_forest_world_model_sequence(
    *,
    bucket: str = "std_16x9_25fps_5s_overlap32_v1",
    attention: str = "sdpa-cudnn",
    seed: int = 20260603,
    continuation_frames: int = 32,
    image_strength: float = 1.0,
    plan_name: str = "right_curve",
    use_initial_image: bool = True,
    plan_limit: int | None = None,
) -> dict[str, Any]:
    if plan_name not in FOREST_WORLD_MODEL_PLANS:
        raise ValueError(f"Unknown forest world-model plan: {plan_name}")
    plan = FOREST_WORLD_MODEL_PLANS[plan_name]
    if plan_limit is not None:
        plan = plan[:plan_limit]
    run_prefix = f"forest_{plan_name}"
    run_id = time.strftime(f"modal_{run_prefix}_%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path("/cache/modal_official_outputs") / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assets = _ensure_assets()
    _worker_env(
        output_root=output_root,
        bucket=bucket,
        checkpoint_path=assets["checkpoint_path"],
        gemma_root=assets["gemma_root"],
        attention=attention,
        emit_overlap_frames=False,
    )

    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    from ltx_serve.buckets import BUCKETS
    from ltx_serve.remote_official_worker import OfficialEngine, build_engine_args
    from ltx_serve.schemas import GenerateRequest

    bucket_spec = BUCKETS[bucket]
    prepared_image: Path | None = None
    image_info: dict[str, Any] | None = None
    if use_initial_image:
        prepared_image = output_root / "forest_init_1280x736.jpg"
        image_info = _prepare_bucket_image(
            Path(f"{REMOTE_REPO}/frame_gallery/forest.png"),
            prepared_image,
            width=int(bucket_spec.width),
            height=int(bucket_spec.height),
        )

    args = build_engine_args()
    args.seed = seed
    engine = OfficialEngine(args)
    engine.load()

    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    previous_output_path: str | None = None
    for index, segment in enumerate(plan):
        _set_variant(engine.args, variant="dense")
        job_id = f"{run_prefix}_25fps_dense_{index:02d}_{segment['name']}"
        response = engine.generate(
            GenerateRequest(
                prompt=str(segment["prompt"]),
                job_id=job_id,
                duration_s=5,
                tier="standard",
                bucket=bucket,
                seed=seed + index,
                audio=True,
                image_conditioning_path=str(prepared_image) if index == 0 and prepared_image is not None else None,
                image_conditioning_frame_idx=0,
                image_conditioning_strength=image_strength,
                continuation_mode="latent",
                continuation_video_path=previous_output_path,
                continuation_frames=continuation_frames if previous_output_path is not None else 0,
                continuation_strength=1.0,
            ),
            output_root / job_id,
        )
        payload = _response_payload(
            response,
            output_root=output_root,
            job_id=job_id,
            prompt_name=f"{run_prefix}/{segment['name']}",
            variant="dense",
        )
        payload["segment_index"] = index
        payload["segment_name"] = segment["name"]
        payload["action_text"] = segment["text"]
        payload["prompt"] = str(segment["prompt"])
        rows.append(payload)
        actions.append(
            {
                "segment_index": index,
                "text": segment["text"],
                "prompt": segment["prompt"],
                "state": "ready",
                "job_id": job_id,
                "bucket": bucket,
                "output_url": payload["output_url"],
                "output_path": payload["output_path"],
                "media_type": payload["media_type"],
                "runtime_s": payload["runtime_s"],
                "time_to_first_video_byte_s": payload["time_to_first_video_byte_s"],
                "stage_times": payload["stage_times"],
                "residual_cache": payload.get("residual_cache"),
            }
        )
        _print_row(payload)
        previous_output_path = payload["output_path"]

    session = {
        "session_id": run_id,
        "base_prompt": plan[0]["prompt"],
        "plan_name": plan_name,
        "tier": "standard",
        "duration_s": 5,
        "target_duration_s": float(bucket_spec.output_duration_s),
        "audio": True,
        "seed": seed,
        "initial_image_path": str(prepared_image) if prepared_image is not None else None,
        "initial_image_strength": image_strength if prepared_image is not None else 0.0,
        "use_initial_image": use_initial_image,
        "continuity_frames": continuation_frames,
        "continuity_strength": 1.0,
        "bucket": bucket,
        "actions": actions,
    }
    (output_root / "session_final.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    (output_root / "prompts.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "plan_name": plan_name,
        "bucket": bucket,
        "attention": attention,
        "quantization": "fp8-scaled-mm",
        "steps": 8,
        "variant": "dense",
        "seed": seed,
        "continuation_frames": continuation_frames,
        "image_strength": image_strength,
        "use_initial_image": use_initial_image,
        "plan_limit": plan_limit,
        "image_info": image_info,
        "assets": assets,
        "trtllm_scaled_mm_usable": trtllm_scaled_mm_usable(),
        "ltx2_commit": LTX2_COMMIT,
        "output_root": str(output_root),
        "volume_output_root": str(output_root.relative_to("/cache")),
        "rows": rows,
        "session": session,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache_volume.commit()
    engine.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    mode: str = "hamster_suite",
    bucket: str = "hd_16x9_15fps_5s_v1",
    attention: str = "sdpa-cudnn",
    seed: int = 20260602,
    include_dense: bool = True,
    continuation_frames: int = 22,
    plan_limit: int = 3,
    steering_variants: str = "quality_spd_calibrate,fast_spd_magcache",
    emit_overlap_frames: bool = False,
) -> None:
    if mode == "hamster_suite":
        result = run_hamster_suite.remote(
            bucket=bucket,
            attention=attention,
            seed=seed,
            include_dense=include_dense,
        )
    elif mode == "prompt_steering":
        if bucket == "hd_16x9_15fps_5s_v1":
            bucket = "hd_16x9_15fps_5s_overlap22_v1"
        result = run_prompt_steering_suite.remote(
            bucket=bucket,
            attention=attention,
            seed=seed,
            continuation_frames=continuation_frames,
            plan_limit=plan_limit,
            steering_variants=steering_variants,
            emit_overlap_frames=emit_overlap_frames,
        )
    elif mode == "rabbit_image_actions":
        result = run_rabbit_image_action_sequence.remote(
            bucket="hd_16x9_15fps_5s_overlap22_v1" if bucket == "hd_16x9_15fps_5s_v1" else bucket,
            attention=attention,
            seed=seed,
            continuation_frames=continuation_frames,
        )
    elif mode == "forest_world_model":
        result = run_forest_world_model_sequence.remote(
            bucket="std_16x9_25fps_5s_overlap32_v1" if bucket == "hd_16x9_15fps_5s_v1" else bucket,
            attention=attention,
            seed=seed,
            continuation_frames=continuation_frames,
            plan_name="right_curve",
        )
    elif mode == "forest_hard_right_model":
        result = run_forest_world_model_sequence.remote(
            bucket="std_16x9_25fps_5s_overlap32_v1" if bucket == "hd_16x9_15fps_5s_v1" else bucket,
            attention=attention,
            seed=seed,
            continuation_frames=continuation_frames,
            plan_name="hard_right",
        )
    elif mode == "forest_prompt_right_model":
        result = run_forest_world_model_sequence.remote(
            bucket="std_16x9_25fps_5s_overlap32_v1" if bucket == "hd_16x9_15fps_5s_v1" else bucket,
            attention=attention,
            seed=seed,
            continuation_frames=continuation_frames,
            plan_name="realworld_prompt_right",
            use_initial_image=False,
            plan_limit=plan_limit,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(json.dumps(result, indent=2))
