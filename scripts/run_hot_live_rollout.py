from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from ltx_serve.buckets import resolve_bucket


DEFAULT_PROMPT = (
    "A realistic close-up documentary video of a small golden hamster running inside a transparent exercise wheel "
    "in a cozy living room. Natural daylight, shallow depth of field, detailed fur, stable camera, real-world "
    "lighting, subtle wheel motion, generated soft wheel squeak and room tone, no text, no captions."
)


def _url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ready_actions(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [action for action in session.get("actions", []) if action.get("state") == "ready"]


def _download(client: httpx.Client, base_url: str, action: dict[str, Any], output_path: Path) -> None:
    raw_url = str(action.get("output_url") or "")
    if not raw_url:
        raise RuntimeError(f"action {action.get('segment_index')} has no output_url")
    url = raw_url if raw_url.startswith(("http://", "https://")) else _url(base_url, raw_url)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with output_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    except httpx.HTTPStatusError:
        job_id = action.get("job_id")
        if not job_id:
            raise
        proxy_url = _url(base_url, f"/api/remote/streams/{job_id}")
        with client.stream("GET", proxy_url) as response:
            response.raise_for_status()
            with output_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    if output_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty segment from {url}")


def _maybe_concat(output_dir: Path, segment_count: int) -> str | None:
    concat_file = output_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file 'segment_{index}.mp4'\n" for index in range(segment_count)),
        encoding="utf-8",
    )
    merged = output_dir / "merged.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file.name,
        "-c",
        "copy",
        merged.name,
    ]
    try:
        subprocess.run(cmd, cwd=output_dir, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return str(merged)


def _write_summary(output_dir: Path, session: dict[str, Any], merged_path: str | None) -> None:
    rows = []
    try:
        bucket = resolve_bucket(int(session.get("duration_s") or 5), str(session.get("tier") or "realtime"), session.get("bucket"))
        final_fps = float(bucket.final_fps)
    except (TypeError, ValueError):
        final_fps = 15.0
    for action in _ready_actions(session):
        stage_times = action.get("stage_times") or {}
        residual_cache = action.get("residual_cache") or {}
        emitted_frames = stage_times.get("emitted_frames")
        emitted_duration_s = (
            float(emitted_frames) / final_fps
            if emitted_frames is not None
            else float(action.get("target_duration_s") or session.get("target_duration_s") or 5.0)
        )
        runtime_s = float(action.get("runtime_s") or 0.0)
        ttfb_s = float(action.get("time_to_first_video_byte_s") or 0.0)
        rows.append(
            "| {segment} | {visible} | {cache} | {reason} | {ttfb:.3f} | {runtime:.3f} | {emitted:.3f} | {live:.3f}x | {denoise:.3f} | {skipped} | {url} |".format(
                segment=action.get("segment_index"),
                visible="yes" if action.get("user_visible") else "no",
                cache=action.get("cache_policy") or "off",
                reason=action.get("cache_reason") or "",
                ttfb=ttfb_s,
                runtime=runtime_s,
                emitted=emitted_duration_s,
                live=runtime_s / emitted_duration_s if emitted_duration_s > 0 else 0.0,
                denoise=float((action.get("stage_times") or {}).get("denoise_s") or 0.0),
                skipped=residual_cache.get("skipped_steps"),
                url=action.get("output_url") or "",
            )
        )
    text = [
        "# Hot Live Rollout",
        "",
        f"Session: `{session.get('session_id')}`",
        f"Bucket: `{session.get('bucket')}`",
        f"Continuity frames: `{session.get('continuity_frames')}`",
        f"Cache enabled: `{session.get('cache_enabled')}`",
        f"Cache ratio version: `{session.get('cache_ratio_version')}`",
        f"Merged: `{merged_path}`" if merged_path else "Merged: unavailable",
        "",
        "| Segment | User visible | Cache | Reason | Startup byte s | Producer interval s | Emitted duration s | Steady factor | Denoise s | Skipped steps | URL |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *rows,
        "",
    ]
    (output_dir / "result.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a bounded hot-path live rollout.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--bucket", default="hd_16x9_15fps_5s_overlap22_v1")
    parser.add_argument("--duration-s", type=int, default=5)
    parser.add_argument("--tier", default="realtime", choices=["realtime", "fast", "standard", "premium"])
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--continuity-frames", type=int, default=22)
    parser.add_argument("--continuity-strength", type=float, default=1.0)
    parser.add_argument("--cache-mode", choices=["off", "magcache", "teacache"])
    parser.add_argument("--cache-threshold", type=float)
    parser.add_argument("--cache-max-skips", type=int)
    parser.add_argument("--cache-retention-ratio", type=float)
    parser.add_argument("--cache-metric-element-stride", type=int)
    parser.add_argument("--cache-refresh-interval", type=int)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_payload = {
        "prompt": args.prompt,
        "duration_s": args.duration_s,
        "tier": args.tier,
        "bucket": args.bucket,
        "seed": args.seed,
        "audio": True,
        "continuity_frames": args.continuity_frames,
        "continuity_strength": args.continuity_strength,
    }
    cache_overrides = {
        "live_cache_mode": args.cache_mode,
        "live_cache_threshold": args.cache_threshold,
        "live_cache_max_skips": args.cache_max_skips,
        "live_cache_retention_ratio": args.cache_retention_ratio,
        "live_cache_metric_element_stride": args.cache_metric_element_stride,
        "live_cache_refresh_interval": args.cache_refresh_interval,
    }
    create_payload.update({key: value for key, value in cache_overrides.items() if value is not None})
    with httpx.Client(timeout=None) as client:
        response = client.post(_url(args.base_url, "/api/live/sessions"), json=create_payload)
        response.raise_for_status()
        session = response.json()
        session_id = session["session_id"]
        _write_json(args.output_dir / "session_create.json", session)

        for index, action in enumerate(args.action, start=1):
            response = client.post(_url(args.base_url, f"/api/live/sessions/{session_id}/actions"), json={"text": action})
            response.raise_for_status()
            _write_json(args.output_dir / f"queue_action_{index}.json", response.json())

        deadline = time.monotonic() + args.timeout_s
        while True:
            response = client.get(_url(args.base_url, f"/api/live/sessions/{session_id}"))
            response.raise_for_status()
            session = response.json()
            _write_json(args.output_dir / "session_latest.json", session)
            ready = _ready_actions(session)
            if len(ready) >= args.segments:
                break
            if any(action.get("state") == "failed" for action in session.get("actions", [])):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {args.segments} ready segments in session {session_id}")
            time.sleep(args.poll_s)

        stop = client.delete(_url(args.base_url, f"/api/live/sessions/{session_id}"))
        stop.raise_for_status()
        final = stop.json()
        _write_json(args.output_dir / "session_final.json", final)
        ready = _ready_actions(final)[: args.segments]
        for action in ready:
            _download(client, args.base_url, action, args.output_dir / f"segment_{int(action['segment_index'])}.mp4")

    merged_path = _maybe_concat(args.output_dir, len(ready))
    _write_summary(args.output_dir, final, merged_path)
    print(args.output_dir / "result.md")


if __name__ == "__main__":
    main()
