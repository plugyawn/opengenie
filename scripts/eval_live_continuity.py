from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffprobe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        text=True,
    )
    return json.loads(raw)


def _stream_summary(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration": float(probe["format"]["duration"]),
        "streams": [
            {
                "codec_type": stream.get("codec_type"),
                "codec_name": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "r_frame_rate": stream.get("r_frame_rate"),
                "duration": float(stream["duration"]) if stream.get("duration") is not None else None,
                "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in {None, "N/A"} else None,
            }
            for stream in probe.get("streams", [])
        ],
    }


def _video_frame_count(path: Path) -> int:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    streams = json.loads(raw).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if count in {None, "N/A"}:
        raise RuntimeError(f"cannot read video frame count for {path}")
    return int(count)


def _extract_frame(video: Path, timestamp_s: float, output: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(timestamp_s, 0.0):.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ]
    )
    if not output.exists():
        raise RuntimeError(f"ffmpeg did not extract {output} from {video} at {timestamp_s:.6f}s")


def _extract_frame_index(video: Path, frame_index: int, output: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(output),
        ]
    )
    if not output.exists():
        raise RuntimeError(f"ffmpeg did not extract frame {frame_index} from {video} to {output}")


def _make_contact_sheet(frame_paths: list[Path], output: Path, *, columns: int = 8) -> None:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        _make_contact_sheet_ffmpeg(frame_paths, output, columns=columns)
        return

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise ValueError("no images for contact sheet")
    thumb_w, thumb_h = images[0].size
    rows = math.ceil(len(images) / columns)
    label_h = 28
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, path) in enumerate(zip(images, frame_paths, strict=True)):
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 6), path.stem, fill=(0, 0, 0))
    sheet.save(output)


def _stack_images_ffmpeg(inputs: list[Path], output: Path, *, direction: str) -> None:
    if direction not in {"h", "v"}:
        raise ValueError(f"unsupported stack direction: {direction}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    labels = "".join(f"[{index}:v]" for index in range(len(inputs)))
    filter_name = "hstack" if direction == "h" else "vstack"
    cmd.extend(["-filter_complex", f"{labels}{filter_name}=inputs={len(inputs)}[out]", "-map", "[out]", str(output)])
    _run(cmd)


def _make_contact_sheet_ffmpeg(frame_paths: list[Path], output: Path, *, columns: int = 8) -> None:
    if not frame_paths:
        raise ValueError("no images for contact sheet")
    rows: list[Path] = []
    for row_index, start in enumerate(range(0, len(frame_paths), columns)):
        row_inputs = frame_paths[start : start + columns]
        row_output = output.with_name(f"{output.stem}_row{row_index}.png")
        _stack_images_ffmpeg(row_inputs, row_output, direction="h")
        rows.append(row_output)
    if len(rows) == 1:
        rows[0].replace(output)
    else:
        _stack_images_ffmpeg(rows, output, direction="v")
        for row in rows:
            row.unlink(missing_ok=True)


def _build_seam_sheet(segments: list[Path], output_dir: Path, *, fps: float) -> list[dict[str, Any]]:
    frame_dir = output_dir / "seam_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seam_index in range(len(segments) - 1):
        prev_path = segments[seam_index]
        next_path = segments[seam_index + 1]
        prev_frames = _video_frame_count(prev_path)
        frame_paths: list[Path] = []
        for offset in [4, 3, 2, 1]:
            frame = frame_dir / f"seam{seam_index}_prev_m{offset}.png"
            _extract_frame_index(prev_path, prev_frames - offset, frame)
            frame_paths.append(frame)
        for offset in [0, 1, 2, 3]:
            frame = frame_dir / f"seam{seam_index}_next_p{offset}.png"
            _extract_frame_index(next_path, offset, frame)
            frame_paths.append(frame)
        sheet = output_dir / f"seam_{seam_index}_{seam_index + 1}.png"
        _make_contact_sheet(frame_paths, sheet)
        rows.append({"seam": [seam_index, seam_index + 1], "contact_sheet": str(sheet)})
    return rows


def _latent_delta_metrics(
    session: dict[str, Any],
    remote_jobs_dir: Path,
    *,
    output_duration_s: float,
    final_fps: float,
) -> list[dict[str, Any]]:
    import torch

    actions = session["actions"]
    rows: list[dict[str, Any]] = []
    for index in range(1, len(actions)):
        prev = actions[index - 1]
        cur = actions[index]
        frames = int(cur.get("continuation_frames") or 0)
        if frames <= 0:
            continue
        prev_dir = remote_jobs_dir / str(prev["job_id"])
        cur_dir = remote_jobs_dir / str(cur["job_id"])
        prev_video = torch.load(prev_dir / "video_latent.pt", map_location="cpu")
        cur_video = torch.load(cur_dir / "video_latent.pt", map_location="cpu")
        video_tail_frames = max(1, math.ceil(frames / 8.0))
        video_ref = prev_video[:, :, -video_tail_frames:, :, :]
        video_new = cur_video[:, :, :video_tail_frames, :, :]
        video_delta = (video_new - video_ref).float()

        row: dict[str, Any] = {
            "seam": [index - 1, index],
            "continuation_frames": frames,
            "video_tail_latent_frames": video_tail_frames,
            "video_exact_equal": bool(torch.equal(video_new, video_ref)),
            "video_max_abs": float(video_delta.abs().max().item()),
            "video_rms": float(torch.sqrt(torch.mean(video_delta.square())).item()),
        }

        prev_audio_path = prev_dir / "audio_latent.pt"
        cur_audio_path = cur_dir / "audio_latent.pt"
        if prev_audio_path.exists() and cur_audio_path.exists():
            prev_audio = torch.load(prev_audio_path, map_location="cpu")
            cur_audio = torch.load(cur_audio_path, map_location="cpu")
            overlap_s = frames / final_fps
            audio_tail_frames = max(1, round(overlap_s * 25.0))
            emitted_audio_frames = min(round(output_duration_s * 25.0), int(prev_audio.shape[2]))
            audio_start = max(emitted_audio_frames - audio_tail_frames, 0)
            audio_stop = min(audio_start + audio_tail_frames, int(prev_audio.shape[2]))
            audio_ref = prev_audio[:, :, audio_start:audio_stop, :]
            audio_new = cur_audio[:, :, : audio_ref.shape[2], :]
            audio_delta = (audio_new - audio_ref).float()
            row.update(
                {
                    "audio_tail_latent_frames": int(audio_ref.shape[2]),
                    "audio_exact_equal": bool(torch.equal(audio_new, audio_ref)),
                    "audio_max_abs": float(audio_delta.abs().max().item()),
                    "audio_rms": float(torch.sqrt(torch.mean(audio_delta.square())).item()),
                }
            )
        rows.append(row)
    return rows


def _write_concat(segments: list[Path], concat_path: Path) -> None:
    concat_path.write_text("".join(f"file '{path.resolve()}'\n" for path in segments), encoding="utf-8")


def _merge_segments_for_inspection(segments: list[Path], concat_path: Path, merged_path: Path, *, fps: float) -> None:
    """Build a stable inspection artifact from fragmented streaming MP4s.

    The worker emits low-latency fragmented MP4s. Stream-copy concatenation can
    preserve odd fragment timing metadata and has produced 60 fps merged files
    from 15 fps segments, which makes frame-quality inspection misleading.
    Re-encode the eval merge to a constant frame rate; production segments stay
    untouched.
    """
    _write_concat(segments, concat_path)
    _run(
        [
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
            str(concat_path),
            "-vf",
            f"fps={fps:g}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(merged_path),
        ]
    )


def _fmt_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "n/a"


def _fmt_factor(value: float | None) -> str:
    return f"{value:.3f}x" if value is not None else "n/a"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"best": None, "p50": None, "worst": None}
    return {"best": min(values), "p50": statistics.median(values), "worst": max(values)}


def _video_stream(summary: dict[str, Any]) -> dict[str, Any] | None:
    for stream in summary.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


def _audio_stream(summary: dict[str, Any]) -> dict[str, Any] | None:
    for stream in summary.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-jobs-dir", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-duration-s", type=float, default=5.0)
    args = parser.parse_args()

    session = _read_json(args.session)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ready_actions = [action for action in session["actions"] if action.get("state") == "ready"]
    segments = [args.output_dir / f"segment_{action['segment_index']}.mp4" for action in ready_actions]
    missing = [str(path) for path in segments if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing segment files: {missing}")

    concat_path = args.output_dir / "concat_eval.txt"
    merged_path = args.output_dir / "merged_eval.mp4"
    _merge_segments_for_inspection(segments, concat_path, merged_path, fps=args.fps)

    segment_probes = [_stream_summary(_ffprobe(path)) for path in segments]
    merged_probe = _stream_summary(_ffprobe(merged_path))
    seams = _build_seam_sheet(segments, args.output_dir, fps=args.fps)
    latent_deltas = []
    if args.remote_jobs_dir is not None and args.remote_jobs_dir.exists():
        latent_deltas = _latent_delta_metrics(
            {"actions": ready_actions},
            args.remote_jobs_dir,
            output_duration_s=args.output_duration_s,
            final_fps=args.fps,
        )

    metrics = {
        "session_id": session["session_id"],
        "bucket": session["bucket"],
        "continuity_frames": session["continuity_frames"],
        "segments": [
            {
                "segment_index": action["segment_index"],
                "state": action["state"],
                "job_id": action["job_id"],
                "time_to_first_video_byte_s": action.get("time_to_first_video_byte_s"),
                "runtime_s": action.get("runtime_s"),
                "denoise_s": (action.get("stage_times") or {}).get("denoise_s"),
                "latent_continuation_s": (action.get("stage_times") or {}).get("latent_continuation_s"),
                "emitted_frames": (action.get("stage_times") or {}).get("emitted_frames"),
                "emitted_duration_s": (
                    float((action.get("stage_times") or {}).get("emitted_frames")) / float(args.fps)
                    if (action.get("stage_times") or {}).get("emitted_frames") is not None
                    else args.output_duration_s
                ),
                "live_capacity_factor": _ratio(
                    action.get("runtime_s"),
                    (
                        float((action.get("stage_times") or {}).get("emitted_frames")) / float(args.fps)
                        if (action.get("stage_times") or {}).get("emitted_frames") is not None
                        else args.output_duration_s
                    ),
                ),
                "first_byte_capacity_factor": _ratio(
                    action.get("time_to_first_video_byte_s"),
                    (
                        float((action.get("stage_times") or {}).get("emitted_frames")) / float(args.fps)
                        if (action.get("stage_times") or {}).get("emitted_frames") is not None
                        else args.output_duration_s
                    ),
                ),
                "continuation_frames": action.get("continuation_frames"),
                "probe": segment_probes[action["segment_index"]],
            }
            for action in ready_actions
        ],
        "merged": {"path": str(merged_path), "probe": merged_probe},
        "seams": seams,
        "latent_deltas": latent_deltas,
    }
    warm_segments = metrics["segments"][1:] if len(metrics["segments"]) > 1 else metrics["segments"]
    first_byte_values = [
        float(row["time_to_first_video_byte_s"]) for row in warm_segments if row["time_to_first_video_byte_s"] is not None
    ]
    denoise_values = [float(row["denoise_s"]) for row in warm_segments if row["denoise_s"] is not None]
    runtime_values = [float(row["runtime_s"]) for row in warm_segments if row["runtime_s"] is not None]
    live_capacity_values = [float(row["live_capacity_factor"]) for row in warm_segments if row["live_capacity_factor"] is not None]
    first_byte_capacity_values = [
        float(row["first_byte_capacity_factor"]) for row in warm_segments if row["first_byte_capacity_factor"] is not None
    ]
    metrics["summary"] = {
        "warm_segments": len(warm_segments),
        "first_byte": _stats(first_byte_values),
        "runtime": _stats(runtime_values),
        "denoise": _stats(denoise_values),
        "live_capacity_factor": _stats(live_capacity_values),
        "first_byte_capacity_factor": _stats(first_byte_capacity_values),
        "live_capacity_passes": sum(value <= 1.0 for value in live_capacity_values),
        "live_capacity_total": len(live_capacity_values),
        "first_byte_passes": sum(value <= 1.0 for value in first_byte_capacity_values),
        "first_byte_total": len(first_byte_capacity_values),
        "all_video_latents_exact": all(row.get("video_exact_equal") for row in latent_deltas) if latent_deltas else None,
        "all_audio_latents_exact": all(row.get("audio_exact_equal") for row in latent_deltas) if latent_deltas else None,
    }
    metrics_path = args.output_dir / "continuity_eval.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = metrics["summary"]
    merged_video = _video_stream(merged_probe) or {}
    merged_audio = _audio_stream(merged_probe) or {}

    report_path = args.output_dir / "continuity_eval.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Live Continuity Eval: {session['session_id']}",
                "",
                f"Bucket: `{session['bucket']}`",
                f"Continuity frames: `{session['continuity_frames']}`",
                "",
                "## Summary",
                "",
                f"- Warm first-byte: best {_fmt_seconds(summary['first_byte']['best'])}, "
                f"p50 {_fmt_seconds(summary['first_byte']['p50'])}, "
                f"worst {_fmt_seconds(summary['first_byte']['worst'])}",
                f"- Warm denoise: best {_fmt_seconds(summary['denoise']['best'])}, "
                f"p50 {_fmt_seconds(summary['denoise']['p50'])}, "
                f"worst {_fmt_seconds(summary['denoise']['worst'])}",
                f"- Steady live capacity factor: best {_fmt_factor(summary['live_capacity_factor']['best'])}, "
                f"p50 {_fmt_factor(summary['live_capacity_factor']['p50'])}, "
                f"worst {_fmt_factor(summary['live_capacity_factor']['worst'])}",
                f"- Steady live pass rate: {summary['live_capacity_passes']}/{summary['live_capacity_total']}",
                f"- First-byte/startup pass rate: {summary['first_byte_passes']}/{summary['first_byte_total']}",
                f"- Exact latent seams: video={summary['all_video_latents_exact']}, audio={summary['all_audio_latents_exact']}",
                f"- Merged AV: duration {merged_probe['duration']:.3f}s, "
                f"video {merged_video.get('width')}x{merged_video.get('height')} "
                f"{merged_video.get('nb_frames')} frames, audio codec {merged_audio.get('codec_name')}",
                "",
                "## Segment Timing",
                "",
                "| Segment | First byte | Runtime | Denoise | Continuation |",
                "| ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {row['segment_index']} | "
                        f"{_fmt_seconds(row['time_to_first_video_byte_s'])} | "
                        f"{_fmt_seconds(row['runtime_s'])} | "
                        f"{_fmt_seconds(row['denoise_s'])} | "
                        f"{row['continuation_frames']} |"
                    )
                    for row in metrics["segments"]
                ],
                "",
                "## Livestream Capacity",
                "",
                "| Segment | Emitted frames | Emitted duration | Runtime factor | First-byte factor |",
                "| ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {row['segment_index']} | "
                        f"{int(row['emitted_frames']) if row['emitted_frames'] is not None else ''} | "
                        f"{row['emitted_duration_s']:.3f}s | "
                        f"{_fmt_factor(row['live_capacity_factor'])} | "
                        f"{_fmt_factor(row['first_byte_capacity_factor'])} |"
                    )
                    for row in metrics["segments"]
                ],
                "",
                "## Latent Prefix Deltas",
                "",
                "| Seam | Video exact | Video max abs | Video RMS | Audio exact | Audio max abs | Audio RMS |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: |",
                *[
                    (
                        f"| {row['seam'][0]}->{row['seam'][1]} | "
                        f"{row['video_exact_equal']} | {row['video_max_abs']:.6g} | {row['video_rms']:.6g} | "
                        f"{row.get('audio_exact_equal')} | {row.get('audio_max_abs', 0.0):.6g} | {row.get('audio_rms', 0.0):.6g} |"
                    )
                    for row in latent_deltas
                ],
                "",
                "## Artifacts",
                "",
                f"- Merged: `{merged_path}`",
                *[f"- Seam {row['seam'][0]}->{row['seam'][1]}: `{row['contact_sheet']}`" for row in seams],
                f"- JSON: `{metrics_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
