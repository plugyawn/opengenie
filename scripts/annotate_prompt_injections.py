from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _short_text(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


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
    count = streams[0].get("nb_read_frames") or streams[0].get("nb_frames")
    if count in {None, "N/A"}:
        raise RuntimeError(f"could not read frame count for {path}")
    return int(count)


def _input_duration_s(path: Path) -> float:
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        text=True,
    )
    return float(json.loads(raw)["format"]["duration"])


def _injection_frames(session: dict[str, Any], segment_dir: Path | None, fixed_segment_frames: int) -> dict[int, int]:
    frame_by_segment: dict[int, int] = {}
    frame = 0
    for action in sorted(session["actions"], key=lambda row: int(row["segment_index"])):
        segment_index = int(action["segment_index"])
        frame_by_segment[segment_index] = frame
        if segment_dir is None:
            frame += fixed_segment_frames
        else:
            frame += _video_frame_count(segment_dir / f"segment_{segment_index}.mp4")
    return frame_by_segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Burn prompt-injection frame markers into a live-session MP4.")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--segment-frames", type=int, default=75)
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=None,
        help="Directory containing segment_N.mp4 files; when set, injection frames use actual segment frame counts.",
    )
    parser.add_argument("--marker-duration-s", type=float, default=1.4)
    args = parser.parse_args()

    session: dict[str, Any] = json.loads(args.session.read_text(encoding="utf-8"))
    frame_by_segment = _injection_frames(session, args.segment_dir, args.segment_frames)
    input_duration_s = _input_duration_s(args.input)
    filters: list[str] = []
    injection_rows: list[dict[str, Any]] = []
    for action in session["actions"]:
        segment_index = int(action["segment_index"])
        frame = frame_by_segment[segment_index]
        start = frame / args.fps
        if start >= input_duration_s:
            continue
        stop = start + args.marker_duration_s
        action_text = str(action["text"])
        label = f"prompt @ frame {frame}: {_short_text(action_text)}"
        injection_rows.append(
            {
                "segment_index": segment_index,
                "frame": frame,
                "time_s": start,
                "text": action_text,
            }
        )
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(label)}':"
            "x=24:y=24:"
            "fontcolor=white:fontsize=24:"
            "box=1:boxcolor=black@0.58:boxborderw=12:"
            f"enable='between(t,{start:.6f},{stop:.6f})'"
        )

    vf = ",".join(filters)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.input),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(args.output),
        ],
        check=True,
    )
    args.output.with_suffix(".injections.json").write_text(json.dumps(injection_rows, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
