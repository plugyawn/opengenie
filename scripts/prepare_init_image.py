from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _content_bbox(arr: np.ndarray, *, threshold: float) -> tuple[int, int, int, int]:
    luminance = arr.astype(np.float32).mean(axis=2)
    rows = np.where(luminance.mean(axis=1) > threshold)[0]
    cols = np.where(luminance.mean(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, arr.shape[1], arr.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


def _subject_bbox(arr: np.ndarray, *, dark_threshold: float) -> tuple[int, int, int, int] | None:
    luminance = arr.astype(np.float32).mean(axis=2)
    mask = luminance < dark_threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _row_median_background(arr: np.ndarray, *, dark_threshold: float) -> Image.Image:
    luminance = arr.astype(np.float32).mean(axis=2)
    bg = np.empty_like(arr)
    for y in range(arr.shape[0]):
        row = arr[y]
        keep = luminance[y] >= dark_threshold
        if keep.any():
            color = np.median(row[keep], axis=0)
        else:
            color = np.median(row, axis=0)
        bg[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)
    return Image.fromarray(bg, mode="RGB")


def prepare(
    input_path: Path,
    crop_output: Path,
    centered_output: Path,
    *,
    width: int,
    height: int,
    bar_threshold: float,
    dark_threshold: float,
) -> dict[str, object]:
    image = Image.open(input_path).convert("RGB")
    arr = np.asarray(image)
    x0, y0, x1, y1 = _content_bbox(arr, threshold=bar_threshold)
    cropped = image.crop((x0, y0, x1, y1)).resize((width, height), Image.Resampling.LANCZOS)
    crop_output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(crop_output, quality=95)

    crop_arr = np.asarray(cropped)
    subject = _subject_bbox(crop_arr, dark_threshold=dark_threshold)
    background = _row_median_background(crop_arr, dark_threshold=dark_threshold)
    centered = background.copy()
    if subject is None:
        shift_x = 0
        shift_y = 0
    else:
        sx0, sy0, sx1, sy1 = subject
        subject_cx = (sx0 + sx1) / 2.0
        subject_cy = (sy0 + sy1) / 2.0
        shift_x = int(round(width / 2.0 - subject_cx))
        # Keep floor contact visually stable; use horizontal recentering as the main correction.
        shift_y = int(round((height * 0.52) - subject_cy))
        shift_y = max(min(shift_y, height // 10), -height // 10)

    src_left = max(0, -shift_x)
    src_top = max(0, -shift_y)
    src_right = min(width, width - shift_x)
    src_bottom = min(height, height - shift_y)
    dst_left = max(0, shift_x)
    dst_top = max(0, shift_y)
    if src_right > src_left and src_bottom > src_top:
        patch = cropped.crop((src_left, src_top, src_right, src_bottom))
        centered.paste(patch, (dst_left, dst_top))
    centered.save(centered_output, quality=95)

    return {
        "input_size": image.size,
        "content_bbox": [x0, y0, x1, y1],
        "target_size": [width, height],
        "subject_bbox": list(subject) if subject is not None else None,
        "shift": [shift_x, shift_y],
        "crop_output": str(crop_output),
        "centered_output": str(centered_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare image-conditioning frames for LTX buckets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--crop-output", type=Path, required=True)
    parser.add_argument("--centered-output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=928)
    parser.add_argument("--height", type=int, default=522)
    parser.add_argument("--bar-threshold", type=float, default=12.0)
    parser.add_argument("--dark-threshold", type=float, default=80.0)
    args = parser.parse_args()

    result = prepare(
        args.input,
        args.crop_output,
        args.centered_output,
        width=args.width,
        height=args.height,
        bar_threshold=args.bar_threshold,
        dark_threshold=args.dark_threshold,
    )
    print(result)


if __name__ == "__main__":
    main()
