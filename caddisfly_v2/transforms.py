"""Aspect-ratio-safe transforms for high-resolution specimen photographs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PAD_FILL = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


@dataclass(frozen=True)
class EnsureRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    weighted_total = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 128
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        between = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if between > best_variance:
            best_variance = between
            best_threshold = threshold
    return int(np.clip(best_threshold, 70, 210))


def _largest_active_run(active: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(np.append(active, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


@dataclass(frozen=True)
class ForegroundCrop:
    """Conservatively crop a dark specimen on a light photographic background.

    Detection happens on a small grayscale proxy and keeps the largest broad
    horizontal foreground run.  Any uncertain case returns the original frame,
    which is safer than removing diagnostic anatomy.
    """

    proxy_size: int = 320
    padding_fraction: float = 0.16
    minimum_width_fraction: float = 0.12
    minimum_height_fraction: float = 0.04

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < 8 or height < 8:
            return image
        proxy = image.convert("L")
        proxy.thumbnail((self.proxy_size, self.proxy_size), Image.Resampling.BILINEAR)
        gray = np.asarray(proxy)
        proxy_height, proxy_width = gray.shape
        threshold = _otsu_threshold(gray)
        dark = gray <= threshold

        # Ignore a very narrow border during detection; the eventual padded box
        # may still extend to the true image boundary.
        bx = max(1, round(proxy_width * 0.015))
        by = max(1, round(proxy_height * 0.015))
        dark[:by, :] = False
        dark[-by:, :] = False
        dark[:, :bx] = False
        dark[:, -bx:] = False

        min_column_pixels = max(2, round(proxy_height * 0.012))
        active_columns = dark.sum(axis=0) >= min_column_pixels
        # Bridge small gaps caused by fine wing pattern and antennae.
        bridge = max(3, round(proxy_width * 0.025))
        active_columns = np.convolve(
            active_columns.astype(np.int16), np.ones(bridge, dtype=np.int16), mode="same"
        ) > 0
        x_run = _largest_active_run(active_columns)
        if x_run is None or x_run[1] - x_run[0] < proxy_width * self.minimum_width_fraction:
            return image

        x0_proxy, x1_proxy = x_run
        restricted = dark[:, x0_proxy:x1_proxy]
        min_row_pixels = max(2, round((x1_proxy - x0_proxy) * 0.012))
        active_rows = restricted.sum(axis=1) >= min_row_pixels
        row_bridge = max(3, round(proxy_height * 0.025))
        active_rows = np.convolve(
            active_rows.astype(np.int16), np.ones(row_bridge, dtype=np.int16), mode="same"
        ) > 0
        y_run = _largest_active_run(active_rows)
        if y_run is None or y_run[1] - y_run[0] < proxy_height * self.minimum_height_fraction:
            return image

        y0_proxy, y1_proxy = y_run
        scale_x = width / proxy_width
        scale_y = height / proxy_height
        x0, x1 = x0_proxy * scale_x, x1_proxy * scale_x
        y0, y1 = y0_proxy * scale_y, y1_proxy * scale_y
        pad_x = (x1 - x0) * self.padding_fraction
        pad_y = (y1 - y0) * self.padding_fraction
        box = (
            max(0, round(x0 - pad_x)),
            max(0, round(y0 - pad_y)),
            min(width, round(x1 + pad_x)),
            min(height, round(y1 + pad_y)),
        )
        cropped_width = box[2] - box[0]
        cropped_height = box[3] - box[1]
        if cropped_width < width * self.minimum_width_fraction:
            return image
        if cropped_height < height * self.minimum_height_fraction:
            return image
        return image.crop(box)


@dataclass(frozen=True)
class FitPad:
    size: int
    fill: tuple[int, int, int] = DEFAULT_PAD_FILL

    def __call__(self, image: Image.Image) -> Image.Image:
        contained = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - contained.width) // 2, (self.size - contained.height) // 2)
        canvas.paste(contained, offset)
        return canvas


def build_transform(
    *,
    image_size: int,
    training: bool,
    crop_mode: str = "foreground",
    normalization_mean: Sequence[float] = IMAGENET_MEAN,
    normalization_std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    if crop_mode not in {"foreground", "letterbox"}:
        raise ValueError("crop_mode must be 'foreground' or 'letterbox'")
    if len(normalization_mean) != 3 or len(normalization_std) != 3:
        raise ValueError("normalization_mean and normalization_std must have three values")
    # A top-level callable is required because Windows DataLoader workers use
    # spawn and must pickle the transform graph.
    steps: list[object] = [EnsureRGB()]
    if crop_mode == "foreground":
        steps.append(ForegroundCrop())
    steps.append(FitPad(image_size))
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=7,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=DEFAULT_PAD_FILL,
                ),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
    return transforms.Compose(steps)
