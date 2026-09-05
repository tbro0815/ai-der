#!/usr/bin/env python3
"""From an image to the patches the Qwen3.8-Flash-Next vision tower expects.

The C engine takes preprocessed f32 patches and knows nothing about JPEG or
resizing; that work lives here, mirroring llama.cpp's qwen3vl path exactly
(tools/mtmd/mtmd-image.cpp, mtmd_image_preprocessor_dyn_size):

  1. decode and convert to RGB
  2. smart-resize: preserve the aspect ratio, ROUND each side to a multiple
     of 32 (patch 16 x merge 2), then floor/ceil-scale into the
     [min_pixels, max_pixels] budget -- and resize the pixels straight to that
     aligned canvas (llama.cpp stretches the sliver the rounding introduces;
     there is NO black padding on this path, unlike glm53's)
  3. bicubic resample (Pillow; llama.cpp reimplements Pillow's kernel
     fixed-point, so the two agree to rounding)
  4. rescale 1/255, normalize with mean/std 0.5 (the mmproj's values)
  5. cut into patches in the tower's order: block-major over the 2x2 merge
     blocks, [channel][row][col] inside each patch (3*16*16 = 768 floats --
     the temporal repeat is folded into the engine's patch matrix, so it is
     NOT duplicated here)

Defaults mirror llama.cpp's qwen3vl limits (8..4096 output tokens);
QWEN38_MAX_IMAGE_TOKENS lowers the cap for bounded encode latency -- the
image is shrunk, not cropped.

USAGE:
  python3 tools/qwen38_image.py photo.jpg --out patches.f32
  python3 tools/qwen38_image.py photo.jpg --json     # grid only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PATCH = 16
MERGE = 2
MIN_TOKENS = 8          # output (merged) tokens, llama.cpp set_limit_image_tokens(8, 4096)
MAX_TOKENS = 4096
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def smart_resize(height, width, *, factor=PATCH * MERGE,
                 min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
    """llama.cpp img_tool::calc_size_preserved_ratio, longest_edge disabled."""
    if height <= 0 or width <= 0:
        raise ValueError("empty image")
    pixels_per_token = factor * factor
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def round_by(v):
        return max(factor, int(round(v / factor)) * factor)

    def ceil_by(v):
        return int(math.ceil(v / factor)) * factor

    def floor_by(v):
        return max(factor, int(math.floor(v / factor)) * factor)

    h_bar, w_bar = round_by(height), round_by(width)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar, w_bar = floor_by(height / beta), floor_by(width / beta)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = ceil_by(height * beta), ceil_by(width * beta)
    return h_bar, w_bar


def patchify(pixels, numpy, patch=PATCH, merge=MERGE):
    """[3, H, W] normalized -> ([grid, 3*16*16], grid_h, grid_w).

    Block-major over the merge blocks -- the 2x2 merger closes over adjacent
    tokens, so the order IS the correctness (vision_qwen3vl.h header)."""
    channels, height, width = pixels.shape
    grid_h, grid_w = height // patch, width // patch
    blocks = pixels.reshape(channels, grid_h // merge, merge, patch,
                            grid_w // merge, merge, patch)
    # -> [gh/m, gw/m, m, m, C, p, p]
    blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)
    return numpy.ascontiguousarray(
        blocks.reshape(grid_h * grid_w, channels * patch * patch),
        dtype=numpy.float32), grid_h, grid_w


def preprocess(source, model_dir=None):
    """Image (path, bytes or PIL object) -> (patches, grid_h, grid_w).

    model_dir is accepted for symmetry with glm53_image.preprocess but unused:
    this checkpoint's constants live in the mmproj GGUF, are fixed above, and
    the engine validates grid vs patch count anyway."""
    del model_dir
    import numpy
    from PIL import Image

    max_tokens = MAX_TOKENS
    override = os.environ.get("QWEN38_MAX_IMAGE_TOKENS")
    if override:
        try:
            max_tokens = max(MIN_TOKENS, int(override))
        except ValueError:
            pass

    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, bytes):
        import io
        image = Image.open(io.BytesIO(source))
    else:
        image = source
    image = image.convert("RGB")

    target_h, target_w = smart_resize(image.height, image.width,
                                      max_tokens=max_tokens)
    if (target_h, target_w) != (image.height, image.width):
        image = image.resize((target_w, target_h), Image.BICUBIC)

    pixels = numpy.asarray(image, dtype=numpy.float32).transpose(2, 0, 1) / 255.0
    for channel in range(3):
        pixels[channel] = (pixels[channel] - MEAN[channel]) / STD[channel]
    return patchify(pixels, numpy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, help="write the f32 patches here")
    parser.add_argument("--json", action="store_true", help="print the grid only")
    arguments = parser.parse_args()

    patches, grid_h, grid_w = preprocess(arguments.image)
    tokens = (grid_h // MERGE) * (grid_w // MERGE)
    if arguments.out:
        arguments.out.write_bytes(patches.tobytes())
    if arguments.json:
        print(json.dumps({"grid_h": grid_h, "grid_w": grid_w,
                          "patches": int(patches.shape[0]),
                          "image_tokens": tokens}))
    else:
        print(f"{arguments.image.name}: grid {grid_h}x{grid_w}, "
              f"{patches.shape[0]} patches of {patches.shape[1]}, "
              f"{tokens} image tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
