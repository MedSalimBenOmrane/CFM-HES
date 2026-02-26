"""
# 1) auto-crop white borders (threshold 240), pad to exact multiple, 1024 patches
python crop_and_tile.py --input /path/to/large_image.png

# 2) add a 10 px margin after border detection
python crop_and_tile.py --input large_image.png --extra-margin 10

# 3) without padding (non-multiple borders are cropped)
python crop_and_tile.py --input large_image.png --no-pad

# 4) custom output folder and filename prefix
python crop_and_tile.py --input large_image.png --outdir "1024 patches" --prefix slideA

python crop_and_tile.py --input "put path of your data here" --no-pad --prefix slide101F




"""
import os
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def autocrop_white(img: Image.Image, white_threshold: int = 240, extra_margin: int = 0) -> Image.Image:
    """
    Remove white borders around the image.
    - white_threshold: 0..255 ; values above are considered white
    - extra_margin: extra pixels to keep around detected content
    """
    arr = np.asarray(img.convert("RGB"))
    # non-white pixels (at least one channel below threshold)
    mask = (arr < white_threshold).any(axis=2)

    if not mask.any():
        # fully white image -> nothing to crop
        return img

    ys, xs = np.where(mask)
    y1, y2 = int(max(ys.min() - extra_margin, 0)), int(min(ys.max() + 1 + extra_margin, arr.shape[0]))
    x1, x2 = int(max(xs.min() - extra_margin, 0)), int(min(xs.max() + 1 + extra_margin, arr.shape[1]))

    return img.crop((x1, y1, x2, y2))


def tile_image(img: Image.Image, patch_size: int = 1024, pad: bool = True,
               pad_color=(255, 255, 255)):
    """
    Split an image into `patch_size` x `patch_size` tiles.
    - pad=True: add `pad_color` border to reach an exact multiple
    - pad=False: crop to keep only the divisible region
    Return a list [(patch, row, col)].
    """
    w, h = img.size

    if pad:
        new_w = math.ceil(w / patch_size) * patch_size
        new_h = math.ceil(h / patch_size) * patch_size
        if (new_w, new_h) != (w, h):
            canvas = Image.new("RGB", (new_w, new_h), pad_color)
            canvas.paste(img, (0, 0))
            img = canvas
            w, h = img.size
    else:
        new_w = (w // patch_size) * patch_size
        new_h = (h // patch_size) * patch_size
        if (new_w, new_h) != (w, h):
            img = img.crop((0, 0, new_w, new_h))
            w, h = img.size

    patches = []
    rows = h // patch_size
    cols = w // patch_size
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * patch_size, r * patch_size
            x2, y2 = x1 + patch_size, y1 + patch_size
            patches.append((img.crop((x1, y1, x2, y2)), r, c))

    return patches


def main():
    parser = argparse.ArgumentParser(description="Auto-crop white borders and split into 1024x1024 patches.")
    parser.add_argument("--input", required=True, help="Path to input image (png/jpg/tif...).")
    parser.add_argument("--outdir", default="1024 patches", help="Output directory (created if missing).")
    parser.add_argument("--patch", type=int, default=1024, help="Patch size (default: 1024).")
    parser.add_argument("--white-th", type=int, default=240, help="White threshold (0..255).")
    parser.add_argument("--extra-margin", type=int, default=0, help="Extra margin after border detection (px).")
    parser.add_argument("--no-pad", action="store_true", help="Do not pad; crop to the nearest exact multiple.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix (default = input stem).")
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = Image.open(inp).convert("RGB")

    # 1) auto-crop white borders
    cropped = autocrop_white(img, white_threshold=args.white_th, extra_margin=args.extra_margin)

    # 2) split into patches
    patches = tile_image(cropped, patch_size=args.patch, pad=not args.no_pad)

    # 3) save patches
    base = args.prefix if args.prefix is not None else inp.stem
    count = 0
    for patch, r, c in patches:
        fname = f"{base}_r{r:03d}_c{c:03d}.png"
        patch.save(outdir / fname)
        count += 1

    print(f"Input image: {inp}")
    print(f"Original size: {img.size[0]}x{img.size[1]} px")
    print(f"Size after crop: {cropped.size[0]}x{cropped.size[1]} px")
    print(f"Saved patches: {count} in '{outdir.resolve()}'")


if __name__ == "__main__":
    main()
