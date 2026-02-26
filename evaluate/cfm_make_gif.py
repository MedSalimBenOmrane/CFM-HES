# -*- coding: utf-8 -*-
"""
cfm_make_gif.py - Generate a GIF showing CFM evolution (HE -> S -> HES)
per iteration, with panels: [HE | S_gen_t | S_GT | HES_gen_t | HES_GT].

Example:
export PYTHONPATH="put path of your data here:$PYTHONPATH"

python cfm_make_gif.py \
--he_path "put path of your data here" \
  --hes_gt_path "put path of your data here" \
  --ckpt_dir "put path of your data here" \
  --device cuda \
  --dim_image 256 \
  --dim 128 --dim_mults 1 1 2 2 4 \
  --sample_steps 50 \
  --save_every 1 --fps 10 --scale 1 \
  --out_dir .

Notes:
- If intermediate frames are not exposed by your inference code, see the
  "OPTIONAL PATCH" note below to enable a callback in inference_fm_cond_s_he.py.
"""

import os, argparse, time, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import torch

# --- Stain utils & CFM inference ---
from separation_model.stain_matrix import Wgt2
from separation_model.global_utils import vectorize

from inference_fm_cond_s_he import (
    InferenceConfig, load_model as load_cfm_model,
    generate_hes_from_he, separate_he_from_rgb
)
from inference_fm_cond_s_he_frames import generate_hes_from_he_frames


# ----------------------- General utils -----------------------
def set_seeds(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pil_from_tensor01(x01):
    """x01: torch [1,3,H,W] or [3,H,W], values in [0,1]."""
    if torch.is_tensor(x01):
        if x01.dim() == 4: x01 = x01[0]
        arr = (x01.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        arr = x01
    return Image.fromarray(arr)

def tensor01_from_pil(img: Image.Image):
    arr = (np.asarray(img.convert("RGB")).astype(np.float32) / 255.0).transpose(2,0,1)
    t = torch.from_numpy(arr).unsqueeze(0)
    return t

def to_u8(np01):
    return (np.clip(np01, 0.0, 1.0) * 255.0).round().astype(np.uint8)

def annotate(img_u8, text):
    """Add a text banner in the top-left corner. ASCII only."""
    text = str(text).replace('—','-').replace('–','-')
    im = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    if hasattr(draw, "textbbox"):
        l,t,r,b = draw.textbbox((0,0), text, font=font)
        tw, th = r-l, b-t
    else:
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([0, 0, tw+2*pad, th+2*pad], fill=(0,0,0,128))
    draw.text((pad, pad), text, fill=(255,255,255), font=font)
    return np.asarray(im)

def hstack(imgs):
    """Horizontally concatenate uint8 RGB numpy images with same height."""
    widths = [im.shape[1] for im in imgs]
    heights = [im.shape[0] for im in imgs]
    H = max(heights); W = sum(widths)
    out = Image.new('RGB', (W, H))
    x = 0
    for im in imgs:
        out.paste(Image.fromarray(im), (x, 0))
        x += im.shape[1]
    return np.asarray(out)

def resize_np(img_u8, scale=1):
    if scale == 1: return img_u8
    H, W = img_u8.shape[:2]
    return np.asarray(Image.fromarray(img_u8).resize((W*scale, H*scale), Image.NEAREST))


# ------------------ Conversions HE/HES/Concentrations ------------------
def rgb_pil_to_conc3_numpy(img: Image.Image) -> np.ndarray:
    """Return cH,cE,cS as numpy [3,H,W] (>=0)."""
    im = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    im = np.clip(im, 1e-6, 1.0)
    OD = -np.log(im)  # [H,W,3]
    V = vectorize((OD*255.0).astype(np.uint8))  # same utility as training (OD already float here)
    # Keep consistency: recompute directly with float OD
    V = OD.reshape(-1,3).T.astype(np.float32)   # (3,N)
    W = np.array(Wgt2, dtype=np.float32)
    Winv = np.linalg.pinv(W).astype(np.float32)
    C3 = Winv @ V                                # (3,N)
    H, Wd = img.size[1], img.size[0]
    C3 = C3.reshape(3, H, Wd)
    C3 = np.clip(C3, 0.0, None)
    return C3

def s_to_rgb_u8(cS: np.ndarray) -> np.ndarray:
    """RGB visualization of S only, using W's S column."""
    W = np.array(Wgt2, dtype=np.float32)
    OD_S = cS[..., None] * W[:,2][None,None,:]  # [H,W,3]
    I = np.exp(-OD_S)
    return (np.clip(I, 0.0, 1.0) * 255.0).astype(np.uint8)


# ------------------ Capture frames during CFM ------------------
def collect_frames_via_return_frames(he_pil, cfg, model, ref_p99_s, save_every):
    """Try generate_hes_from_he(..., return_frames=True)."""
    try:
        out = generate_hes_from_he(
            he_pil, cfg, model=model, ref_p99_s=ref_p99_s,
            return_frames=True, save_every=save_every
        )
        # expected: hes_gen_pil, s_vis_pil, frames
        if isinstance(out, tuple) and len(out) >= 3:
            hes_final, s_vis_final, frames = out[0], out[1], out[2]
            return hes_final, s_vis_final, frames
    except TypeError:
        pass
    return None, None, None

def collect_frames_via_callback(he_pil, cfg, model, ref_p99_s, save_every):
    """Try generate_hes_from_he(..., capture_callback=...)."""
    frames = []
    def cb(iter_idx, payload):
        # payload may contain 'hes_rgb' (PIL/np) and/or 'cS' (np)
        hes_img = payload.get('hes_rgb', None)
        cS = payload.get('cS', None)
        if hes_img is not None and isinstance(hes_img, Image.Image):
            frames.append({'iter': iter_idx, 'hes_rgb': hes_img})
        elif hes_img is not None and isinstance(hes_img, np.ndarray):
            frames.append({'iter': iter_idx, 'hes_rgb': Image.fromarray(hes_img)})
        if cS is not None:
            frames[-1]['cS'] = cS  # [H,W] float
    try:
        hes_final, s_vis_final = generate_hes_from_he(
            he_pil, cfg, model=model, ref_p99_s=ref_p99_s,
            capture_callback=cb, save_every=save_every
        )
        return hes_final, s_vis_final, frames
    except TypeError:
        pass
    return None, None, None


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--he_path', required=True, type=str)
    ap.add_argument('--hes_gt_path', type=str, default=None)
    ap.add_argument('--ckpt_dir', required=True, type=str)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dim_image', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--dim_mults', nargs='+', type=int, default=[1,1,2,2,4])
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--ref_p99_s', type=float, default=1.0)
    ap.add_argument('--use_gt_p99_s', action='store_true')
    ap.add_argument('--save_every', type=int, default=1, help='save one frame every N iterations')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--scale', type=int, default=1)
    ap.add_argument('--out_dir', type=str, default='.')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Load CFM model ---
    cfg = InferenceConfig(
        CHECKPOINT_DIR=args.ckpt_dir,
        DIM=args.dim,
        DIM_MULTS=tuple(args.dim_mults),
        DROPOUT=args.dropout,
        SAMPLE_STEPS=args.sample_steps,
        IMG_SIZE=args.dim_image,
        DEVICE=str(device),
        REF_P99_S=args.ref_p99_s,
    )
    model = load_cfm_model(cfg)  # .eval() is called internally

    # --- Load images ---
    he_pil = Image.open(args.he_path).convert('RGB').resize((args.dim_image, args.dim_image), Image.BICUBIC)
    hes_gt_pil = None
    if args.hes_gt_path and os.path.exists(args.hes_gt_path):
        hes_gt_pil = Image.open(args.hes_gt_path).convert('RGB').resize((args.dim_image, args.dim_image), Image.BICUBIC)

    # Calibrate p99_S from GT if requested
    ref_p99 = None
    if args.use_gt_p99_s and hes_gt_pil is not None:
        C3_gt = rgb_pil_to_conc3_numpy(hes_gt_pil)
        ref_p99 = float(np.percentile(np.clip(C3_gt[2], 0, None), 99.0))

    # --- Capture CFM frames ---
    t0 = time.time()
    hes_final, s_vis_final, frames = collect_frames_via_return_frames(
        he_pil, cfg, model, ref_p99, args.save_every
    )
    if frames is None:
      hes_final, s_vis_final, frames = generate_hes_from_he_frames(
        he_pil, cfg, model=model, ref_p99_s=ref_p99, save_every=args.save_every, max_steps=args.sample_steps
    )
    # Fallback: no intermediate frames -> compute only the final output
    if frames is None:
        # appel simple
            hes_final, s_vis_final, frames = generate_hes_from_he_frames(
        he_pil, cfg, model=model, ref_p99_s=ref_p99, save_every=args.save_every, max_steps=args.sample_steps
    )
    dt = time.time() - t0
    print(f"[INFO] CFM inference finished in {dt:.2f}s. Captured frames: {len(frames)}")

    # --- Prepare HE and GT as uint8 numpy arrays ---
    he_u8 = np.asarray(he_pil, dtype=np.uint8)
    s_gt_u8 = None
    if hes_gt_pil is not None:
        C3_gt = rgb_pil_to_conc3_numpy(hes_gt_pil)
        s_gt_u8 = s_to_rgb_u8(C3_gt[2])
    hes_gt_u8 = np.asarray(hes_gt_pil, dtype=np.uint8) if hes_gt_pil is not None else None

    # --- Build GIF frames ---
    gif_frames = []

    # Add initial frame if nothing was captured
    if len(frames) == 0:
        # Rebuild final S and HES (single frame)
        C3_fin = rgb_pil_to_conc3_numpy(hes_final)
        s_fin_u8 = s_to_rgb_u8(C3_fin[2])
        panel = [he_u8, s_fin_u8, s_gt_u8 if s_gt_u8 is not None else he_u8,
                 np.asarray(hes_final), hes_gt_u8 if hes_gt_u8 is not None else he_u8]
        row = hstack([annotate(img, lab) for img,lab in zip(
            panel,
                ["HE", "S gen (final)", "S GT" if s_gt_u8 is not None else "S GT n/a",
                 "HES gen (final)", "HES GT" if hes_gt_u8 is not None else "HES GT n/a"]
        )])
        gif_frames.append(resize_np(row, args.scale))
    else:
        for k, step in enumerate(frames):
            # Get generated HES
            hes_k = step.get('hes_rgb', None)
            if isinstance(hes_k, Image.Image):
                hes_k_u8 = np.asarray(hes_k, dtype=np.uint8)
            elif isinstance(hes_k, np.ndarray):
                hes_k_u8 = hes_k.astype(np.uint8)
            else:
                # skip frame if no HES image is present
                continue

            # Get generated S: direct if provided, otherwise from HES_k
            if 'cS' in step and step['cS'] is not None:
                s_k_u8 = s_to_rgb_u8(step['cS'])
            else:
                C3_k = rgb_pil_to_conc3_numpy(Image.fromarray(hes_k_u8))
                s_k_u8 = s_to_rgb_u8(C3_k[2])

            # Build 5-column panel
            cols = [
                annotate(he_u8, f"HE (fixe)"),
                annotate(s_k_u8, f"S gen - iter {k*args.save_every}/{args.sample_steps}"),
                annotate(s_gt_u8 if s_gt_u8 is not None else he_u8,
                         "S GT" if s_gt_u8 is not None else "S GT n/a"),
                annotate(hes_k_u8, f"HES gen - iter {k*args.save_every}/{args.sample_steps}"),
                annotate(hes_gt_u8 if hes_gt_u8 is not None else he_u8,
                         "HES GT" if hes_gt_u8 is not None else "HES GT n/a"),
            ]
            row = hstack(cols)
            gif_frames.append(resize_np(row, args.scale))

    # --- Save GIF ---
    gif_path = os.path.join(args.out_dir, "cfm_evolution_HE_S_HES.gif")
    imageio.mimsave(gif_path, gif_frames, fps=args.fps, loop=0)
    print(f"[OK] GIF saved: {gif_path}")


if __name__ == "__main__":
    main()
