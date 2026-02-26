"""
Inference HES <- HE (256x256) via Conditional Flow Matching (S|H,E).

- Build the same U-Net as training (in: 3=[S_t,H_norm,E_norm], out:1=v_S).
- Load the latest checkpoint from CHECKPOINT_DIR.
- Separate H & E from HE (OD + Wgt2 + pseudo-inverse).
- Sample S with Euler (steps=SAMPLE_STEPS).
- Recompose HES in RGB (inverse Beer-Lambert).
"""

import os
import re
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]      # .../CFM HES
PROJECT_DIR = ROOT_DIR.parent                        # .../PNP_FM
for p in (str(ROOT_DIR), str(PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from separation_model.stain_matrix import Wgt2
from separation_model.global_utils import vectorize


@dataclass
class InferenceConfig:
    ROOT: str = str(ROOT_DIR)
    DATASET: str = "hes"
    MODEL_NAME: str = "fm_cond_s_he"
    CHECKPOINT_DIR: str = "put path of your data here"

    DIM: int = 128
    DIM_MULTS: Tuple[int, ...] = (1, 1, 2, 2, 4)
    DROPOUT: float = 0.0
    SAMPLE_STEPS: int = 50
    IMG_SIZE: int = 256

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REF_P99_S: float = 1.0


def _compose_rgb_from_conc(cH: np.ndarray, cE: np.ndarray, cS: np.ndarray, W: np.ndarray) -> np.ndarray:
    od = (
        cH[..., None] * W[:, 0][None, None, :]
        + cE[..., None] * W[:, 1][None, None, :]
        + cS[..., None] * W[:, 2][None, None, :]
    )
    rgb = np.exp(-od) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _percentile_np(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x.reshape(-1), q))


def _ensure_w_matrix(w_np: Optional[np.ndarray] = None) -> np.ndarray:
    w = np.array(Wgt2 if w_np is None else w_np, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] != 3:
        raise ValueError(f"Invalid W shape: {w.shape}")
    if w.shape[1] == 2:
        s_col = np.array(Wgt2, dtype=np.float32)[:, 2:3]
        w = np.concatenate([w, s_col], axis=1)
    if w.shape[1] != 3:
        raise ValueError(f"Invalid W shape: {w.shape}")
    return w


def _find_latest_ckpt(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None

    cands = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    best = []
    for p in cands:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        if m:
            best.append((int(m.group(1)), p))

    if best:
        return max(best, key=lambda t: t[0])[1]

    final = os.path.join(ckpt_dir, "model_final.pt")
    return final if os.path.isfile(final) else None


def build_unet_fm_cond_s_he(cfg: InferenceConfig, device: str) -> nn.Module:
    from unet.unet_core import Unet as BaseUNet

    model = BaseUNet(
        dim=cfg.DIM,
        dim_mults=tuple(cfg.DIM_MULTS),
        in_channels=3,
        out_channels=1,
        dropout=cfg.DROPOUT,
    ).to(device)
    return model


def load_model(cfg: InferenceConfig) -> nn.Module:
    device = cfg.DEVICE
    model = build_unet_fm_cond_s_he(cfg, device)

    ckpt_path = _find_latest_ckpt(cfg.CHECKPOINT_DIR)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {cfg.CHECKPOINT_DIR}")

    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def separate_he_from_rgb(he_img: Image.Image, to_size: int = 256, w_np: Optional[np.ndarray] = None):
    he = he_img.convert("RGB").resize((to_size, to_size), Image.BICUBIC)
    im_u8 = np.asarray(he)

    v = vectorize(im_u8).astype(np.float32)
    W = _ensure_w_matrix(w_np)
    winv = np.linalg.pinv(W)

    c3 = winv @ v
    h, w = to_size, to_size
    cH = c3[0].reshape(h, w)
    cE = c3[1].reshape(h, w)

    pH = max(1e-6, _percentile_np(np.clip(cH, 0, None), 99.0))
    pE = max(1e-6, _percentile_np(np.clip(cE, 0, None), 99.0))
    cH_norm = np.clip(cH / pH, 0.0, 1.0)
    cE_norm = np.clip(cE / pE, 0.0, 1.0)

    cH_norm_t = torch.from_numpy(cH_norm)[None, None].float()
    cE_norm_t = torch.from_numpy(cE_norm)[None, None].float()

    return cH, cE, cH_norm_t, cE_norm_t


@torch.no_grad()
def sample_S(
    model: nn.Module,
    cH_norm: torch.Tensor,
    cE_norm: torch.Tensor,
    steps: int = 50,
    device: Optional[str] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    Hm11 = cH_norm.to(device) * 2.0 - 1.0
    Em11 = cE_norm.to(device) * 2.0 - 1.0

    B, _, H, W = Hm11.shape
    x = torch.randn(B, 1, H, W, device=device)
    dt = 1.0 / steps

    model.eval()
    for s in range(steps):
        t = torch.full((B,), (s + 0.5) * dt, device=device)
        x_in = torch.cat([x, Hm11, Em11], dim=1)
        v = model(x_in, t)
        x = x + v * dt
    return x


def generate_hes_from_he(
    he_img: Image.Image,
    cfg: InferenceConfig,
    model: Optional[nn.Module] = None,
    ref_p99_s: Optional[float] = None,
    w_np: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Image.Image]:
    device = cfg.DEVICE
    model = model or load_model(cfg)
    ref_p99_s = float(cfg.REF_P99_S if ref_p99_s is None else ref_p99_s)

    cH_raw, cE_raw, cH_norm_t, cE_norm_t = separate_he_from_rgb(he_img, to_size=cfg.IMG_SIZE, w_np=w_np)

    S_m11 = sample_S(model, cH_norm_t, cE_norm_t, steps=cfg.SAMPLE_STEPS, device=device)
    S_norm = ((S_m11 + 1.0) / 2.0).clamp(0.0, 1.0)

    cS_hat = (S_norm[0, 0].cpu().numpy() * ref_p99_s).astype(np.float32)

    W = _ensure_w_matrix(w_np)
    hes_rgb = _compose_rgb_from_conc(
        cH_raw.astype(np.float32),
        cE_raw.astype(np.float32),
        cS_hat,
        W,
    )

    od_s = cS_hat[..., None] * W[:, 2][None, None, :]
    s_rgb = np.exp(-od_s) * 255.0
    s_rgb = np.clip(s_rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(hes_rgb), Image.fromarray(s_rgb)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HES from HE 256x256 with CFM.")
    parser.add_argument("--he", required=True, help="Input HE image (png/jpg).")
    parser.add_argument("--out", default="hes_generated.png", help="HES output path.")
    parser.add_argument("--outs", default="s_generated.png", help="S output path.")
    parser.add_argument("--ref-p99-s", type=float, default=None, help="Reference p99_S override.")
    args = parser.parse_args()

    cfg = InferenceConfig()
    model = load_model(cfg)

    he_pil = Image.open(args.he).convert("RGB")
    hes_pil, s_pil = generate_hes_from_he(he_pil, cfg, model=model, ref_p99_s=args.ref_p99_s)

    hes_pil.save(args.out)
    s_pil.save(args.outs)
    print(f"[OK] HES saved -> {args.out}")
    print(f"[OK] S saved   -> {args.outs}")
