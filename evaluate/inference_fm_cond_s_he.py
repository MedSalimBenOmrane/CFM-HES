"""
Inference HES <- HE (256x256) via Conditional Flow Matching (S|H,E).

- Build the same U-Net as training (in: 3=[S_t,H_norm,E_norm], out:1=v_S).
- Load the **latest checkpoint** from CHECKPOINT_DIR.
- Separate H & E from the HE image (OD + Wgt2 + pseudo-inverse).
- Sample S with Euler (steps=SAMPLE_STEPS).
- Recompose HES in RGB (inverse Beer-Lambert).
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]      # .../CFM HES
PROJECT_DIR = ROOT_DIR.parent                       # .../PNP_FM
for p in (str(ROOT_DIR), str(PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os, re, glob, math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn

# --- utility imports ---
from separation_model.stain_matrix import Wgt2          # (3,3) OD columns: H,E,S
from separation_model.global_utils import vectorize     # uint8 RGB -> OD (3,H*W)

# --- U-Net used during training ---
from unet.unet_core import Unet as BaseUNet


# ===================== Config (same options as training) =====================

@dataclass
class InferenceConfig:
    # paths
    ROOT: str = str(ROOT_DIR)
    DATASET: str = "hes"
    MODEL_NAME: str = "fm_cond_s_he"
    # checkpoint directory (must contain model_*.pt or model_final.pt)
    CHECKPOINT_DIR: str = "put path of your data here"

    # architecture / hyperparameters
    DIM: int = 128
    DIM_MULTS: Tuple[int, ...] = (1, 1, 2, 2, 4)
    DROPOUT: float = 0.0
    SAMPLE_STEPS: int = 50
    IMG_SIZE: int = 256

    # device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # calibration: reference p99_S to denormalize test-time S_norm
    REF_P99_S: float = 1.0  # set the training p99_S mean here if available


# ===================== Numeric utils =====================

def _torch_to_rgb01(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) [-1,1] or [0,1] -> [0,1]."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0) if x.min() < 0 else x.clamp(0.0, 1.0)

def _compose_rgb_from_conc(cH: np.ndarray, cE: np.ndarray, cS: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Beer–Lambert inverse -> RGB uint8 (H,W,3)."""
    OD = (cH[..., None] * W[:, 0][None, None, :] +
          cE[..., None] * W[:, 1][None, None, :] +
          cS[..., None] * W[:, 2][None, None, :])
    I = np.exp(-OD) * 255.0
    return np.clip(I, 0, 255).astype(np.uint8)

def _percentile_np(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x.reshape(-1), q))

def _find_latest_ckpt(ckpt_dir: str) -> Optional[str]:
    """Return latest 'model_*.pt' path, or 'model_final.pt' as fallback."""
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


# ===================== Build / Load model =====================

def build_unet_fm_cond_s_he(cfg: InferenceConfig, device: str) -> nn.Module:
    """Same architecture as training."""
    model = BaseUNet(
        dim=cfg.DIM,
        dim_mults=tuple(cfg.DIM_MULTS),
        in_channels=3,   # [S_t, H_norm, E_norm]
        out_channels=1,  # v_S
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
    # supports 2 formats: full dict {'model': state_dict, ...} or raw state_dict
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ===================== H/E separation from HE =====================

def separate_he_from_rgb(he_img: Image.Image, to_size: int = 256):
    """
    he_img: PIL RGB (any size) -> resized to `to_size`.
    Returns:
      cH_raw, cE_raw : np.float32 (H,W)
      cH_norm, cE_norm : torch.float32 (1,1,H,W) in [0,1]
    """
    he = he_img.convert("RGB").resize((to_size, to_size), Image.BICUBIC)
    im_u8 = np.array(he)  # (H,W,3) uint8

    V = vectorize(im_u8).astype(np.float32)   # (3, H*W)
    W = np.array(Wgt2, dtype=np.float32)      # (3,3)
    Winv = np.linalg.pinv(W)

    C3 = Winv @ V                               # (3, H*W) -> estimated H,E,S
    H, Wd = to_size, to_size
    cH = C3[0].reshape(H, Wd)
    cE = C3[1].reshape(H, Wd)

    # p99 normalization (same as training)
    pH = max(1e-6, _percentile_np(np.clip(cH, 0, None), 99.0))
    pE = max(1e-6, _percentile_np(np.clip(cE, 0, None), 99.0))
    cH_norm = np.clip(cH / pH, 0.0, 1.0)
    cE_norm = np.clip(cE / pE, 0.0, 1.0)

    # to torch (1,1,H,W)
    cH_norm_t = torch.from_numpy(cH_norm)[None, None].float()
    cE_norm_t = torch.from_numpy(cE_norm)[None, None].float()

    return cH, cE, cH_norm_t, cE_norm_t


# ===================== S sampling (Euler) =====================

@torch.no_grad()
def sample_S(model: nn.Module, cH_norm: torch.Tensor, cE_norm: torch.Tensor,
             steps: int = 50, device: Optional[str] = None) -> torch.Tensor:
    """
    cH_norm, cE_norm : (1,1,H,W) in [0,1]
    Returns S_gen_m11 : (1,1,H,W) in [-1,1]
    """
    device = device or next(model.parameters()).device
    Hm11 = cH_norm.to(device) * 2.0 - 1.0
    Em11 = cE_norm.to(device) * 2.0 - 1.0

    B, _, H, W = Hm11.shape
    x = torch.randn(B, 1, H, W, device=device)  # x(0) ~ N(0,1)
    dt = 1.0 / steps

    model.eval()
    for s in range(steps):
        t = torch.full((B,), (s + 0.5) * dt, device=device)
        x_in = torch.cat([x, Hm11, Em11], dim=1)
        v = model(x_in, t)
        x = x + v * dt
    return x  # [-1,1]


# ===================== Pipeline: HE -> HES =====================

def generate_hes_from_he(
    he_img: Image.Image,
    cfg: InferenceConfig,
    model: Optional[nn.Module] = None,
    ref_p99_s: Optional[float] = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Input:
      - he_img : PIL RGB (any size) - resized to 256x256
      - cfg : InferenceConfig (dim, steps, device, checkpoint, etc.)
      - model : optional, if already loaded
      - ref_p99_s : si None -> cfg.REF_P99_S
    Outputs:
      - hes_rgb : PIL RGB 256x256 (H+E+S)
      - s_rgb   : PIL RGB 256x256 (Saffron only, colorized)
    """
    device = cfg.DEVICE
    model = model or load_model(cfg)
    ref_p99_s = float(cfg.REF_P99_S if ref_p99_s is None else ref_p99_s)

    # 1) separate H,E from HE
    cH_raw, cE_raw, cH_norm_t, cE_norm_t = separate_he_from_rgb(he_img, to_size=cfg.IMG_SIZE)

    # 2) generate S_norm with Euler
    S_m11 = sample_S(model, cH_norm_t, cE_norm_t, steps=cfg.SAMPLE_STEPS, device=device)
    S_norm = ((S_m11 + 1.0) / 2.0).clamp(0.0, 1.0)  # (1,1,H,W)

    # 3) denormalize S with reference p99_S
    cS_hat = (S_norm[0,0].cpu().numpy() * ref_p99_s).astype(np.float32)

    # 4) recompose HES (inverse Beer-Lambert)
    W = np.array(Wgt2, dtype=np.float32)
    hes_rgb = _compose_rgb_from_conc(cH_raw.astype(np.float32),
                                     cE_raw.astype(np.float32),
                                     cS_hat, W)
    # 5) S-only image (useful for debug/visualization)
    OD_S = cS_hat[..., None] * W[:, 2][None, None, :]   # (H,W,3)
    s_rgb = np.exp(-OD_S) * 255.0
    s_rgb = np.clip(s_rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(hes_rgb), Image.fromarray(s_rgb)


# ===================== Script usage example =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate HES from HE (256x256).")
    parser.add_argument("--he", required=True, help="Input HE image (png/jpg).")
    parser.add_argument("--out", default="hes_generated.png", help="HES output path.")
    parser.add_argument("--outs", default="s_generated.png", help="S output path (visualization).")
    parser.add_argument("--ref-p99-s", type=float, default=None, help="Reference p99_S (default: cfg.REF_P99_S).")
    args = parser.parse_args()

    cfg = InferenceConfig()
    model = load_model(cfg)

    he_pil = Image.open(args.he).convert("RGB")
    hes_pil, s_pil = generate_hes_from_he(he_pil, cfg, model=model, ref_p99_s=args.ref_p99_s)

    hes_pil.save(args.out)
    s_pil.save(args.outs)
    print(f"[OK] HES saved -> {args.out}")
    print(f"[OK] S (visualization) -> {args.outs}")
