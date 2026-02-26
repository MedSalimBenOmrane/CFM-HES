# -*- coding: utf-8 -*-
"""
inference_fm_cond_s_he_frames.py
Capture intermediate frames for CFM (HE -> S -> HES) without modifying
the existing code. Three strategies are attempted, in this order:
  1) generate_hes_from_he(..., return_frames=True, save_every=...)
  2) generate_hes_from_he(..., capture_callback=cb, save_every=...)
  3) Slow fallback: rerun generate_hes_from_he with SAMPLE_STEPS = 1..T
     (saving every N iterations via save_every).

Returns:
  hes_final_pil, s_vis_final_pil, frames
where frames is a list of dicts:
{'iter': k, 'hes_rgb': PIL.Image, 'cS': np.ndarray[H,W]}.
"""

import copy
import numpy as np
from PIL import Image

# imports from the existing inference module
from inference_fm_cond_s_he import (
    InferenceConfig, load_model as load_cfm_model,
    generate_hes_from_he, separate_he_from_rgb  # separate_he_from_rgb is unused here but available
)

# stain utils
from separation_model.stain_matrix import Wgt2


# ---------- S / concentration utilities ----------
def rgb_pil_to_conc3_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert an RGB (PIL) image to cH, cE, cS concentrations (numpy [3,H,W], >=0).
    """
    im = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    im = np.clip(im, 1e-6, 1.0)
    OD = -np.log(im)  # [H,W,3]
    V = OD.reshape(-1,3).T.astype(np.float32)  # (3,N)
    W = np.array(Wgt2, dtype=np.float32)
    Winv = np.linalg.pinv(W).astype(np.float32)
    C3 = Winv @ V                                # (3,N)
    H, Wd = img.size[1], img.size[0]
    C3 = C3.reshape(3, H, Wd)
    C3 = np.clip(C3, 0.0, None)
    return C3

def s_to_rgb_u8(cS: np.ndarray) -> np.ndarray:
    """
    RGB visualization of S only, using the S column of W.
    """
    W = np.array(Wgt2, dtype=np.float32)
    OD_S = cS[..., None] * W[:,2][None,None,:]  # [H,W,3]
    I = np.exp(-OD_S)
    return (np.clip(I, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------- config helpers ----------
def clone_cfg(cfg: InferenceConfig, **updates) -> InferenceConfig:
    """
    Recreate an InferenceConfig with the same fields, allowing overrides
    (especially SAMPLE_STEPS).
    """
    # Assume InferenceConfig accepts exactly these kwargs:
    kwargs = dict(
        CHECKPOINT_DIR = getattr(cfg, 'CHECKPOINT_DIR', None),
        DIM            = getattr(cfg, 'DIM', None),
        DIM_MULTS      = tuple(getattr(cfg, 'DIM_MULTS', (1,1,2,2,4))),
        DROPOUT        = getattr(cfg, 'DROPOUT', 0.0),
        SAMPLE_STEPS   = getattr(cfg, 'SAMPLE_STEPS', 50),
        IMG_SIZE       = getattr(cfg, 'IMG_SIZE', 256),
        DEVICE         = str(getattr(cfg, 'DEVICE', 'cuda')),
        REF_P99_S      = getattr(cfg, 'REF_P99_S', 1.0),
    )
    kwargs.update(updates)
    return InferenceConfig(**kwargs)


# ---------- strategy 1: via return_frames ----------
def _try_return_frames(he_pil, cfg, model, ref_p99_s, save_every):
    try:
        out = generate_hes_from_he(
            he_pil, cfg, model=model, ref_p99_s=ref_p99_s,
            return_frames=True, save_every=save_every
        )
        # expected output: (hes_final_pil, s_vis_final_pil, frames)
        if isinstance(out, tuple) and len(out) >= 3:
            hes_final, s_vis_final, frames = out[0], out[1], out[2]
            # frames are already in [{'iter':..., 'hes_rgb':..., 'cS':...}, ...] format
            return hes_final, s_vis_final, frames
    except TypeError:
        pass
    return None, None, None


# ---------- strategy 2: via capture_callback ----------
def _try_callback(he_pil, cfg, model, ref_p99_s, save_every):
    frames = []
    def cb(iter_idx, payload):
        # expected payload: 'hes_rgb' PIL/np and optionally 'cS' np[H,W]
        entry = {'iter': int(iter_idx)}
        hes_img = payload.get('hes_rgb', None)
        if hes_img is not None:
            if isinstance(hes_img, Image.Image):
                entry['hes_rgb'] = hes_img
            else:
                entry['hes_rgb'] = Image.fromarray(hes_img)
        cS = payload.get('cS', None)
        if cS is not None:
            entry['cS'] = cS.astype(np.float32)
        frames.append(entry)

    try:
        hes_final, s_vis_final = generate_hes_from_he(
            he_pil, cfg, model=model, ref_p99_s=ref_p99_s,
            capture_callback=cb, save_every=save_every
        )
        # if cS is missing in payload, it can be reconstructed on demand
        return hes_final, s_vis_final, frames
    except TypeError:
        pass
    return None, None, None


# ---------- strategy 3: slow fallback (replay 1..T steps) ----------
def _slow_replay(he_pil, cfg, model, ref_p99_s, save_every, max_steps=None):
    """
    Rerun inference with SAMPLE_STEPS = 1..T (T=cfg.SAMPLE_STEPS if max_steps is None).
    Capture each run's final frame as an approximation of state at iteration k.
    This is slow but 100% compatible without modifying original inference.
    """
    T = int(max_steps or getattr(cfg, 'SAMPLE_STEPS', 50))
    frames = []
    hes_final, s_vis_final = None, None
    for k in range(1, T+1):
        if (k % save_every) != 0 and k != 1 and k != T:
            continue
        cfg_k = clone_cfg(cfg, SAMPLE_STEPS=k)
        hes_k, s_vis_k = generate_hes_from_he(he_pil, cfg_k, model=model, ref_p99_s=ref_p99_s)
        # reconstruct cS from HES_k
        C3_k = rgb_pil_to_conc3_numpy(hes_k)
        frames.append({'iter': k, 'hes_rgb': hes_k, 'cS': C3_k[2]})
        hes_final, s_vis_final = hes_k, s_vis_k  # last run is the final output
    return hes_final, s_vis_final, frames


# ---------- public API ----------
def generate_hes_from_he_frames(
    he_pil: Image.Image,
    cfg: InferenceConfig,
    model=None,
    ref_p99_s=None,
    save_every: int = 1,
    max_steps: int = None
):
    """
    Produce final HES + final S visualization + intermediate FRAMES.
    Try return_frames, then capture_callback, then replay fallback.
    """
    # Load model if not provided
    if model is None:
        model = load_cfm_model(cfg)

    # 1) try return_frames
    hes_final, s_vis_final, frames = _try_return_frames(he_pil, cfg, model, ref_p99_s, save_every)
    if frames is not None and len(frames) > 0:
        return hes_final, s_vis_final, frames

    # 2) try callback
    hes_final, s_vis_final, frames = _try_callback(he_pil, cfg, model, ref_p99_s, save_every)
    if frames is not None and len(frames) > 0:
        return hes_final, s_vis_final, frames

    # 3) slow fallback
    return _slow_replay(he_pil, cfg, model, ref_p99_s, save_every, max_steps=max_steps)
