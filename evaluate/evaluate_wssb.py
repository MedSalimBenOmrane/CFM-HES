#!/usr/bin/env python3
"""
export PYTHONPATH="put path of your data here:$PYTHONPATH"

python CFM/evaluate/evaluate_wssb.py \
  --wssb_root "put path of your data here" \
  --hes_ref_root "put path of your data here" \
  --out_dir "wssb_eval" \
  --ckpt_dir "put path of your data here" \
  --device cuda

Evaluate CFM on WSSB HE images -> generate HES, then compute FID/KID
against two distributions:
  1) HE distribution (WSSB input, per organ)
  2) HES distribution (external HES dataset, global reference)

Outputs:
  - Generated HES images in: out_dir/<ORGAN>/
  - metrics.json with per-organ and mean FID/KID
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from separation_model.global_utils import vectorize
from separation_model.stain_matrix import safran as SAFRAN_RGB

from inference_fm_cond_s_he import InferenceConfig, load_model, sample_S


# ------------------------- Metrics -------------------------
_HAS_TORCHMETRICS = False
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    _HAS_TORCHMETRICS = True
except Exception as e:  # pragma: no cover
    print("[WARN] torchmetrics not available, FID/KID will be NaN:", repr(e))


class ImageFolderDataset(Dataset):
    def __init__(self, paths: List[Path], image_size: int = None):
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        # (H,W,3) -> (3,H,W)
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t


def compute_fid_kid(
    real_paths: List[Path],
    fake_paths: List[Path],
    device: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    kid_subset_size: int,
    max_images: int = 0,
    pbar_desc: str = "",
) -> Tuple[float, float]:
    if not _HAS_TORCHMETRICS:
        return float("nan"), float("nan")

    if max_images and max_images > 0:
        real_paths = real_paths[:max_images]
        fake_paths = fake_paths[:max_images]

    if len(real_paths) == 0 or len(fake_paths) == 0:
        return float("nan"), float("nan")

    real_ds = ImageFolderDataset(real_paths, image_size=image_size)
    fake_ds = ImageFolderDataset(fake_paths, image_size=image_size)

    real_loader = DataLoader(
        real_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )
    fake_loader = DataLoader(
        fake_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = None
    n = min(len(real_paths), len(fake_paths))
    if n >= 2:
        subset_size = min(kid_subset_size, n - 1)
        if subset_size >= 1:
            kid = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)

    desc_real = f"{pbar_desc} real".strip()
    desc_fake = f"{pbar_desc} fake".strip()
    for batch in tqdm(real_loader, desc=desc_real, leave=False):
        fid.update(batch.to(device), real=True)
        if kid is not None:
            kid.update(batch.to(device), real=True)

    for batch in tqdm(fake_loader, desc=desc_fake, leave=False):
        fid.update(batch.to(device), real=False)
        if kid is not None:
            kid.update(batch.to(device), real=False)

    fid_score = float(fid.compute().detach().cpu().item())
    if kid is None:
        kid_score = float("nan")
    else:
        kid_mean, _ = kid.compute()
        kid_score = float(kid_mean.detach().cpu().item())
    return fid_score, kid_score


# ------------------------- Stain utils -------------------------
def load_he_matrix(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append([float(x) for x in parts])
    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (3, 2):
        raise ValueError(f"Expected 3x2 matrix in {path}, got {mat.shape}")
    return mat


def build_w_hes(W_he: np.ndarray) -> np.ndarray:
    safran = np.array(SAFRAN_RGB, dtype=np.float32)
    safran_od = -np.log(safran / 255.0)
    W_hes = np.concatenate([W_he, safran_od[:, None]], axis=1)
    return W_hes.astype(np.float32)


def separate_he_from_rgb_custom(img: Image.Image, W_he: np.ndarray, to_size: int):
    he = img.convert("RGB").resize((to_size, to_size), Image.BICUBIC)
    im_u8 = np.array(he)
    V = vectorize(im_u8).astype(np.float32)  # (3, N)
    Winv = np.linalg.pinv(W_he).astype(np.float32)  # (2,3)
    C2 = Winv @ V  # (2, N)
    H, Wd = to_size, to_size
    cH = C2[0].reshape(H, Wd)
    cE = C2[1].reshape(H, Wd)
    cH = np.clip(cH, 0.0, None)
    cE = np.clip(cE, 0.0, None)

    pH = max(1e-6, float(np.percentile(cH.reshape(-1), 99.0)))
    pE = max(1e-6, float(np.percentile(cE.reshape(-1), 99.0)))
    cH_norm = np.clip(cH / pH, 0.0, 1.0)
    cE_norm = np.clip(cE / pE, 0.0, 1.0)

    cH_norm_t = torch.from_numpy(cH_norm)[None, None].float()
    cE_norm_t = torch.from_numpy(cE_norm)[None, None].float()
    return cH, cE, cH_norm_t, cE_norm_t


def compose_hes(cH: np.ndarray, cE: np.ndarray, cS: np.ndarray, W_hes: np.ndarray) -> np.ndarray:
    OD = (
        cH[..., None] * W_hes[:, 0][None, None, :] +
        cE[..., None] * W_hes[:, 1][None, None, :] +
        cS[..., None] * W_hes[:, 2][None, None, :]
    )
    I = np.exp(-OD) * 255.0
    return np.clip(I, 0, 255).astype(np.uint8)


@torch.no_grad()
def generate_hes_from_he_custom(
    he_img: Image.Image,
    W_he: np.ndarray,
    W_hes: np.ndarray,
    cfg: InferenceConfig,
    model: torch.nn.Module,
    ref_p99_s: float,
):
    cH_raw, cE_raw, cH_norm_t, cE_norm_t = separate_he_from_rgb_custom(
        he_img, W_he=W_he, to_size=cfg.IMG_SIZE
    )
    S_m11 = sample_S(model, cH_norm_t, cE_norm_t, steps=cfg.SAMPLE_STEPS, device=cfg.DEVICE)
    S_norm = ((S_m11 + 1.0) / 2.0).clamp(0.0, 1.0)
    cS_hat = (S_norm[0, 0].cpu().numpy() * ref_p99_s).astype(np.float32)
    hes_rgb = compose_hes(cH_raw.astype(np.float32), cE_raw.astype(np.float32), cS_hat, W_hes)
    return Image.fromarray(hes_rgb)


# ------------------------- Data helpers -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_image_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def find_matrix_file(wssb_root: Path, organ_name: str) -> Path:
    target = organ_name.lower()
    for p in wssb_root.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "mean_stain_matrix" in name and target in name:
            return p
    raise FileNotFoundError(f"No mean_stain_matrix file for organ '{organ_name}' in {wssb_root}")


def organ_dirs(wssb_root: Path, only_organs: List[str] = None) -> List[Path]:
    orgs = []
    for p in wssb_root.iterdir():
        if p.is_dir():
            if only_organs and p.name.lower() not in [o.lower() for o in only_organs]:
                continue
            if list_image_files(p):
                orgs.append(p)
    return sorted(orgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wssb_root",
        default="put path of your data here",
    )
    ap.add_argument(
        "--hes_ref_root",
        default="put path of your data here",
    )
    ap.add_argument("--out_dir", default="wssb_eval")

    ap.add_argument("--ckpt_dir", default=InferenceConfig.CHECKPOINT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dim", type=int, default=InferenceConfig.DIM)
    ap.add_argument("--dim_mults", type=int, nargs="+", default=list(InferenceConfig.DIM_MULTS))
    ap.add_argument("--dropout", type=float, default=InferenceConfig.DROPOUT)
    ap.add_argument("--sample_steps", type=int, default=InferenceConfig.SAMPLE_STEPS)
    ap.add_argument("--img_size", type=int, default=InferenceConfig.IMG_SIZE)
    ap.add_argument("--ref_p99_s", type=float, default=InferenceConfig.REF_P99_S)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--kid_subset_size", type=int, default=50)
    ap.add_argument("--max_images", type=int, default=0)

    ap.add_argument("--skip_generation", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only_organs", nargs="*", default=None)

    args = ap.parse_args()

    wssb_root = Path(args.wssb_root)
    hes_ref_root = Path(args.hes_ref_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config for CFM model
    cfg = InferenceConfig(
        CHECKPOINT_DIR=args.ckpt_dir,
        DIM=args.dim,
        DIM_MULTS=tuple(args.dim_mults),
        DROPOUT=args.dropout,
        SAMPLE_STEPS=args.sample_steps,
        IMG_SIZE=args.img_size,
        DEVICE=args.device,
        REF_P99_S=args.ref_p99_s,
    )

    model = load_model(cfg)
    model.eval()

    organs = organ_dirs(wssb_root, only_organs=args.only_organs)
    if not organs:
        raise FileNotFoundError(f"No organ folders found in {wssb_root}")

    # Preload HES reference list
    hes_ref_paths = list_image_files(hes_ref_root)

    results = {
        "distribution_HE": {},
        "distribution_HES": {},
        "counts": {
            "HES_ref": len(hes_ref_paths),
        },
        "config": {
            "wssb_root": str(wssb_root),
            "hes_ref_root": str(hes_ref_root),
            "out_dir": str(out_dir),
            "ckpt_dir": cfg.CHECKPOINT_DIR,
            "device": cfg.DEVICE,
            "sample_steps": cfg.SAMPLE_STEPS,
            "img_size": cfg.IMG_SIZE,
            "ref_p99_s": cfg.REF_P99_S,
        },
    }

    # Generation per organ
    for organ_dir in organs:
        organ_key = organ_dir.name.upper()
        print(f"[Organ] {organ_key}")
        W_he_path = find_matrix_file(wssb_root, organ_dir.name)
        W_he = load_he_matrix(W_he_path)
        W_hes = build_w_hes(W_he)

        he_paths = list_image_files(organ_dir)
        if args.max_images and args.max_images > 0:
            he_paths = he_paths[: args.max_images]

        out_org_dir = out_dir / organ_key
        out_org_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_generation:
            for he_path in tqdm(he_paths, desc=f"Generate {organ_key}"):
                out_path = out_org_dir / he_path.name
                if out_path.exists() and not args.overwrite:
                    continue
                he_img = Image.open(he_path).convert("RGB")
                hes_img = generate_hes_from_he_custom(
                    he_img,
                    W_he=W_he,
                    W_hes=W_hes,
                    cfg=cfg,
                    model=model,
                    ref_p99_s=args.ref_p99_s,
                )
                hes_img.save(out_path)

        gen_paths = list_image_files(out_org_dir)
        results["counts"][organ_key] = {
            "he": len(he_paths),
            "gen": len(gen_paths),
        }

        # Metrics vs HE distribution (per organ)
        fid_he, kid_he = compute_fid_kid(
            real_paths=he_paths,
            fake_paths=gen_paths,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.img_size,
            kid_subset_size=args.kid_subset_size,
            max_images=args.max_images,
        )
        results["distribution_HE"][organ_key] = {"FID": fid_he, "KID": kid_he}

        # Metrics vs HES distribution (global ref)
        fid_hes, kid_hes = compute_fid_kid(
            real_paths=hes_ref_paths,
            fake_paths=gen_paths,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.img_size,
            kid_subset_size=args.kid_subset_size,
            max_images=args.max_images,
        )
        results["distribution_HES"][organ_key] = {"FID": fid_hes, "KID": kid_hes}

    # Compute mean across organs
    def mean_metric(dist: Dict[str, Dict[str, float]], key: str) -> float:
        vals = [v[key] for v in dist.values() if isinstance(v, dict) and not np.isnan(v[key])]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    results["distribution_HE"]["FID_mean"] = mean_metric(results["distribution_HE"], "FID")
    results["distribution_HE"]["KID_mean"] = mean_metric(results["distribution_HE"], "KID")
    results["distribution_HES"]["FID_mean"] = mean_metric(results["distribution_HES"], "FID")
    results["distribution_HES"]["KID_mean"] = mean_metric(results["distribution_HES"], "KID")

    out_json = out_dir / "metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
