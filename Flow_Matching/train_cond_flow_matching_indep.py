""""
 PYTHONPATH="put path of your data here:$PYTHONPATH" python main.py --opts root "put path of your data here" train True eval False dataset hes method pnp_flow model fm_cond_s_he data_root "put path of your data here" batch_size_train 16 batch_size_test 16 num_workers 1 num_epoch 100 lr 0.00005 dim 128 dim_mults "(1,1,2,2,4)" dropout 0.0 sample_steps 50 dim_image 256
"""
# train_fm_cond_s_he_aymen.py
import os, re, glob, math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.cuda.amp as amp
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# --- stain separation imports ---
from separation_model.stain_matrix import Wgt2          # (3,3) OD matrix columns: H,E,S
from separation_model.global_utils import vectorize     # np RGB uint8 -> OD (3, H*W)

# --- local U-Net (in=3: [S_t,H,E], out=1: v_S) ---
from unet.unet_core import Unet as BaseUNet


# =========================================================
#            UTILS: conversion / HES <-> H,E,S
# =========================================================

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert float [0,1] image to uint8 (or keep uint8)."""
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0.0, 1.0) * 255.0
    return img.astype(np.uint8)

def _torch_to_rgb01(x: torch.Tensor) -> torch.Tensor:
    """Ensure [0,1] range for a tensor (B,3,H,W) that may be in [-1,1] or [0,1]."""
    if x.min() < 0:
        return (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def _percentile_np(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x.reshape(-1), q))

def _compose_rgb_from_conc(cH: np.ndarray, cE: np.ndarray, cS: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Beer–Lambert inverse: I = 255 * exp( - (cH*W[:,0] + cE*W[:,1] + cS*W[:,2]) )
    Inputs cH,cE,cS: (H,W), W: (3,3)
    Returns RGB uint8 (H,W,3)
    """
    H, Wd = cH.shape
    OD = (cH[..., None] * W[:, 0][None, None, :]
        + cE[..., None] * W[:, 1][None, None, :]
        + cS[..., None] * W[:, 2][None, None, :])  # (H,W,3)
    I = np.exp(-OD) * 255.0
    return np.clip(I, 0, 255).astype(np.uint8)

def _compose_rgb_stain_only(cK: np.ndarray, col: int, W: np.ndarray) -> np.ndarray:
    """Color image of a single stain k∈{0:H,1:E,2:S} (H,W,3) uint8."""
    H, Wd = cK.shape
    OD = cK[..., None] * W[:, col][None, None, :]  # (H,W,3)
    I = np.exp(-OD) * 255.0
    return np.clip(I, 0, 255).astype(np.uint8)

def _separate_batch_aymen(
    hes_batch: torch.Tensor,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Input : hes_batch (B,3,H,W) in [-1,1] or [0,1]
    Output:
      cH_raw, cE_raw, cS_raw : (B,1,H,W)  (OD concentrations, non-normalized)
      cH_norm, cE_norm, cS_norm : (B,1,H,W)  (per-image p99 normalization)
      p99_H, p99_E, p99_S : (B,1,1,1)  (for denormalization when needed)
      HE_rgb01 : (B,3,H,W) in [0,1] (HE recomposition, S=0) — input visualization
    """
    W = np.array(Wgt2, dtype=np.float32)        # (3,3)
    Winv = np.linalg.pinv(W).astype(np.float32) # (3,3)

    hes01 = _torch_to_rgb01(hes_batch).detach().cpu().numpy()  # (B,3,H,W) float
    B, _, H, Wd = hes01.shape

    cH_raw_list, cE_raw_list, cS_raw_list = [], [], []
    HE_rgb_list = []
    p99H, p99E, p99S = [], [], []

    for b in range(B):
        # -> (H,W,3) float -> uint8
        im = np.transpose(hes01[b], (1,2,0))  # (H,W,3)
        im_u8 = _to_uint8_rgb(im)

        # Vectorized OD (3, H*W)
        V = vectorize(im_u8).astype(np.float32)

        # Concentrations (3,H*W)
        C3 = (Winv @ V)  # H,E,S
        cH = C3[0].reshape(H, Wd)
        cE = C3[1].reshape(H, Wd)
        cS = C3[2].reshape(H, Wd)

        # Per-image p99 (reduces large outliers) and normalization
        pH = max(1e-6, _percentile_np(np.clip(cH, 0, None), 99.0))
        pE = max(1e-6, _percentile_np(np.clip(cE, 0, None), 99.0))
        pS = max(1e-6, _percentile_np(np.clip(cS, 0, None), 99.0))

        cH_norm = np.clip(cH / pH, 0.0, 1.0)
        cE_norm = np.clip(cE / pE, 0.0, 1.0)
        cS_norm = np.clip(cS / pS, 0.0, 1.0)

        # HE (S=0) for input display
        HE_rgb = _compose_rgb_from_conc(cH, cE, np.zeros_like(cS), W)  # uint8

        cH_raw_list.append(torch.from_numpy(cH)[None, None])
        cE_raw_list.append(torch.from_numpy(cE)[None, None])
        cS_raw_list.append(torch.from_numpy(cS)[None, None])

        HE_rgb_list.append(torch.from_numpy(HE_rgb).permute(2,0,1)[None])  # (1,3,H,W)

        p99H.append(torch.tensor([[ [ [pH] ] ]], dtype=torch.float32))
        p99E.append(torch.tensor([[ [ [pE] ] ]], dtype=torch.float32))
        p99S.append(torch.tensor([[ [ [pS] ] ]], dtype=torch.float32))

    cH_raw = torch.cat(cH_raw_list, dim=0).to(device)
    cE_raw = torch.cat(cE_raw_list, dim=0).to(device)
    cS_raw = torch.cat(cS_raw_list, dim=0).to(device)

    cH_norm = torch.clamp(cH_raw / torch.tensor(1.0, device=device), 0, 1)  # placeholder (recomputed below)
    cE_norm = torch.clamp(cE_raw / torch.tensor(1.0, device=device), 0, 1)
    cS_norm = torch.clamp(cS_raw / torch.tensor(1.0, device=device), 0, 1)

    p99_H = torch.cat(p99H, dim=0).to(device)
    p99_E = torch.cat(p99E, dim=0).to(device)
    p99_S = torch.cat(p99S, dim=0).to(device)

    # Actual p99 normalization
    cH_norm = torch.clamp(cH_raw / p99_H, 0.0, 1.0)
    cE_norm = torch.clamp(cE_raw / p99_E, 0.0, 1.0)
    cS_norm = torch.clamp(cS_raw / p99_S, 0.0, 1.0)

    HE_rgb01 = torch.cat(HE_rgb_list, dim=0).float().to(device) / 255.0

    return dict(
        cH_raw=cH_raw, cE_raw=cE_raw, cS_raw=cS_raw,
        cH_norm=cH_norm, cE_norm=cE_norm, cS_norm=cS_norm,
        p99_H=p99_H, p99_E=p99_E, p99_S=p99_S,
        HE_rgb01=HE_rgb01
    )


# =========================================================
#                      TRAINER
# =========================================================

class FMCondSfromHE_Aymen:
    """
    Train a U-Net that predicts S velocity (1 channel) conditioned on H & E.
    Network input = concat([S_t, H_norm, E_norm]) in [-1,1].
    FM target = (S_gt - x0) in [-1,1].
    """

    def __init__(self, model: nn.Module, device: torch.device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.scaler = amp.GradScaler()

        # paths
        checkpointpath = getattr(args, "ckpt_dir", "put path of your data here")
        self.save_path = os.path.join(args.root, f"results/{args.dataset}/fm_cond_s_he/")
        self.model_path = checkpointpath
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    # ---------- step FM

    def _fm_step(self, he_batch: torch.Tensor, hes_batch: torch.Tensor):
        """
        he_batch : (B,3,H,W) image HE (condition)
        hes_batch: (B,3,H,W) HES image (S target)
        Returns loss + useful debug/visualization elements
        """
        sep_he = _separate_batch_aymen(he_batch, self.device)
        sep_hes = _separate_batch_aymen(hes_batch, self.device)
        # Normalize [0,1] -> [-1,1]
        to_m11 = lambda x: x * 2.0 - 1.0
        Hm11 = to_m11(sep_he["cH_norm"])
        Em11 = to_m11(sep_he["cE_norm"])
        Sgt = to_m11(sep_hes["cS_norm"])  # S target from HES

        B = Sgt.size(0)
        # Initial noise
        x0 = torch.randn_like(Sgt)
        # t ~ U(0,1)
        t  = torch.rand(B, 1, 1, 1, device=self.device)
        St = t * Sgt + (1.0 - t) * x0

        x_in = torch.cat([St, Hm11, Em11], dim=1)  # (B,3,H,W)
        t_flat = t.view(B)

        with amp.autocast():
            v_pred = self.model(x_in, t_flat)   # (B,1,H,W)
            target = (Sgt - x0)
            loss = F.mse_loss(v_pred, target)

        return loss, sep_he, sep_hes, v_pred, St, x0, t

    # ---------- sampling Euler

    @torch.no_grad()
    def _sample_S(self, cH_norm: torch.Tensor, cE_norm: torch.Tensor, steps: int = 50):
        """
        Euler integration of dx/dt = vθ(x,t,HE), t∈[0,1].
        cH_norm, cE_norm: (B,1,H,W) in [0,1] (converted to [-1,1] here)
        """
        self.model.eval()
        Hm11 = cH_norm * 2.0 - 1.0
        Em11 = cE_norm * 2.0 - 1.0
        B, _, H, W = Hm11.shape

        x = torch.randn(B, 1, H, W, device=self.device)  # x(0)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((B,), (s + 0.5) * dt, device=self.device)
            x_in = torch.cat([x, Hm11, Em11], dim=1)
            v = self.model(x_in, t)
            x = x + v * dt
        return x  # [-1,1]

    # ---------- visualization grid

    @torch.no_grad()
    def _save_grid_epoch(self, epoch: int, test_loader: DataLoader, num_examples: int = 8, steps: int = 50):
        """
        Cols: [HE_in (RGB) | S_gen (color) | S_gt (color) | HES_gen (RGB) | HES_gt (RGB)]
        """
        self.model.eval()

        # Retrieve N images
        he_xs, hes_xs = [], []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                he, hes = batch[0], batch[1]
            else:
                # fallback old format: no separate HE
                hes = batch[0] if isinstance(batch, (list, tuple)) else batch
                he = hes
            he_xs.append(he.to(self.device))
            hes_xs.append(hes.to(self.device))
            if sum(b.shape[0] for b in hes_xs) >= num_examples:
                break
        he = torch.cat(he_xs, dim=0)[:num_examples]
        hes = torch.cat(hes_xs, dim=0)[:num_examples]  # (N,3,H,W)

        # H,E from HE; S from HES
        sep_he = _separate_batch_aymen(he, self.device)
        sep_hes = _separate_batch_aymen(hes, self.device)
        cH_raw, cE_raw = sep_he["cH_raw"], sep_he["cE_raw"]
        cS_raw = sep_hes["cS_raw"]
        cH_norm, cE_norm = sep_he["cH_norm"], sep_he["cE_norm"]
        pS = sep_hes["p99_S"]  # (N,1,1,1)
        HE_rgb01 = _torch_to_rgb01(he)
        HES_gt01 = _torch_to_rgb01(hes)

        # Generate S_norm with Euler
        S_gen_m11 = self._sample_S(cH_norm, cE_norm, steps=steps)        # [-1,1]
        S_gen_norm = torch.clamp((S_gen_m11 + 1.0) / 2.0, 0.0, 1.0)      # [0,1]

        # Recompose S_gen color + HES_gen
        W = np.array(Wgt2, dtype=np.float32)
        rows = []
        for i in range(num_examples):
            # (1,H,W) -> (H,W)
            cH_r = cH_raw[i,0].detach().cpu().numpy()
            cE_r = cE_raw[i,0].detach().cpu().numpy()
            cS_r = cS_raw[i,0].detach().cpu().numpy()

            # Denormalize S_gen: cS_hat = S_gen_norm * p99_S(i)
            cS_hat = (S_gen_norm[i,0].detach().cpu().numpy() * float(pS[i].item()))

            # Color images (uint8)
            S_gen_rgb = _compose_rgb_stain_only(cS_hat, 2, W)                 # col=2 for Safran
            S_gt_rgb  = _compose_rgb_stain_only(cS_r,   2, W)
            HES_gen   = _compose_rgb_from_conc(cH_r, cE_r, cS_hat, W)

            # tensors [0,1] (3,H,W)
            HE_vis  = HE_rgb01[i]
            Sg_vis  = torch.from_numpy(S_gen_rgb).permute(2,0,1).float()/255.0
            Sgt_vis = torch.from_numpy(S_gt_rgb ).permute(2,0,1).float()/255.0
            HESg_vis= torch.from_numpy(HES_gen  ).permute(2,0,1).float()/255.0
            HESgt_vis = HES_gt01[i]

            rows += [HE_vis.cpu(), Sg_vis, Sgt_vis, HESg_vis, HESgt_vis.cpu()]

        grid = vutils.make_grid(rows, nrow=5, normalize=False)
        npgrid = grid.permute(1,2,0).cpu().numpy()
        npgrid = np.clip(npgrid, 0.0, 1.0)

        out_dir = os.path.join(self.save_path, 'grid_conditional')
        os.makedirs(out_dir, exist_ok=True)
        plt.imsave(os.path.join(out_dir, f'conditional_epoch_{epoch}.png'), npgrid)
        self.model.train()

    # ---------- ckpt utils

    def _latest_ckpt(self):
        paths = glob.glob(os.path.join(self.model_path, 'model_*.pt'))
        best = []
        for p in paths:
            m = re.search(r'model_(\d+)\.pt$', p)
            if m:
                best.append((int(m.group(1)), p))
        if not best:
            return None
        return max(best, key=lambda x: x[0])

    # ---------- loop

    def train(self, data_loaders: Dict[str, DataLoader]):
        train_loader: DataLoader = data_loaders['train']
        test_loader : DataLoader = data_loaders['test']

        # log model
        with open(os.path.join(self.save_path, 'model_info.txt'), 'w') as f:
            f.write("PARAMETERS\n")
            f.write(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n")
            f.write(f"Number of epochs: {self.args.num_epoch}\n")
            f.write(f"Batch size: {self.args.batch_size_train}\n")
            f.write(f"Learning rate: {self.args.lr}\n")

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        loss_log = os.path.join(self.save_path, 'loss_training.txt')

        # resume
        start_ep = 0
        last = self._latest_ckpt()
        if last:
            ep, path = last
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                print(f"→ Resume checkpoint from epoch {ep}: {path}")
                ck = torch.load(path, map_location=self.device)
                self.model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['optimizer'])
                self.scaler.load_state_dict(ck['scaler'])
                start_ep = ck['epoch']
            else:
                print(f"⚠️ Checkpoint {path} is missing/empty. Starting fresh.")

        # training loop
        for ep in range(start_ep, self.args.num_epoch):
            print(f"\n→ Epoch {ep+1}/{self.args.num_epoch}")
            self.model.train()

            bar = tqdm(train_loader, desc="   Batches", unit="batch", leave=False)
            for it, batch in enumerate(bar):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    he, hes = batch[0], batch[1]
                else:
                    hes = batch[0] if isinstance(batch, (list, tuple)) else batch
                    he = hes
                he = he.to(self.device, non_blocking=True)
                hes = hes.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with amp.autocast():
                    loss, *_ = self._fm_step(he, hes)

                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()

                with open(loss_log, 'a') as f:
                    f.write(f'Epoch: {ep}, iter: {it}, Loss: {loss.item():.6f}\n')
                bar.set_postfix(loss=f"{loss.item():.4f}")

            # ckpt + grid (8 ex.)
            ck = {
                'epoch': ep+1,
                'model': self.model.state_dict(),
                'optimizer': opt.state_dict(),
                'scaler': self.scaler.state_dict(),
            }
            torch.save(ck, os.path.join(self.model_path, f'model_{ep+1}.pt'))
            self._save_grid_epoch(ep+1, test_loader, num_examples=8, steps=getattr(self.args, "sample_steps", 50))

        # final save
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model_final.pt'))


# =========================================================
#           MODEL BUILDER (custom variant)
# =========================================================
def build_unet_fm_cond_s_he(args, device):
    """
    U-Net:
      in_channels=3  -> [S_t, H_norm, E_norm]
      out_channels=1 -> S velocity
    """
    model = BaseUNet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        in_channels=3,
        out_channels=1,
        dropout=args.dropout,
    ).to(device)
    return model
