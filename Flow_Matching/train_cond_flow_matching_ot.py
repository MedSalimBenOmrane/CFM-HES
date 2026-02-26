"""
PYTHONPATH="put path of your data here:$PYTHONPATH" python main.py --opts \
  root "put path of your data here" \
  train True eval False dataset hes method pnp_flow model fm_cond_ot_s_he \
  data_root "put path of your data here" \
  batch_size_train 16 batch_size_test 16 num_workers 1 num_epoch 100 \
  lr 0.00005 dim 128 dim_mults "(1,1,2,2,4)" dropout 0.0 dim_image 256 \
  sample_steps 50 loss_l1_weight 1.0 loss_l2_weight 1.0 ema_decay 0.999 \
  checkpoint
"""

import copy
import glob
import os
import re
from typing import Dict

import numpy as np
import ot
import torch
import torch.cuda.amp as amp
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from separation_model.stain_matrix import Wgt2


DEFAULT_CKPT_DIR = "put path of your data here"


def _build_w(device: torch.device):
    w = torch.tensor(np.array(Wgt2, dtype=np.float32), dtype=torch.float32, device=device)
    w_inv = torch.linalg.pinv(w)
    return w, w_inv


def _to_rgb01(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    if x.max() > 1.5:
        return torch.clamp(x / 255.0, 0.0, 1.0)
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def _rgb_to_hes_batch(x_rgb: torch.Tensor, w_inv: torch.Tensor) -> torch.Tensor:
    x = _to_rgb01(x_rgb)
    x = torch.clamp(x, 1e-6, 1.0)
    od = -torch.log(x)
    b, _, h, w = od.shape
    od_flat = od.view(b, 3, -1)
    c = torch.einsum("ij,bjn->bin", w_inv, od_flat)
    c = torch.clamp(c, min=0.0)
    return c.view(b, 3, h, w)


@torch.no_grad()
def _hes_to_rgb_batch(c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    b, _, h, w_ = c.shape
    c_flat = c.view(b, 3, -1)
    od = torch.einsum("ij,bjn->bin", w, c_flat).view(b, 3, h, w_)
    return torch.clamp(torch.exp(-od), 0.0, 1.0)


@torch.no_grad()
def _single_comp_to_rgb(c_k: torch.Tensor, k: int, w: torch.Tensor) -> torch.Tensor:
    w_k = w[:, k].view(1, 3, 1, 1)
    od = c_k * w_k
    return torch.clamp(torch.exp(-od), 0.0, 1.0)


def _norm_conc(c_raw: torch.Tensor) -> torch.Tensor:
    # No p99: bounded monotonic transform in [0,1)
    return c_raw / (1.0 + c_raw)


def _denorm_conc(c_norm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return c_norm / torch.clamp(1.0 - c_norm, min=eps)


class FMCondSfromHE_OT:
    """
    Conditional OT Flow Matching:
    - condition: H,E
    - predicted channel: S
    - model input: [S_t, H_norm, E_norm] (3 channels)
    - model output: v_S (1 channel)
    """

    def __init__(self, model: nn.Module, device: torch.device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.scaler = amp.GradScaler()

        self.w, self.w_inv = _build_w(device)
        self.loss_l1_weight = float(getattr(args, "loss_l1_weight", 1.0))
        self.loss_l2_weight = float(getattr(args, "loss_l2_weight", 1.0))
        self.ema_decay = float(getattr(args, "ema_decay", 0.999))

        ckpt_dir = getattr(args, "ckpt_dir", None)
        if ckpt_dir is None:
            ckpt_dir = DEFAULT_CKPT_DIR
        self.model_path = ckpt_dir
        self.save_path = os.path.join(args.root, f"results/{args.dataset}/fm_cond_ot_s_he/")
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.data.copy_(b.data)

    @torch.no_grad()
    def _sample_s(self, c_h_norm: torch.Tensor, c_e_norm: torch.Tensor, steps: int = 50, use_ema: bool = True):
        model = self.ema_model if use_ema else self.model
        model.eval()

        h_m11 = c_h_norm * 2.0 - 1.0
        e_m11 = c_e_norm * 2.0 - 1.0
        b, _, h, w = h_m11.shape
        x = torch.randn(b, 1, h, w, device=self.device)
        dt = 1.0 / max(steps, 1)

        for s in range(steps):
            t = torch.full((b,), (s + 0.5) * dt, device=self.device)
            x_in = torch.cat([x, h_m11, e_m11], dim=1)
            v = model(x_in, t)
            x = x + v * dt
        return x

    @torch.no_grad()
    def _save_grid_epoch(self, epoch: int, test_loader: DataLoader, num_examples: int = 8, steps: int = 50):
        self.model.eval()
        self.ema_model.eval()

        he_xs, hes_xs = [], []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                he, hes = batch[0], batch[1]
            else:
                hes = batch[0] if isinstance(batch, (list, tuple)) else batch
                he = hes
            he_xs.append(he.to(self.device))
            hes_xs.append(hes.to(self.device))
            if sum(b.shape[0] for b in hes_xs) >= num_examples:
                break
        he = torch.cat(he_xs, dim=0)[:num_examples]
        hes = torch.cat(hes_xs, dim=0)[:num_examples]

        c_raw_he = _rgb_to_hes_batch(he, self.w_inv)
        c_raw_hes = _rgb_to_hes_batch(hes, self.w_inv)
        c_norm_he = _norm_conc(c_raw_he)
        c_h_raw, c_e_raw = c_raw_he[:, 0:1], c_raw_he[:, 1:2]
        c_s_raw = c_raw_hes[:, 2:3]
        c_h_norm, c_e_norm = c_norm_he[:, 0:1], c_norm_he[:, 1:2]

        s_gen_m11 = self._sample_s(c_h_norm, c_e_norm, steps=steps, use_ema=True)
        s_gen_norm = torch.clamp((s_gen_m11 + 1.0) / 2.0, 0.0, 1.0)
        c_s_gen_raw = _denorm_conc(s_gen_norm)

        c_he = torch.cat([c_h_raw, c_e_raw, torch.zeros_like(c_h_raw)], dim=1)
        c_hes_gt = torch.cat([c_h_raw, c_e_raw, c_s_raw], dim=1)
        c_hes_gen = torch.cat([c_h_raw, c_e_raw, c_s_gen_raw], dim=1)

        he_vis = _to_rgb01(he)
        s_gt_vis = _single_comp_to_rgb(c_s_raw, 2, self.w)
        s_gen_vis = _single_comp_to_rgb(c_s_gen_raw, 2, self.w)
        hes_gt_vis = _to_rgb01(hes)
        hes_gen_vis = _hes_to_rgb_batch(c_hes_gen, self.w)

        rows = []
        for i in range(num_examples):
            rows += [
                he_vis[i].cpu(),
                s_gt_vis[i].cpu(),
                s_gen_vis[i].cpu(),
                hes_gt_vis[i].cpu(),
                hes_gen_vis[i].cpu(),
            ]
        grid = vutils.make_grid(rows, nrow=5, normalize=False)
        np_grid = np.clip(grid.permute(1, 2, 0).numpy(), 0.0, 1.0)

        out_dir = os.path.join(self.save_path, "grid_conditional")
        os.makedirs(out_dir, exist_ok=True)
        plt.imsave(os.path.join(out_dir, f"conditional_epoch_{epoch}.png"), np_grid)

    def _latest_ckpt(self):
        paths = glob.glob(os.path.join(self.model_path, "model_*.pt"))
        cands = []
        for p in paths:
            m = re.search(r"model_(\d+)\.pt$", p)
            if m:
                cands.append((int(m.group(1)), p))
        if not cands:
            return None
        return max(cands, key=lambda x: x[0])

    def train(self, data_loaders: Dict[str, DataLoader]):
        train_loader: DataLoader = data_loaders["train"]
        test_loader: DataLoader = data_loaders["test"]

        with open(os.path.join(self.save_path, "model_info.txt"), "w") as f:
            f.write("PARAMETERS\n")
            f.write(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n")
            f.write(f"Number of epochs: {self.args.num_epoch}\n")
            f.write(f"Batch size: {self.args.batch_size_train}\n")
            f.write(f"Learning rate: {self.args.lr}\n")
            f.write(f"Conditioning: HE -> S (OT coupling)\n")
            f.write(f"Loss robust S: {self.loss_l2_weight}*L2 + {self.loss_l1_weight}*L1\n")
            f.write(f"EMA decay: {self.ema_decay}\n")
            f.write(f"Checkpoint dir: {self.model_path}\n")

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        loss_log = os.path.join(self.save_path, "loss_training.txt")

        start_ep = 0
        last = self._latest_ckpt()
        if last:
            ep, path = last
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                print(f"→ Resume checkpoint from epoch {ep}: {path}")
                ck = torch.load(path, map_location=self.device)
                self.model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optimizer"])
                self.scaler.load_state_dict(ck["scaler"])
                if "ema_decay" in ck:
                    self.ema_decay = float(ck["ema_decay"])
                if ck.get("ema_model", None) is not None:
                    self.ema_model.load_state_dict(ck["ema_model"])
                else:
                    self.ema_model.load_state_dict(ck["model"])
                start_ep = int(ck["epoch"])
                self._save_grid_epoch(
                    start_ep,
                    test_loader,
                    num_examples=8,
                    steps=int(getattr(self.args, "sample_steps", 50)),
                )
            else:
                print(f"⚠️ Checkpoint {path} is missing/empty. Starting fresh.")

        for ep in range(start_ep, self.args.num_epoch):
            print(f"\n→ Epoch {ep + 1}/{self.args.num_epoch}")
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

                c_raw_he = _rgb_to_hes_batch(he, self.w_inv)
                c_raw_hes = _rgb_to_hes_batch(hes, self.w_inv)
                c_norm_he = _norm_conc(c_raw_he)
                c_norm_hes = _norm_conc(c_raw_hes)
                h_m11 = c_norm_he[:, 0:1] * 2.0 - 1.0
                e_m11 = c_norm_he[:, 1:2] * 2.0 - 1.0
                s_gt = c_norm_hes[:, 2:3] * 2.0 - 1.0

                x0 = torch.randn_like(s_gt)
                t = torch.rand(s_gt.shape[0], 1, 1, 1, device=self.device)

                x0_flat = x0.view(x0.size(0), -1)
                s_flat = s_gt.view(s_gt.size(0), -1)
                m_t = torch.cdist(x0_flat, s_flat, p=2) ** 2
                m = m_t.detach().cpu().numpy()
                a = b = np.ones(x0_flat.size(0), dtype=np.float64) / x0_flat.size(0)
                plan = ot.emd(a, b, m)
                p = plan.flatten()
                p = p / max(p.sum(), 1e-12)
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1],
                    p=p,
                    size=len(x0),
                    replace=True,
                )
                i, j = np.divmod(choices, plan.shape[1])

                x0 = x0[i]
                s_gt = s_gt[j]
                h_m11 = h_m11[j]
                e_m11 = e_m11[j]

                s_t = t * s_gt + (1.0 - t) * x0
                x_in = torch.cat([s_t, h_m11, e_m11], dim=1)
                t_flat = t.view(s_gt.size(0))
                target = s_gt - x0

                opt.zero_grad(set_to_none=True)
                with amp.autocast():
                    v_pred = self.model(x_in, t_flat)
                    loss_l2 = torch.mean((v_pred - target) ** 2)
                    loss_l1 = torch.mean(torch.abs(v_pred - target))
                    loss = self.loss_l2_weight * loss_l2 + self.loss_l1_weight * loss_l1

                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                self._update_ema()

                with open(loss_log, "a") as f:
                    f.write(
                        f"Epoch: {ep}, iter: {it}, Loss: {loss.item():.6f}, "
                        f"L1: {loss_l1.item():.6f}, L2: {loss_l2.item():.6f}\n"
                    )
                bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{loss_l1.item():.4f}",
                    l2=f"{loss_l2.item():.4f}",
                )

            # sampling each epoch (8 examples)
            self._save_grid_epoch(
                ep + 1,
                test_loader,
                num_examples=8,
                steps=int(getattr(self.args, "sample_steps", 50)),
            )

            # checkpoint every 5 epochs
            if (ep + 1) % 5 == 0:
                ck = {
                    "epoch": ep + 1,
                    "model": self.model.state_dict(),
                    "ema_model": self.ema_model.state_dict(),
                    "ema_decay": self.ema_decay,
                    "optimizer": opt.state_dict(),
                    "scaler": self.scaler.state_dict(),
                }
                torch.save(ck, os.path.join(self.model_path, f"model_{ep + 1}.pt"))

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_final.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(self.model_path, "model_final_ema.pt"))
