# evaluate_cfm_he2hes.py
# Conditional Flow Matching model evaluation: HE (RGB) -> S (conc.) -> HES (RGB)
# - CSV for all patches: PSNR, SSIM, PieAPP, LPIPS, MSE, SDC, time
# - PDF limited to TOP-K patches by a chosen criterion (PSNR by default)
# - Global summary (means) in JSON + TXT
"""
export PYTHONPATH="put path of your data here:$PYTHONPATH"

python evaluate.py \
  --data_root "put path of your data here" \
  --split test \
  --dim_image 256 --batch_size 8 --num_workers 4 \
  --ckpt_dir "put path of your data here" \
  --device cuda \
  --dim 128 --dim_mults 1 1 2 2 4 \
  --dropout 0.0 \
  --sample_steps 50 \
  --ref_p99_s 1.0 \
  --s_mask_method otsu --s_mask_quantile 0.5 \
  --topk 20 --topk_metric composite \
  --out_dir "cfm_eval_officiel" \
  --use_gt_p99_s
"""
import os, time, json, csv, math, argparse, random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Optional metrics ----------
_HAS_LPIPS = False
try:
    import lpips
    _LPIPS_MODEL = lpips.LPIPS(net='alex')
    _HAS_LPIPS = True
except Exception as e:
    _LPIPS_MODEL = None
    print("[WARN] LPIPS not available:", repr(e))

_HAS_PIEAPP = False
try:
    import piq  # PieAPP via PIQ
    _PIQ_PIEAPP = piq.PieAPP(reduction='mean')
    _HAS_PIEAPP = True
except Exception as e:
    _PIQ_PIEAPP = None
    print("[WARN] PieAPP (PIQ) not available:", repr(e))

_HAS_SKIMAGE = False
try:
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except Exception as e:
    print("[WARN] scikit-image not available; SDC falls back to quantile.", repr(e))

# ---------- Stain utils ----------
from separation_model.stain_matrix import Wgt2
from separation_model.global_utils import vectorize

# ---------- Conditional FM inference ----------
try:
    from inference_fm_cond_s_he import (
        InferenceConfig, load_model as load_cfm_model,
        generate_hes_from_he, separate_he_from_rgb
    )
except Exception:
    from inference_fm_cond_s_he import (
        InferenceConfig, load_model as load_cfm_model,
        generate_hes_from_he, separate_he_from_rgb
    )

# ---------- DataLoader aligned with PnP ----------
try:
    from Flow_Matching.methods.pnp_dataloaders import get_pnp_loader
except Exception:
    from pnp_dataloaders import get_pnp_loader


# ====================== General helpers ======================
def set_seeds(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to01(x_m1: torch.Tensor) -> torch.Tensor:
    if x_m1.max() > 1.5:  # [0,255]
        x = x_m1 / 255.0
    else:                 # [-1,1]
        x = (x_m1 + 1.0) / 2.0
    return x.clamp(0.0, 1.0)

def torch_img01_to_pil(x01: torch.Tensor) -> Image.Image:
    if x01.dim() == 4: x01 = x01[0]
    arr = (x01.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_torch01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def psnr_torch(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = F.mse_loss(x01, y01).item()
    return float('inf') if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)

def ssim_torch(x01: torch.Tensor, y01: torch.Tensor, win=11, sigma=1.5, K1=0.01, K2=0.03) -> float:
    device = x01.device
    C1 = (K1**2); C2 = (K2**2)
    coords = torch.arange(win, device=device) - win//2
    g = torch.exp(-(coords**2)/(2*sigma**2)); g = (g / g.sum()).view(1,1,1,win)
    window = (g.transpose(-1,-2) @ g).view(1,1,win,win); window = window / window.sum()
    def filt(im): return F.conv2d(im, window.expand(im.size(1),1,win,win), padding=win//2, groups=im.size(1))
    mu_x, mu_y = filt(x01), filt(y01)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x2 = filt(x01*x01) - mu_x2
    sigma_y2 = filt(y01*y01) - mu_y2
    sigma_xy = filt(x01*y01) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().item())

def mse_img(x01: torch.Tensor, y01: torch.Tensor) -> float:
    return float(F.mse_loss(x01, y01).item())

def lpips_metric(x01: torch.Tensor, y01: torch.Tensor) -> float:
    if not _HAS_LPIPS or _LPIPS_MODEL is None: return float('nan')
    with torch.no_grad():
        xm1 = (x01 * 2.0 - 1.0).clamp(-1, 1)
        ym1 = (y01 * 2.0 - 1.0).clamp(-1, 1)
        dev = next(_LPIPS_MODEL.parameters()).device
        v = _LPIPS_MODEL(xm1.to(dev), ym1.to(dev))
        return float(v.mean().item())

def pieapp_metric(x01: torch.Tensor, y01: torch.Tensor) -> float:
    if not _HAS_PIEAPP or _PIQ_PIEAPP is None: return float('nan')
    with torch.no_grad():
        dev = next(_PIQ_PIEAPP.parameters()).device if any(p.requires_grad for p in _PIQ_PIEAPP.parameters()) \
              else torch.device('cpu')
        v = _PIQ_PIEAPP(x01.to(dev), y01.to(dev))
        return float(v.item())

def rgb_pil_to_conc3_numpy(img: Image.Image) -> np.ndarray:
    im = np.asarray(img.convert('RGB'))
    V = vectorize(im).astype(np.float32)          # (3,N)
    W = np.array(Wgt2, dtype=np.float32)
    Winv = np.linalg.pinv(W).astype(np.float32)
    C3 = Winv @ V                                 # (3,N)
    H, Wd = img.size[1], img.size[0]
    C3 = C3.reshape(3, H, Wd).astype(np.float32)
    C3 = np.clip(C3, 0.0, None)
    return C3

def concHE_to_rgb_numpy(cH: np.ndarray, cE: np.ndarray) -> np.ndarray:
    W = np.array(Wgt2, dtype=np.float32)
    OD_HE = (cH[..., None] * W[:,0][None,None,:] + cE[..., None] * W[:,1][None,None,:])
    I = np.exp(-OD_HE) * 255.0
    return np.clip(I, 0, 255).astype(np.uint8)

def saffron_dice_from_conc(cS_gen: np.ndarray, cS_gt: np.ndarray, method='otsu', q=0.5) -> float:
    g = cS_gen.reshape(-1).astype(np.float32)
    t = cS_gt.reshape(-1).astype(np.float32)
    if method=='otsu' and _HAS_SKIMAGE:
        try:
            tg = threshold_otsu(g); tt = threshold_otsu(t)
        except Exception:
            tg = np.quantile(g, q); tt = np.quantile(t, q)
    else:
        tg = np.quantile(g, q); tt = np.quantile(t, q)
    Mg = (g>tg).astype(np.uint8); Mt=(t>tt).astype(np.uint8)
    inter = (Mg & Mt).sum()
    return float((2.0*inter)/(Mg.sum()+Mt.sum()+1e-12))


# ---------- TOP-K scoring ----------
def score_from_metrics(kind, psnr, ssim, lpips, pieapp, mse, sdc):
    kind = kind.lower()
    if kind == 'psnr':     return psnr                     # plus grand = mieux
    if kind == 'ssim':     return ssim
    if kind == 'lpips':    return -lpips                   # plus petit = mieux
    if kind == 'pieapp':   return -pieapp
    if kind == 'mse':      return -mse
    if kind == 'sdc':      return sdc
    if kind == 'composite':
        # Simple stable combination without normalization: coarse weighting
        # (you can tune these weights for your dataset)
        return (psnr
                + 20.0*ssim
                + 5.0*sdc
                - 50.0*(0.0 if np.isnan(lpips) else lpips)
                - 10.0*(0.0 if np.isnan(pieapp) else pieapp)
                - 10.0*mse)
    # fallback
    return psnr


# ====================== Main script ======================
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--dim_image', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    # conditional FM (inference config)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--dim_mults', nargs='+', type=int, default=[1,1,2,2,4])
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--sample_steps', type=int, default=50)   # Euler steps
    ap.add_argument('--ref_p99_s', type=float, default=1.0)   # calibration S
    ap.add_argument('--use_gt_p99_s', action='store_true',
                    help='Calibrate S with p99_S extracted from HES GT (per patch)')
    # metrics / SDC
    ap.add_argument('--s_mask_method', choices=['otsu','quantile'], default='otsu')
    ap.add_argument('--s_mask_quantile', type=float, default=0.5)
    # bookkeeping / top-k
    ap.add_argument('--topk', type=int, default=20, help='Number of best patches to include in PDF')
    ap.add_argument('--topk_metric', type=str, default='psnr',
                    choices=['psnr','ssim','lpips','pieapp','mse','sdc','composite'],
                    help='Ranking criterion for best patches')
    ap.add_argument('--max_images', type=int, default=None)
    ap.add_argument('--out_dir', type=str, default='cfm_eval_results')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Move LPIPS & PieAPP to selected device
    global _LPIPS_MODEL, _PIQ_PIEAPP
    if _HAS_LPIPS and _LPIPS_MODEL is not None:
        _LPIPS_MODEL = _LPIPS_MODEL.to(device).eval()
    if _HAS_PIEAPP and _PIQ_PIEAPP is not None:
        _PIQ_PIEAPP = _PIQ_PIEAPP.to(device).eval()

    # --- Build CFM config and load model ---
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

    # --- DataLoader ---
    loader = get_pnp_loader(
        data_root=args.data_root, split=args.split,
        batch_size=args.batch_size, dim_image=args.dim_image,
        num_workers=args.num_workers, shuffle=False
    )

    # --- Outputs ---
    pdf_path = os.path.join(args.out_dir, "eval_visualizationals_topk.pdf")
    csv_path = os.path.join(args.out_dir, "metrics_per_patch.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["idx","Score","PSNR","SSIM","PieAPP","LPIPS","MSE","SDC","GenTime_s"])

    # Global accumulators
    PSNRs, SSIMs, PIEAPPs, LPIPSs, MSEs, SDCs, GTs = [], [], [], [], [], [], []
    global_idx = 0

    # TOP-K buffer (keep only K best entries in memory for the PDF)
    topk_buf = []  # list of dicts: {'score':..., 'idx':..., 'imgs':{...}, 'metrics':{...}}

    def maybe_push_topk(payload):
        nonlocal topk_buf
        if len(topk_buf) < args.topk:
            topk_buf.append(payload)
        else:
            # replace the worst one if current is better
            worst_i = min(range(len(topk_buf)), key=lambda i: topk_buf[i]['score'])
            if payload['score'] > topk_buf[worst_i]['score']:
                topk_buf[worst_i] = payload

    # ---------- Loop ----------
    for it, (hes_rgb_m1, he_rgb_m1) in enumerate(loader):
        hes01 = to01(hes_rgb_m1).to(device)
        he01  = to01(he_rgb_m1).to(device)

        B = he01.size(0)
        for b in range(B):
            if args.max_images is not None and global_idx >= args.max_images: break

            he_b01  = he01[b:b+1]
            hes_b01 = hes01[b:b+1]

            he_pil = torch_img01_to_pil(he_b01)

            # p99_S calibration from GT (optional)
            p99_s_runtime = None
            if args.use_gt_p99_s:
                hes_gt_pil_for_p99 = torch_img01_to_pil(hes_b01)
                C3_gt_for_p99 = rgb_pil_to_conc3_numpy(hes_gt_pil_for_p99)   # (3,H,W)
                p99_s_runtime = float(np.percentile(np.clip(C3_gt_for_p99[2], 0, None), 99.0))

            # Inference
            t0 = time.time()
            hes_gen_pil, s_vis_pil = generate_hes_from_he(
                he_pil, cfg, model=model, ref_p99_s=p99_s_runtime
            )
            gen_time = time.time() - t0

            # Tensors [0,1]
            hes_gen01 = pil_to_torch01(hes_gen_pil).to(device)
            hes_gt01  = hes_b01

            # Reconstructed HE (for PDF row 1)
            cH_raw, cE_raw, _, _ = separate_he_from_rgb(he_pil, to_size=args.dim_image)
            he_recon_u8 = concHE_to_rgb_numpy(cH_raw.astype(np.float32), cE_raw.astype(np.float32))
            he_recon_pil = Image.fromarray(he_recon_u8)

            # SDC: extract generated/GT cS
            C3_gen = rgb_pil_to_conc3_numpy(hes_gen_pil)
            C3_gt  = rgb_pil_to_conc3_numpy(torch_img01_to_pil(hes_gt01))
            sdc_i = saffron_dice_from_conc(C3_gen[2], C3_gt[2],
                                           method=args.s_mask_method, q=args.s_mask_quantile)

            # Component renderings for PDF (row 2)
            H_only_u8 = concHE_to_rgb_numpy(cH_raw.astype(np.float32), np.zeros_like(cE_raw))
            E_only_u8 = concHE_to_rgb_numpy(np.zeros_like(cH_raw), cE_raw.astype(np.float32))
            # S GT visualization
            W = np.array(Wgt2, dtype=np.float32)
            S_gt_vis_u8 = (np.exp(-(C3_gt[2][...,None]*W[:,2][None,None,:]))*255.0).clip(0,255).astype(np.uint8)

            # Metrics
            psnr_i  = psnr_torch(hes_gen01, hes_gt01)
            ssim_i  = ssim_torch(hes_gen01, hes_gt01)
            mse_i   = mse_img(hes_gen01,  hes_gt01)
            lpips_i = lpips_metric(hes_gen01, hes_gt01)
            pie_i   = pieapp_metric(hes_gen01, hes_gt01)

            # Accumulate means
            PSNRs.append(psnr_i); SSIMs.append(ssim_i); MSEs.append(mse_i)
            LPIPSs.append(lpips_i); PIEAPPs.append(pie_i); SDCs.append(sdc_i)
            GTs.append(gen_time)

            # Score for TOP-K
            score = score_from_metrics(args.topk_metric, psnr_i, ssim_i, lpips_i, pie_i, mse_i, sdc_i)

            # CSV (all images)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([global_idx, f"{score:.6f}",
                                        f"{psnr_i:.6f}", f"{ssim_i:.6f}", f"{pie_i:.6f}",
                                        f"{lpips_i:.6f}", f"{mse_i:.6f}", f"{sdc_i:.6f}",
                                        f"{gen_time:.3f}"])

            # Push candidate to TOP-K (keep only K best in RAM)
            payload = {
                'score': score,
                'idx': global_idx,
                'imgs': {
                    'he_real': he_pil,
                    'he_recon': he_recon_pil,
                    'h_only': Image.fromarray(H_only_u8),
                    'e_only': Image.fromarray(E_only_u8),
                    's_gen': s_vis_pil,
                    's_gt': Image.fromarray(S_gt_vis_u8),
                    'hes_gen': hes_gen_pil,
                    'hes_gt': torch_img01_to_pil(hes_gt01),
                },
                'metrics': {
                    'PSNR': psnr_i, 'SSIM': ssim_i, 'LPIPS': lpips_i,
                    'PieAPP': pie_i, 'MSE': mse_i, 'SDC': sdc_i, 'GenTime_s': gen_time
                }
            }
            maybe_push_topk(payload)
            global_idx += 1

        if args.max_images is not None and global_idx >= args.max_images:
            break

    # --------- PDF: save only TOP-K ---------
    topk_sorted = sorted(topk_buf, key=lambda d: d['score'], reverse=True)
    pdf_path = os.path.join(args.out_dir, f"eval_visualizationals_top{args.topk}_{args.topk_metric}.pdf")
    with PdfPages(pdf_path) as pdf:
        for entry in topk_sorted:
            m = entry['metrics']; im = entry['imgs']; idx = entry['idx']; sc = entry['score']
            fig, axes = plt.subplots(3,4, figsize=(16,12))
            for a in axes.ravel(): a.axis('off')

            # Row 1
            axes[0,0].imshow(im['he_real']);   axes[0,0].set_title(f"HE real (idx={idx})")
            axes[0,1].imshow(im['he_recon']);  axes[0,1].set_title("HE reconstructed (H+E)")

            # Row 2
            axes[1,0].imshow(im['h_only']);    axes[1,0].set_title("H (from HE)")
            axes[1,1].imshow(im['e_only']);    axes[1,1].set_title("E (from HE)")
            axes[1,2].imshow(im['s_gen']);     axes[1,2].set_title("Generated S")
            axes[1,3].imshow(im['s_gt']);      axes[1,3].set_title("S GT")

            # Row 3
            axes[2,0].imshow(im['hes_gen']);   axes[2,0].set_title(
                f"Generated HES (score={sc:.3f})\n"
                f"PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.3f}\n"
                f"LPIPS={m['LPIPS']:.3f}  PieAPP={m['PieAPP']:.3f}\n"
                f"MSE={m['MSE']:.4f}  SDC={m['SDC']:.3f}\n"
                f"t_gen={m['GenTime_s']:.2f}s")
            axes[2,1].imshow(im['hes_gt']);    axes[2,1].set_title("HES GT")

            plt.tight_layout(); pdf.savefig(fig, dpi=250); plt.close(fig)

    # --------- Means ---------
    def _nanmean(v): return float(np.nanmean(v)) if len(v)>0 else float('nan')
    summary = {
        "N": global_idx,
        "PSNR_mean": _nanmean(PSNRs),
        "SSIM_mean": _nanmean(SSIMs),
        "PieAPP_mean": _nanmean(PIEAPPs),
        "LPIPS_mean": _nanmean(LPIPSs),
        "MSE_mean": _nanmean(MSEs),
        "SDC_mean": _nanmean(SDCs),
        "GenTime_mean_s": _nanmean(GTs),
        "TopK": args.topk,
        "TopK_metric": args.topk_metric,
        "PDF_path": pdf_path,
        "CSV_path": csv_path
    }
    with open(os.path.join(args.out_dir,"metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir,"metrics_summary.txt"), "w") as f:
        f.write("=== EVALUATION SUMMARY (CFM HE→HES) ===\n")
        for k,v in summary.items(): f.write(f"{k}: {v}\n")

    print("Done. Summary:", summary)
    print("• TOP-K PDF:", pdf_path)
    print("• Per-image CSV:", csv_path)


if __name__ == "__main__":
    main()
