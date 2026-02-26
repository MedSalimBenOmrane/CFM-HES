import io
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Keep startup light: no torch / heavy ML imports at module import time.
BASE_DIR = Path(__file__).resolve().parents[1]      # .../CFM HES
PROJECT_DIR = BASE_DIR.parent                        # .../PNP_FM
for p in (str(BASE_DIR), str(PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

PATCH = 256
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


def _parse_dim_mults(s: str):
    vals = re.findall(r"\d+", s or "")
    return tuple(int(v) for v in vals) if vals else (1, 1, 2, 2, 4)


def load_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")


def ensure_size(img: Image.Image, size: Tuple[int, int], allow_resize: bool) -> Image.Image:
    if img.size == size:
        return img
    if not allow_resize:
        raise ValueError(f"Image must be {size}, got {img.size}")
    return img.resize(size, Image.BICUBIC)


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n)
    win = np.outer(w, w)
    win = (win - win.min()) / (win.max() - win.min() + 1e-12)
    return win.astype(np.float32)


def patch_coords(img_wh: Tuple[int, int], patch: int = PATCH, overlap: int = 0) -> List[Tuple[int, int]]:
    w, h = img_wh
    step = patch - overlap
    xs = list(range(0, max(w - patch + 1, 1), step))
    ys = list(range(0, max(h - patch + 1, 1), step))
    if xs and xs[-1] != w - patch:
        xs.append(w - patch)
    if ys and ys[-1] != h - patch:
        ys.append(h - patch)
    return [(x, y) for y in ys for x in xs]


def crop_patch(img: Image.Image, x: int, y: int, patch: int = PATCH) -> Image.Image:
    return img.crop((x, y, x + patch, y + patch))


def recon_from_patches(
    patches: List[Tuple[int, int, Image.Image]],
    canvas_wh: Tuple[int, int],
    patch: int = PATCH,
    overlap: int = 0,
) -> Image.Image:
    w, h = canvas_wh
    acc = np.zeros((h, w, 3), np.float32)
    wgt = np.zeros((h, w, 1), np.float32)
    win = _hann2d(patch) if overlap > 0 else np.ones((patch, patch), np.float32)
    win3 = win[..., None]

    for x, y, p in patches:
        arr = np.asarray(p, dtype=np.float32)
        acc[y:y + patch, x:x + patch] += arr * win3
        wgt[y:y + patch, x:x + patch] += win3

    out = acc / np.clip(wgt, 1e-6, None)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


@st.cache_resource(show_spinner=False)
def _load_cfm_cached(ckpt_dir: str, device: str, dim: int, dim_mults: tuple, dropout: float, sample_steps: int):
    # Lazy import to avoid startup blocking.
    from inference_fm_cond_s_he import InferenceConfig, load_model as load_cfm_model

    cfg = InferenceConfig(
        CHECKPOINT_DIR=ckpt_dir,
        DIM=dim,
        DIM_MULTS=tuple(dim_mults),
        DROPOUT=dropout,
        SAMPLE_STEPS=sample_steps,
        IMG_SIZE=256,
        DEVICE=device,
        REF_P99_S=1.0,
    )
    model = load_cfm_model(cfg)
    return cfg, model


def _estimate_ref_p99_s(hes_gt: Image.Image, to_size: int = 256, w_np: Optional[np.ndarray] = None) -> float:
    from separation_model.stain_matrix import Wgt2
    from separation_model.global_utils import vectorize

    gt = hes_gt.resize((to_size, to_size), Image.BICUBIC)
    im = np.asarray(gt.convert("RGB"))
    v = vectorize(im).astype(np.float32)
    W = _ensure_w_matrix(w_np if w_np is not None else np.array(Wgt2, dtype=np.float32))
    winv = np.linalg.pinv(W)
    c3 = winv @ v
    cS = np.clip(c3[2].reshape(to_size, to_size).astype(np.float32), 0.0, None)
    return float(np.percentile(cS.reshape(-1), 99.0))


def _extract_s_and_metrics(hes_out: Image.Image, gt_img: Optional[Image.Image], w_np: Optional[np.ndarray] = None):
    from separation_model.stain_matrix import Wgt2
    from separation_model.global_utils import vectorize

    W = _ensure_w_matrix(w_np if w_np is not None else np.array(Wgt2, dtype=np.float32))

    def _extract_cS(img: Image.Image):
        im = np.asarray(img.convert("RGB"))
        v = vectorize(im).astype(np.float32)
        winv = np.linalg.pinv(W)
        c3 = winv @ v
        h, w = im.shape[:2]
        return np.clip(c3[2].reshape(h, w).astype(np.float32), 0.0, None)

    cS_pred = _extract_cS(hes_out)
    od_s_pred = cS_pred[..., None] * W[:, 2][None, None, :]
    s_img_pred = Image.fromarray(np.clip(np.exp(-od_s_pred) * 255.0, 0, 255).astype(np.uint8))

    s_img_gt = None
    metrics = None
    if gt_img is not None:
        cS_gt = _extract_cS(gt_img)
        od_s_gt = cS_gt[..., None] * W[:, 2][None, None, :]
        s_img_gt = Image.fromarray(np.clip(np.exp(-od_s_gt) * 255.0, 0, 255).astype(np.uint8))

        pred = np.asarray(hes_out).astype(np.float32)
        gt = np.asarray(gt_img).astype(np.float32)
        mse = float(np.mean(((pred - gt) / 255.0) ** 2))
        metrics = {"MSE": mse, "SSIM": "N/A", "LPIPS": "N/A", "PIEAPP": "N/A"}
        try:
            from skimage.metrics import structural_similarity as ssim
            metrics["SSIM"] = float(ssim(gt, pred, data_range=255, channel_axis=2))
        except Exception:
            pass
        try:
            import torch
            import lpips as _lpips

            lpips_model = _lpips.LPIPS(net="alex").eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            lpips_model = lpips_model.to(dev)
            with torch.no_grad():
                x = torch.from_numpy((pred / 255.0).transpose(2, 0, 1))[None].to(dev)
                y = torch.from_numpy((gt / 255.0).transpose(2, 0, 1))[None].to(dev)
                xm1 = (x * 2 - 1).clamp(-1, 1)
                ym1 = (y * 2 - 1).clamp(-1, 1)
                metrics["LPIPS"] = float(lpips_model(xm1, ym1).mean().item())
        except Exception:
            pass
        try:
            import torch
            import piq as _piq

            pie_model = _piq.PieAPP(reduction="mean").eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            pie_model = pie_model.to(dev)
            with torch.no_grad():
                x = torch.from_numpy((pred / 255.0).transpose(2, 0, 1))[None].to(dev)
                y = torch.from_numpy((gt / 255.0).transpose(2, 0, 1))[None].to(dev)
                metrics["PIEAPP"] = float(pie_model(x.float(), y.float()).item())
        except Exception:
            pass

    return s_img_pred, s_img_gt, metrics


def _ensure_w_matrix(w_np: np.ndarray) -> np.ndarray:
    from separation_model.stain_matrix import Wgt2

    w = np.array(w_np, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] != 3:
        raise ValueError(f"Invalid W shape: {w.shape}")
    if w.shape[1] == 2:
        s_col = np.array(Wgt2, dtype=np.float32)[:, 2:3]
        w = np.concatenate([w, s_col], axis=1)
    if w.shape[1] != 3:
        raise ValueError(f"Invalid W shape: {w.shape}")
    return w


def _default_w_matrix() -> np.ndarray:
    from separation_model.stain_matrix import Wgt2

    return np.array(Wgt2, dtype=np.float32)


def _load_wssb_w(organ: str) -> np.ndarray:
    mapping = {
        "breast": ("Breast", "Breast_mean_stain_matrix.txt"),
        "lung": ("Lung", "Lung_mean_stain_matrix.txt"),
    }
    key = organ.lower()
    if key not in mapping:
        raise ValueError(f"Unknown organ: {organ}")
    folder, fname = mapping[key]
    w_path_local = BASE_DIR / "wssb_dataset" / "original_data_RGB_images" / folder / fname
    w_path_parent = PROJECT_DIR / "wssb_dataset" / "original_data_RGB_images" / folder / fname
    w_path = w_path_local if w_path_local.exists() else w_path_parent
    if not w_path.exists():
        raise FileNotFoundError(f"W matrix not found: {w_path_local}")
    w = np.loadtxt(w_path)
    return _ensure_w_matrix(w)


def _separate_hes_to_he_s(hes_img: Image.Image, w_np: np.ndarray) -> Tuple[Image.Image, Image.Image]:
    from separation_model.global_utils import vectorize

    im = np.asarray(hes_img.convert("RGB"))
    W = _ensure_w_matrix(w_np)
    v = vectorize(im).astype(np.float32)
    winv = np.linalg.pinv(W)
    c3 = winv @ v
    h, w = im.shape[:2]
    cH = c3[0].reshape(h, w)
    cE = c3[1].reshape(h, w)
    cS = c3[2].reshape(h, w)

    od_he = cH[..., None] * W[:, 0][None, None, :] + cE[..., None] * W[:, 1][None, None, :]
    od_s = cS[..., None] * W[:, 2][None, None, :]
    he_rgb = np.clip(np.exp(-od_he) * 255.0, 0, 255).astype(np.uint8)
    s_rgb = np.clip(np.exp(-od_s) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(he_rgb), Image.fromarray(s_rgb)


def _separate_he_to_h_e(he_img: Image.Image, w_np: np.ndarray) -> Tuple[Image.Image, Image.Image]:
    from separation_model.global_utils import vectorize

    im = np.asarray(he_img.convert("RGB"))
    W = _ensure_w_matrix(w_np)
    v = vectorize(im).astype(np.float32)
    winv = np.linalg.pinv(W)
    c3 = winv @ v
    h, w = im.shape[:2]
    cH = c3[0].reshape(h, w)
    cE = c3[1].reshape(h, w)

    od_h = cH[..., None] * W[:, 0][None, None, :]
    od_e = cE[..., None] * W[:, 1][None, None, :]
    h_rgb = np.clip(np.exp(-od_h) * 255.0, 0, 255).astype(np.uint8)
    e_rgb = np.clip(np.exp(-od_e) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(h_rgb), Image.fromarray(e_rgb)


def _render_generation_tab(
    prefix: str,
    w_np: Optional[np.ndarray],
    auto_res: bool,
    target_side,
    allow_resize: bool,
    overlap: int,
    ckpt_dir: str,
    sample_steps: int,
    device: str,
    dim: int,
    dim_mults_str: str,
    dropout: float,
) -> None:
    left, right = st.columns(2)
    with left:
        he_file = st.file_uploader("Upload HE", type=ALLOWED_IMAGE_TYPES, key=f"{prefix}_he")
    with right:
        gt_file = st.file_uploader("Upload HES GT (optional)", type=ALLOWED_IMAGE_TYPES, key=f"{prefix}_gt")

    run_btn = st.button("Generate with CFM", type="primary", key=f"{prefix}_run")

    if he_file or gt_file:
        pv_he, pv_gt = st.columns(2)
        if he_file:
            he_raw = load_image(he_file)
            if auto_res:
                he_proc = he_raw
            else:
                try:
                    he_proc = ensure_size(he_raw, (int(target_side), int(target_side)), allow_resize or auto_res)
                except Exception:
                    he_proc = he_raw
            pv_he.image(he_raw, caption="HE upload", use_container_width=True)
            if he_proc.size != he_raw.size:
                pv_he.caption(f"HE used for inference resized to {he_proc.size[0]}x{he_proc.size[1]}")
            else:
                pv_he.caption(f"HE used for inference: {he_proc.size[0]}x{he_proc.size[1]}")
        if gt_file:
            gt_raw = load_image(gt_file)
            pv_gt.image(gt_raw, caption="HES GT upload", use_container_width=True)

    if not run_btn:
        return

    if he_file is None:
        st.error("Please upload HE image.")
        st.stop()

    if prefix == "wssb" and w_np is None:
        st.error("Please select one WSSB organ (breast or lung).")
        st.stop()

    he_img = load_image(he_file)
    gt_img = load_image(gt_file) if gt_file else None

    if auto_res:
        target_wh = he_img.size
    else:
        target_wh = (int(target_side), int(target_side))

    he_img = ensure_size(he_img, target_wh, allow_resize or auto_res)
    if gt_img is not None:
        gt_img = ensure_size(gt_img, target_wh, allow_resize or auto_res)

    ref_p99_runtime = 1.0
    if gt_img is not None:
        try:
            ref_p99_runtime = _estimate_ref_p99_s(gt_img, to_size=256, w_np=w_np)
        except Exception as e:
            st.warning(f"GT p99 calibration failed ({e}); using fallback {ref_p99_runtime:.4f}.")

    dim_mults = _parse_dim_mults(dim_mults_str)
    try:
        cfg_cfm, model_cfm = _load_cfm_cached(
            ckpt_dir=ckpt_dir,
            device=device,
            dim=int(dim),
            dim_mults=dim_mults,
            dropout=float(dropout),
            sample_steps=int(sample_steps),
        )
        from inference_fm_cond_s_he import generate_hes_from_he
    except Exception as e:
        st.error(f"CFM model loading failed: {e}")
        st.stop()

    coords = patch_coords(target_wh, patch=PATCH, overlap=int(overlap))
    total_n = len(coords)

    progress = st.progress(0.0, text="Starting generation...")
    out_placeholder = st.empty()

    out_patches = []
    t0 = time.time()
    for k, (x, y) in enumerate(coords, start=1):
        p = crop_patch(he_img, x, y, PATCH)
        hes_patch, _s_vis = generate_hes_from_he(
            p,
            cfg_cfm,
            model=model_cfm,
            ref_p99_s=ref_p99_runtime,
            w_np=w_np,
        )
        out_patches.append((x, y, hes_patch))

        canvas = recon_from_patches(out_patches, target_wh, patch=PATCH, overlap=int(overlap))
        out_placeholder.image(canvas, caption=f"CFM stream - patch {k}/{total_n}", use_container_width=True)
        progress.progress(k / total_n, text=f"Generating {k}/{total_n}")

    gen_time = time.time() - t0
    hes_out = recon_from_patches(out_patches, target_wh, patch=PATCH, overlap=int(overlap))

    s_img = None
    s_img_gt = None
    metrics = None
    try:
        s_img, s_img_gt, metrics = _extract_s_and_metrics(hes_out, gt_img, w_np=w_np)
    except Exception as e:
        st.warning(f"Post-processing skipped: {e}")

    st.subheader("CFM Result")
    if gt_img is not None:
        h1, h2 = st.columns(2)
        h1.image(gt_img, caption="HES Ground Truth", use_container_width=True)
        h2.image(hes_out, caption=f"HES generee (CFM) - {gen_time:.2f}s", use_container_width=True)
    else:
        st.image(hes_out, caption=f"HES generee (CFM) - {gen_time:.2f}s", use_container_width=True)

    if s_img is not None:
        if s_img_gt is not None:
            s1, s2 = st.columns(2)
            s1.image(s_img_gt, caption="Safran GT", use_container_width=True)
            s2.image(s_img, caption="Safran genere", use_container_width=True)
        else:
            st.image(s_img, caption="Safran genere", use_container_width=True)

    if metrics is not None:
        st.markdown("### Metrics")
        cols = st.columns(4)
        keys = ["SSIM", "MSE", "PIEAPP", "LPIPS"]
        for i, k in enumerate(keys):
            if k in metrics:
                if isinstance(metrics[k], (int, float, np.floating)):
                    cols[i].metric(k, f"{metrics[k]:.4f}")
                else:
                    cols[i].metric(k, str(metrics[k]))

    b_hes = io.BytesIO()
    hes_out.save(b_hes, format="PNG")
    st.download_button(
        "Download HES (CFM)",
        data=b_hes.getvalue(),
        file_name=f"hes_generated_{prefix}_cfm.png",
        mime="image/png",
        key=f"{prefix}_dl_hes",
    )

    if s_img is not None:
        b_s = io.BytesIO()
        s_img.save(b_s, format="PNG")
        st.download_button(
            "Download S (CFM)",
            data=b_s.getvalue(),
            file_name=f"s_generated_{prefix}_cfm.png",
            mime="image/png",
            key=f"{prefix}_dl_s",
        )


def _render_separation_tab(prefix: str, w_np: Optional[np.ndarray]) -> None:
    is_wssb = prefix == "wssb"
    if is_wssb:
        st.caption("WSSB separation: HE -> H + E")
        file_label = "Upload HE"
    else:
        st.caption("Hospital Data separation: HES -> HE + S")
        file_label = "Upload HES"

    side_choice = st.selectbox(
        "Separation resolution",
        ["Auto", 256, 512, 1024],
        index=3,
        key=f"{prefix}_sep_side",
    )
    sep_auto = side_choice == "Auto"
    sep_resize = st.toggle(
        "Resize to selected resolution" if not sep_auto else "Resize (disabled in Auto)",
        value=not sep_auto,
        disabled=sep_auto,
        key=f"{prefix}_sep_resize",
    )
    sep_file = st.file_uploader(file_label, type=ALLOWED_IMAGE_TYPES, key=f"{prefix}_sep_file")
    run_sep = st.button("Run separation", key=f"{prefix}_sep_run")

    if sep_file:
        sep_raw = load_image(sep_file)
        if sep_auto:
            sep_proc = sep_raw
        else:
            try:
                sep_proc = ensure_size(sep_raw, (int(side_choice), int(side_choice)), sep_resize or sep_auto)
            except Exception:
                sep_proc = sep_raw
        p1, p2 = st.columns(2)
        p1.image(sep_raw, caption="Input", use_container_width=True)
        p2.image(sep_proc, caption="Loaded", use_container_width=True)

    if not run_sep:
        return

    if sep_file is None:
        st.error("Please upload an image first.")
        st.stop()
    if w_np is None:
        st.error("Please select one WSSB organ (breast or lung).")
        st.stop()

    img_in = load_image(sep_file)
    if sep_auto:
        target_wh = img_in.size
    else:
        target_wh = (int(side_choice), int(side_choice))
    img_in = ensure_size(img_in, target_wh, sep_resize or sep_auto)

    if is_wssb:
        h_img, e_img = _separate_he_to_h_e(img_in, w_np)
        c1, c2, c3 = st.columns(3)
        c1.image(img_in, caption="HE input", use_container_width=True)
        c2.image(h_img, caption="H channel", use_container_width=True)
        c3.image(e_img, caption="E channel", use_container_width=True)

        b1 = io.BytesIO(); img_in.save(b1, format="PNG")
        b2 = io.BytesIO(); h_img.save(b2, format="PNG")
        b3 = io.BytesIO(); e_img.save(b3, format="PNG")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download HE", data=b1.getvalue(), file_name=f"{prefix}_HE_input.png", mime="image/png", key=f"{prefix}_sep_dl_he")
        d2.download_button("Download H", data=b2.getvalue(), file_name=f"{prefix}_H.png", mime="image/png", key=f"{prefix}_sep_dl_h")
        d3.download_button("Download E", data=b3.getvalue(), file_name=f"{prefix}_E.png", mime="image/png", key=f"{prefix}_sep_dl_e")
    else:
        he_img, s_img = _separate_hes_to_he_s(img_in, w_np)
        c1, c2, c3 = st.columns(3)
        c1.image(img_in, caption="HES input", use_container_width=True)
        c2.image(he_img, caption="HE reconstructed", use_container_width=True)
        c3.image(s_img, caption="Safran", use_container_width=True)

        b1 = io.BytesIO(); img_in.save(b1, format="PNG")
        b2 = io.BytesIO(); he_img.save(b2, format="PNG")
        b3 = io.BytesIO(); s_img.save(b3, format="PNG")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download HES", data=b1.getvalue(), file_name=f"{prefix}_HES_input.png", mime="image/png", key=f"{prefix}_sep_dl_hes")
        d2.download_button("Download HE", data=b2.getvalue(), file_name=f"{prefix}_HE.png", mime="image/png", key=f"{prefix}_sep_dl_he2")
        d3.download_button("Download S", data=b3.getvalue(), file_name=f"{prefix}_S.png", mime="image/png", key=f"{prefix}_sep_dl_s")


st.set_page_config(page_title="CFM Demo (HE -> HES)", layout="wide")
st.title("CFM Demo - HE to HES (Ours)")
st.caption("CFM-only app with Generation + Separation tabs, and WSSB organ selection.")

with st.sidebar:
    st.header("CFM Settings")
    target_side = st.selectbox("Inference resolution", ["Auto", 256, 512, 1024], index=3)
    auto_res = target_side == "Auto"
    allow_resize = st.toggle(
        "Resize to selected resolution" if not auto_res else "Resize (disabled in Auto)",
        value=not auto_res,
        disabled=auto_res,
    )
    overlap = st.slider("Patch overlap", min_value=0, max_value=64, value=0, step=16)

    ckpt_dir = st.text_input(
        "CFM checkpoint directory",
        value="put path of your data here",
    )
    sample_steps = st.number_input("sample_steps (Euler)", min_value=1, max_value=500, value=50, step=1)

    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    dim = st.number_input("U-Net dim", min_value=16, max_value=512, value=128, step=16)
    dim_mults_str = st.text_input("dim_mults", value="(1,1,2,2,4)")
    dropout = st.number_input("dropout", min_value=0.0, max_value=0.95, value=0.0, step=0.05)

bic_tab, wssb_tab = st.tabs(["Hospital Data", "WSSB"])

with bic_tab:
    try:
        w_bic = _default_w_matrix()
    except Exception as e:
        st.error(f"Failed to load default W matrix: {e}")
        w_bic = None
    gen_tab_bic, sep_tab_bic = st.tabs(["Generation", "Separation"])
    with gen_tab_bic:
        _render_generation_tab(
            prefix="bic",
            w_np=w_bic,
            auto_res=auto_res,
            target_side=target_side,
            allow_resize=allow_resize,
            overlap=overlap,
            ckpt_dir=ckpt_dir,
            sample_steps=int(sample_steps),
            device=device,
            dim=int(dim),
            dim_mults_str=dim_mults_str,
            dropout=float(dropout),
        )
    with sep_tab_bic:
        _render_separation_tab(prefix="bic", w_np=w_bic)

with wssb_tab:
    st.subheader("WSSB organ selection")
    selected_organ = st.radio("Choose one organ", ["breast", "lung"], horizontal=True, key="wssb_organ_radio")
    w_wssb = None
    try:
        w_wssb = _load_wssb_w(selected_organ)
        st.success(f"W matrix loaded for organ: {selected_organ}")
    except Exception as e:
        st.error(f"W matrix loading failed: {e}")

    gen_tab_wssb, sep_tab_wssb = st.tabs(["Generation", "Separation"])
    with gen_tab_wssb:
        _render_generation_tab(
            prefix="wssb",
            w_np=w_wssb,
            auto_res=auto_res,
            target_side=target_side,
            allow_resize=allow_resize,
            overlap=overlap,
            ckpt_dir=ckpt_dir,
            sample_steps=int(sample_steps),
            device=device,
            dim=int(dim),
            dim_mults_str=dim_mults_str,
            dropout=float(dropout),
        )
    with sep_tab_wssb:
        _render_separation_tab(prefix="wssb", w_np=w_wssb)
