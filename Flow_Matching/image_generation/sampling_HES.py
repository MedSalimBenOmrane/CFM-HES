#!/usr/bin/env python3
# coding=utf-8

import os, random, numpy as np, torch
import torchvision.utils as vutils

from Flow_Matching.utils import load_cfg_from_cfg_file, merge_cfg_from_list, define_model
from Flow_Matching.image_generation.sde_lib import RectifiedFlow
from Flow_Matching.image_generation.sampling import get_rectified_flow_sampler

# --- 1) SETTINGS ---
MODEL_PATH  = "put path of your data here"
PROJECT_ROOT= os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG_PATH    = os.path.join(PROJECT_ROOT, "config", "main_config.yaml")
OUT_DIR     = os.path.join(os.path.dirname(MODEL_PATH), "latest_sampling")
os.makedirs(OUT_DIR, exist_ok=True)

USE_SDE    = True             # True -> stochastic diffusion, False -> deterministic flow
SIGMA_VAR  = 1.0 if USE_SDE else 0.0
ODE_METHOD = 'euler'          # keep Euler: handles noisy/non-noisy modes
SEEDS      = [0, 1234, 2025, 4000]

# --- 2) CONFIG & MODEL ---
cfg = load_cfg_from_cfg_file(CFG_PATH)
cfg = merge_cfg_from_list(cfg, ["dataset", "hes", "model", "ot"])
cfg.image_size, cfg.dim_image, cfg.num_channels = 256, 256, 3

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

model, _ = define_model(cfg)
state     = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    try:    model.load_state_dict(state)
    except: model = state
model.to(device).eval()
print(f"[INFO] Model loaded from {MODEL_PATH}")

# --- 3) SDE & SAMPLER INSTANTIATION ---
sde = RectifiedFlow(
    init_type='gaussian',
    noise_scale=1.0,
    reflow_flag=True,
    reflow_t_schedule='uniform',
    reflow_loss='l2',
    use_ode_sampler=ODE_METHOD,  # 'euler' here
    sigma_var=SIGMA_VAR,         # 0.0 or >0.0
    ode_tol=1e-5,
    sample_N=1000                # 300 -> 1000 or 2000 for finer samples
)

shape           = (1, cfg.num_channels, cfg.image_size, cfg.image_size)
inverse_scaler  = lambda x: (x + 1.0) / 2.0
sampler         = get_rectified_flow_sampler(sde, shape, inverse_scaler, device)

# --- 4) MULTI-SEED LOOP ---
for seed in SEEDS:
    seed_everything(seed)
    with torch.no_grad():
        sample, nfe = sampler(model)
    mode = "SDE" if USE_SDE else "ODE"
    out  = os.path.join(OUT_DIR, f"{mode.lower()}_seed_{seed}.png")
    vutils.save_image(sample, out, nrow=1, normalize=False, scale_each=False)
    print(f"✅ {mode} seed={seed} (NFE={nfe}) → {out}")
