#!/usr/bin/env python3
# coding=utf-8
"python Flow_Matching/image_generation/generate_HES.py     --cfg     config/main_config.yaml     --model   model/hes/ot/model_85.pt     --out_dir out/high_quality_single     --n_samples 1     --batch_size 1     --seed    50     --use_sde     --sample_N 8000"

import os
import argparse
import random
import numpy as np
import torch
import torchvision.utils as vutils

from Flow_Matching.utils import load_cfg_from_cfg_file, merge_cfg_from_list, define_model
from Flow_Matching.image_generation.sde_lib import RectifiedFlow
from Flow_Matching.image_generation.sampling import get_rectified_flow_sampler
from Flow_Matching.image_generation.models.utils import get_model_fn
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def build_generator(cfg, model_path, device, use_sde, ode_method, sigma_var, sample_N):
    """
    Load the model and instantiate the RectifiedFlow sampler.
    Returns (model, sampler).
    """
    # Load model
    model, _ = define_model(cfg)
    state = torch.load(model_path, map_location=device)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device).eval()

    # Instancie la flow/SDE et le sampler
    sde = RectifiedFlow(
        init_type='gaussian',
        noise_scale=1.0,
        reflow_flag=True,
        reflow_t_schedule='uniform',
        reflow_loss='l2',
        use_ode_sampler=ode_method,
        sigma_var=sigma_var,
        ode_tol=1e-5,
        sample_N=sample_N
    )
    shape = (1, cfg.num_channels, cfg.image_size, cfg.image_size)
    inverse_scaler = lambda x: (x + 1.0) / 2.0
    sampler = get_rectified_flow_sampler(sde, shape, inverse_scaler, device)
    return model, sampler

def main():
    parser = argparse.ArgumentParser(
        description="Generate images with your RectifiedFlow model"
    )
    parser.add_argument('--cfg',        type=str, required=True,
                        help='path to the YAML config file')
    parser.add_argument('--model',      type=str, required=True,
                        help='checkpoint .pt file to load')
    parser.add_argument('--out_dir',    type=str, required=True,
                        help='output directory for generated samples')
    parser.add_argument('--n_samples',  type=int, default=1000,
                        help='total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images per sampler call')
    parser.add_argument('--use_sde',    action='store_true',
                        help='enable stochastic diffusion (otherwise ODE mode)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--sample_N',   type=int, default=1000,
                        help='number of SDE/flow steps')
    args = parser.parse_args()

    # Setup device and seed
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Load config and adjust key attributes
    cfg = load_cfg_from_cfg_file(args.cfg)
    cfg = merge_cfg_from_list(cfg, ["dataset", "hes", "model", "ot"])
    cfg.image_size, cfg.dim_image, cfg.num_channels = 256, 256, 3

    # Instantiate model + sampler (keep direct model handle here)
#    sigma_var  = 1.0 if args.use_sde else 0.0
#    ode_method = 'euler'
#    model, sampler = build_generator(
#        cfg, args.model, device,
#        args.use_sde, ode_method,
#        sigma_var, sample_N=args.sample_N
#    )
    # 1) Instantiate model manually
    model, _ = define_model(cfg)
    state = torch.load(args.model, map_location=device)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device).eval()

    # 2) Rebuild SDE + shape + inverse_scaler
    from Flow_Matching.image_generation.sde_lib import RectifiedFlow
    sigma_var  = 1.0 if args.use_sde else 0.0
    ode_method = 'euler'
    sde = RectifiedFlow(
        init_type='gaussian',
        noise_scale=1.0,
        reflow_flag=True,
        reflow_t_schedule='uniform',
        reflow_loss='l2',
        use_ode_sampler=ode_method,
        sigma_var=sigma_var,
        ode_tol=1e-5,
        sample_N=args.sample_N
    )
    shape = (args.batch_size, cfg.num_channels, cfg.image_size, cfg.image_size)
    inverse_scaler = lambda x: (x + 1.0) / 2.0

    # 3) Define local sampler drawing new noise at each call via sde.get_z0
    sampler = get_rectified_flow_sampler(
    sde=sde,
    shape=shape,
    inverse_scaler=inverse_scaler,
    device=device
)
    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)
    total      = args.n_samples
    per_batch  = args.batch_size
    batches    = int(np.ceil(total / per_batch))
    count      = 0

    print(f"-> Generating {total} images in {args.out_dir}")
    print(f"   Mode   : {'SDE' if args.use_sde else 'ODE'}")
    print(f"   Steps  : {args.sample_N}")
    print(f"   Batch  : {per_batch}")
    print(f"   Seed   : {args.seed}\n")

    # Generation loop
    for b in range(batches):
        with torch.no_grad():
            # Pass the actual model to sampler
            samples, nfe = sampler(model)
        imgs = samples.clamp(0, 1)  # ensure values stay in [0, 1]
        for i in range(imgs.size(0)):
            if count >= total:
                break
            filename = f"sample_{count:05d}.png"
            path = os.path.join(args.out_dir, filename)
            vutils.save_image(imgs[i:i+1],
                              path,
                              normalize=False,
                              scale_each=False)
            print(f"[{count+1}/{total}] saved → {filename} (NFE={nfe})")
            count += 1

    print(f"\nGeneration complete: {count} images saved in '{args.out_dir}'")

if __name__ == '__main__':
    main()
