#!/usr/bin/env python3
import os
import time
import torch
from tqdm import tqdm

# --- monkey-patch torchdiffeq.odeint_adjoint for fixed RK4 ---
import torchdiffeq
from torchdiffeq import odeint
_orig_adjoint = torchdiffeq.odeint_adjoint

def _patched_odeint_adjoint(func, y0, t, *args, **kwargs):
    # Remove any existing config to avoid duplicates
    kwargs.pop('method', None)
    kwargs.pop('rtol', None)
    kwargs.pop('atol', None)
    # Force RK4
    return odeint(func, y0, t, method='rk4', **kwargs)

torchdiffeq.odeint_adjoint = _patched_odeint_adjoint
# --------------------------------------------------------------------

# Imports from this repository
from main import parse_args
from Flow_Matching.utils import define_model
from Flow_Matching.dataloaders import DataLoaders
from Flow_Matching.train_flow_matching_indep import FLOW_MATCHING as FLOW_MATCHING_INDEP

def main():
    # 1) Load config and device
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 2) Instantiate model
    model, _ = define_model(args)

    # 3) Prepare evaluator and load test set
    fm = FLOW_MATCHING_INDEP(model, device, args)
    data = DataLoaders(
        args.dataset,
        args.data_root,
        batch_size_train=16,
        batch_size_test=64,
        dim_image=args.dim_image,
        num_workers=args.num_workers
    ).load_data()
    fm.full_train_set = data['test']
    n_test = 5000  # number of images (ex. 3146)

    # 4) Paths
    ckpt_dir = "put path of your data here"
    log_file = "put path of your data here"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 5) Loop over checkpoints with progress display
    epochs = list(range(5, 101, 5))
    total = len(epochs)
    pbar = tqdm(total=total,
                desc="FID Evaluation",
                unit="epoch",
                ncols=80)

    for idx, epoch in enumerate(epochs, 1):
        print(f"[{idx}/{total}] -> Evaluating checkpoint model_{epoch}.pt")
        ckpt_path = os.path.join(ckpt_dir, f"model_{epoch}.pt")
        if not os.path.isfile(ckpt_path):
            print(f"   ⚠️  Checkpoint missing : {ckpt_path}")
            pbar.update(1)
            continue

        # 5.1) Model loading
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)

        # 5.2) Compute FID
        print("   [RUN] Computing FID...")
        t0 = time.time()
        fid = fm.compute_fast_fid(n_test,n_iter=100)
        dt = time.time() - t0

        # 5.3) Log to disk
        with open(log_file, "a") as f:
            f.write(f"Epoch: {epoch} : FID: {fid:.4f}\n")

        # 5.4) Reporting console
        print(f"   ✅  Epoch {epoch:>3} | FID: {fid:.4f} | duration: {dt:5.1f}s")

        # 5.5) Progress bar update
        pbar.update(1)
        pbar.set_postfix({"FID": f"{fid:.3f}", "t": f"{dt:.1f}s"})

    pbar.close()


if __name__ == "__main__":
    main()
