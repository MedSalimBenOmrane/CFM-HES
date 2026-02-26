"""
cd "put path of your data here"

PYTHONPATH="put path of your data here:$PYTHONPATH" \
python main.py --opts \
  root "put path of your data here" \
  train True eval False dataset hes method pnp_flow model fm_cond_ot_s_he \
  data_root "put path of your data here" \
  batch_size_train 16 batch_size_test 16 num_workers 1 num_epoch 100 \
  lr 0.00005 dim 128 dim_mults "(1,1,2,2,4)" dropout 0.0 dim_image 256 \
  sample_steps 50 loss_l1_weight 1.0 loss_l2_weight 1.0 ema_decay 0.999 \
  ckpt_dir "put path of your data here"

"""
import os
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
# imports
from Flow_Matching.train_flow_matching_indep import FLOW_MATCHING as FLOW_MATCHING_INDEP
from Flow_Matching.utils import load_cfg_from_cfg_file, merge_cfg_from_list
from Flow_Matching.dataloaders import DataLoaders
from Flow_Matching.train_cond_flow_matching_indep  import FMCondSfromHE_Aymen, build_unet_fm_cond_s_he
from Flow_Matching.train_cond_flow_matching_ot import FMCondSfromHE_OT
from Flow_Matching.utils import define_model
import warnings
warnings.filterwarnings("ignore", module="matplotlib\\..*")

torch.cuda.empty_cache()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('./' + 'config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + \
        'config/dataset_config/{}.yaml'.format(
            cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_cfg = {}
    method_name = getattr(cfg, "method", None)
    method_config_file = None
    if method_name is not None:
        method_config_file = cfg.root + \
            'config/method_config/{}.yaml'.format(
                method_name)
        if os.path.isfile(method_config_file):
            method_cfg = load_cfg_from_cfg_file(method_config_file)
            cfg.update(method_cfg)

    if args.opts is not None:
        # override config with command line input
        cfg = merge_cfg_from_list(cfg, args.opts)

    # for all keys in the method config file, create a dictionary {key: value} in the cfg object cfg.dict_cfg_method
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    # attach device to args so define_model() can use it
    args.device = device
    if getattr(args, "model", None) in ("fm_cond_s_he", "fm_cond_ot_s_he"):
        model = build_unet_fm_cond_s_he(args, device)  # in=3 [S_t,H,E], out=1
        state = None
    else:
        (model, state) = define_model(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    

    if args.train:
        args.batch_size = args.batch_size_train
        print('Training...')
        data_loaders = DataLoaders(
    args.dataset,
    args.data_root,
    args.batch_size_train,
    args.batch_size_test,
    args.dim_image, 
    args.num_workers,
    model_name=getattr(args, "model", None),
    he_root=getattr(args, "he_root", None),
).load_data()
        if args.model == "fm_cond_s_he":
            # conditional trainer (HE ➜ S)
            generative_method = FMCondSfromHE_Aymen(model, device, args)
        elif args.model == "fm_cond_ot_s_he":
            # OT conditional trainer (HE ➜ S)
            generative_method = FMCondSfromHE_OT(model, device, args)
        elif args.model == "fm_indep":
            # independent coupling (standard FM)
            generative_method = FLOW_MATCHING_INDEP(model, device, args)
        else:
            raise ValueError(
                "Model not implemented yet: choose one of 'fm_indep', 'fm_cond_s_he', 'fm_cond_ot_s_he'")
        generative_method.train(data_loaders)
        print('Training done!')

    if args.eval:
        raise NotImplementedError(
            "Eval/inverse-problem pipeline was removed. This repo is now focused on FM/CFM training only."
        )


if __name__ == "__main__":
    main()
