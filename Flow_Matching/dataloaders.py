import torch
import torchvision
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import pandas as pd
import os
import warnings
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import logging
import glob

class DataLoaders:
    def __init__(self,
                 dataset_name: str,
                 data_root: str,
                 batch_size_train: int,
                 batch_size_test: int,
                 dim_image: int,
                 num_workers: int = 4,
                 model_name: str = None,
                 he_root: str = None):
        self.dataset_name      = dataset_name
        self.data_root         = data_root          # path to ./dataHes
        self.batch_size_train  = batch_size_train
        self.batch_size_test   = batch_size_test
        self.dim_image         = dim_image
        self.num_workers       = num_workers        # can come from config
        self.model_name        = model_name
        self.he_root           = he_root

    def load_data(self):

        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = './data/celeba/img_align_celeba/'
            partition_csv = './data/celeba/list_eval_partition.csv'

            # Datasets
            train_dataset = CelebADataset(
                img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(
                img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(
                img_dir, partition_csv, partition=2, transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'celebahq':

            transform = v2.Compose([
                v2.Resize(256),
                v2.ToTensor(),         # Convert images to PyTorch tensor
            ])

            test_dir = './data/celebahq/test/'
            test_dataset = CelebAHQDataset(
                test_dir, batchsize=self.batch_size_test, transform=transform)
            train_loader = None
            val_loader = None
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # transform = False
            img_dir_test = './data/afhq_cat/test/cat/'
            img_dir_val = './data/afhq_cat/val/cat/'
            img_dir_train = './data/afhq_cat/train/cat/'
            test_dataset = AFHQDataset(
                img_dir_test, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(
                img_dir_val, batchsize=self.batch_size_test, transform=transform)
            train_dataset = AFHQDataset(
                img_dir_train, batchsize=self.batch_size_test, transform=transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)
        elif self.dataset_name == 'hes':
            # DDPM pipeline in FM
            from torchvision import transforms as T

            # Training: random flip + crop + normalization
            from torchvision import transforms as T
        
            # resize to 256×256, then apply flip augmentation
            train_transform = T.Compose([
                T.Resize((self.dim_image, self.dim_image)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

            # Validation/Test: center crop + normalization
            # for validation/test, only resize + normalization
            eval_transform = T.Compose([
                T.Resize((self.dim_image, self.dim_image)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

            use_paired_he_hes = self.model_name in ("fm_cond_s_he", "fm_cond_ot_s_he")
            if use_paired_he_hes:
                if not self.he_root:
                    raise ValueError(
                        "he_root is required for conditional models fm_cond_s_he / fm_cond_ot_s_he"
                    )
                train_dataset = HESPairedDataset(
                    hes_root=self.data_root,
                    he_root=self.he_root,
                    split='train',
                    transform_hes=train_transform,
                    transform_he=train_transform,
                )
                val_dataset = HESPairedDataset(
                    hes_root=self.data_root,
                    he_root=self.he_root,
                    split='val',
                    transform_hes=eval_transform,
                    transform_he=eval_transform,
                )
                test_dataset = HESPairedDataset(
                    hes_root=self.data_root,
                    he_root=self.he_root,
                    split='test',
                    transform_hes=eval_transform,
                    transform_he=eval_transform,
                )
            else:
                train_dataset = HESDataset(self.data_root, split='train', transform=train_transform)
                val_dataset   = HESDataset(self.data_root, split='val',   transform=eval_transform)
                test_dataset  = HESDataset(self.data_root, split='test',  transform=eval_transform)

            # DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                num_workers=self.num_workers,      # or cfg.dataset.num_workers
                collate_fn=custom_collate  # if needed, otherwise remove
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=custom_collate
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=custom_collate
            )
        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders

class HESDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # search in dataHes/train/*_patches/*.*
        pattern = os.path.join(root_dir, split, '*_patches', '*.*')
        self.img_paths = glob.glob(pattern)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No image found with pattern {pattern}")
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return image AND a dummy label so loops like "for x,_ in loader" keep working
        return img, 0


class HESPairedDataset(Dataset):
    """Paired HE/HES dataset aligned by split + relative patch path."""

    def __init__(self, hes_root, he_root, split='train', transform_hes=None, transform_he=None):
        hes_pattern = os.path.join(hes_root, split, '*_patches', '*.*')
        hes_paths = sorted(glob.glob(hes_pattern))
        if len(hes_paths) == 0:
            raise RuntimeError(f"No HES image found with pattern {hes_pattern}")

        hes_split_root = os.path.join(hes_root, split)
        he_split_root = os.path.join(he_root, split)
        pairs = []
        missing = 0
        for hes_path in hes_paths:
            rel = os.path.relpath(hes_path, hes_split_root)
            he_path = os.path.join(he_split_root, rel)
            if os.path.exists(he_path):
                pairs.append((he_path, hes_path))
            else:
                missing += 1

        if len(pairs) == 0:
            raise RuntimeError(
                f"No HE/HES pair found for split='{split}'. "
                f"hes_root={hes_root}, he_root={he_root}"
            )
        if missing > 0:
            logging.warning(
                "HESPairedDataset(%s): %d HES files without HE pair were ignored",
                split, missing
            )

        self.pairs = pairs
        self.transform_hes = transform_hes
        self.transform_he = transform_he if transform_he is not None else transform_hes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        he_path, hes_path = self.pairs[idx]
        he_img = Image.open(he_path).convert('RGB')
        hes_img = Image.open(hes_path).convert('RGB')

        if self.transform_he:
            he_img = self.transform_he(he_img)
        if self.transform_hes:
            hes_img = self.transform_hes(hes_img)
        return he_img, hes_img, 0
class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load partition file
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CelebAHQDataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.files = os.listdir(data_dir)
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = 2 * image - 1
        image = image.float()

        return image, 0


class AFHQDataset(Dataset):
    """AFHQ Cat dataset."""

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        self.files = os.listdir(img_dir)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


def custom_collate(batch):
    # Filter out invalid values
    batch = [
        x for x in batch
        if x is not None and (not isinstance(x, tuple) or len(x) == 0 or x[0] is not None)
    ]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)
