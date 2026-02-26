# dataloaders.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class HESPaireDataset(Dataset):
    """
    Dataset loading paired (HES, HE) samples for a given split.
    Assumes data_root contains two subfolders:
      - data_root/HES/{train,val,test}/..._patches/...
      - data_root/HE/{train,val,test}/..._patches/...
    """

    def __init__(self, data_root, split='test',
                 dim_image=256,
                 transform_hes=None, transform_he=None):
        self.hes_paths = sorted(
            glob.glob(os.path.join(data_root, 'HES',  split, '*_patches', '*.*'))
        )
        self.he_paths  = sorted(
            glob.glob(os.path.join(data_root, 'HE',   split, '*_patches', '*.*'))
        )
        if len(self.hes_paths) != len(self.he_paths):
            raise RuntimeError(
                f"HES count ({len(self.hes_paths)}) != HE count ({len(self.he_paths)})"
            )

        # Default transforms when none are provided
        self.transform_hes = transform_hes or T.Compose([
            T.Resize((dim_image, dim_image)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.transform_he = transform_he or T.Compose([
            T.Resize((dim_image, dim_image)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.hes_paths)

    def __getitem__(self, idx):
        hes_img = Image.open(self.hes_paths[idx]).convert('RGB')
        he_img  = Image.open(self.he_paths[idx]).convert('RGB')
        hes = self.transform_hes(hes_img)
        he  = self.transform_he(he_img)
        return hes, he

def get_pnp_loader(data_root: str,
                   split: str = 'test',
                   batch_size: int = 1,
                   dim_image: int = 256,
                   num_workers: int = 4,
                   shuffle: bool = False) -> DataLoader:
    """
    Return a DataLoader over (HES, HE) pairs for the PnP-FM stage.
    """
    dataset = HESPaireDataset(
        data_root=data_root,
        split=split,
        dim_image=dim_image
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
