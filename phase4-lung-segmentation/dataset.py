import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random

# ── constants ──────────────────────────────────────────────────────────────
DATA_ROOT  = "/teamspace/studios/this_studio/data/montgomery/Montgomery/MontgomerySet"
IMG_SIZE   = 512    # chest X-rays have more detail than pet photos — use 512
BATCH_SIZE = 8      # smaller batch than before — images are larger (512x512)
NUM_WORKERS = 2

# ── dataset ────────────────────────────────────────────────────────────────
class MontgomeryDataset(Dataset):
    """
    Montgomery County Chest X-Ray dataset for lung segmentation.
    Input:  grayscale chest X-ray (1 channel)
    Output: binary lung mask — 1 = lung tissue, 0 = everything else
    
    Each image has TWO mask files (left lung, right lung).
    We combine them into one binary mask — any pixel that is lung
    in either mask gets labeled as foreground.
    """
    def __init__(self, image_paths, augment=False):
        # image_paths: list of full paths to each CXR image
        self.image_paths = image_paths
        self.augment = augment

        # build mask paths from image paths
        # image: CXR_png/MCUCXR_0001_0.png
        # left:  ManualMask/leftMask/MCUCXR_0001_0.png
        # right: ManualMask/rightMask/MCUCXR_0001_0.png
        self.left_mask_paths  = [
            p.replace("CXR_png", "ManualMask/leftMask") for p in image_paths
        ]
        self.right_mask_paths = [
            p.replace("CXR_png", "ManualMask/rightMask") for p in image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ── load image ──────────────────────────────────────────────
        # convert to grayscale ('L') — chest X-rays are single channel
        # no RGB information, just intensity (how much X-ray was absorbed)
        image = Image.open(self.image_paths[idx]).convert("L")

        # ── load and combine masks ───────────────────────────────────
        # load left and right lung masks separately
        left_mask  = np.array(Image.open(self.left_mask_paths[idx]).convert("L"))
        right_mask = np.array(Image.open(self.right_mask_paths[idx]).convert("L"))

        # combine: pixel is lung if it appears in EITHER left OR right mask
        # np.maximum takes the element-wise max — equivalent to logical OR
        # for binary masks: max(0,0)=0, max(255,0)=255, max(0,255)=255
        combined_mask = np.maximum(left_mask, right_mask)
        mask = Image.fromarray(combined_mask)

        # ── resize ──────────────────────────────────────────────────
        # bilinear for image (smooth interpolation between pixels)
        image = TF.resize(image, [IMG_SIZE, IMG_SIZE])
        # nearest for mask — preserve exact 0/255 boundary, no fractional values
        mask  = TF.resize(mask,  [IMG_SIZE, IMG_SIZE], interpolation=Image.NEAREST)

        # ── augmentation ────────────────────────────────────────────
        # same rule as before: apply identical transform to image AND mask
        # chest X-rays: only horizontal flip makes clinical sense
        # vertical flip would produce an upside-down X-ray (never seen in training)
        # strong rotations also unrealistic — keep small angles only
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask,  angle)

        # ── image → tensor ──────────────────────────────────────────
        # to_tensor: PIL grayscale → [1, H, W] float tensor, range [0,1]
        image = TF.to_tensor(image)   # shape: [1, 512, 512]

        # normalize with chest X-ray appropriate stats
        # 0.5 mean/std centers the distribution around 0
        # (can't use ImageNet stats — those are for 3-channel RGB images)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # ── mask → binary tensor ─────────────────────────────────────
        # mask pixel values: 255 = lung, 0 = background
        # convert to float and scale to 0/1 by dividing by 255
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        mask = (mask > 127).float()   # threshold: anything above halfway = lung
        mask = mask.unsqueeze(0)      # [H, W] → [1, H, W] to match model output shape

        return image, mask


# ── dataloaders ────────────────────────────────────────────────────────────
def get_dataloaders():
    """
    Builds train, val, and test dataloaders for Montgomery dataset.
    138 total images — small dataset, so we use 80/10/10 split.
    """
    # gather all image paths
    image_dir = os.path.join(DATA_ROOT, "CXR_png")
    all_images = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    print(f"Total images found: {len(all_images)}")

    # verify masks exist for every image — catches path mismatches early
    for path in all_images:
        left  = path.replace("CXR_png", "ManualMask/leftMask")
        right = path.replace("CXR_png", "ManualMask/rightMask")
        assert os.path.exists(left),  f"Missing left mask:  {left}"
        assert os.path.exists(right), f"Missing right mask: {right}"

    # shuffle then split: 80% train, 10% val, 10% test
    # fixed seed for reproducibility — same split every run
    random.seed(42)
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    train_paths = all_images[:train_end]
    val_paths   = all_images[train_end:val_end]
    test_paths  = all_images[val_end:]

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = MontgomeryDataset(train_paths, augment=True)
    val_ds   = MontgomeryDataset(val_paths,   augment=False)
    test_ds  = MontgomeryDataset(test_paths,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader