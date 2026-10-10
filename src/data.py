"""SEM/depth pairing, augmentation and PyTorch datasets for the DACON simulation data."""
import os
import re
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data


def depth_path_for(sem_path):
    """Map simulation_data/SEM/.../x_itr0.png to simulation_data/Depth/.../x.png (two SEMs share one depth)."""
    head, name = os.path.split(sem_path)
    head = re.sub(r'([\\/])SEM([\\/])', r'\1Depth\2', head)
    return os.path.join(head, re.sub(r'_itr\d+(\.png)$', r'\1', name))


def prepare_sem_data(data_dir, val_ratio=0.2):
    """Return (train, valid, test) lists of (sem_path, depth_path); sorted split, last val_ratio is validation."""
    sem_paths = sorted(glob(os.path.join(data_dir, 'simulation_data', 'SEM', '*', '*', '*.png')))
    pairs = [(p, depth_path_for(p)) for p in sem_paths]
    split = int(len(pairs) * (1 - val_ratio))
    test_paths = sorted(glob(os.path.join(data_dir, 'test', 'SEM', '*.png')))
    return pairs[:split], pairs[split:], [(p, '') for p in test_paths]


def ordinal_label(depth, step=10, n_bins=17):
    """Uniform ordinal bins: depth in (10k, 10(k+1)] -> k for k = 1..16, everything else -> 0."""
    edges = np.arange(1, n_bins + 1) * step
    label = np.zeros(depth.shape, dtype=np.float32)
    for k in range(n_bins - 1):
        label[(depth > edges[k]) & (depth <= edges[k + 1])] = k + 1
    return label


def _gaussian_blur(img, rng):
    sigma = rng.uniform(0, 3.0)
    return img if sigma < 1e-2 else cv2.GaussianBlur(img, (0, 0), sigma)


def _average_blur(img, rng):
    k = int(rng.integers(2, 8))
    return cv2.blur(img, (k, k))


def _median_blur(img, rng):
    return cv2.medianBlur(img, int(rng.choice([3, 5, 7, 9, 11])))


def _blur(img, rng):
    return [_gaussian_blur, _average_blur, _median_blur][rng.integers(3)](img, rng)


def _gaussian_noise(img, rng):
    noise = rng.normal(0, rng.uniform(0, 0.05 * 255), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _pixel_dropout(img, rng):
    keep = rng.random(img.shape) >= rng.uniform(0.01, 0.1)
    return (img * keep).astype(np.uint8)


def _linear_contrast(img, rng):
    alpha = rng.uniform(0.5, 2.0)
    return np.clip(128 + alpha * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)


PHOTOMETRIC_AUGS = [_blur, _gaussian_noise, _pixel_dropout, _linear_contrast]


def augment_pair(sem, depth, rng):
    """Random h/v flips on both images, then 0-4 photometric ops (random order) on the SEM only."""
    if rng.random() < 0.5:
        sem, depth = sem[:, ::-1], depth[:, ::-1]
    if rng.random() < 0.5:
        sem, depth = sem[::-1], depth[::-1]
    sem = np.ascontiguousarray(sem)
    n_ops = min(int(rng.integers(0, 6)), len(PHOTOMETRIC_AUGS))
    for idx in rng.permutation(len(PHOTOMETRIC_AUGS))[:n_ops]:
        sem = PHOTOMETRIC_AUGS[idx](sem, rng)
    return sem, np.ascontiguousarray(depth)


class SEMDepthDataset(data.Dataset):
    """Yields (sem, depth, ordinal) as 1xHxW float arrays; sem/depth are min-max scaled to [0, 1]."""

    def __init__(self, pairs, sem_max=212.0, depth_max=170.0, k_ordinal_class=17, augment=False):
        self.pairs = pairs
        self.sem_max, self.depth_max = sem_max, depth_max
        self.k_ordinal_class = k_ordinal_class
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sem_path, depth_path = self.pairs[idx]
        sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if self.augment:
            rng = np.random.default_rng(int(torch.randint(2 ** 31, (1,)).item()))  # follows torch seeding per worker/epoch
            sem, depth = augment_pair(sem, depth, rng)
        ordinal = ordinal_label(depth, n_bins=self.k_ordinal_class)
        return (sem[None].astype(np.float32) / self.sem_max,
                depth[None].astype(np.float32) / self.depth_max,
                ordinal[None].astype(np.int64))


class SEMTestDataset(data.Dataset):
    """Yields (sem, file_name) for submission inference."""

    def __init__(self, paths, sem_max=212.0):
        self.paths = [p[0] if isinstance(p, tuple) else p for p in paths]
        self.sem_max = sem_max

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sem = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return sem[None].astype(np.float32) / self.sem_max, os.path.basename(self.paths[idx])
