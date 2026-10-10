"""Training, validation (loss + RMSE), prediction and submission writing."""
import json
import os
import random
import zipfile

import cv2
import numpy as np
import torch
import torch.utils.data as data
import yaml
from tqdm import tqdm

from src.data import SEMDepthDataset, SEMTestDataset, prepare_sem_data
from src.model import DepthNet, depth_loss


def load_config(path, overrides=None):
    """Read a YAML config and apply non-None CLI overrides."""
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def set_seed(seed):
    """Seed python, numpy and torch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(cfg, device, checkpoint=None):
    """DepthNet on device (DataParallel on multi-GPU), optionally loaded from a checkpoint."""
    net = DepthNet(k_ordinal_class=cfg['k_ordinal_class'], model_name=cfg['model_name'])
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net.load_state_dict({k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()})
    net = net.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net


def train_one_epoch(net, loader, optimizer, device):
    """One pass over the loader; returns mean (loss, l1, ord)."""
    net.train()
    totals = np.zeros(3)
    for sem, depth, ordinal in tqdm(loader, desc='train', leave=False):
        sem, depth, ordinal = sem.to(device), depth.to(device), ordinal.to(device)
        optimizer.zero_grad()
        losses = depth_loss(*net(sem), depth, ordinal)
        losses[0].backward()
        optimizer.step()
        totals += [l.item() * len(sem) for l in losses]
    return {k: float(v) for k, v in zip(['loss', 'l1', 'ord'], totals / len(loader.dataset))}


@torch.no_grad()
def evaluate(net, loader, device, depth_max):
    """Mean (loss, l1, ord) and pixel RMSE in depth units (0-255 scale) on a labelled loader."""
    net.eval()
    totals, sq_err, n_pix = np.zeros(3), 0.0, 0
    for sem, depth, ordinal in tqdm(loader, desc='valid', leave=False):
        sem, depth, ordinal = sem.to(device), depth.to(device), ordinal.to(device)
        pred_depth, pred_ord = net(sem)
        totals += [l.item() * len(sem) for l in depth_loss(pred_depth, pred_ord, depth, ordinal)]
        sq_err += ((pred_depth - depth) * depth_max).pow(2).sum().item()
        n_pix += depth.numel()
    stats = {k: float(v) for k, v in zip(['loss', 'l1', 'ord'], totals / len(loader.dataset))}
    stats['rmse'] = float(np.sqrt(sq_err / max(n_pix, 1)))
    return stats


@torch.no_grad()
def predict(net, loader, device, output_depth_max):
    """Yield (file_name, uint8 depth map) with sigmoid output rescaled by output_depth_max."""
    net.eval()
    for sem, names in tqdm(loader, desc='predict', leave=False):
        pred_depth, _ = net(sem.to(device))
        maps = np.clip(np.round(pred_depth[:, 0].cpu().numpy() * output_depth_max), 0, 255).astype(np.uint8)
        yield from zip(names, maps)


def write_submission(predictions, out_dir):
    """Write depth PNGs named like the test SEM files and zip them flat into submission.zip."""
    png_dir = os.path.join(out_dir, 'submission')
    os.makedirs(png_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, 'submission.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for name, depth_map in predictions:
            path = os.path.join(png_dir, name)
            cv2.imwrite(path, depth_map)
            zf.write(path, arcname=name)
    return zip_path


def make_loaders(cfg):
    """Train (augmented, shuffled) and validation loaders from the sorted SEM split."""
    train_pairs, valid_pairs, _ = prepare_sem_data(cfg['data_dir'], cfg['val_ratio'])
    if not train_pairs:
        raise FileNotFoundError(f"No SEM images under {cfg['data_dir']}/simulation_data/SEM/*/*/")
    kw = dict(sem_max=cfg['sem_max'], depth_max=cfg['depth_max'], k_ordinal_class=cfg['k_ordinal_class'])
    train_loader = data.DataLoader(SEMDepthDataset(train_pairs, augment=True, **kw), batch_size=cfg['batch_size'],
                                   shuffle=True, num_workers=cfg['num_workers'])
    valid_loader = data.DataLoader(SEMDepthDataset(valid_pairs, **kw), batch_size=cfg['infer_batch_size'],
                                   shuffle=False, num_workers=cfg['num_workers'])
    return train_loader, valid_loader


def fit(cfg):
    """Train for cfg['epochs'], keep the checkpoint with the lowest validation loss; returns its path."""
    set_seed(cfg['seed'])
    device = get_device()
    os.makedirs(cfg['output_dir'], exist_ok=True)
    train_loader, valid_loader = make_loaders(cfg)
    net = build_model(cfg, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    best_path = os.path.join(cfg['output_dir'], 'best.pth')
    best_loss, history = float('inf'), {}
    for epoch in range(1, cfg['epochs'] + 1):
        train_stats = train_one_epoch(net, train_loader, optimizer, device)
        valid_stats = evaluate(net, valid_loader, device, cfg['depth_max']) if len(valid_loader.dataset) else train_stats
        history[epoch] = {'train': train_stats, 'valid': valid_stats}
        print(f"epoch {epoch}: train loss {train_stats['loss']:.4f} | valid loss {valid_stats['loss']:.4f} "
              f"rmse {valid_stats.get('rmse', float('nan')):.3f}")
        state = (net.module if isinstance(net, torch.nn.DataParallel) else net).state_dict()
        torch.save(state, os.path.join(cfg['output_dir'], 'last.pth'))
        if valid_stats['loss'] < best_loss:
            best_loss = valid_stats['loss']
            torch.save(state, best_path)
        with open(os.path.join(cfg['output_dir'], 'stats.json'), 'w') as f:
            json.dump(history, f, indent=1)
    return best_path


def run_prediction(cfg, checkpoint):
    """Predict every test/SEM image with a checkpoint and write submission.zip; returns the zip path."""
    device = get_device()
    _, _, test_pairs = prepare_sem_data(cfg['data_dir'], cfg['val_ratio'])
    if not test_pairs:
        raise FileNotFoundError(f"No test images under {cfg['data_dir']}/test/SEM/")
    loader = data.DataLoader(SEMTestDataset(test_pairs, sem_max=cfg['sem_max']), batch_size=cfg['infer_batch_size'],
                             shuffle=False, num_workers=cfg['num_workers'])
    net = build_model(cfg, device, checkpoint)
    return write_submission(predict(net, loader, device, cfg['output_depth_max']), cfg['output_dir'])
