"""End-to-end smoke test on a tiny fake DACON dataset: train steps, validation RMSE, submission zip."""
import os
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import prepare_sem_data  # noqa: E402
from src.engine import (build_model, evaluate, load_config, make_loaders, predict,  # noqa: E402
                        set_seed, train_one_epoch, write_submission)
from src.data import SEMTestDataset  # noqa: E402

H, W = 72, 48  # DACON SEM / depth image size


def make_fake_dataset(root, n_cases=2, n_depth=3, n_test=4, seed=0):
    """simulation_data/{SEM,Depth}/Case_i/80/*.png (2 SEMs per depth) + test/SEM/*.png."""
    rng = np.random.default_rng(seed)
    for c in range(1, n_cases + 1):
        sem_dir = os.path.join(root, 'simulation_data', 'SEM', f'Case_{c}', '80')
        depth_dir = os.path.join(root, 'simulation_data', 'Depth', f'Case_{c}', '80')
        os.makedirs(sem_dir), os.makedirs(depth_dir)
        for i in range(n_depth):
            name = f'20201001_202940_NE142400C_RAE01_1_S01_M0005-01MS_{c}{i}'
            depth = rng.integers(0, 171, (H, W), dtype=np.uint8)
            cv2.imwrite(os.path.join(depth_dir, f'{name}.png'), depth)
            for itr in range(2):
                sem = np.clip(depth * 1.2 + rng.normal(0, 5, (H, W)), 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(sem_dir, f'{name}_itr{itr}.png'), sem)
    test_dir = os.path.join(root, 'test', 'SEM')
    os.makedirs(test_dir)
    for i in range(n_test):
        cv2.imwrite(os.path.join(test_dir, f'{i:05d}.png'), rng.integers(0, 213, (H, W), dtype=np.uint8))


def main():
    base = os.environ.get('SMOKE_TMP') or tempfile.gettempdir()
    root = tempfile.mkdtemp(prefix='samsung_smoke_', dir=base)
    try:
        data_dir = os.path.join(root, 'data')
        make_fake_dataset(data_dir)
        cfg = load_config(os.path.join(ROOT, 'config.yaml'), {
            'data_dir': data_dir, 'output_dir': os.path.join(root, 'outputs'),
            'batch_size': 4, 'infer_batch_size': 4, 'num_workers': 0})

        train_pairs, valid_pairs, test_pairs = prepare_sem_data(data_dir, cfg['val_ratio'])
        assert len(train_pairs) + len(valid_pairs) == 12 and len(test_pairs) == 4
        assert all(os.path.exists(d) for _, d in train_pairs + valid_pairs), 'SEM/Depth pairing failed'

        set_seed(cfg['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, valid_loader = make_loaders(cfg)
        sem, depth, ordinal = next(iter(train_loader))
        assert sem.shape == depth.shape == ordinal.shape == (4, 1, H, W)
        assert 0 <= ordinal.min() and ordinal.max() < cfg['k_ordinal_class']

        net = build_model(cfg, device)
        pred_depth, pred_ord = net(sem.to(device))
        assert pred_depth.shape == (4, 1, H, W) and pred_ord.shape == (4, cfg['k_ordinal_class'], H, W)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
        train_stats = train_one_epoch(net, train_loader, optimizer, device)
        assert all(np.isfinite(v) for v in train_stats.values()), train_stats
        valid_stats = evaluate(net, valid_loader, device, cfg['depth_max'])
        assert np.isfinite(valid_stats['rmse']) and valid_stats['rmse'] >= 0, valid_stats
        print('train', train_stats, '\nvalid', valid_stats)

        test_loader = torch.utils.data.DataLoader(SEMTestDataset(test_pairs, cfg['sem_max']), batch_size=4)
        zip_path = write_submission(predict(net, test_loader, device, cfg['output_depth_max']), cfg['output_dir'])
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            assert names == [f'{i:05d}.png' for i in range(4)], names
            img = cv2.imdecode(np.frombuffer(zf.read(names[0]), np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape == (H, W) and img.dtype == np.uint8
        print(f'OK: submission with {len(names)} depth maps of shape {img.shape}')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
