"""Predict depth maps for data/test/SEM and write outputs/.../submission.zip."""
import argparse
import os

from src.engine import load_config, run_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', help='defaults to <output_dir>/best.pth')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--num_workers', type=int)
    args = vars(parser.parse_args())
    checkpoint = args.pop('checkpoint')
    cfg = load_config(args.pop('config'), args)
    checkpoint = checkpoint or os.path.join(cfg['output_dir'], 'best.pth')
    print(f'submission: {run_prediction(cfg, checkpoint)}')


if __name__ == '__main__':
    main()
