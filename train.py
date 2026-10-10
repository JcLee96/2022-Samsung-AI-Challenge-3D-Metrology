"""Train the depth network and (optionally) write the test submission with the best checkpoint."""
import argparse

from src.engine import fit, load_config, run_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--predict', action='store_true', help='write submission.zip after training')
    args = vars(parser.parse_args())
    cfg = load_config(args.pop('config'), args)
    best = fit(cfg)
    print(f'best checkpoint: {best}')
    if cfg['predict']:
        print(f'submission: {run_prediction(cfg, best)}')


if __name__ == '__main__':
    main()
