<div align="center">

# 2022 Samsung AI Challenge — 3D Metrology

**Final Winner · 8th place · team 웅싸**

[![Competition](https://img.shields.io/badge/DACON-Competition-1f6feb)](https://dacon.io/competitions/official/235954/overview/description)
[![Leaderboard](https://img.shields.io/badge/DACON-Leaderboard-2ea043)](https://dacon.io/competitions/official/235954/leaderboard)
[![Solution](https://img.shields.io/badge/DACON-Solution%20post-f0883e)](https://dacon.io/competitions/official/235954/talkboard/406931)
[![Data](https://img.shields.io/badge/DACON-Dataset-8250df)](https://dacon.io/competitions/official/235954/data)

</div>

Predicts the depth map of a semiconductor structure from a single 72x48 SEM image, trained only on the simulation data.

## Approach

- **Model**: EfficientNet-B0 encoder (GroupNorm instead of BatchNorm, stride-1 stem so the small 72x48 input keeps resolution) + ASPP (image pooling and dilations 1/3/6/9) + U-Net style decoder of transposed convs and residual blocks with dense (concatenated) skips. Trained from scratch.
- **Two heads, one loss**: a sigmoid depth head trained with L1 on depth scaled by 1/170, plus an ordinal regression head (17 uniform depth bins of width 10, DORN-style ordinal loss) as an auxiliary task. Total loss = L1 + ordinal.
- **Data**: SEM scaled by 1/212; random h/v flips (applied to SEM and depth) and 0–4 random photometric ops on the SEM (Gaussian/average/median blur, Gaussian noise, pixel dropout, linear contrast). The sorted simulation list is split 80/20 into train/validation.
- **Training**: Adam, lr 1e-3, batch 512, 50 epochs, seed 5; the checkpoint with the lowest validation loss is used. Submission depth = sigmoid output x 212.

## Project structure

```
├── config.yaml          # all hyperparameters and paths (final setting)
├── train.py             # train (+ --predict to write the submission)
├── predict.py           # checkpoint -> submission.zip
├── train.sh             # final run: python train.py --predict
├── requirements.txt
├── src/
│   ├── data.py          # SEM/Depth pairing, split, augmentation, ordinal labels, datasets
│   ├── backbone.py      # modified EfficientNet encoder (GroupNorm, stride-1 stem)
│   ├── model.py         # DepthNet (encoder + ASPP + decoder) and L1 + ordinal losses
│   └── engine.py        # train / validate (loss, RMSE) / predict loops, submission writer
└── tests/
    └── test_smoke.py    # end-to-end run on a tiny synthetic dataset
```

## Environment

```bash
git clone https://github.com/JcLee96/2022-Samsung-AI-Challenge-3D-Metrology.git
cd 2022-Samsung-AI-Challenge-3D-Metrology

conda create -n samsung3d python=3.9 -y && conda activate samsung3d
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt
```

## Dataset

Download the data from the [DACON data page](https://dacon.io/competitions/official/235954/data) and unpack it under `data/`:

```
data/
├── simulation_data/
│   ├── SEM/Case_*/<depth>/<name>_itr0.png, <name>_itr1.png   # two SEM images per depth map
│   └── Depth/Case_*/<depth>/<name>.png
└── test/
    └── SEM/*.png
```

Only `simulation_data/` (training) and `test/SEM/` (submission) are used. Each SEM image is paired with the depth map of the same name without the `_itrN` suffix.

## Usage

```bash
# train with the final setting and write outputs/final/submission.zip
python train.py --config config.yaml --predict      # or: sh train.sh

# predict from a checkpoint (defaults to <output_dir>/best.pth)
python predict.py --config config.yaml --checkpoint outputs/final/best.pth
```

`outputs/final/` holds `best.pth`, `last.pth`, `stats.json` (per-epoch train/valid loss and validation RMSE) and `submission.zip` — the predicted depth PNGs named exactly like the test SEM files, zipped flat as DACON expects. Any config value such as `--data_dir`, `--epochs`, `--batch_size` or `--num_workers` can be overridden from the command line.

## Smoke test

Builds a tiny synthetic dataset in the DACON layout, runs a training epoch, validation (RMSE) and writes a submission zip:

```bash
python tests/test_smoke.py
```

## License
Released under the [MIT License](LICENSE).
