#!/bin/bash
# Final setting: 50 epochs, Adam 1e-3, batch 512, seed 5, then write outputs/final/submission.zip
python train.py --config config.yaml --predict
