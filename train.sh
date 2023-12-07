#!/bin/bash

taskset -c 21-40 python samsung_trainer.py --gpu '3, 4' --nr_epochs 50 --ordinal_class 17 --lr 1.0e-3 --seed 5
taskset -c 21-40 python samsung_tester.py --gpu '3, 4' --ordinal_class 17 --seed 5
