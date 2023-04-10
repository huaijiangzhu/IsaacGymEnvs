#!/bin/bash

for seed in 0 1 2 42
do
    python train.py wandb_activate=True seed=$seed
done