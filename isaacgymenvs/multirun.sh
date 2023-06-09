#!/bin/bash

for seed in 42 0 1
do
    python train.py wandb_activate=True seed=$seed
done