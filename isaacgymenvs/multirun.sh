#!/bin/bash

for seed in 42 0 1
do
    python train.py task.env.enable_location_qp=True wandb_activate=True seed=$seed
done

for seed in 42 0 1
do
    python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True seed=$seed
done