# #!/bin/bash

# for seed in 42 0 1
# do
#     python train.py task.env.enable_location_qp=True wandb_activate=True seed=$seed
# done

# for seed in 42 0 1
# do
#     python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True seed=$seed
# done

# for damping in 3.0 4.0 5.0 6.0
# do
#     python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True task.env.ftip_damping=$damping
# done

# for scale in 10.0 15.0 20.0 25.0
# do
#     python train.py task.env.enable_location_qp=True wandb_activate=True task.env.ftip_reaching_scale=$scale
# done


# for scale in 1.0 2.0 5.0 10.0
# do
#     python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True task.env.force_qp_scale=$scale
# done

# for scale in 0.2 0.5
# do
#     python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True task.env.contact_rwd_weight=$scale
# done

# for scale in 0.2 0.5 1.0
# do
#     python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True  wandb_activate=True task.env.force_qp_torque_ref=$scale
# done

for seed in 0 1
do
    python train.py task.env.enable_location_qp=True wandb_group="location_qp" wandb_activate=True seed=$seed
done

for seed in 0 1
do
    python train.py task.env.enable_location_qp=True task.env.enable_force_qp=True wandb_group="force_location_qp" wandb_activate=True seed=$seed
done

# for seed in 0 1
# do
#     python train.py task.env.enable_ftip_damping=False wandb_group="no_qp" wandb_activate=True seed=$seed
# done