import isaacgym
import isaacgymenvs
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt

cfg = OmegaConf.load("cfg/config.yaml")
cfg.task_name = "TrifingerNYU"
cfg.num_envs = 1
cfg.task = OmegaConf.load("cfg/task/TrifingerNYU.yaml")
cfg.task.env.command_mode = "fingertip_diff"
# cfg.task.env.command_mode = "torque"

device = cfg.sim_device
cfg.headless = False

def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        return envs

envs = create_env_thunk()

# get fingertip states
N = 500
action_buffer = torch.zeros(N, 9).to(device)
ftip_pos_buffer = torch.zeros(N, 9).to(device)
ftip_vel_buffer = torch.zeros(N, 9).to(device)
jacobian_buffer = torch.zeros(N, 9, 9).to(device)
dof_vel_buffer = torch.zeros(N, 9).to(device)

for n in range(N):
    action = torch.rand(1, 9).to(device) * 2 - 1
    obs, rwds, resets, info = envs.step(torch.rand(1, 9).to(cfg.sim_device) * 2 - 1)
    action_buffer[n] = action
    
    q = envs._dof_position
    dq = envs._dof_velocity
    dof_vel_buffer[n] = dq[0]
    
    fingertip_state = envs._rigid_body_state[:, envs._fingertip_indices]
    fingertip_position = fingertip_state[:, :, 0:3].reshape(envs.num_envs, 9)
    fingertip_velocity = fingertip_state[:, :, 7:10].reshape(envs.num_envs, 9)
    ftip_pos_buffer[n] = fingertip_position[0]
    ftip_vel_buffer[n] = fingertip_velocity[0]
    
    fid = [5, 12, 19]
    jacobian_fingertip_linear = envs._jacobian[:, fid, :3, :]
    jacobian_fingertip_linear = jacobian_fingertip_linear.view(
                    envs.num_envs, 
                    3 * envs._dims.NumFingers.value, 
                    envs._dims.GeneralizedCoordinatesDim.value)
    jacobian_buffer[n] = jacobian_fingertip_linear[0]