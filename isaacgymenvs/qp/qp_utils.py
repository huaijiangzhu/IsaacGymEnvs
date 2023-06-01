from typing import List
from .vecrobotics import *

@torch.jit.script
def get_cube_contact_normals(ftip_pos: torch.Tensor, threshold: float = 0.0435):
    batch_size = len(ftip_pos)
    contact_normals = torch.zeros(batch_size, 3).to(ftip_pos.device)
        
    _, max_indices = torch.max(torch.abs(ftip_pos), dim=1)
    max_values = torch.squeeze(torch.gather(ftip_pos, 1, max_indices.unsqueeze(1)))

    mask_pos = (torch.abs(max_values) <= threshold) * (max_values > 0)
    mask_neg = (torch.abs(max_values) <= threshold) * (max_values < 0)

    # contact normal points to the same direction as the contact force, hence into the object
    contact_normals[mask_pos, max_indices[mask_pos]] = -1.0
    contact_normals[mask_neg, max_indices[mask_neg]] = 1.0
    
    return contact_normals

@torch.jit.script
def get_contact_frame_orn(contact_normals: torch.Tensor):
    # get the orientation of the contact frames expressed in the object frame
    z_axis = contact_normals
    zero_indices = torch.argmax(torch.eq(z_axis, 0).int(), dim=1)
    y_axis = torch.eye(3).to(z_axis.device)[zero_indices]
    x_axis = torch.cross(y_axis, z_axis, dim=1)
    y_axis = torch.cross(z_axis, x_axis, dim=1) # this makes sure if z is all zero, then orn is a zero matrix
    orn = torch.stack((x_axis, y_axis, z_axis), dim=2)
    return orn

@torch.jit.script
def get_force_qp_data(ftip_pos: torch.Tensor, object_pose: torch.Tensor, total_force_des: torch.Tensor, weights: List[float]):
    # get ftip positin in the object frame
    batch_size, num_ftip, _ = ftip_pos.shape
    num_vars = num_ftip * 3

    p = SE3_inverse_transform(object_pose.repeat_interleave(3, dim=0), ftip_pos.view(-1, 3))
    contact_normals = get_cube_contact_normals(p)
    
    R = get_contact_frame_orn(contact_normals)
    R_vstacked = R.transpose(1, 2).reshape(-1, 3 * num_ftip, 3)
    Q1 = R_vstacked @ R_vstacked.transpose(1, 2)
    
    pxR = vec2skewsym_mat(p) @ R
    pxR_vstacked = pxR.transpose(1, 2).reshape(-1, 3 * num_ftip, 3)
    Q2 = pxR_vstacked @ pxR_vstacked.transpose(1, 2)
    
    w1, w2, w3 = weights
    Q = w1 * Q1 + w2 * Q2 + w3 * torch.eye(3 * num_ftip).repeat(batch_size, 1, 1).to(Q1.device)

    # for Q == 0, hence R1, R2, R3 == 0, fill the diagnoal of Q with ones. This produces f == 0
    reshaped_tensor = Q.view(batch_size, -1)
    diagonal_elements_zero = torch.all(reshaped_tensor[:, ::num_vars+1] == 0, dim=1)
    mask = diagonal_elements_zero[:, None].repeat(1, num_vars)
    diag_idx = torch.arange(num_vars)
    Q[:, diag_idx, diag_idx] = Q[:, diag_idx, diag_idx].masked_fill_(mask, 1)
    
    object_orn = quat2mat(object_pose[:, 3:])
    total_force_des_object_frame = bmv(object_orn.transpose(1,2), total_force_des)
    q = -2 * bmv(R_vstacked, total_force_des_object_frame)
    
    return Q, q, R_vstacked, pxR_vstacked, contact_normals

@torch.jit.script
def get_location_qp_data(ftip_pos: torch.Tensor, ftip_pos_des: torch.Tensor, torque_ref: torch.Tensor, jacobian: torch.Tensor, weights: List[float]):
    batch_size, num_ftip, _ = ftip_pos.shape
    num_vars = num_ftip * 3
    num_vars = 9
    diag_idx = torch.arange(num_vars)

    jacobian_transpose = torch.transpose(jacobian, 1, 2)
    Q1 = jacobian @ jacobian_transpose
    q1 = -2 * bmv(jacobian_transpose, torque_ref)

    ftip_pos_diff = (ftip_pos_des - ftip_pos).reshape(batch_size, 9)
    task_space_force = torch.tensor([100, 100, 200] * 3, dtype=torch.float32, device=torque_ref.device) * ftip_pos_diff
    Q2 = torch.zeros_like(Q1)
    Q2[:, diag_idx, diag_idx] = 1.
    q2 = -2 * task_space_force

    w1, w2 = weights
    Q = w1*Q1 + w2*Q2
    q = w1*q1 + w2*q2

    return Q, q

