import torch

@torch.jit.script
def bmv(mat: torch.Tensor, vec: torch.Tensor):
    return torch.einsum('bij, bj -> bi', mat, vec)

# def quat2mat2(quat: torch.Tensor):
#     x, y, z, w = torch.unbind(quat, dim=-1)
#     return torch.vmap(_quat2mat)(x, y, z, w)

# def _quat2mat(x, y, z, w):
#     x2, y2, z2 = x**2, y**2, z**2
#     wx, wy, wz = w*x, w*y, w*z
#     xy, xz, yz = x*y, x*z, y*z
#     rotation_matrix = torch.stack([
#         1-2*y2-2*z2, 2*(xy-wz), 2*(xz+wy),
#         2*(xy+wz), 1-2*x2-2*z2, 2*(yz-wx),
#         2*(xz-wy), 2*(yz+wx), 1-2*x2-2*y2]
#     )
#     return rotation_matrix.view(3, 3)

@torch.jit.script
def quat2mat(quat: torch.Tensor):
    batch_size = len(quat)
    rot_mat = torch.zeros(batch_size, 3, 3).to(quat.device)
    
    x, y, z, w = torch.unbind(quat, dim=-1)    
    x2, y2, z2 = x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    rot_mat[:, 0, 0] = 1 - 2*y2 - 2*z2
    rot_mat[:, 0, 1] = 2*(xy - wz)
    rot_mat[:, 0, 2] = 2*(xz + wy)
    rot_mat[:, 1, 0] = 2*(xy + wz)
    rot_mat[:, 1, 1] = 1 - 2*x2 - 2*z2
    rot_mat[:, 1, 2] = 2*(yz - wx)
    rot_mat[:, 2, 0] = 2*(xz - wy)
    rot_mat[:, 2, 1] = 2*(yz + wx)
    rot_mat[:, 2, 2] = 1 - 2*x2 - 2*y2
    
    return rot_mat

@torch.jit.script
def local2world(
    local_frame_pose: torch.Tensor,
    position_local: torch.Tensor
):
    local_frame_pos = local_frame_pose[:, 0:3]
    local_frame_orn = local_frame_pose[:, 3:7]
    rot = quat2mat(local_frame_orn)
    
    position_world = local_frame_pos + bmv(rot, position_local)

    return position_world

@torch.jit.script
def world2local(
    local_frame_pose: torch.Tensor,
    position_world: torch.Tensor
):
    local_frame_pos = local_frame_pose[:, 0:3]
    local_frame_orn = local_frame_pose[:, 3:7]
    rot = quat2mat(local_frame_orn)
    rot_inv = torch.transpose(rot, 1, 2)
    
    position_local = -bmv(rot_inv, local_frame_pos) + bmv(rot_inv, position_world)

    return position_local
