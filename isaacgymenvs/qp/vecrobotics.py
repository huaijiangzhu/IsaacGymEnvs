import torch

@torch.jit.script
def bmv(mat: torch.Tensor, vec: torch.Tensor):
    return torch.einsum('bij, bj -> bi', mat, vec)

@torch.jit.script
def stacked_bmv(mat: torch.Tensor, vec: torch.Tensor):
    batch_size, dmat1, dmat2 = mat.shape
    batch_size, dvec_stacked = vec.shape # dvec_stacked should be multiples of dmat2
    return bmv(mat, vec.view(-1, dmat2)).view(-1, dvec_stacked) 

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
def SE3_transform(
    pose: torch.Tensor,
    p: torch.Tensor
):
    pos = pose[:, 0:3]
    quat = pose[:, 3:7]
    rot = quat2mat(quat)
    
    transformed = pos + bmv(rot, p)

    return transformed

@torch.jit.script
def SE3_inverse_transform(
    pose: torch.Tensor,
    p: torch.Tensor
):
    pos = pose[:, 0:3]
    quat = pose[:, 3:7]
    rot = quat2mat(quat)
    rot_inv = torch.transpose(rot, 1, 2)
    
    transformed = -bmv(rot_inv, pos) + bmv(rot_inv, p)

    return transformed

@torch.jit.script
def vec2skewsym_mat(vec: torch.Tensor):
    batch_size, _ = vec.shape
    mat = torch.zeros(batch_size, 3, 3, device=vec.device)
    
    mat[:, 0, 1] = -vec[:, 2]
    mat[:, 0, 2] = vec[:, 1]
    mat[:, 1, 0] = vec[:, 2]
    mat[:, 1, 2] = -vec[:, 0]
    mat[:, 2, 0] = -vec[:, 1]
    mat[:, 2, 1] = vec[:, 0]
    
    return mat
