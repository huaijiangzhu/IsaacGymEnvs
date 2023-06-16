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
    batch_size = len(vec)
    mat = torch.zeros(batch_size, 3, 3, device=vec.device)
    
    mat[:, 0, 1] = -vec[:, 2]
    mat[:, 0, 2] = vec[:, 1]
    mat[:, 1, 0] = vec[:, 2]
    mat[:, 1, 2] = -vec[:, 0]
    mat[:, 2, 0] = -vec[:, 1]
    mat[:, 2, 1] = vec[:, 0]
    
    return mat

@torch.jit.script
def skewsym_mat2vec(mat: torch.Tensor):
    batch_size = len(mat)
    vec = torch.zeros(batch_size, 3, device=mat.device)
    
    vec[:, 0] = -mat[:, 1, 2]
    vec[:, 1] = mat[:, 0, 2] 
    vec[:, 2] = mat[:, 1, 0]
    
    return vec

@torch.jit.script
def log3(rot_mat: torch.Tensor):
    batch_size = len(rot_mat)
    tr = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    theta = torch.acos((tr - 1) / 2)

    zero_theta_mask = theta.abs() < 1e-6
    non_zero_theta_mask = ~zero_theta_mask
    coeff_non_zero_theta = theta[non_zero_theta_mask] / (2 * torch.sin(theta[non_zero_theta_mask]))
    non_zero_log_rot = coeff_non_zero_theta.view(-1, 1, 1) * (rot_mat[non_zero_theta_mask] - rot_mat[non_zero_theta_mask].transpose(1, 2))
    
    pi_theta_mask = (theta == torch.pi)
    omega = torch.zeros(len(rot_mat[pi_theta_mask]), 3).to(rot_mat.device)
    omega[:, 0] = rot_mat[pi_theta_mask][:, 0, 2]
    omega[:, 1] = rot_mat[pi_theta_mask][:, 1, 2]
    omega[:, 2] = 1 + rot_mat[pi_theta_mask][:, 2, 2]
    coeff_pi_theta = theta[pi_theta_mask] / torch.sqrt(2 + 2*rot_mat[pi_theta_mask][:, 2, 2])
    
    log_rot = torch.zeros(batch_size, 3).to(rot_mat.device)
    log_rot[non_zero_theta_mask] = skewsym_mat2vec(non_zero_log_rot)
    log_rot[pi_theta_mask] = -coeff_pi_theta.view(-1, 1) * omega # both omega and -omega are correct, use this convention to match pinocchio log3

    return log_rot
