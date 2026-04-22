"""
DMG 特征变换模块
复用 MLD 的 feats2joints 功能

功能：
1. RIFKE 特征（263-dim）→ 3D 关节坐标
2. 支持 HumanML3D 的 22 关节骨架
3. 支持 3D 可视化和 Blender 渲染

依赖：
- MLD/mld/data/humanml/scripts/motion_process.py 的核心逻辑
"""

import torch
import numpy as np
from typing import Union


# ============================================================================
# 核心数学工具（从 MLD 移植）
# ============================================================================

def qrot(q, r):
    """对位置 r 应用四元数 q 旋转"""
    shape = q.shape
    q_w = q[..., 0:1]
    q_vec = q[..., 1:]
    a = v_mul(q_vec, r) * 2.0
    b = cross(q_vec, r)
    c = q_vec * torch.broadcast_to(q_w, q_vec.shape)
    return r + a + b + c


def v_mul(p, q):
    """向量乘法"""
    return torch.cat([
        p[..., 1:2] * q[..., 2:3] - p[..., 2:3] * q[..., 1:2],
        p[..., 2:3] * q[..., 0:1] - p[..., 0:1] * q[..., 2:3],
        p[..., 0:1] * q[..., 1:2] - p[..., 1:2] * q[..., 0:1],
    ], dim=-1)


def cross(p, q):
    """向量叉积"""
    return torch.cat([
        p[..., 1:2] * q[..., 2:3] - p[..., 2:3] * q[..., 1:2],
        p[..., 2:3] * q[..., 0:1] - p[..., 0:1] * q[..., 2:3],
        p[..., 0:1] * q[..., 1:2] - p[..., 1:2] * q[..., 0:1],
    ], dim=-1)


def qinv(q):
    """四元数求逆（假设 q 是单位四元数）"""
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def recover_root_rot_pos(data):
    """从 RIFKE 数据恢复根节点旋转和位置"""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    
    # 从旋转速度累积角度
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # 构建四元数 (cos(ang), 0, sin(ang), 0) - Y 轴旋转
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # 恢复根节点位置
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22):
    """
    从 RIFKE 特征恢复 3D 关节位置

    Args:
        data: [T, 263] 或 [B, T, 263] RIFKE 特征（已归一化）
        joints_num: 关节数量（HumanML3D 为 22）

    Returns:
        positions: [T, joints_num, 3] 或 [B, T, joints_num, 3] 关节 3D 坐标
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # 提取相对位置（从第 4 个元素开始，每 3 个元素为一组关节坐标）
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # 应用根节点旋转
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )

    # 添加根节点 XZ 位置
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # 拼接根节点和关节
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


# ============================================================================
# 高级接口
# ============================================================================

def feats2joints(
    motion: Union[np.ndarray, torch.Tensor],
    mean: np.ndarray,
    std: np.ndarray,
    joints_num: int = 22,
    scale: float = 1.0,
) -> np.ndarray:
    """
    将 RIFKE 特征转换为 3D 关节坐标（用于可视化）

    Args:
        motion: [T, 263] 或 [B, T, 263] RIFKE 特征（归一化后的数据）
        mean: [263] 归一化均值
        std: [263] 归一化标准差
        joints_num: 关节数量（HumanML3D 为 22）
        scale: 缩放因子（MLD 使用 1.3 用于可视化）

    Returns:
        joints: [T, joints_num, 3] 或 [B, T, joints_num, 3] 关节坐标
    """
    # 转换为 tensor
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # 反归一化（将归一化数据还原为原始物理尺度）
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).float()
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).float()
    
    motion_denorm = motion * std + mean
    
    # 恢复关节坐标
    joints = recover_from_ric(motion_denorm, joints_num)
    
    # 应用缩放
    if scale != 1.0:
        joints = joints * scale
    
    return joints.cpu().numpy()


def feats2joints_simple(
    motion_norm: Union[np.ndarray, torch.Tensor],
    joints_num: int = 22,
) -> np.ndarray:
    """
    将已归一化的 RIFKE 特征直接转换为 3D 关节坐标

    Args:
        motion_norm: [T, 263] 或 [B, T, 263] 已归一化的 RIFKE 特征
        joints_num: 关节数量

    Returns:
        joints: [T, joints_num, 3] 或 [B, T, joints_num, 3] 关节坐标
    """
    if isinstance(motion_norm, np.ndarray):
        motion_norm = torch.from_numpy(motion_norm).float()
    
    joints = recover_from_ric(motion_norm, joints_num)
    return joints.cpu().numpy()


# ============================================================================
# Blender 渲染支持（网格模式）
# ============================================================================

# HumanML3D 关节名称
HUMANML3D_JOINT_NAMES = [
    'root',  # 0
    'l_hip', 'r_hip',  # 1, 2
    'spine1', 'l_knee', 'r_knee',  # 3, 4, 5
    'spine2', 'l_ankle', 'r_ankle',  # 6, 7, 8
    'spine3', 'l_foot', 'r_foot',  # 9, 10, 11
    'neck', 'l_collar', 'r_collar',  # 12, 13, 14
    'jaw', # 15
    'l_shoulder', 'r_shoulder',  # 16, 17
    'l_elbow', 'r_elbow',  # 18, 19
    'l_wrist', 'r_wrist',  # 20, 21
]


def joints_to_smplh_format(joints: np.ndarray) -> np.ndarray:
    """
    将 HumanML3D 关节转换为 SMPLH 格式

    HumanML3D: 22 关节
    SMPLH: 52 关节（包含手部）

    这里做简单映射，Hand 设为 0
    """
    # 简单情况：直接返回（后续可扩展）
    return joints
