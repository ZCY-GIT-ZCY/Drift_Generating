"""
DMG Sequence Utilities
复用 MLD 的 tems_utils
"""

from typing import Dict, List
import numpy as np
import torch
from torch import Tensor


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    """
    根据序列长度生成布尔掩码

    Args:
        lengths: 每个序列的实际长度
        device: 计算设备
        max_len: 最大长度，默认为最长序列长度

    Returns:
        mask: (batch_size, max_len) 的布尔张量，True 表示有效位置
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def detach_to_numpy(tensor):
    """将张量分离并转换为 numpy"""
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    """
    去除批量张量中的填充部分

    Args:
        tensors: (batch_size, max_len, feature_dim) 的张量
        lengths: 每个序列的实际长度列表

    Returns:
        list: 变长序列列表
    """
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def nfeats_of(rottype):
    """返回旋转表示的特征维度"""
    if rottype in ["rotvec", "axisangle"]:
        return 3
    elif rottype in ["rotquat", "quaternion"]:
        return 4
    elif rottype in ["rot6d", "6drot", "rotation6d"]:
        return 6
    elif rottype in ["rotmat"]:
        return 9
    else:
        return TypeError("This rotation type doesn't have features.")


def subsample(num_frames, last_framerate, new_framerate):
    """子采样帧"""
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


def upsample(motion, last_framerate, new_framerate):
    """上采样帧"""
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    output = np.concatenate((output, motion[[-1]]))
    return output
