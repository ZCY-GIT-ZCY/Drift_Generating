"""
DMG Data Collate Functions
数据整理函数

包含 DMG 训练专用的 collate 函数。
MLD 风格的 mld_collate 已移除，DMG 使用 collate_sliding_window。
"""

import torch


def pad_sequences(sequences, padding_value=0, max_len=None):
    """
    填充序列到相同长度

    Args:
        sequences: 序列列表 [torch.Tensor]
        padding_value: 填充值
        max_len: 最大长度，默认使用最长序列

    Returns:
        padded: [N, max_len, D] 填充后的序列
        mask: [N, max_len] 有效位置掩码
    """
    if max_len is None:
        max_len = max(s.shape[0] for s in sequences)

    batch_size = len(sequences)
    feat_dim = sequences[0].shape[-1]

    padded = torch.full((batch_size, max_len, feat_dim), padding_value, dtype=sequences[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length] = seq
        mask[i, :length] = True

    return padded, mask


def uncollate_tensors(batch, lengths):
    """
    还原填充前的批次

    Args:
        batch: [B, T, D] 批次张量
        lengths: [B] 各序列实际长度

    Returns:
        sequences: 变长序列列表
    """
    sequences = []
    for i, length in enumerate(lengths):
        if isinstance(length, torch.Tensor):
            length = length.item()
        sequences.append(batch[i, :length])
    return sequences
