"""
DMG Position Encoding
复用 MLD 的位置编码
"""

import math
import torch
import torch.nn as nn


class PositionEmbeddingSine1D(nn.Module):
    """
    正弦位置编码 (1D)

    用于为序列中的每个位置添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 500, temperature: int = 10000):
        """
        初始化位置编码

        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            temperature: 温度参数
        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.temperature = temperature

        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(temperature) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [len, B, D] 输入张量

        Returns:
            x: [len, B, D] 添加位置编码的张量
        """
        len_seq = x.shape[0]
        return x + self.pe[:len_seq, :].unsqueeze(1)


class PositionEmbeddingLearned1D(nn.Module):
    """
    可学习位置编码 (1D)
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        len_seq = x.shape[0]
        return x + self.pe[:len_seq]


def get_position_encoding(
    d_model: int,
    max_len: int = 500,
    pe_type: str = 'sine'
):
    """
    获取位置编码模块

    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        pe_type: 编码类型 ('sine' 或 'learned')

    Returns:
        position_encoding: 位置编码模块
    """
    if pe_type in ('v2', 'sine', 'sinusoidal'):
        return PositionEmbeddingSine1D(d_model, max_len)
    elif pe_type in ('v3', 'learned'):
        return PositionEmbeddingLearned1D(d_model, max_len)
    else:
        raise ValueError(f"Unknown position encoding type: {pe_type}")
