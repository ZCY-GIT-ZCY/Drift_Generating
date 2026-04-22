"""
DMG Transformer Operators
复用 MLD 的 Transformer 组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _get_clones(module, n):
    """
    克隆模块 n 次

    Args:
        module: 要克隆的模块
        n: 克隆数量

    Returns:
        nn.ModuleList: 克隆模块列表
    """
    return nn.ModuleList([module for _ in range(n)])


def _get_clone(module, n=1):
    """
    克隆模块 n 次（不共享权重）

    Args:
        module: 要克隆的模块
        n: 克隆数量

    Returns:
        nn.ModuleList: 克隆模块列表
    """
    return nn.ModuleList([module for _ in range(n)])


class SkipTransformerEncoder(nn.Module):
    """
    带跳跃连接的 Transformer 编码器（来自 MLD）

    架构: Input → [Block × N//2] → Middle → [Block × N//2] → Output
                   ↑______________|  |______________↑

    特点:
    - 中间层作为信息瓶颈
    - 跳跃连接保留多尺度信息
    - 通过线性层融合不同尺度的特征
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        初始化跳跃连接编码器

        Args:
            encoder_layer: Transformer 编码器层
            num_layers: 总层数（必须是奇数：前半 + 中间 + 后半）
            norm: 归一化层
        """
        super().__init__()

        # num_layers 必须是奇数（前半 + 中间 + 后半）
        if num_layers % 2 == 0:
            raise ValueError(
                f"SkipTransformerEncoder requires odd num_layers (got {num_layers}). "
                "The architecture splits layers into: input_blocks + middle_block + output_blocks."
            )
        num_block = (num_layers - 1) // 2

        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.num_blocks = num_block + 2  # input + middle + output

        self.linear_blocks = _get_clones(
            nn.Linear(encoder_layer.d_model * 2, encoder_layer.d_model),
            num_block
        )

        self.norm = norm

    def forward(self, src, src_key_padding_mask=None):
        """
        前向传播

        Args:
            src: [len, B, D] 输入序列
            src_key_padding_mask: [B, len] 填充掩码

        Returns:
            output: [len, B, D] 输出序列
        """
        x = src
        xs = []

        # 编码器前半部分
        for module in self.input_blocks:
            x = module(x, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        # 中间层（_get_clone 返回 ModuleList，需取 [0]）
        x = self.middle_block[0](x, src_key_padding_mask=src_key_padding_mask)

        # 解码器后半部分（带跳跃连接）
        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerEncoder(nn.Module):
    """
    标准 Transformer 编码器（无跳跃连接）
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            src = self.norm(src)

        return src


class TransformerDecoder(nn.Module):
    """
    标准 Transformer 解码器
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class SkipTransformerDecoder(nn.Module):
    """
    带跳跃连接的 Transformer 解码器
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()

        num_block = (num_layers - 1) // 2

        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)

        self.linear_blocks = _get_clones(
            nn.Linear(decoder_layer.d_model * 2, decoder_layer.d_model),
            num_block
        )

        self.norm = norm

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        xs = []

        for module in self.input_blocks:
            x = module(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            xs.append(x)

        x = self.middle_block[0](
            x,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        if self.norm is not None:
            x = self.norm(x)

        return x
