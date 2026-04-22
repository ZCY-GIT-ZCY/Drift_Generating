"""
阶段4：PyTorch ↔ JAX Drift Loss 桥接

负责：
1. PyTorch Tensor → JAX Array 格式转换
2. JAX drift_loss 调用（jit 编译）
3. 梯度计算：JAX grad → NumPy → PyTorch

梯度桥接机制：
  JAX drift_loss 的梯度通过 np.array 转换后，
  外部通过 torch.Tensor.backward() 注入到 PyTorch generator 参数中。

数据格式约定：
- PyTorch 侧：gen [B, 256]
- JAX 侧：drift_loss 期望 [B, C, S] 即 [B, 256, 1]
"""

import numpy as np
import torch
from jax import grad, jit
import jax.numpy as jnp
from torch import Tensor

from ....Drifting_Model import drift_loss as jax_drift_loss


# =============================================================================
# JAX drift_loss JIT 编译（模块级，避免重复编译）
# =============================================================================

_jax_drift_loss_jit = jit(
    jax_drift_loss.drift_loss,
    static_argnames=("R_list",)
)


def _jax_grad_inner(
    gen_jax,
    pos_jax,
    neg_jax,
    weight_gen_jax,
    weight_pos_jax,
    weight_neg_jax,
    R_list,
):
    """JAX grad(drift_loss) 对 gen 的梯度（内部函数）"""
    def scalar_loss(g):
        l, _ = jax_drift_loss.drift_loss(
            gen=g,
            fixed_pos=pos_jax,
            fixed_neg=neg_jax,
            weight_gen=weight_gen_jax,
            weight_pos=weight_pos_jax,
            weight_neg=weight_neg_jax,
            R_list=R_list,
        )
        return l.sum()
    return grad(scalar_loss)(gen_jax)


_jax_grad_jit = jit(_jax_grad_inner, static_argnames=("R_list",))


# =============================================================================
# 格式转换辅助
# =============================================================================

def _np(x):
    """将 numpy array 或 PyTorch Tensor 转换为 numpy"""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# 对外接口
# =============================================================================


def compute_drift_loss_and_gradients(
    feat_gen: Tensor,
    future_pos,
    future_neg,
    weight_pos: Tensor,
    weight_neg: Tensor,
    R_list=(0.1, 0.5, 1.0),
) -> tuple[float, dict, np.ndarray]:
    """
    计算 JAX drift_loss 及其对 feat_gen 的梯度

    Args:
        feat_gen:   [B, 256] 生成特征（来自 z_T 再过 VAE Encoder）
        future_pos:  [N_pos, 256] Bank 采样的未来帧正样本
        future_neg:  [N_neg, 256] Bank 采样的未来帧负样本
        weight_pos:  [N_pos] 正样本核权重
        weight_neg:  [N_neg] 负样本权重
        R_list:     R 值列表

    Returns:
        loss_value: float 标量 loss（batch 内均值）
        info:       dict 各 R 分量损失和 scale
        grad_gen:   [B, 256] JAX grad of loss w.r.t. feat_gen
                    **调用方负责将此梯度通过 backward() 注入 PyTorch 计算图**
    """
    R_list = tuple(R_list)
    B = feat_gen.shape[0]

    # ---- PyTorch → JAX 格式 ----
    # feat_gen: [B, 256] → [B, 256, 1]
    gen_jax = jnp.array(_np(feat_gen))[:, :, None]

    # future_pos: [N_pos, 256] → [B, 256, N_pos]
    pos_np = _np(future_pos)
    N_pos = pos_np.shape[0]
    pos_jax = jnp.broadcast_to(
        jnp.array(pos_np)[None, :, :],
        (B, N_pos, pos_np.shape[1])
    ).transpose(0, 2, 1)  # → [B, 256, N_pos]

    # future_neg: [N_neg, 256] → [B, 256, N_neg]
    neg_np = _np(future_neg)
    N_neg = neg_np.shape[0]
    neg_jax = jnp.broadcast_to(
        jnp.array(neg_np)[None, :, :],
        (B, N_neg, neg_np.shape[1])
    ).transpose(0, 2, 1)

    # weight_gen: [B, 256] 全1
    weight_gen_jax = jnp.ones((B, 256), dtype=jnp.float32)

    # weight_pos: [N_pos] → [B, 1, N_pos]
    wp_np = _np(weight_pos)
    weight_pos_jax = jnp.broadcast_to(
        jnp.array(wp_np)[None, None, :],
        (B, 1, N_pos)
    ).astype(jnp.float32)

    # weight_neg: [N_neg] → [B, 256]
    wn_np = _np(weight_neg) if weight_neg is not None else np.ones(N_neg)
    weight_neg_jax = jnp.broadcast_to(
        jnp.array(wn_np)[None, :],
        (B, N_neg)
    ).astype(jnp.float32)

    # ---- JAX 前向计算（无梯度） ----
    loss_jax, info_jax = _jax_drift_loss_jit(
        gen_jax,
        pos_jax,
        neg_jax,
        weight_gen_jax,
        weight_pos_jax,
        weight_neg_jax,
        R_list,
    )

    # ---- JAX 梯度计算 ----
    grad_gen_jax = _jax_grad_jit(
        gen_jax,
        pos_jax,
        neg_jax,
        weight_gen_jax,
        weight_pos_jax,
        weight_neg_jax,
        R_list,
    )  # [B, 256, 1]

    # ---- JAX → NumPy ----
    grad_gen_np = np.array(grad_gen_jax.squeeze(-1))  # [B, 256]

    # ---- info 转换 ----
    info = {k: float(v) for k, v in info_jax.items()}

    return float(np.array(loss_jax).mean()), info, grad_gen_np


# =============================================================================
# nn.Module 版本（保留，用于 future 扩展）
# =============================================================================

import torch.nn as nn


class DriftLossBridge(nn.Module):
    """
    PyTorch ↔ JAX Drift Loss 桥接模块

    使用方式：
        bridge = DriftLossBridge(R_list=[0.1, 0.5, 1.0])
        loss, info, grad_gen = bridge(
            feat_gen, future_pos, future_neg, weight_pos, weight_neg
        )
        feat_gen.backward(grad_gen)  # 梯度注入
    """

    def __init__(self, R_list=(0.1, 0.5, 1.0)):
        super().__init__()
        self.R_list = tuple(R_list)

    def forward(
        self,
        feat_gen: Tensor,
        future_pos,
        future_neg,
        weight_pos: Tensor,
        weight_neg: Tensor,
    ) -> tuple[Tensor, dict, Tensor]:
        loss_val, info, grad_np = compute_drift_loss_and_gradients(
            feat_gen=feat_gen,
            future_pos=future_pos,
            future_neg=future_neg,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            R_list=self.R_list,
        )
        grad_gen = torch.from_numpy(grad_np).to(feat_gen.device).float()
        loss_t = torch.tensor(loss_val, device=feat_gen.device, requires_grad=True)
        return loss_t, info, grad_gen
