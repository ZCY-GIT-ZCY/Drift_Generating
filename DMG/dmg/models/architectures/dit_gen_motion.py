"""
阶段3：DitGenMotion — 1D DiT 动作生成模型

将 LightningDiT（2D 图像生成）改造为 1D 时序生成，用于条件动作预测。

核心改造（pipeline §4.1）：
  1. Patchify: 2D 网格 → 1D 时序 token
  2. 位置编码: 2D SinCos → 1D SinCos
  3. RoPE: 2D RoPE → 1D RoPE
  4. 条件注入: 单一类别嵌入 → 文本 + 历史帧 + CFG 三重条件（AdaLN）

条件构建（pipeline §4.3）：
  text_cond  = Linear(512 → 768)(text_emb)
  his_cond   = Linear(512 → 768)(cat(mean(z_h), std(z_h)))
  cfg_emb    = Linear(1   → 768)(cfg_scale) * 0.02
  cond       = text_cond + his_cond + cfg_emb

输入：
  z_noise  [B, 1, 256]  — 标准正态噪声（DiT 起点）
  z_h      [B, 1, 256]  — VAE 历史帧 latent
  text_emb [B, 512]     — CLIP 文本向量
  cfg_dropout_mask [B]  — 训练时 10% 概率 drop 文本条件

输出：
  z_T [B, 1, 256]  — 生成帧的 latent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 1. 底层基础组件
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    公式: x / rms(x) * weight,  其中 rms(x) = sqrt(E[x^2])
    比 LayerNorm 少计算均值，更高效。

    对应 JAX 版 RMSNorm (generator.py §1)：
      var = jnp.mean(jax.lax.square(var_x), axis=-1, keepdims=True)
      normed = x * jax.lax.rsqrt(var + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入模块（对应 generator.py §3 TimestepEmbedder）

    使用频率嵌入 + MLP 将标量时间步映射到隐藏维度。

    原版 JAX 实现：
      embedding = concat([cos(args), sin(args)])   # [B, frequency_embedding_size]
      t_emb = MLP(hidden_size → hidden_size)(embedding)

    PyTorch 版本语义完全一致。
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # 权重初始化（与原版 "normal" 一致）
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 标量时间步（或 CFG 尺度），值域任意

        Returns:
            t_emb: [B, hidden_size] 嵌入向量
        """
        half = self.frequency_embedding_size // 2
        # 频率衰减：exp(-log(10000) * i / half)
        freqs = torch.exp(
            torch.arange(0, half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )  # [half]

        args = t.float()[:, None] * freqs[None]  # [B, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, frequency_embedding_size]

        # 如果 frequency_embedding_size 是奇数，补零对齐
        if self.frequency_embedding_size % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        t_emb = self.mlp(embedding)  # [B, hidden_size]
        return t_emb


def modulate_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制（对应 generator.py §1 的 modulate）

    公式: x * (1 + scale) + shift
    shift/scale 均沿 token 维度广播（dim=1）。
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNModulation(nn.Module):
    """
    AdaLN 调制模块

    将条件向量 c 映射为 shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    用于在每个 LightningDiTBlock 中注入条件。
    """

    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        # 初始化为零，保证训练初期 AdaLN 为恒等映射（稳定）
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: [B, cond_dim] 条件向量

        Returns:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            均为 [B, hidden_size]
        """
        out = self.net(c)  # [B, 6 * hidden_size]
        chunks = out.chunk(6, dim=-1)
        return chunks  # 6 × [B, hidden_size]


# =============================================================================
# 1D RoPE (Rotary Position Embedding)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    旋转半个维度（对应 generator.py §1 rotate_half）

    公式: concat([-x[..., D/2:], x[..., :D/2]])
    """
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D Rotary Position Embedding（参考 generator.py §1 apply_rope）

    在注意力计算前，对 q 和 k 沿序列维度进行旋转编码。
    使注意力权重自然地包含相对位置信息，无需可学习的位置嵌入。

    公式:
      rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]])
      x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)

    输入形状: q/k [B, N, H, D]  (N=序列长度, H=头数, D=每头维度)
    """
    B, N, H, D = q.shape

    # 构建旋转频率（与原版 generator.py 完全一致）
    freqs = torch.exp(
        torch.arange(0, D // 2, device=q.device, dtype=q.dtype)
        * (-math.log(10000.0) / (D // 2))
    )  # [D/2]

    t = torch.arange(N, device=q.device, dtype=q.dtype)  # [N]
    freqs = torch.outer(t, freqs)  # [N, D/2] — 外积得到每位置的旋转角
    emb = torch.cat([freqs, freqs], dim=-1)  # [N, D]

    cos = emb.cos()[None, :, None, :]   # [1, N, 1, D]
    sin = emb.sin()[None, :, None, :]   # [1, N, 1, D]

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed


# =============================================================================
# 3. 1D SwiGLU FFN
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN（对应 generator.py §1 SwiGLUFFN）

    SwiGLU(x) = SiLU(W1(x)) * W3(x)
    输出 = SwiGLU(x) * W2

    相比标准 GELU MLP，SwiGLU 在大模型中表现更好。
    """

    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(2 / 3 * hidden_size * 4.0)
            intermediate_size = (intermediate_size + 31) // 32 * 32

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardFFN(nn.Module):
    """
    标准 FFN（GELU 激活）

    作为 SwiGLU 的备选，当 use_swiglu=False 时使用。
    """

    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# =============================================================================
# 4. 1D Patchify / Unpatchify
# =============================================================================

class Patchify1D(nn.Module):
    """
    1D Patchify（pipeline §3.1）

    将输入序列划分为非重叠的 patch，每个 patch 线性投影到 hidden_size。

    当 latent_seq_len=1（单 token VAE latent）时，退化为恒等投影。

    输入: x [B, T, C]
    输出: x [B, N, hidden_size]  其中 N = T / patch_size
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if self.patch_size == 1:
            # 恒等映射 + 线性投影
            return self.proj(x)

        # 非重叠分块
        T = x.shape[1]
        assert T % self.patch_size == 0, f"T={T} 不是 patch_size={self.patch_size} 的倍数"
        N = T // self.patch_size
        x = x.view(x.shape[0], N, self.patch_size * x.shape[2])
        return self.proj(x)


class Unpatchify1D(nn.Module):
    """
    1D Unpatchify（pipeline §3.3 输出层）

    将 token 序列还原为原始维度。

    当 latent_seq_len=1 时，退化为恒等投影。

    输入: x [B, N, hidden_size]
    输出: x [B, T, C]  其中 T = N * patch_size
    """

    def __init__(self, hidden_size: int, out_channels: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(hidden_size, patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, hidden_size]
        if self.patch_size == 1:
            return self.proj(x)  # [B, N, out_channels]

        x = self.proj(x)  # [B, N, patch_size * out_channels]
        T = x.shape[1] * self.patch_size
        return x.view(x.shape[0], T, x.shape[2] // self.patch_size)


# =============================================================================
# 5. LightningDiTBlock1D（核心 Transformer Block）
# =============================================================================

class LightningDiTBlock1D(nn.Module):
    """
    1D LightningDiT Block（pipeline §3.2, §3.3）

    由以下组件构成：
      1. 自注意力（Self-Attention）+ 1D RoPE（可选）+ QK Norm（可选）
      2. AdaLN 调制（shift/scale/gate）
      3. FFN（SwiGLU 或标准 GELU）

    对应 generator.py §2 LightningDiTBlock
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        cond_dim: int = 768,
        attn_fp32: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.use_qknorm = use_qknorm
        self.use_rmsnorm = use_rmsnorm
        self.attn_fp32 = attn_fp32

        # ---- 归一化 ----
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # ---- QKV 投影 ----
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

        # ---- QK Norm（仅归一化 q/k，不归一化 v）----
        if use_qknorm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

        # ---- 输出投影 ----
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # ---- AdaLN 调制 ----
        self.adaLN_modulation = AdaLNModulation(cond_dim, hidden_size)

        # ---- FFN ----
        intermediate_size = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, intermediate_size)
        else:
            self.mlp = StandardFFN(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, N, hidden_size] 输入 token 序列
            cond: [B, cond_dim] 条件向量

        Returns:
            x:    [B, N, hidden_size] 输出 token 序列
        """
        # ---- AdaLN 调制参数（全 fp32 保证精度）----
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \
            = self.adaLN_modulation(cond.float())
        # 均为 [B, hidden_size]

        # ---- 自注意力路径 ----
        x_norm = self.norm1(x)                                    # Pre-LN
        x_norm_s = modulate_adaln(x_norm, shift_msa, scale_msa)    # AdaLN 调制

        # QKV
        qkv = self.qkv(x_norm_s)                                   # [B, N, 3D]
        qkv = qkv.view(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]                          # 各 [B, H, N, head_dim]

        # QK Norm
        if self.use_qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 1D RoPE
        if self.use_rope:
            q, k = apply_rope_1d(q, k)

        # 注意力（全 fp32）
        q_in = q.float() * (self.head_dim ** -0.5)
        k_in = k.float()
        v_in = v.float()
        attn = torch.matmul(q_in, k_in.transpose(-2, -1))
        attn = F.softmax(attn.float(), dim=-1)
        attn = torch.matmul(attn, v_in)

        # 恢复形状并投影
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.hidden_size)
        attn = self.proj(attn)

        # 残差连接（含 gate）
        x = x + gate_msa.unsqueeze(1).float() * attn.float()

        # ---- FFN 路径 ----
        x_norm2 = self.norm2(x)
        x_norm2_s = modulate_adaln(x_norm2, shift_mlp, scale_mlp)
        ff_out = self.mlp(x_norm2_s)
        x = x + gate_mlp.unsqueeze(1).float() * ff_out.float()

        return x


class FinalLayer1D(nn.Module):
    """
    最终输出层（pipeline §3.3 输出层）

    将 hidden_size 还原为 latent 维度 out_channels。
    对应 generator.py §2 FinalLayer 的 1D 版本。
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        cond_dim: int,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, N, hidden_size]
            cond: [B, cond_dim]

        Returns:
            [B, N, out_channels]
        """
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = modulate_adaln(x.float(), shift.float(), scale.float())
        return self.linear(x)


# =============================================================================
# 6. DitGenMotion 主模型
# =============================================================================

class DitGenMotion(nn.Module):
    """
    DitGenMotion: 1D DiT 动作生成模型（pipeline §3）

    将 LightningDiT 改造为专用于条件动作生成的 1D 时序模型。

    配置（pipeline §4.2）：
      hidden_size=768, depth=12, num_heads=12, cond_dim=768
      use_qknorm=True, use_rmsnorm=True, use_rope=True, use_swiglu=True

    前向路径（pipeline §3.3）：
      z_noise → patchify → + pos_embed → DiTBlock×12 → unpatchify → z_T
                                    ↑
                          AdaLN 注入条件 (text + his + cfg)
    """

    def __init__(
        self,
        # 模型维度（pipeline §4.2）
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        cond_dim: int = 768,
        mlp_ratio: float = 4.0,
        # VAE latent 维度
        latent_dim: int = 256,
        latent_seq_len: int = 1,   # VAE latent 的序列长度（当前 = 1）
        patch_size: int = 1,
        # 高级特性开关（pipeline §3.3）
        use_qknorm: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        attn_fp32: bool = True,
        # 文本条件投影
        text_dim: int = 512,
        # 丢弃率
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.latent_seq_len = latent_seq_len
        self.patch_size = patch_size
        self.cfg_dropout = cfg_dropout

        # ---- Patchify ----
        self.patchify = Patchify1D(
            in_channels=latent_dim,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )

        # ---- 1D SinCos 位置编码（pipeline §3.1）----
        # 对于 N=1 的情况，使用可学习的位置编码更稳定
        max_len = 1024
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_len, hidden_size)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # ---- 条件投影 ----
        # text_cond: CLIP 512d → cond_dim (pipeline §4.3)
        self.text_proj = nn.Linear(text_dim, cond_dim, bias=True)

        # his_cond: VAE latent (mean+std) 512d → cond_dim (pipeline §4.3)
        # latent: [B, latent_seq_len, latent_dim] → mean+std: [B, 2*latent_dim]
        self.his_proj = nn.Linear(2 * latent_dim, cond_dim, bias=True)

        # ---- CFG 嵌入（对应 generator.py DitGen c_cfg_noise_to_cond）----
        # 原版使用 TimestepEmbedder + RMSNorm，完全复用原版结构
        self.cfg_embedder = TimestepEmbedder(cond_dim, frequency_embedding_size=256)
        self.cfg_norm = RMSNorm(cond_dim)

        # ---- Transformer Block 堆叠 ----
        self.blocks = nn.ModuleList([
            LightningDiTBlock1D(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
                cond_dim=cond_dim,
                attn_fp32=attn_fp32,
            )
            for _ in range(depth)
        ])

        # ---- 最终输出层 ----
        self.final_layer = FinalLayer1D(
            hidden_size=hidden_size,
            out_channels=latent_dim,
            cond_dim=cond_dim,
            use_rmsnorm=use_rmsnorm,
        )

        self.adaLN_zero = nn.Parameter(torch.zeros(cond_dim))

    def build_cond(
        self,
        text_emb: torch.Tensor,
        z_h: torch.Tensor,
        cfg_scale: Optional[torch.Tensor] = None,
        cfg_drop: bool = False,
    ) -> torch.Tensor:
        """
        构建 AdaLN 条件向量（pipeline §4.3）

        cond = text_cond + his_cond + cfg_emb * 0.02

        Args:
            text_emb: [B, 512] CLIP 文本向量
            z_h:      [B, latent_seq_len, latent_dim] VAE 历史帧 latent
            cfg_scale: [B] 或 float, CFG 尺度（可选，推理时使用）
            cfg_drop:  是否 drop 文本条件（训练时 10% 概率）

        Returns:
            cond: [B, cond_dim]
        """
        # text_cond
        if cfg_drop:
            text_cond = torch.zeros(
                text_emb.shape[0],
                self.cond_dim,
                device=text_emb.device,
                dtype=text_emb.dtype,
            )
        else:
            text_cond = self.text_proj(text_emb)

        # his_cond: mean + std over sequence dimension
        z_h_mean = z_h.mean(dim=1)          # [B, latent_dim]
        z_h_std = z_h.std(dim=1, unbiased=False) + 1e-6    # [B, latent_dim]
        his_cond = self.his_proj(torch.cat([z_h_mean, z_h_std], dim=-1))  # [B, cond_dim]

        # cfg_emb（对应 generator.py DitGen.c_cfg_noise_to_cond）
        # 原版: cfg_scale_t = self.cfg_norm(self.cfg_embedder(cfg_scale_t))
        #       cond = cond + cfg_scale_t * 0.02
        if cfg_scale is None:
            cfg_scale_t = torch.ones(text_emb.shape[0], device=text_emb.device, dtype=torch.float32)
        elif isinstance(cfg_scale, float):
            cfg_scale_t = torch.full((text_emb.shape[0],), cfg_scale, device=text_emb.device, dtype=torch.float32)
        else:
            cfg_scale_t = cfg_scale.view(-1).float()

        cfg_emb = self.cfg_norm(self.cfg_embedder(cfg_scale_t)) * 0.02  # [B, cond_dim]

        # 条件向量融合
        cond = text_cond + his_cond + cfg_emb

        # 残差可学习的零初始化（参考 DiT 的 adaLN_modulation 技巧）
        cond = cond + self.adaLN_zero

        return cond

    def forward(
        self,
        z_noise: torch.Tensor,
        z_h: torch.Tensor,
        text_emb: torch.Tensor,
        cfg_scale: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        单步前向传播（pipeline §3.4 单步前向验证）

        Args:
            z_noise:    [B, latent_seq_len, latent_dim] 输入噪声
            z_h:         [B, latent_seq_len, latent_dim] VAE 历史帧 latent
            text_emb:    [B, 512] CLIP 文本向量
            cfg_scale:   [B] 或 None, CFG 尺度（None 时不使用 cfg）
            return_hidden: 是否返回中间 hidden states（用于 debug）

        Returns:
            z_T: [B, latent_seq_len, latent_dim] 生成帧 latent
        """
        B = z_noise.shape[0]
        N = z_noise.shape[1]  # latent_seq_len

        # CFG drop（训练时，batch 级别决策：要么整个 batch 有文本，要么全部无条件）
        cfg_drop = self.training and torch.rand(1).item() < self.cfg_dropout

        # Patchify: [B, N, latent_dim] → [B, N, hidden_size]
        x = self.patchify(z_noise)

        # 位置编码
        x = x + self.pos_embed[:, :N, :]

        # 条件向量
        cond = self.build_cond(text_emb, z_h, cfg_scale, cfg_drop=cfg_drop)

        # DiT Block 堆叠
        hidden_states = []
        for block in self.blocks:
            x = block(x, cond)
            if return_hidden:
                hidden_states.append(x)

        # 最终输出层
        z_T = self.final_layer(x, cond)  # [B, N, latent_dim]

        if return_hidden:
            return z_T, hidden_states
        return z_T


# =============================================================================
# 7. 推理辅助：CFG 组合
# =============================================================================

def apply_cfg(
    model: DitGenMotion,
    z_noise: torch.Tensor,
    z_h: torch.Tensor,
    text_emb: torch.Tensor,
    cfg_scale: float = 2.0,
    cfg_dropout: float = 0.0,
) -> torch.Tensor:
    """
    使用 Classifier-Free Guidance 进行推理（pipeline §5.3）

    两次前向：
      1. 有条件：正常文本
      2. 无条件：文本向量置零

    z_cfg = z_uncond + cfg_scale * (z_cond - z_uncond)

    Args:
        model:      DitGenMotion 实例
        z_noise:    [B, 1, 256]
        z_h:        [B, 1, 256]
        text_emb:   [B, 512]
        cfg_scale:  CFG 尺度（默认 2.0）
        cfg_dropout: 覆盖模型的 cfg_dropout（设为 0.0 强制有条件）

    Returns:
        z_T: [B, 1, 256] CFG 加权的生成 latent
    """
    orig_cfg_dropout = model.cfg_dropout
    model.cfg_dropout = 0.0  # 强制无 drop

    # 有条件生成
    model.eval()
    with torch.no_grad():
        z_cond = model(z_noise, z_h, text_emb)

    # 无条件生成（文本向量置零）
    text_emb_zero = torch.zeros_like(text_emb)
    with torch.no_grad():
        z_uncond = model(z_noise, z_h, text_emb_zero)

    model.train()
    model.cfg_dropout = orig_cfg_dropout

    # CFG 组合
    z_T = z_uncond + cfg_scale * (z_cond - z_uncond)
    return z_T
