"""
DMG MLD VAE 模型
复用 MLD 的 MldVae

此模块用于：
1. 阶段0-1: 验证 VAE 加载和特征提取
2. 阶段1-5: 作为冻结的特征提取器和解码器
"""

import torch
import torch.nn as nn

from ...utils.position_encoding import get_position_encoding
from ...utils.operator import SkipTransformerEncoder, SkipTransformerDecoder


class MldVae(nn.Module):
    """
    MLD VAE (Motion Latent Diffusion Variational Autoencoder)

    复用 MLD 的 VAE 架构，用于动作压缩和重建
    """

    def __init__(
        self,
        nfeats: int = 263,
        latent_dim: list = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
        arch: str = 'all_encoder',
        mlp_dist: bool = False,
        **kwargs
    ):
        """
        初始化 MLD VAE

        Args:
            nfeats: 输入特征维度 (HumanML3D: 263)
            latent_dim: latent 维度 [token数, 通道数], 如 [1, 256]
            ff_size: 前馈网络维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            dropout: Dropout 概率
            activation: 激活函数类型
            arch: 架构类型 ('all_encoder' 或 'encoder_decoder')
            mlp_dist: 是否使用 MLP 预测分布参数
        """
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim  # [1, 256]
        self.latent_size = latent_dim[0]  # 1
        self.mlp_dist = mlp_dist

        # 激活函数
        activation_fn = nn.ReLU() if activation == 'relu' else nn.GELU()

        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim[1],
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation_fn,
            batch_first=True
        )
        self.encoder = SkipTransformerEncoder(
            encoder_layer,
            num_layers,
            norm=nn.LayerNorm(latent_dim[1])
        )

        # 解码器
        if arch == "all_encoder":
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim[1],
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation_fn,
                batch_first=True
            )
            self.decoder = SkipTransformerEncoder(
                decoder_layer,
                num_layers,
                norm=nn.LayerNorm(latent_dim[1])
            )
        elif arch == "encoder_decoder":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=latent_dim[1],
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation_fn,
                batch_first=True
            )
            self.decoder = SkipTransformerDecoder(
                decoder_layer,
                num_layers,
                norm=nn.LayerNorm(latent_dim[1])
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.arch = arch

        # 分布预测层和 Motion Token
        if self.mlp_dist:
            self.dist_layer = nn.Linear(latent_dim[1], 2 * latent_dim[1])
            self.global_motion_token = nn.Parameter(
                torch.randn(latent_dim[0], latent_dim[1])
            )
        else:
            # 前半 token 为 mu，后半为 logvar
            self.global_motion_token = nn.Parameter(
                torch.randn(latent_dim[0] * 2, latent_dim[1])
            )

        # 特征嵌入
        self.skel_embedding = nn.Linear(nfeats, latent_dim[1])
        self.final_layer = nn.Linear(latent_dim[1], nfeats)

        # 位置编码
        self.query_pos_encoder = get_position_encoding(
            latent_dim[1],
            max_len=500,
            pe_type='sine'
        )

    def encode(self, features, lengths=None):
        """
        编码器：将动作特征编码为 latent

        Args:
            features: [B, T, nfeats] 动作特征
            lengths: 可选，序列长度列表

        Returns:
            latent: [B, latent_size, latent_dim] latent 向量
            dist: 分布对象（可用于采样）
        """
        bs = features.shape[0]

        # 特征嵌入
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)  # [T, B, D]

        # 拼接 Motion Token
        dist = torch.tile(
            self.global_motion_token[:, None, :],
            (1, bs, 1)
        )  # [N, B, D]
        xseq = torch.cat((dist, x), 0)  # [N+T, B, D]

        # 位置编码
        xseq = self.query_pos_encoder(xseq)

        # Transformer 编码
        aug_mask = None
        if lengths is not None:
            max_len = features.shape[1]
            mask = torch.zeros(bs, max_len, device=features.device, dtype=torch.bool)
            for i, length in enumerate(lengths):
                if length < max_len:
                    mask[i, length:] = True
            aug_mask = torch.cat([torch.zeros(bs, self.latent_size, device=features.device, dtype=torch.bool), mask], dim=1)

        dist = self.encoder(xseq, src_key_padding_mask=aug_mask)[:self.latent_size]

        # 分布预测
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim[1]]
            logvar = tokens_dist[:, :, self.latent_dim[1]:]
        else:
            mu = dist[0:self.latent_size]
            logvar = dist[self.latent_size:]

        # 重参数化采样
        import torch.distributions as distributions
        distribution = distributions.Normal(mu, logvar.exp().pow(0.5))
        latent = distribution.rsample()

        return latent, distribution

    def decode(self, z, lengths):
        """
        解码器：将 latent 解码为动作特征

        Args:
            z: [B, latent_size, latent_dim] latent 向量
            lengths: 目标输出长度列表

        Returns:
            features: [B, T, nfeats] 重建的动作特征
        """
        bs = z.shape[0]
        nframes = max(lengths)

        # 生成掩码
        mask = torch.zeros(bs, nframes, device=z.device, dtype=torch.bool)
        for i, length in enumerate(lengths):
            if length < nframes:
                mask[i, length:] = True

        # 生成查询
        queries = torch.zeros(nframes, bs, self.latent_dim[1], device=z.device)

        # 拼接 latent 和查询
        xseq = torch.cat((z, queries), axis=0)  # [N+T, B, D]

        # Transformer 解码
        augmask = torch.zeros(bs, self.latent_size + nframes, device=z.device, dtype=torch.bool)
        augmask[:, :self.latent_size] = False
        augmask[:, self.latent_size:] = mask

        output = self.decoder(xseq, src_key_padding_mask=augmask)[self.latent_size:]

        # 输出投影
        output = self.final_layer(output)
        output[mask.T] = 0  # 填充置零

        return output.permute(1, 0, 2)  # [B, T, D]

    def forward(self, features, lengths=None):
        """
        前向传播：编码 + 解码

        Args:
            features: [B, T, nfeats] 动作特征
            lengths: 序列长度列表

        Returns:
            output: [B, T, nfeats] 重建的动作特征
            latent: [B, latent_size, latent_dim] latent 向量
        """
        latent, _ = self.encode(features, lengths)
        output = self.decode(latent, lengths)
        return output, latent

    def load_pretrained(self, checkpoint_path):
        """
        加载预训练权重

        Args:
            checkpoint_path: 预训练模型路径

        Returns:
            bool: 是否成功加载
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Pretrained VAE checkpoint not found at {checkpoint_path}")
            print("VAE will be initialized from scratch.")
            return False

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

            # 处理不同格式的 checkpoint
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                # 有时权重嵌套在 'model' 下
                state_dict = state_dict['model']

            # 提取 VAE 相关权重，支持多种前缀格式
            vae_dict = {}
            unmatched_keys = []

            for k, v in state_dict.items():
                # 跳过非 VAE 权重
                if 'vae' in k.lower():
                    name = k.replace('vae.', '').replace('VAE.', '')
                    vae_dict[name] = v
                elif k.startswith('model.') and 'encoder' in k.lower():
                    name = k.replace('model.', '')
                    vae_dict[name] = v
                elif k.startswith('module.') and 'encoder' in k.lower():
                    name = k.replace('module.', '')
                    vae_dict[name] = v
                elif '.' not in k:
                    # 顶级权重（无前缀）
                    vae_dict[k] = v
                else:
                    # 记录未匹配的权重，用于调试
                    if 'encoder' in k.lower() or 'decoder' in k.lower() or 'embedding' in k.lower():
                        unmatched_keys.append(k)

            # 尝试加载权重
            load_result = self.load_state_dict(vae_dict, strict=False)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys

            if missing_keys:
                print(f"Warning: {len(missing_keys)} VAE weights not loaded (missing in checkpoint)")
                if len(missing_keys) <= 5:
                    for k in missing_keys:
                        print(f"  - Missing: {k}")

            if unexpected_keys:
                print(f"Info: {len(unexpected_keys)} extra weights in checkpoint (ignored)")
                if len(unexpected_keys) <= 5:
                    for k in unexpected_keys:
                        print(f"  - Extra: {k}")

            if not missing_keys and not unexpected_keys:
                print(f"Successfully loaded pretrained VAE from {checkpoint_path}")
            else:
                print(f"Partially loaded pretrained VAE from {checkpoint_path}")

            return True

        except Exception as e:
            print(f"Error loading pretrained VAE from {checkpoint_path}: {e}")
            print("VAE will be initialized from scratch.")
            return False
