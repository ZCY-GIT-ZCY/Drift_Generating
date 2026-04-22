"""
DMG Motion Encoder

阶段7：动作生成评估特征提取器

用于计算评估指标时的动作特征提取。
两种模式：
  1. 'vae': 直接复用 VAE Encoder 提取特征（DMG 训练时已冻结）
  2. 't2m': 复用 MLD 的 T2M_MotionEncoder（更专业的动作特征，用于评估）

pipeline.md §8.2 评估指标：
  - FID: 特征分布距离
  - R-Precision: 文本-动作检索精度 top-k
  - MM Dist: 文本-动作余弦距离
  - Diversity: 生成样本多样性

这些指标都需要将动作序列编码为特征向量。
"""

import torch
import torch.nn as nn
from typing import Optional


class DMGMotionEncoder(nn.Module):
    """
    DMG 动作特征编码器

    输入：[B, T, 263] RIFKE 特征
    输出：[B, D] 特征向量（用于 FID/R-Precision/Diversity 计算）

    支持两种模式：
    - 'vae': VAE encoder freeze → mean pooling → [B, 256]
    - 't2m': 复用 MLD T2M_MotionEncoder → [B, 512]
    """

    def __init__(
        self,
        vae_encoder: nn.Module,
        mode: str = 'vae',
        output_dim: int = 256,
        t2m_encoder_path: Optional[str] = None,
    ):
        """
        Args:
            vae_encoder: VAE 编码器（来自 dmg.models.architectures.mld_vae）
            mode: 'vae' 或 't2m'
            output_dim: 输出特征维度
            t2m_encoder_path: MLD T2M_MotionEncoder 权重路径（mode='t2m' 时需要）
        """
        super().__init__()
        self.mode = mode
        self.vae = vae_encoder
        self.output_dim = output_dim

        if mode == 'vae':
            # 直接用 VAE latent + mean pooling
            self.feature_dim = 256
        elif mode == 't2m':
            self.feature_dim = 512
            self._load_t2m_encoder(t2m_encoder_path)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'vae' or 't2m'.")

        # 投影层：统一输出维度
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def _load_t2m_encoder(self, model_path: Optional[str]):
        """加载 MLD T2M_MotionEncoder"""
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            mld_path = project_root / "MLD"
            if mld_path.exists() and str(mld_path) not in sys.path:
                sys.path.insert(0, str(project_root))

            from mld.models.architectures.t2m_motionenc import T2M_MotionEncoder
            self.t2m_encoder = T2M_MotionEncoder(
                inputEmb=263,
                hiddenSize=1024,
                outputEmb=self.feature_dim,
            )

            if model_path and torch.cuda.is_available():
                state = torch.load(model_path, map_location='cuda')
                self.t2m_encoder.load_state_dict(state, strict=False)
                print("[DMGMotionEncoder] T2M encoder loaded")
            else:
                print("[DMGMotionEncoder] T2M encoder initialized (no pretrained weights)")

        except ImportError as e:
            print(f"[DMGMotionEncoder] Warning: Could not load T2M encoder ({e})")
            print("  Falling back to VAE mode")
            self.mode = 'vae'
            self.feature_dim = 256

    @torch.no_grad()
    def encode(
        self,
        motion: torch.Tensor,
        lengths: list[int],
    ) -> torch.Tensor:
        """
        编码动作序列为特征向量

        Args:
            motion: [B, T, 263] RIFKE 特征
            lengths: 各序列实际帧数 [B]

        Returns:
            features: [B, output_dim] 特征向量
        """
        if self.mode == 'vae':
            return self._encode_vae(motion, lengths)
        else:
            return self._encode_t2m(motion, lengths)

    def _encode_vae(
        self,
        motion: torch.Tensor,
        lengths: list[int],
    ) -> torch.Tensor:
        """VAE 模式：VAE encode → mean pooling"""
        latent, _ = self.vae.encode(motion, lengths)  # [B, 1, 256]
        feat = latent.squeeze(1)  # [B, 256]
        return self.projection(feat)

    def _encode_t2m(
        self,
        motion: torch.Tensor,
        lengths: list[int],
    ) -> torch.Tensor:
        """T2M 模式：MLD T2M_MotionEncoder"""
        if hasattr(self, 't2m_encoder'):
            return self.t2m_encoder(motion, lengths)
        else:
            return self._encode_vae(motion, lengths)


def create_motion_encoder(
    vae_encoder: nn.Module,
    mode: str = 'vae',
    output_dim: int = 256,
    device: str = 'cuda',
) -> DMGMotionEncoder:
    """工厂函数：创建 motion encoder"""
    encoder = DMGMotionEncoder(
        vae_encoder=vae_encoder,
        mode=mode,
        output_dim=output_dim,
    )
    return encoder.to(device)
