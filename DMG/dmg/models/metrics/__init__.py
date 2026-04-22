"""
DMG Evaluation Metrics Module

阶段7：评估指标计算

主要组件：
  - evaluator.py: DMGEvaluator、DMGOfflineEvaluator、FID/R-Precision/MM Dist/Diversity 计算
  - dmg_motion_encoder.py: 动作特征提取器（VAE freeze / T2M MotionEncoder）
"""

from .evaluator import (
    DMGEvaluator,
    DMGOfflineEvaluator,
    calculate_fid,
    calculate_r_precision,
    calculate_matching_score,
    calculate_diversity,
)
from ..architectures.dmg_motion_encoder import DMGMotionEncoder, create_motion_encoder

__all__ = [
    'DMGEvaluator',
    'DMGOfflineEvaluator',
    'DMGMotionEncoder',
    'create_motion_encoder',
    'calculate_fid',
    'calculate_r_precision',
    'calculate_matching_score',
    'calculate_diversity',
]
