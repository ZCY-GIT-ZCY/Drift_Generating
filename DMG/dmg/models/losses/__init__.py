"""
DMG Losses Module

包含：
- drift_loss_bridge: PyTorch ↔ JAX drift_loss 桥接（阶段4）
"""

from .drift_loss_bridge import DriftLossBridge, compute_drift_loss_and_gradients

__all__ = [
    'DriftLossBridge',
    'compute_drift_loss_and_gradients',
]
