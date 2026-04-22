# DMG Data Module
# 复用 MLD 的数据处理管线

from .sliding_window import SlidingWindowDataset, SlidingWindowDataModule, collate_sliding_window
from .bank import MotionDriftBank, BankBuilder
from .get_data import get_global_mean_std

__all__ = [
    # 滑动窗口数据集（DMG 训练专用）
    'SlidingWindowDataset',
    'SlidingWindowDataModule',
    'collate_sliding_window',
    # Bank
    'MotionDriftBank',
    'BankBuilder',
    # 数据工具
    'get_global_mean_std',
    # 废弃：仅供 validate.py 调用（见 get_data.py 中的 DeprecationWarning）
    'get_datasets',
    'get_WordVectorizer',
]
