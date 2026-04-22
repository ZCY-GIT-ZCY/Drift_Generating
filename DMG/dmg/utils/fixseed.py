"""
DMG Random Seed Utilities
"""

import random
import numpy as np
import torch


def fixseed(seed):
    """
    固定随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 默认种子
SEED = 1234
EVALSEED = 0

# 应用默认种子
fixseed(SEED)
