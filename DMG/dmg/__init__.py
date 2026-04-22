# DMG: Drifting Motion Generation

"""
DMG - Drifting Motion Generation

将 Drifting 范式迁移到动作生成领域
以 HumanML3D 为数据集，MLD 的数据处理和 VAE 为基础
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import utils

__all__ = ['data', 'models', 'utils']
