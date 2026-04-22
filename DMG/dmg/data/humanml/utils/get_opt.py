"""
DMG HumanML3D Utils - get_opt
复用 MLD 的配置加载逻辑
"""

import os
from os.path import join as pjoin


class ObjectView(object):
    """
    将字典转换为对象，方便使用点号访问
    """
    def __init__(self, d):
        self.__dict__ = d


def get_opt(opt_path, opt_key=None):
    """
    加载 yaml 配置文件

    Args:
        opt_path: 配置文件路径
        opt_key: 可选的配置键

    Returns:
        配置对象
    """
    from omegaconf import OmegaConf
    import yaml

    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Config file not found: {opt_path}")

    with open(opt_path, 'r') as f:
        if opt_path.endswith('.yaml') or opt_path.endswith('.yml'):
            cfg = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {opt_path}")

    if opt_key:
        cfg = cfg.get(opt_key, {})

    return ObjectView(cfg)
