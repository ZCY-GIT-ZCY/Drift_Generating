"""
DMG 模型加载工厂
"""

import importlib


def get_model(cfg, datamodule, phase="train"):
    """
    获取模型实例

    Args:
        cfg: 配置对象
        datamodule: 数据模块
        phase: 阶段 ('train', 'test')

    Returns:
        model: 模型实例
    """
    modeltype = cfg.model.model_type
    if modeltype == "dmg":
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    """
    根据配置获取模型模块

    Args:
        cfg: 配置对象
        datamodule: 数据模块

    Returns:
        model: 模型实例
    """
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="dmg.models")
    Model = getattr(model_module, modeltype.upper())
    return Model(cfg=cfg, datamodule=datamodule)
