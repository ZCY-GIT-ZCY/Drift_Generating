"""
DMG 数据集工具函数

包含 DMG 各阶段复用的数据加载辅助函数。
MLD 风格的数据集工厂函数（get_datasets, get_collate_fn, mld_collate）已废弃，
DMG 训练统一使用 SlidingWindowDataModule。
"""

import os
from os.path import join as pjoin
from typing import Tuple

import numpy as np


def get_global_mean_std(dataset_name: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取数据集的全局均值和标准差（用于 Bank 构建和滑动窗口归一化）

    与 MLD Text2MotionDatasetV2 保持一致：
    - HumanML3D 的 Mean.npy/Std.npy 是预计算好的全局统计量
    - Bank 构建和滑动窗口采样时使用全局统计量进行 Z-score 归一化

    Args:
        dataset_name: 数据集名称 ('humanml3d', 'kit')
        cfg: 配置对象

    Returns:
        mean, std: 全局归一化参数
    """
    name = "t2m" if dataset_name == "humanml3d" else dataset_name

    dataset_key = dataset_name.upper()
    if hasattr(cfg.DATASET, dataset_key):
        dataset_cfg = getattr(cfg.DATASET, dataset_key)
        data_root = dataset_cfg.ROOT
    else:
        raise KeyError(f"DATASET.{dataset_key} not found in config")

    # 从 DATASET.ROOT 加载全局统计量（HumanML3D 结构）
    mean_path = pjoin(data_root, "Mean.npy")
    std_path = pjoin(data_root, "Std.npy")

    # 备选：从 meta 子目录加载
    if not os.path.exists(mean_path):
        meta_root = pjoin(data_root, "meta")
        mean_path = pjoin(meta_root, "mean.npy")
        std_path = pjoin(meta_root, "std.npy")

    if not os.path.exists(mean_path):
        raise FileNotFoundError(
            f"Global mean/std not found at {mean_path}. "
            f"Please ensure HumanML3D dataset is properly set up."
        )

    mean = np.load(mean_path)
    std = np.load(std_path)

    return mean, std


# =============================================================================
# 以下为 MLD 风格数据集工厂函数，已废弃
# DMG 训练不再使用，仅保留供阶段0验证脚本（validate.py）调用
# =============================================================================

def get_WordVectorizer(cfg, phase, dataset_name):
    """
    获取文本向量器（MLD 风格，已废弃）

    Args:
        cfg: 配置对象
        phase: 阶段
        dataset_name: 数据集名称

    Returns:
        WordVectorizer 实例
    """
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            from .humanml.utils.word_vectorizer import WordVectorizer
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_datasets(cfg, logger=None, phase="train"):
    """
    获取数据集模块列表（MLD 风格，已废弃）

    **已废弃：DMG 训练不再使用此函数。**
    请使用 SlidingWindowDataModule 代替。

    此函数保留仅供阶段0验证脚本（validate.py）调用，
    用于验证 HumanML3D 数据集的完整性。

    Args:
        cfg: 配置对象
        logger: 日志记录器
        phase: 阶段 ('train', 'test', 'demo')

    Returns:
        datasets: 数据集模块列表
    """
    import warnings
    warnings.warn(
        "get_datasets() is deprecated. DMG training uses SlidingWindowDataModule instead. "
        "This function is kept only for Stage 0 validation (validate.py).",
        DeprecationWarning,
        stacklevel=2,
    )

    from .humanml.datamodule import HumanML3DDataModule

    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []

    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            dataset_key = dataset_name.upper()
            if hasattr(cfg.DATASET, dataset_key):
                dataset_cfg = getattr(cfg.DATASET, dataset_key)
                data_root = dataset_cfg.ROOT
            else:
                raise KeyError(f"DATASET.{dataset_key} not found in config")

            # 获取均值和标准差
            mean, std = get_global_mean_std(dataset_name, cfg)
            mean_eval, std_eval = get_global_mean_std(dataset_name, cfg)

            # 获取文本向量器
            word_vectorizer = get_WordVectorizer(cfg, phase, dataset_name)

            # 获取 collate 函数
            collate_fn = _mld_collate

            # 创建数据集模块
            motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}
            dataset = HumanML3DDataModule(
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=word_vectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=getattr(cfg.DATASET, dataset_key).UNIT_LEN,
            )
            datasets.append(dataset)
        else:
            raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    # 设置全局特征维度
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints

    return datasets


def _mld_collate(batch):
    """MLD 风格的 collate 函数（内部使用）"""
    from torch.utils.data._utils.collate import default_collate
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)
