"""
DMG HumanML3D DataModule
复用 MLD 的 HumanML3DDataModule
"""

import numpy as np
import torch

from ...utils.motion_process import process_file, recover_from_ric
from .dataset import Text2MotionDatasetV2, TextOnlyDataset
from ..base import BASEDataModule


class HumanML3DDataModule(BASEDataModule):
    """
    HumanML3D 数据模块

    负责加载和预处理 HumanML3D 数据集
    """

    def __init__(
        self,
        cfg,
        batch_size,
        num_workers,
        collate_fn=None,
        phase="train",
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        # 注意：LightningDataModule 没有 save_hyperparameters 方法，与 LightningModule 不同
        # 直接存储 hparams 到实例属性
        self.hparams = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'phase': phase,
        }
        self.phase = phase
        self.name = "humanml3d"
        self.njoints = 22
        self.cfg = cfg

        # 根据阶段选择数据集类型
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = Text2MotionDatasetV2

        # 获取样本集信息
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats

    def feats2joints(self, features):
        """
        将 RIFKE 特征转换为关节坐标

        Args:
            features: [B, T, 263] RIFKE 特征

        Returns:
            joints: [B, T, 22, 3] 关节坐标
        """
        mean = torch.tensor(self.hparams.mean).to(features.device)
        std = torch.tensor(self.hparams.std).to(features.device)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        """
        将关节坐标转换为 RIFKE 特征

        Args:
            joints: 关节坐标

        Returns:
            features: [B, T, 263] RIFKE 特征
        """
        features = process_file(features, self.njoints)[0]
        return features

    def renorm4t2m(self, features):
        """
        为 T2M 评估器重新归一化

        Args:
            features: [B, T, 263] RIFKE 特征

        Returns:
            features: 重新归一化的特征
        """
        ori_mean = torch.tensor(self.hparams.mean).to(features.device)
        ori_std = torch.tensor(self.hparams.std).to(features.device)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features.device)
        eval_std = torch.tensor(self.hparams.std_eval).to(features.device)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        """
        设置多模态模式

        Args:
            mm_on: 是否开启多模态模式
        """
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(
                self.name_list,
                self.cfg.TEST.MM_NUM_SAMPLES,
                replace=False
            )
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
