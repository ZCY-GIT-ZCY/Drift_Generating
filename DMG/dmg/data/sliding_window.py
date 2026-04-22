"""
阶段2：滑动窗口数据集
用于 MotionDriftBank 离线构建

设计要点：
1. 加载 HumanML3D 原始长序列（已归一化的 RIFKE 特征）
2. 对每条长序列进行重叠滑动窗口采样，生成 (his, future, text) 三元组
3. 每个窗口输出：his RIFKE、future RIFKE、全文描述
4. 用于后续 CLIP 聚类和 VAE 编码

阶段5：训练数据加载
5. 复用 SlidingWindowDataset + SlidingWindowDataModule 提供训练数据
6. collate_fn 返回 {'his', 'future', 'text'} 字典，匹配 DMG.training_step 期望
"""

import codecs as cs
import os
from os.path import join as pjoin
from typing import Dict, List, Tuple, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader


def collate_sliding_window(batch):
    """
    滑动窗口批次整理函数

    返回：
        his_batch: [B, his_len, 263] 填充到 batch 内最大长度
        future_batch: [B, future_len, 263] 填充到 batch 内最大长度
        captions: [B] 字符串列表
        sources: [B] 原始序列名称
    """
    # 按 future 长度排序（降序，保持填充后对齐）
    batch.sort(key=lambda x: x[1].shape[0], reverse=True)

    # 提取各字段
    his_list = [item[0] for item in batch]
    future_list = [item[1] for item in batch]
    captions = [item[2] for item in batch]
    sources = [item[3] for item in batch]

    # 填充到相同长度（所有 his 等长，所有 future 等长）
    his_max = his_list[0].shape[0]
    future_max = future_list[0].shape[0]
    feat_dim = his_list[0].shape[1]

    his_padded = torch.zeros(len(his_list), his_max, feat_dim, dtype=torch.float32)
    future_padded = torch.zeros(len(future_list), future_max, feat_dim, dtype=torch.float32)

    for i, (h, f) in enumerate(zip(his_list, future_list)):
        # 转换为 float32，与 his_padded/future_padded 的 dtype 保持一致
        his_padded[i, :h.shape[0]] = torch.from_numpy(h).float()
        future_padded[i, :f.shape[0]] = torch.from_numpy(f).float()

    return {
        'his': his_padded,
        'future': future_padded,
        'text': captions,
    }


class SlidingWindowDataset(data.Dataset):
    """
    滑动窗口数据集

    对 HumanML3D 原始长序列进行重叠滑动窗口采样，
    生成 his/future/text 三元组用于 Bank 构建。

    输入：
        - motion_dir: .npy 序列目录（每条序列 [T, 263] RIFKE 特征）
        - text_dir: .txt 描述目录（每条序列对应一个文本文件）
        - split_file: 序列 ID 列表文件

    参数：
        - his_len: 历史帧长度（默认 20）
        - future_len: 未来帧长度（默认 25）
        - stride: 滑动步长（默认 5）
        - max_total_len: his + future，用于内存对齐

    输出（每个样本）：
        - his: [his_len, 263] 历史帧 RIFKE
        - future: [future_len, 263] 未来帧 RIFKE
        - caption: str 全文描述
        - source: str 原始序列名称
    """

    def __init__(
        self,
        motion_dir: str,
        text_dir: str,
        split_file: str,
        his_len: int = 20,
        future_len: int = 25,
        stride: int = 5,
        max_total_len: Optional[int] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        min_motion_length: int = 45,
        max_motion_length: int = 300,
        tiny: bool = False,
        debug_samples: int = 100,
        progress_bar: bool = True,
        caption_mode: str = "full",
    ):
        """
        初始化滑动窗口数据集

        Args:
            motion_dir: motion .npy 文件目录（HumanML3D raw RIFKE 数据）
            text_dir: text .txt 文件目录
            split_file: 序列 ID 列表文件路径
            his_len: 历史帧长度
            future_len: 未来帧长度
            stride: 滑动步长
            max_total_len: his_len + future_len
            mean: Z-score 归一化均值（与 MLD Text2MotionDatasetV2 一致）
            std: Z-score 归一化标准差
            min_motion_length: 最小序列长度（过滤短序列）
            max_motion_length: 最大序列长度
            tiny: 调试模式
            debug_samples: 调试模式样本数
            progress_bar: 是否显示进度条
            caption_mode: 文本关联模式（当前仅支持 "full"）
                - "full": 使用整条序列的完整描述
        """
        self.motion_dir = motion_dir
        self.text_dir = text_dir
        self.his_len = his_len
        self.future_len = future_len
        self.stride = stride
        self.max_total_len = max_total_len or (his_len + future_len)
        self.mean = mean
        self.std = std
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.tiny = tiny
        self.debug_samples = debug_samples
        self.caption_mode = caption_mode

        # 读取序列 ID 列表
        with cs.open(split_file, "r") as f:
            id_list = [line.strip() for line in f.readlines()]
        self.id_list = id_list

        # 调试模式
        if tiny:
            id_list = id_list[:debug_samples]

        # 构建滑动窗口三元组
        self.windows = []  # List[(his, future, caption, source)]

        print(f"[SlidingWindowDataset] Loading {len(id_list)} sequences, "
              f"his={his_len}, future={future_len}, stride={stride}")

        total_windows = 0
        skipped_short = 0
        skipped_bad = 0

        # 进度条（参考 MLD 使用 tqdm 或简单 enumerate）
        # 使用 tqdm 风格的简单实现，与 MLD dataset.py 的 track 风格保持一致
        iterator = id_list
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(id_list, desc="Generating windows")
            except ImportError:
                pass  # 无 tqdm 时使用普通迭代

        for seq_name in iterator:
            try:
                # 加载 motion
                motion_path = pjoin(self.motion_dir, seq_name + ".npy")
                motion = np.load(motion_path)  # [T, 263]

                seq_len = len(motion)
                min_needed = self.his_len + self.future_len
                if seq_len < min_needed:
                    skipped_short += 1
                    continue

                # 加载文本
                text_path = pjoin(self.text_dir, seq_name + ".txt")
                captions_data = self._load_texts(text_path, self.caption_mode)

                if not captions_data:
                    skipped_bad += 1
                    continue

                # 滑动窗口采样
                for start in range(0, seq_len - self.his_len - self.future_len + 1, self.stride):
                    his_end = start + self.his_len
                    future_end = his_end + self.future_len

                    his = motion[start:his_end]  # [his_len, 263]
                    future = motion[his_end:future_end]  # [future_len, 263]

                    # 与 MLD Text2MotionDatasetV2 保持一致：进行 Z-score 归一化
                    # MLD 在 dataset.__getitem__ 中执行: motion = (motion - self.mean) / self.std
                    if self.mean is not None and self.std is not None:
                        his = (his - self.mean) / self.std
                        future = (future - self.mean) / self.std

                    # 选择文本（当前仅支持 "full" 模式）
                    caption = captions_data[0]["caption"]

                    self.windows.append((his, future, caption, seq_name))
                    total_windows += 1

            except Exception as e:
                print(f"Error loading {seq_name}: {e}")
                skipped_bad += 1
                continue

        print(f"[SlidingWindowDataset] Generated {total_windows} windows "
              f"(skipped {skipped_short} short, {skipped_bad} bad sequences)")

    def _load_texts(self, text_path: str, mode: str) -> List[Dict]:
        """
        加载文本描述

        Args:
            text_path: 文本文件路径
            mode: 加载模式
                - "full": 返回整条序列的完整描述列表
                - "overlap": 返回带时间重叠的描述

        Returns:
            文本字典列表 [{"caption": str, "f_tag": float, "to_tag": float}]
        """
        texts = []
        try:
            with cs.open(text_path, "r") as f:
                for line in f.readlines():
                    line_split = line.strip().split("#")
                    caption = line_split[0].strip()
                    tokens = line_split[1].split(" ") if len(line_split) > 1 else []
                    f_tag = float(line_split[2]) if len(line_split) > 2 else 0.0
                    to_tag = float(line_split[3]) if len(line_split) > 3 else 0.0

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    if not caption or len(tokens) == 0:
                        continue

                    texts.append({
                        "caption": caption,
                        "tokens": tokens,
                        "f_tag": f_tag,
                        "to_tag": to_tag,
                    })
        except Exception:
            pass

        return texts

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        返回一个滑动窗口三元组

        Args:
            idx: 窗口索引

        Returns:
            his: [his_len, 263] 历史帧
            future: [future_len, 263] 未来帧
            caption: str 文本描述
            source: str 原始序列名
        """
        return self.windows[idx]

    def get_all_texts(self) -> List[str]:
        """获取所有窗口的文本描述（用于 CLIP 聚类）"""
        return [w[2] for w in self.windows]

    def get_text_class_labels(self, class_labels: np.ndarray) -> np.ndarray:
        """
        为每个窗口分配聚类标签

        Args:
            class_labels: [N_clusters] 的聚类标签，N_clusters 应等于窗口数

        Returns:
            聚类标签数组 [num_windows]
        """
        assert len(class_labels) == len(self.windows), (
            f"Class labels count {len(class_labels)} != windows count {len(self.windows)}"
        )
        return class_labels


class SlidingWindowDataModule(pl.LightningDataModule):
    """
    滑动窗口数据模块（阶段5训练用）

    包装 SlidingWindowDataset 为 LightningDataModule 接口，
    提供 train/val/test dataloader，与 DMG.training_step 的
    batch 格式 {'his', 'future', 'text'} 完全匹配。

    使用说明：
        from dmg.data.sliding_window import SlidingWindowDataModule
        dm = SlidingWindowDataModule(cfg, phase="train")
        trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        cfg,
        batch_size: int = 32,
        num_workers: int = 4,
        phase: str = "train",
        split: str = "train",
        dataset_name: str = "humanml3d",
    ):
        """
        初始化滑动窗口数据模块

        Args:
            cfg: DMG 配置对象
            batch_size: 批次大小
            num_workers: 数据加载线程数
            phase: 阶段 ('train', 'test')
            split: 数据划分 ('train', 'val', 'test')
            dataset_name: 数据集名称 ('humanml3d', 'kit')
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase
        self.split = split
        self.dataset_name = dataset_name

        # 从配置获取数据集路径
        dataset_key = dataset_name.upper()
        if hasattr(cfg.DATASET, dataset_key):
            dataset_cfg = getattr(cfg.DATASET, dataset_key)
        else:
            raise KeyError(f"DATASET.{dataset_key} not found in config")

        self.data_root = dataset_cfg.ROOT
        self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
        self.text_dir = pjoin(self.data_root, "texts")
        self.split_file = pjoin(dataset_cfg.SPLIT_ROOT, f"{split}.txt")

        # 滑动窗口参数（从配置读取或使用默认值）
        self.his_len = cfg.WINDOW.HIS_LEN
        self.future_len = cfg.WINDOW.FUTURE_LEN
        self.stride = cfg.WINDOW.STRIDE

        # 归一化参数
        from .get_data import get_global_mean_std
        self.mean, self.std = get_global_mean_std(dataset_name, cfg)

        # 数据集实例（延迟初始化）
        self._dataset = None
        self._test_dataset = None

    @property
    def nfeats(self) -> int:
        """特征维度（固定为 263）"""
        return 263

    @property
    def test_dataset(self):
        """测试数据集（需先调用 test_dataloader()）"""
        return self._test_dataset

    @property
    def njoints(self) -> int:
        """关节数（固定为 22）"""
        return 22

    def setup(self, stage=None):
        """初始化数据集"""
        if self._dataset is None:
            self._dataset = SlidingWindowDataset(
                motion_dir=self.motion_dir,
                text_dir=self.text_dir,
                split_file=self.split_file,
                his_len=self.his_len,
                future_len=self.future_len,
                stride=self.stride,
                mean=self.mean,
                std=self.std,
                min_motion_length=self.cfg.DATASET.SAMPLER.MIN_LEN,
                max_motion_length=self.cfg.DATASET.SAMPLER.MAX_LEN,
                tiny=False,
                progress_bar=True,
            )

    def train_dataloader(self):
        self._dataset = None  # 重置，确保 setup 重新构建训练集
        self.setup()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_sliding_window,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # 不调用 setup()，直接构建 val dataset（避免重复创建 train dataset）
        self._dataset = None
        dataset_key = self.dataset_name.upper()
        dataset_cfg = getattr(self.cfg.DATASET, dataset_key)
        val_split_file = pjoin(dataset_cfg.SPLIT_ROOT, "val.txt")
        val_dataset = SlidingWindowDataset(
            motion_dir=self.motion_dir,
            text_dir=self.text_dir,
            split_file=val_split_file,
            his_len=self.his_len,
            future_len=self.future_len,
            stride=self.stride,
            mean=self.mean,
            std=self.std,
            min_motion_length=self.cfg.DATASET.SAMPLER.MIN_LEN,
            max_motion_length=self.cfg.DATASET.SAMPLER.MAX_LEN,
            tiny=False,
            progress_bar=True,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_sliding_window,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """测试集 dataloader（按需创建并缓存）"""
        if self._test_dataset is None:
            self._build_test_dataset()
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_sliding_window,
            persistent_workers=True,
        )

    def _build_test_dataset(self):
        """构建测试集（内部方法，按需调用）"""
        dataset_key = self.dataset_name.upper()
        dataset_cfg = getattr(self.cfg.DATASET, dataset_key)
        test_split_file = pjoin(dataset_cfg.SPLIT_ROOT, "test.txt")
        self._test_dataset = SlidingWindowDataset(
            motion_dir=self.motion_dir,
            text_dir=self.text_dir,
            split_file=test_split_file,
            his_len=self.his_len,
            future_len=self.future_len,
            stride=self.stride,
            mean=self.mean,
            std=self.std,
            min_motion_length=self.cfg.DATASET.SAMPLER.MIN_LEN,
            max_motion_length=self.cfg.DATASET.SAMPLER.MAX_LEN,
            tiny=False,
            progress_bar=True,
        )
