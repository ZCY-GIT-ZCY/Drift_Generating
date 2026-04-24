"""
阶段2：MotionDriftBank 核心构建器

职责：
1. CLIP 文本编码
2. K-means 聚类
3. VAE latent 提取
4. Bank 组装与持久化
"""

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class BankEntry:
    """
    单个 Bank 条目（属于同一个 text_class）

    存储：
        - his_features: [N, 256] 历史帧 latent
        - future_latents: [N, 256] 未来帧 latent
    """
    his_features: np.ndarray      # [N, 256]
    future_latents: np.ndarray   # [N, 256]


@dataclass
class MotionDriftBank:
    """
    MotionDriftBank 数据结构

    按 text_class 分组存储 latent 特征，供 drift_loss 使用。

    结构：
        {
            text_class_id: BankEntry {
                his_features: [N_class, 256],
                future_latents: [N_class, 256],
            },
            ...
        }

    用途（阶段5）：
        - 按 text_class 采样 future_latents 作为 drift_loss 的正/负样本
        - 使用 his_features 计算核权重 k_hist(his_real, his_gen)
    """

    # 原始数据（用于聚类）
    _class_labels: np.ndarray = field(default=None, repr=False)
    _his_features: np.ndarray = field(default=None, repr=False)
    _future_latents: np.ndarray = field(default=None, repr=False)
    _text_embeddings: np.ndarray = field(default=None, repr=False)

    # 按类别分组的数据
    bank: Dict[int, BankEntry] = field(default_factory=dict)

    # 元信息
    num_classes: int = 512
    max_size_per_class: int = 64
    his_len: int = 20
    future_len: int = 25
    latent_dim: int = 256
    clip_dim: int = 512
    num_windows: int = 0

    # 聚类器（可选，用于推理时计算输入文本的类别）
    _kmeans: Optional[MiniBatchKMeans] = field(default=None, repr=False)

    def build(
        self,
        class_labels: np.ndarray,
        his_features: np.ndarray,
        future_latents: np.ndarray,
        text_embeddings: Optional[np.ndarray] = None,
        kmeans_model: Optional[MiniBatchKMeans] = None,
    ):
        """
        从编码结果构建 Bank

        Args:
            class_labels: [N] 每个窗口的聚类标签
            his_features: [N, 256] 每个窗口 his 的 VAE latent
            future_latents: [N, 256] 每个窗口 future 的 VAE latent
            text_embeddings: [N, 512] 每个窗口的 CLIP 文本向量（可选，用于调试）
            kmeans_model: 训练好的 K-means 模型（用于推理时预测类别）
        """
        assert len(class_labels) == len(his_features) == len(future_latents)
        self.num_windows = len(class_labels)
        self._class_labels = class_labels
        self._his_features = his_features
        self._future_latents = future_latents
        self._text_embeddings = text_embeddings
        self._kmeans = kmeans_model

        # 按类别分组
        unique_classes = np.unique(class_labels)
        self.num_classes = len(unique_classes)

        print(f"[MotionDriftBank] Building bank with {self.num_classes} classes, "
              f"{self.num_windows} total windows, max_size={self.max_size_per_class}")

        for cls_id in unique_classes:
            indices = np.where(class_labels == cls_id)[0]

            # 限制每个类的样本数
            if len(indices) > self.max_size_per_class:
                indices = indices[:self.max_size_per_class]

            his_cls = his_features[indices]
            future_cls = future_latents[indices]

            self.bank[cls_id] = BankEntry(
                his_features=his_cls.astype(np.float32),
                future_latents=future_cls.astype(np.float32),
            )

        print(f"[MotionDriftBank] Bank built successfully")

    def save(self, path: str):
        """
        保存 Bank 到磁盘

        Args:
            path: 保存路径 (.pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 pickle 保存（排除非序列化对象）
        save_dict = {
            "bank": self.bank,
            "num_classes": self.num_classes,
            "max_size_per_class": self.max_size_per_class,
            "his_len": self.his_len,
            "future_len": self.future_len,
            "latent_dim": self.latent_dim,
            "clip_dim": self.clip_dim,
            "num_windows": self.num_windows,
            "_his_features": self._his_features,
            "_future_latents": self._future_latents,
            "_text_embeddings": self._text_embeddings,
            "_kmeans": self._kmeans,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"[MotionDriftBank] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MotionDriftBank":
        """
        从磁盘加载 Bank

        Args:
            path: Bank 文件路径

        Returns:
            MotionDriftBank 实例
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        bank = cls()
        bank.bank = save_dict["bank"]
        bank.num_classes = save_dict["num_classes"]
        bank.max_size_per_class = save_dict["max_size_per_class"]
        bank.his_len = save_dict["his_len"]
        bank.future_len = save_dict["future_len"]
        bank.latent_dim = save_dict["latent_dim"]
        bank.clip_dim = save_dict["clip_dim"]
        bank.num_windows = save_dict["num_windows"]
        bank._his_features = save_dict.get("_his_features")
        bank._future_latents = save_dict.get("_future_latents")
        bank._text_embeddings = save_dict.get("_text_embeddings")
        bank._kmeans = save_dict.get("_kmeans")

        print(f"[MotionDriftBank] Loaded from {path}: "
              f"{bank.num_classes} classes, {bank.num_windows} windows")

        return bank

    def get_class(self, class_id: int) -> Optional[BankEntry]:
        """获取指定类别的条目"""
        return self.bank.get(class_id)

    def predict_text_class(self, text_embeddings: np.ndarray) -> np.ndarray:
        """
        用 K-means 预测文本的类别

        Args:
            text_embeddings: [B, 512] CLIP 文本向量

        Returns:
            class_ids: [B] 预测的类别 ID
        """
        if self._kmeans is None:
            raise ValueError("K-means model not available. "
                             "Load a bank that was built with K-means.")

        text_embeddings = np.asarray(text_embeddings)
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings[np.newaxis, :]

        return self._kmeans.predict(text_embeddings)

    def sample(
        self,
        class_id: int,
        num_pos: int = 16,
        num_neg: int = 32,
        hard_neg: bool = True,
        his_reference: Optional[np.ndarray] = None,
        sigma: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按类别采样正负样本（供阶段5训练使用）

        Args:
            class_id: 目标类别 ID
            num_pos: 正样本数量
            num_neg: 负样本数量
            hard_neg: 是否启用 Hard Negative Mining
            his_reference: [B, 256] 参考的历史帧 latent（用于 hard_neg）
            sigma: 高斯核宽度

        Returns:
            future_latents_pos: [num_pos, 256] 正样本
            future_latents_neg: [num_neg, 256] 负样本
            kernel_weights_pos: [num_pos] 正样本核权重
        """
        entry = self.bank.get(class_id)
        if entry is None or len(entry.his_features) == 0:
            # 某些聚类中心可能是空类（尤其 tiny/debug 或文本重复较高时）
            # 为避免训练阶段直接崩溃，回退到样本数最多的非空类。
            available = [cid for cid, e in self.bank.items() if len(e.his_features) > 0]
            if len(available) == 0:
                raise ValueError("[MotionDriftBank] Fatal: bank is empty, no class can be sampled.")

            fallback_class = max(available, key=lambda cid: len(self.bank[cid].his_features))
            print(
                f"[MotionDriftBank] Warning: class {class_id} missing/empty, "
                f"fallback to class {fallback_class}."
            )
            entry = self.bank[fallback_class]
            class_id = fallback_class

        N = len(entry.his_features)

        # 采样正样本（不足时填充：repeat 不足样本直到达到 num_pos）
        if N >= num_pos:
            pos_indices = np.random.choice(N, num_pos, replace=False)
        else:
            # 正样本不足：随机采样 + repeat 直到达到 num_pos
            pos_indices = np.random.choice(N, num_pos, replace=True)
            print(f"[MotionDriftBank] Warning: class {class_id} has only {N} samples "
                  f"but requested {num_pos} positive samples. "
                  f"Using repeat sampling to fill to {num_pos}.")

        future_pos = entry.future_latents[pos_indices]
        his_pos = entry.his_features[pos_indices]

        # 核权重
        if his_reference is not None:
            # 计算 k_hist(his_ref, his_pos)
            his_ref = np.asarray(his_reference).reshape(-1, self.latent_dim)
            his_pos_rep = np.tile(his_pos[:, np.newaxis, :], (1, len(his_ref), 1))
            his_ref_rep = np.tile(his_ref[np.newaxis, :, :], (len(his_pos), 1, 1))
            dist_sq = np.sum((his_pos_rep - his_ref_rep) ** 2, axis=-1)
            kernel_weights = np.exp(-dist_sq / (2 * sigma ** 2))
            kernel_weights_pos = kernel_weights.mean(axis=1)
        else:
            kernel_weights_pos = np.ones(len(future_pos))

        # 采样负样本
        if hard_neg and his_reference is not None:
            # 从同类中采样 his 最相似的作为 hard negatives
            his_ref = np.asarray(his_reference).reshape(-1, self.latent_dim)
            all_his = entry.his_features  # [N_all, 256]
            all_future = entry.future_latents  # [N_all, 256]

            # 计算与 his_ref 的距离
            dist_sq = np.sum(
                (all_his[:, np.newaxis, :] - his_ref[np.newaxis, :, :]) ** 2,
                axis=-1
            )  # [N_all, B]
            dist_mean = dist_sq.mean(axis=1)  # [N_all]

            # 排除自己（正样本）的 indices
            exclude = set(pos_indices)

            # 按距离排序（距离越小 = his 越相似 = 越 hard）
            sorted_idx = np.argsort(dist_mean)
            hard_indices = [i for i in sorted_idx if i not in exclude]

            if len(hard_indices) >= num_neg:
                neg_indices = np.array(hard_indices[:num_neg])
            else:
                # 先用所有 hard negatives，再从剩余样本中随机补充
                neg_indices = np.array(hard_indices, dtype=np.int64)
                remaining_neg_needed = num_neg - len(hard_indices)
                if remaining_neg_needed > 0:
                    remaining_pool = np.array([
                        i for i in range(len(all_his))
                        if i not in exclude and i not in set(hard_indices)
                    ])
                    if len(remaining_pool) >= remaining_neg_needed:
                        extra_indices = np.random.choice(remaining_pool, remaining_neg_needed, replace=False)
                    else:
                        # 样本不足：从可用池随机补（允许重复）
                        # 若可用池为空（例如该类极小且正样本覆盖全部样本），回退到全量池。
                        fallback_pool = np.array([i for i in range(len(all_his)) if i not in exclude])
                        if len(fallback_pool) == 0:
                            fallback_pool = np.arange(len(all_his))
                        extra_indices = np.random.choice(
                            fallback_pool,
                            remaining_neg_needed,
                            replace=True,
                        )
                    neg_indices = np.concatenate([neg_indices, extra_indices])

            # 兜底：确保返回固定 num_neg 个负样本
            if len(neg_indices) < num_neg:
                pad_pool = np.arange(len(all_his))
                pad = np.random.choice(pad_pool, num_neg - len(neg_indices), replace=True)
                neg_indices = np.concatenate([neg_indices, pad])
            elif len(neg_indices) > num_neg:
                neg_indices = neg_indices[:num_neg]
            neg_indices = neg_indices.astype(np.int64, copy=False)
        else:
            # 随机采样
            if N >= num_neg:
                neg_indices = np.random.choice(N, num_neg, replace=False)
            else:
                neg_indices = np.random.choice(N, num_neg, replace=True)

        future_neg = entry.future_latents[neg_indices]

        return future_pos, future_neg, kernel_weights_pos.astype(np.float32)

    def compute_kernel_weights(
        self,
        his_real: np.ndarray,
        his_gen: np.ndarray,
        sigma: float = 0.1,
    ) -> np.ndarray:
        """
        计算历史帧核权重 k_hist(his_real, his_gen)

        用于 drift_loss 的 weight_pos / weight_neg

        Args:
            his_real: [B, 256] 真实历史帧 latent
            his_gen: [N, 256] 生成/采样的历史帧 latent
            sigma: 高斯核宽度

        Returns:
            weights: [B, N] 核权重矩阵
        """
        his_real = np.asarray(his_real).reshape(-1, self.latent_dim)
        his_gen = np.asarray(his_gen).reshape(-1, self.latent_dim)

        # 计算欧氏距离平方
        dist_sq = np.sum(
            (his_real[:, np.newaxis, :] - his_gen[np.newaxis, :, :]) ** 2,
            axis=-1
        )

        # 高斯核
        weights = np.exp(-dist_sq / (2 * sigma ** 2))
        return weights.astype(np.float32)

    def get_stats(self) -> Dict:
        """获取 Bank 统计信息"""
        class_sizes = [len(e.his_features) for e in self.bank.values()]
        return {
            "num_classes": self.num_classes,
            "num_windows": self.num_windows,
            "max_size_per_class": self.max_size_per_class,
            "mean_size_per_class": np.mean(class_sizes) if class_sizes else 0,
            "min_size_per_class": np.min(class_sizes) if class_sizes else 0,
            "max_size_per_class_actual": np.max(class_sizes) if class_sizes else 0,
            "his_len": self.his_len,
            "future_len": self.future_len,
            "latent_dim": self.latent_dim,
        }


class BankBuilder:
    """
    MotionDriftBank 构建器

    负责：
    1. 加载滑动窗口数据集
    2. CLIP 文本编码（批量）
    3. K-means 聚类
    4. VAE latent 提取（批量）
    5. Bank 组装与保存
    """

    def __init__(
        self,
        vae_encoder: torch.nn.Module,
        clip_encoder: torch.nn.Module,
        device: str = "cuda",
    ):
        """
        初始化 Bank 构建器

        Args:
            vae_encoder: 冻结的 VAE 编码器
            clip_encoder: 冻结的 CLIP 文本编码器
            device: 计算设备
        """
        self.vae = vae_encoder
        self.clip = clip_encoder
        self.device = device

        # 冻结
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        if hasattr(self.clip, 'clip_model') and self.clip.clip_model is not None:
            for param in self.clip.clip_model.parameters():
                param.requires_grad = False
            self.clip.clip_model.eval()

    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """
        批量 CLIP 文本编码

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            embeddings: [N, 512] CLIP 文本向量
        """
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP encoding"):
            batch_texts = texts[i:i + batch_size]
            batch_emb = self.clip.encode_text(batch_texts)
            embeddings.append(batch_emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    @torch.no_grad()
    def encode_motions_vae(
        self,
        dataloader: DataLoader,
        latent_dim: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量 VAE latent 提取

        Args:
            dataloader: 滑动窗口 DataLoader
            latent_dim: latent 维度

        Returns:
            his_features: [N, 256] his 的 VAE latent
            future_latents: [N, 256] future 的 VAE latent
        """
        all_his = []
        all_future = []

        def _to_float_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.from_numpy(x).float()

        for batch in tqdm(dataloader, desc="VAE encoding"):
            # DataLoader 返回两种格式：
            # 1. 无 collate_fn：batch 是列表，每个元素是 (his, future, caption, source) 元组
            # 2. 有 collate_fn 返回字典：batch 是 {'his': ..., 'future': ..., 'text': ...}
            if isinstance(batch, list):
                # 默认无 collate_fn 时，DataLoader 返回列表
                his_list = [item[0] for item in batch]
                future_list = [item[1] for item in batch]
            elif isinstance(batch, dict):
                # collate_sliding_window 返回字典格式 {'his': [B,T,263], 'future': [B,T,263], ...}
                his_batch = batch['his']
                future_batch = batch['future']
                his_list = [his_batch[i] for i in range(his_batch.shape[0])]
                future_list = [future_batch[i] for i in range(future_batch.shape[0])]
            else:
                # 有 collate_fn 返回元组：(his_batch, future_batch, captions, sources)
                his_batch, future_batch, _, _ = batch
                his_list = [his_batch[i] for i in range(his_batch.shape[0])]
                future_list = [future_batch[i] for i in range(future_batch.shape[0])]

            # 合并为批次张量（his/future 都是固定长度）
            his_tensor = torch.stack([_to_float_tensor(h) for h in his_list])
            future_tensor = torch.stack([_to_float_tensor(f) for f in future_list])

            his_batch = his_tensor.to(self.device)
            future_batch = future_tensor.to(self.device)

            # his 编码
            his_lengths = [his_batch.shape[1]] * his_batch.shape[0]
            his_latent, _ = self.vae.encode(his_batch, his_lengths)
            his_latent = his_latent.cpu().numpy().squeeze(1)  # [B, 256]

            # future 编码
            fut_lengths = [future_batch.shape[1]] * future_batch.shape[0]
            fut_latent, _ = self.vae.encode(future_batch, fut_lengths)
            fut_latent = fut_latent.cpu().numpy().squeeze(1)  # [B, 256]

            all_his.append(his_latent)
            all_future.append(fut_latent)

        return np.concatenate(all_his, axis=0), np.concatenate(all_future, axis=0)

    def build_bank(
        self,
        dataset,
        num_classes: int = 512,
        max_size_per_class: int = 64,
        his_len: int = 20,
        future_len: int = 25,
        latent_dim: int = 256,
        batch_size: int = 64,
        clip_batch_size: int = 256,
        num_workers: int = 8,
        save_path: Optional[str] = None,
    ) -> MotionDriftBank:
        """
        完整 Bank 构建流程

        Args:
            dataset: SlidingWindowDataset 实例
            num_classes: 聚类数量
            max_size_per_class: 每个类的最大样本数
            his_len: 历史帧长度
            future_len: 未来帧长度
            latent_dim: latent 维度
            batch_size: VAE 编码批次大小
            clip_batch_size: CLIP 编码批次大小
            num_workers: DataLoader worker 数
            save_path: 保存路径（可选）

        Returns:
            MotionDriftBank 实例
        """
        print("[BankBuilder] Starting Bank construction...")
        print(f"  Dataset: {len(dataset)} windows")
        print(f"  num_classes: {num_classes}, max_size: {max_size_per_class}")

        # Step 1: CLIP 文本编码
        print("\n[BankBuilder] Step 1/4: CLIP text encoding...")
        texts = dataset.get_all_texts()
        text_embeddings = self.encode_texts_batch(texts, batch_size=clip_batch_size)
        print(f"  Text embeddings shape: {text_embeddings.shape}")

        # Step 2: K-means 聚类
        print("\n[BankBuilder] Step 2/4: K-means clustering...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_classes,
            random_state=42,
            batch_size=1024,
            n_init=3,
        )
        class_labels = kmeans.fit_predict(text_embeddings)
        unique, counts = np.unique(class_labels, return_counts=True)
        print(f"  Clustered into {len(unique)} classes")
        print(f"  Min samples per class: {counts.min()}, Max: {counts.max()}, "
              f"Mean: {counts.mean():.1f}")

        # Step 3: VAE latent 提取
        print("\n[BankBuilder] Step 3/4: VAE latent encoding...")
        from dmg.data.sliding_window import collate_sliding_window
        window_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sliding_window,
        )
        his_features, future_latents = self.encode_motions_vae(
            window_loader, latent_dim=latent_dim
        )
        print(f"  his_features shape: {his_features.shape}")
        print(f"  future_latents shape: {future_latents.shape}")

        # Step 4: 组装 Bank
        print("\n[BankBuilder] Step 4/4: Building MotionDriftBank...")
        bank = MotionDriftBank(
            num_classes=num_classes,
            max_size_per_class=max_size_per_class,
            his_len=his_len,
            future_len=future_len,
            latent_dim=latent_dim,
        )
        bank.build(
            class_labels=class_labels,
            his_features=his_features,
            future_latents=future_latents,
            text_embeddings=text_embeddings,
            kmeans_model=kmeans,
        )

        # 保存
        if save_path:
            bank.save(save_path)
            print(f"[BankBuilder] Bank saved to {save_path}")

        return bank
