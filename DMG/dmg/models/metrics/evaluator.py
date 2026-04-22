"""
DMG Evaluation Metrics

阶段7：动作生成评估指标

复用 MLD 评估管线中可迁移的部分，补充 DMG 特有的评估接口。

评估指标（pipeline.md §8.2）：
  - FID: 特征分布 Fréchet 距离
  - R-Precision: top-1/2/3 文本-动作检索精度
  - MM Dist: 文本-动作余弦匹配距离
  - Diversity: 生成样本多样性

关键设计：
  1. motion encoder 提取动作特征（VAE freeze 或 T2M）
  2. CLIP 提取文本特征
  3. 指标计算全部在 PyTorch 中，无需 MLD 依赖
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ..architectures.dmg_motion_encoder import DMGMotionEncoder


# =============================================================================
# 核心评估指标计算
# =============================================================================


def calculate_fid(
    features_gen: np.ndarray,
    features_real: np.ndarray,
) -> float:
    """
    计算 FID (Fréchet Inception Distance)

    使用 numpy 实现，避免 PyTorch GPU/CPU 转换开销。

    Args:
        features_gen:  [N, D] 生成样本特征
        features_real: [M, D] 真实样本特征

    Returns:
        fid: float
    """
    mu_gen = np.mean(features_gen, axis=0)
    mu_real = np.mean(features_real, axis=0)

    sigma_gen = np.cov(features_gen, rowvar=False)
    sigma_real = np.cov(features_real, rowvar=False)

    diff = mu_gen - mu_real
    covmean, _ = _sqrt_cov_matrix(sigma_gen @ sigma_real)

    fid = float(np.sum(diff ** 2) + np.trace(
        sigma_gen + sigma_real - 2 * covmean
    ))
    return fid


def _sqrt_cov_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算协方差矩阵的平方根（Frechet 距离核心）"""
    eigval, eigvec = np.linalg.eigh(M)
    eigval = np.maximum(eigval, 0)
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T, eigvec


def calculate_r_precision(
    motion_features: np.ndarray,
    text_features: np.ndarray,
    top_k: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, float]:
    """
    计算 R-Precision（文本-动作检索精度）

    对每个文本，找到最近的 K 个动作样本，检查正确匹配是否在 top-K 内。

    Args:
        motion_features: [N, D] 动作特征
        text_features:  [N, D] 文本特征（一一对应）

    Returns:
        {'R_precision_top_1': float, 'R_precision_top_2': float, ...}
    """
    # 归一化 → 余弦相似度矩阵
    motion_norm = motion_features / (np.linalg.norm(motion_features, axis=1, keepdims=True) + 1e-8)
    text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
    sim_mat = motion_norm @ text_norm.T  # [N, N]

    N = sim_mat.shape[0]
    results = {}

    for k in top_k:
        correct = 0
        for i in range(N):
            # 找到 top-k 最相似的动作
            topk_indices = np.argsort(sim_mat[i])[::-1][:k]
            # 检查正确匹配（对角线元素）是否在 top-k
            if i in topk_indices:
                correct += 1
        results[f'R_precision_top_{k}'] = correct / N

    return results


def calculate_matching_score(
    motion_features: np.ndarray,
    text_features: np.ndarray,
) -> float:
    """
    计算 Matching Score / MM Dist

    对角线元素的均值（正确匹配的平均相似度）。

    Args:
        motion_features: [N, D] 动作特征
        text_features:  [N, D] 文本特征

    Returns:
        matching_score: float
    """
    motion_norm = motion_features / (np.linalg.norm(motion_features, axis=1, keepdims=True) + 1e-8)
    text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
    sim_mat = motion_norm @ text_norm.T
    matching_score = float(np.mean(np.diag(sim_mat)))
    return matching_score


def calculate_diversity(
    features: np.ndarray,
    num_samples: int = 300,
) -> float:
    """
    计算 Diversity（生成多样性）

    随机采样对，测量平均成对距离。

    Args:
        features: [N, D] 特征
        num_samples: 采样对数

    Returns:
        diversity: float
    """
    N = len(features)
    if N < 2:
        return 0.0

    indices = np.random.permutation(N)
    diversity_sum = 0.0
    count = 0

    for _ in range(num_samples):
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i != j:
            dist = np.linalg.norm(features[i] - features[j])
            diversity_sum += dist
            count += 1

    return float(diversity_sum / count) if count > 0 else 0.0


# =============================================================================
# 评估器类
# =============================================================================


class DMGEvaluator:
    """
    DMG 动作生成评估器

    使用方法：
        evaluator = DMGEvaluator(motion_encoder, clip_encoder, device='cuda')
        evaluator.reset()

        for batch in test_loader:
            motions_gen, motions_real, texts, lengths = batch
            evaluator.update(motions_gen, motions_real, texts, lengths)

        metrics = evaluator.compute()
        # {'FID': float, 'R_precision_top_1': float, ...}
    """

    def __init__(
        self,
        motion_encoder: DMGMotionEncoder,
        clip_encoder,          # MldTextEncoder
        device: str = 'cuda',
        diversity_times: int = 300,
        seed: int = 42,
    ):
        self.encoder = motion_encoder
        self.clip = clip_encoder
        self.device = device
        self.diversity_times = diversity_times

        # 随机种子（确保可复现）
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 存储特征（整个测试集累积）
        self._gen_features: List[np.ndarray] = []
        self._real_features: List[np.ndarray] = []
        self._texts: List[str] = []
        self._text_features: List[np.ndarray] = []

    def reset(self):
        """重置评估器状态"""
        self._gen_features.clear()
        self._real_features.clear()
        self._texts.clear()
        self._text_features.clear()

    @torch.no_grad()
    def update(
        self,
        motions_gen: Tensor,
        motions_real: Tensor,
        texts: List[str],
        lengths: List[int],
    ):
        """
        更新评估器（逐 batch）

        Args:
            motions_gen:  [B, T, 263] 生成的未来帧
            motions_real: [B, T, 263] 真实未来帧
            texts:         文本描述列表 [B]
            lengths:       各序列帧数 [B]
        """
        motions_gen = motions_gen.to(self.device)
        motions_real = motions_real.to(self.device)

        # 动作编码
        feat_gen = self.encoder.encode(motions_gen, lengths)  # [B, D]
        feat_real = self.encoder.encode(motions_real, lengths)  # [B, D]

        # 文本编码
        text_emb = self.clip.encode_text(texts)  # [B, 512]
        text_emb_np = text_emb.cpu().numpy()

        # 存储
        self._gen_features.append(feat_gen.cpu().numpy())
        self._real_features.append(feat_real.cpu().numpy())
        self._texts.extend(texts)
        self._text_features.append(text_emb_np)

    def compute(self) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            metrics: {'FID': float, 'R_precision_top_1': float, ...}
        """
        all_gen = np.concatenate(self._gen_features, axis=0)
        all_real = np.concatenate(self._real_features, axis=0)
        all_text = np.concatenate(self._text_features, axis=0)

        # 1. FID
        fid = calculate_fid(all_gen, all_real)

        # 2. R-Precision（使用真实动作的特征）
        r_prec = calculate_r_precision(all_gen, all_text, top_k=(1, 2, 3))

        # 3. MM Dist / Matching Score
        mm_dist = calculate_matching_score(all_gen, all_text)

        # 4. Diversity（使用生成样本）
        diversity = calculate_diversity(all_gen, self.diversity_times)

        # 5. GT Diversity（对照组）
        gt_diversity = calculate_diversity(all_real, self.diversity_times)

        metrics = {
            'FID': fid,
            'R_precision_top_1': r_prec.get('R_precision_top_1', 0.0),
            'R_precision_top_2': r_prec.get('R_precision_top_2', 0.0),
            'R_precision_top_3': r_prec.get('R_precision_top_3', 0.0),
            'MM_Dist': mm_dist,
            'Diversity': diversity,
            'gt_Diversity': gt_diversity,
        }

        return metrics

    @property
    def num_samples(self) -> int:
        return len(self._texts)


class DMGOfflineEvaluator:
    """
    离线评估器（不需要 PyTorch 模型）

    给定已保存的 .npy 特征文件，直接计算指标。
    用于 pipeline.md §7.1 完整评估。

    用法：
        evaluator = DMGOfflineEvaluator()
        evaluator.add_generated(features_gen, text_features)
        evaluator.add_real(features_real)
        evaluator.add_text_features(text_features)
        metrics = evaluator.compute()
    """

    def __init__(self, diversity_times: int = 300, seed: int = 42):
        self.diversity_times = diversity_times
        np.random.seed(seed)
        self.gen_features: List[np.ndarray] = []
        self.real_features: List[np.ndarray] = []
        self.text_features: List[np.ndarray] = []

    def add_generated(self, features: np.ndarray):
        """添加生成样本特征"""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        self.gen_features.append(features)

    def add_real(self, features: np.ndarray):
        """添加真实样本特征"""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        self.real_features.append(features)

    def add_text_features(self, features: np.ndarray):
        """添加文本特征"""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        self.text_features.append(features)

    def compute(self) -> Dict[str, float]:
        all_gen = np.concatenate(self.gen_features, axis=0)
        all_real = np.concatenate(self.real_features, axis=0) if self.real_features else None
        all_text = np.concatenate(self.text_features, axis=0)

        metrics = {}

        if all_real is not None and len(all_real) > 0:
            metrics['FID'] = calculate_fid(all_gen, all_real)
        else:
            metrics['FID'] = -1.0

        metrics.update(calculate_r_precision(all_gen, all_text, top_k=(1, 2, 3)))
        metrics['MM_Dist'] = calculate_matching_score(all_gen, all_text)
        metrics['Diversity'] = calculate_diversity(all_gen, self.diversity_times)

        if all_real is not None and len(all_real) > 0:
            metrics['gt_Diversity'] = calculate_diversity(all_real, self.diversity_times)

        return metrics
