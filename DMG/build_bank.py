"""
阶段2：MotionDriftBank 构建脚本

用法：
    python build_bank.py --config configs/config_bank.yaml

功能：
    1. 加载 HumanML3D 数据集
    2. 滑动窗口采样，生成 his/future/text 三元组
    3. CLIP 编码文本 + K-means 聚类
    4. VAE 编码 his/future
    5. 组装 MotionDriftBank 并保存

依赖：
    - 阶段1的 VAE 和 CLIP 编码器
    - HumanML3D 数据集（已归一化的 RIFKE 特征）
"""

import os
import sys
from datetime import datetime
from os.path import join as pjoin
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# 导入 DMG 模块
from dmg.config import parse_args
from dmg.data.sliding_window import SlidingWindowDataset
from dmg.data.bank import BankBuilder, MotionDriftBank
from dmg.models.architectures.mld_vae import MldVae
from dmg.models.architectures.mld_clip import MldTextEncoder
from dmg.utils.logger import create_logger

console = Console()


def load_bank_config(config_path: str) -> dict:
    """加载 Bank 构建配置"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_vae(cfg: dict, device: str) -> torch.nn.Module:
    """加载 VAE 编码器"""
    console.print("\n[cyan]Loading VAE encoder...[/cyan]")

    vae = MldVae(
        nfeats=263,
        latent_dim=cfg['vae']['latent_dim'],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        arch='all_encoder',
    )

    # 尝试加载预训练权重
    pretrained_path = cfg['vae'].get('pretrained_path', '')
    if pretrained_path and os.path.exists(pretrained_path):
        console.print(f"  Loading pretrained VAE from {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            vae_dict = {}
            for k, v in state_dict.items():
                if k.startswith('vae.'):
                    vae_dict[k.replace('vae.', '')] = v
                elif '.' not in k:
                    vae_dict[k] = v
            vae.load_state_dict(vae_dict, strict=False)
            console.print(f"  [green]✓[/green] VAE pretrained weights loaded")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Failed to load VAE weights: {e}")
    else:
        console.print(f"  [yellow]⚠[/yellow] No pretrained VAE specified, using random initialization")

    vae = vae.to(device)
    vae.eval()

    # 冻结
    for param in vae.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in vae.parameters())
    console.print(f"  VAE parameters: {num_params:,}")

    return vae


def load_clip(cfg: dict, device: str) -> torch.nn.Module:
    """加载 CLIP 文本编码器"""
    console.print("\n[cyan]Loading CLIP text encoder...[/cyan]")

    clip_encoder = MldTextEncoder(
        modelpath=cfg['clip']['model_path'],
        precision='fp32',
    )

    clip_device = cfg['clip'].get('device', device)
    if clip_device == 'cpu':
        console.print(f"  [yellow]⚠[/yellow] CLIP on CPU, encoding will be slow")
    else:
        # CLIP 模型在 CPU 上加载
        console.print(f"  CLIP device: cpu (CLIP text encoding is CPU-friendly)")

    clip_encoder.clip_device = 'cpu'
    if hasattr(clip_encoder, 'clip_model') and clip_encoder.clip_model is not None:
        for param in clip_encoder.clip_model.parameters():
            param.requires_grad = False
        clip_encoder.clip_model.eval()

    console.print(f"  [green]✓[/green] CLIP encoder ready")

    return clip_encoder


def build_dataset(cfg: dict, split_file: str, mean: np.ndarray, std: np.ndarray):
    """构建滑动窗口数据集"""
    console.print("\n[cyan]Building sliding window dataset...[/cyan]")

    bank_cfg = cfg['bank']
    motion_dir = cfg['dataset']['motion_dir']
    text_dir = cfg['dataset']['text_dir']

    dataset = SlidingWindowDataset(
        motion_dir=motion_dir,
        text_dir=text_dir,
        split_file=split_file,
        his_len=bank_cfg['his_len'],
        future_len=bank_cfg['future_len'],
        stride=bank_cfg['stride'],
        max_total_len=bank_cfg['max_total_len'],
        mean=mean,
        std=std,
        min_motion_length=bank_cfg.get('min_motion_length', 45),
        max_motion_length=bank_cfg.get('max_motion_length', 300),
        tiny=cfg['process']['tiny'],
        debug_samples=cfg['process']['debug_samples'],
        progress_bar=cfg['process'].get('progress_bar', True),
    )

    console.print(f"  [green]✓[/green] Generated {len(dataset)} windows")
    return dataset


def get_dataset_metadata(cfg: dict, split: str):
    """
    从 DMG 配置获取数据集路径和归一化参数

    Returns:
        (data_root, motion_dir, text_dir, split_file, mean, std)
    """
    from dmg.data.get_data import get_global_mean_std

    dm_cfg = parse_args(phase="train")

    # 获取数据集配置
    dataset_name = cfg['dataset'].get('name', 'humanml3d')
    dataset_key = dataset_name.upper()
    if not hasattr(dm_cfg.DATASET, dataset_key):
        raise KeyError(f"DATASET.{dataset_key} not found in config")

    dataset_cfg = getattr(dm_cfg.DATASET, dataset_key)
    data_root = dataset_cfg.ROOT

    # 获取全局均值和标准差（用于 Bank 构建的归一化）
    # 与 MLD Text2MotionDatasetV2 保持一致，使用 HumanML3D 预计算的全局统计量
    mean, std = get_global_mean_std(dataset_name, dm_cfg)

    # 构建路径
    motion_dir = pjoin(data_root, "new_joint_vecs")
    text_dir = pjoin(data_root, "texts")
    split_file = pjoin(dataset_cfg.SPLIT_ROOT, f"{split}.txt")

    return data_root, motion_dir, text_dir, split_file, mean, std


def main():
    console.print(Panel.fit(
        "[bold cyan]DMG Stage 2: MotionDriftBank Construction[/bold cyan]\n"
        "滑动窗口采样 + CLIP 聚类 + VAE latent 提取",
        border_style="cyan"
    ))

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Build MotionDriftBank")
    parser.add_argument('--config', type=str,
                        default='configs/config_bank.yaml',
                        help='Path to bank config file')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split (train/val/test)')
    parser.add_argument('--tiny', action='store_true',
                        help='Debug mode with fewer samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--skip_vae', action='store_true',
                        help='Skip VAE encoding (use cached features)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output bank path (overrides config)')
    args = parser.parse_args()

    # 加载配置
    cfg = load_bank_config(args.config)
    if args.tiny:
        cfg['process']['tiny'] = True
    if args.device:
        cfg.setdefault('device', args.device)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[cyan]Device:[/cyan] {device}")

    # 获取数据集路径和归一化参数
    console.print("\n[cyan]Loading dataset metadata...[/cyan]")

    try:
        data_root, motion_dir, text_dir, split_file_cfg, mean, std = get_dataset_metadata(cfg, args.split)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load from DMG config: {e}[/yellow]")
        console.print(f"  Please ensure cfg['dataset'] contains the correct paths")
        console.print(f"  Or run from DMG directory: python build_bank.py")
        return 1

    console.print(f"  Motion dir: {motion_dir}")
    console.print(f"  Text dir: {text_dir}")
    console.print(f"  Split file: {split_file_cfg}")
    console.print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")

    # 检查数据目录存在
    if not os.path.exists(motion_dir):
        console.print(f"[red]Error: Motion directory not found: {motion_dir}[/red]")
        return 1
    if not os.path.exists(text_dir):
        console.print(f"[red]Error: Text directory not found: {text_dir}[/red]")
        return 1
    if not os.path.exists(split_file_cfg):
        console.print(f"[red]Error: Split file not found: {split_file_cfg}[/red]")
        return 1

    # 加载模型
    vae = load_vae(cfg, device)
    clip_encoder = load_clip(cfg, device)

    # 构建滑动窗口数据集
    dataset = build_dataset(cfg, split_file_cfg, mean, std)

    if len(dataset) == 0:
        console.print("[red]Error: No windows generated. Check dataset paths.[/red]")
        return 1

    # 创建 Bank 构建器
    bank_builder = BankBuilder(
        vae_encoder=vae,
        clip_encoder=clip_encoder,
        device=device,
    )

    # 构建 Bank
    output_dir = Path(cfg['bank']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(output_dir / cfg['bank']['bank_filename'])

    console.print(f"\n[cyan]Building MotionDriftBank...[/cyan]")
    console.print(f"  Output: {output_path}")

    bank_cfg = cfg['bank']
    bank = bank_builder.build_bank(
        dataset=dataset,
        num_classes=bank_cfg['num_text_classes'],
        max_size_per_class=bank_cfg['max_size_per_class'],
        his_len=bank_cfg['his_len'],
        future_len=bank_cfg['future_len'],
        latent_dim=cfg['vae']['latent_dim'][1],  # [token, dim] → 取 dim 部分
        batch_size=bank_cfg.get('batch_size', 64),
        num_workers=cfg['process']['num_workers'],
        save_path=output_path,
    )

    # 打印统计信息
    stats = bank.get_stats()
    console.print("\n[bold green]Bank construction complete![/bold green]")
    console.print(f"  Classes: {stats['num_classes']}")
    console.print(f"  Total windows: {stats['num_windows']}")
    console.print(f"  Mean size per class: {stats['mean_size_per_class']:.1f}")
    console.print(f"  His/Future length: {stats['his_len']}/{stats['future_len']}")
    console.print(f"  Latent dim: {stats['latent_dim']}")
    console.print(f"  Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())