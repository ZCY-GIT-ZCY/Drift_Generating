"""
DMG Evaluation Script

阶段7：完整评估

在 HumanML3D 测试集上计算：
  - FID (Fréchet Inception Distance)
  - R-Precision (top-1/2/3)
  - MM Dist (Matching Score)
  - Diversity / gt_Diversity

支持：
  - 多种子评估（REPLICATION_TIMES，报告均值 ± 95%CI）
  - 多种 motion encoder（VAE freeze / T2M_MotionEncoder）
  - 结果保存为 JSON

用法：
    python test.py \
        --config configs/config_dmg_humanml3d.yaml \
        --checkpoint ./experiments/xxx/checkpoints/last.ckpt
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dmg.config import parse_args
from dmg.data.sliding_window import SlidingWindowDataModule
from dmg.models.get_model import get_model
from dmg.models.metrics import (
    DMGEvaluator,
    create_motion_encoder,
)
from dmg.utils.logger import create_logger

console = Console()


def print_metrics(title: str, metrics: dict):
    """打印指标表格"""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("指标", style="cyan", no_wrap=True)
    table.add_column("值", style="magenta")
    table.add_column("95% CI", style="dim")

    for key, val in metrics.items():
        if isinstance(val, dict) and 'mean' in val and 'ci' in val:
            table.add_row(key, f"{val['mean']:.4f}", f"± {val['ci']:.4f}")
        elif isinstance(val, (float, int)):
            table.add_row(key, f"{val:.4f}", "")
        else:
            table.add_row(key, str(val), "")

    console.print(table)


def get_metric_stats(all_values: list, replication_times: int):
    """计算均值和 95% 置信区间"""
    values = np.array(all_values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci = 1.96 * std / np.sqrt(replication_times)
    return {'mean': mean, 'ci': ci}


def collect_test_features(
    model,
    datamodule: SlidingWindowDataModule,
    motion_encoder,
    clip_encoder,
    device: str,
    replication_times: int,
    cfg_scale: float = 2.0,
    diversity_times: int = 300,
) -> dict:
    """
    在测试集上收集特征，计算所有评估指标

    Args:
        model: DMG 模型
        datamodule: 测试数据模块
        motion_encoder: 动作特征提取器
        clip_encoder: CLIP 文本编码器
        device: 计算设备
        replication_times: 评估次数（多种子取均值）
        cfg_scale: CFG 推理尺度

    Returns:
        all_metrics: 所有指标（包含各次运行的值和统计量）
    """
    evaluator = DMGEvaluator(
        motion_encoder=motion_encoder,
        clip_encoder=clip_encoder,
        device=device,
        diversity_times=diversity_times,
        seed=cfg.SEED_VALUE,
    )

    all_runs = []

    for run_id in range(replication_times):
        evaluator.reset()
        torch.manual_seed(cfg.SEED_VALUE + run_id)
        np.random.seed(cfg.SEED_VALUE + run_id)

        console.print(f"[cyan]Run {run_id + 1}/{replication_times}...[/cyan]")

        test_loader = datamodule.test_dataloader()
        for batch in test_loader:
            his = batch['his'].to(device)
            future = batch['future'].to(device)
            texts = batch['text']

            # 获取实际帧数
            B = his.shape[0]
            lengths_his = [his.shape[1]] * B
            lengths_future = [future.shape[1]] * B

            with torch.no_grad():
                # forward 的 future 参数传入真实 future，
                # 这样 z_T 会经过 VAE decode → 生成重建帧 recon
                outputs = model(
                    his=his,
                    future=future,  # 传入真实未来帧，触发 VAE decode 路径
                    texts=texts,
                    lengths_his=lengths_his,
                    lengths_future=lengths_future,
                    use_cfg=True,
                    cfg_scale=cfg_scale,
                )

                # recon = z_T → VAE decode → 重建的未来帧
                # future = 真实未来帧（ground truth）
                future_gen = outputs['recon']  # [B, T, 263] 生成
                # future 变量就是 ground truth

                evaluator.update(
                    motions_gen=future_gen,
                    motions_real=future,
                    texts=texts,
                    lengths=lengths_future,
                )

        # 本次运行的指标
        run_metrics = evaluator.compute()
        all_runs.append(run_metrics)

        console.print(f"  FID={run_metrics['FID']:.2f}, "
                      f"R@1={run_metrics['R_precision_top_1']:.3f}, "
                      f"Div={run_metrics['Diversity']:.3f}")

    # 汇总统计
    keys = all_runs[0].keys()
    aggregated = {}
    for key in keys:
        all_vals = [r[key] for r in all_runs]
        aggregated[key] = get_metric_stats(all_vals, replication_times)

    return aggregated


def main():
    # 解析配置
    cfg = parse_args(phase="test")

    # 创建日志
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))

    # 随机种子
    pl.seed_everything(cfg.SEED_VALUE)

    # GPU 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.TEST.get('FOLDER', './results'))
    output_dir = output_dir / cfg.model.model_type / cfg.NAME / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]DMG Evaluation[/bold cyan]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Device: {device}")

    # 创建数据集
    dm = SlidingWindowDataModule(
        cfg=cfg,
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        phase="test",
        split=cfg.TEST.SPLIT,
        dataset_name=cfg.TEST.DATASETS[0],
    )
    dm.setup("test")
    # 获取测试样本数（通过 dataloader 的 dataset）
    test_loader = dm.test_dataloader()
    test_sample_count = len(test_loader.dataset)
    console.print(f"  Test samples: {test_sample_count}")

    # 创建模型
    model = get_model(cfg, dm)
    logger.info(f"model {cfg.model.model_type} loaded")

    # 加载检查点
    checkpoint_path = cfg.TEST.CHECKPOINTS
    if not checkpoint_path:
        console.print("[red]Error: --checkpoint is required[/red]")
        return 1

    if not os.path.exists(checkpoint_path):
        console.print(f"[red]Error: checkpoint not found: {checkpoint_path}[/red]")
        return 1

    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    console.print(f"  Checkpoint loaded: {checkpoint_path}")

    # 创建 motion encoder
    eval_cfg = cfg.get('EVAL', {})
    encoder_mode = eval_cfg.get('encoder', 'vae')  # 'vae' 或 't2m'
    motion_encoder = create_motion_encoder(
        vae_encoder=model.vae,
        mode=encoder_mode,
        output_dim=256,
        device=device,
    )
    motion_encoder.eval()
    console.print(f"  Motion encoder: {encoder_mode}")

    # 评估配置
    replication_times = cfg.TEST.get('REPLICATION_TIMES', 20)
    cfg_scale = cfg.TEST.get('CFG_SCALE', 2.0)

    console.print(f"\n[bold]Evaluation[/bold]")
    console.print(f"  Replication times: {replication_times}")
    console.print(f"  CFG scale: {cfg_scale}")

    # 执行评估
    all_metrics = collect_test_features(
        model=model,
        datamodule=dm,
        motion_encoder=motion_encoder,
        clip_encoder=model.clip,
        device=device,
        replication_times=replication_times,
        cfg_scale=cfg_scale,
        diversity_times=cfg.TEST.get('DIVERSITY_TIMES', 300),
    )

    # 打印结果
    print_metrics("DMG Evaluation Results", all_metrics)

    # 保存结果
    result_file = output_dir / "metrics.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]✓[/green] Results saved to {result_file}")

    logger.info(f"Evaluation complete. Results: {all_metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
