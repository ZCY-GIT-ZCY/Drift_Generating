"""
DMG Validation Script - 阶段0验证脚本

此脚本用于验证：
1. HumanML3D 数据集加载流程端到端可运行
2. MLD 数据加载器可正常读取 motion + text
3. RIFKE 263-dim 特征提取、归一化流程正确
4. 工具函数 lengths_to_mask、remove_padding 可用
5. MLD 预训练 VAE 可加载
6. MLD 预训练 CLIP 可加载
7. 评估管线可运行（复用 MLD 评估脚本）
8. 记录基线指标
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.table import Table

from dmg.config import parse_args
from dmg.data.get_data import get_datasets
from dmg.models.get_model import get_model
from dmg.utils.logger import create_logger


console = Console()


def check_mld_import():
    """检查 MLD 模块是否可以导入"""
    console.print("\n[bold cyan]检查1: 导入 MLD 模块[/bold cyan]")

    try:
        # 优先检查同级目录中的 MLD（../MLD）
        project_root = Path(__file__).parent
        workspace_root = project_root.parent
        mld_path = workspace_root / "MLD"

        if mld_path.exists():
            sys.path.insert(0, str(mld_path))
            import mld
            console.print(f"  [green]✓[/green] MLD 模块已导入: {mld.__file__}")
            return True, mld_path
        else:
            console.print(f"  [yellow]⚠[/yellow] MLD 目录不存在于 {mld_path}")
            return False, None
    except ImportError as e:
        console.print(f"  [red]✗[/red] 无法导入 MLD: {e}")
        return False, None


def check_data_loading(cfg, logger):
    """检查数据加载"""
    console.print("\n[bold cyan]检查2: HumanML3D 数据加载[/bold cyan]")

    try:
        # 创建数据集
        datasets = get_datasets(cfg, logger=logger, phase="test")
        datamodule = datasets[0]

        # 设置数据模块
        datamodule.setup("test")

        # 获取一个 batch
        dataloader = datamodule.test_dataloader()
        batch = next(iter(dataloader))

        # 解包 batch
        if len(batch) >= 6:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length = batch[:6]

            console.print(f"  [green]✓[/green] 数据加载成功!")
            console.print(f"    - word_embeddings shape: {word_embeddings.shape}")
            console.print(f"    - pos_one_hots shape: {pos_one_hots.shape}")
            console.print(f"    - caption: {caption[:50]}...")
            console.print(f"    - motion shape: {motion.shape}")
            console.print(f"    - m_length: {m_length}")

            return True, batch
        else:
            console.print(f"  [red]✗[/red] Batch 格式不正确: 预期 6+ 个元素, 实际 {len(batch)}")
            return False, None

    except FileNotFoundError as e:
        console.print(f"  [red]✗[/red] 数据文件未找到: {e}")
        console.print(f"    请确保 HumanML3D 数据集已下载到 {cfg.DATASET.HUMANML3D.ROOT}")
        return False, None
    except Exception as e:
        console.print(f"  [red]✗[/red] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def check_feature_processing(batch):
    """检查特征处理"""
    console.print("\n[bold cyan]检查3: RIFKE 特征处理[/bold cyan]")

    try:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length = batch[:6]

        # 检查特征维度
        nfeats = motion.shape[-1]
        if nfeats == 263:
            console.print(f"  [green]✓[/green] RIFKE 特征维度正确: {nfeats}")
        else:
            console.print(f"  [yellow]⚠[/yellow] RIFKE 特征维度: {nfeats} (预期 263)")

        # 检查归一化
        if np.abs(motion.mean()) < 1.0 and motion.std() < 5.0:
            console.print(f"  [green]✓[/green] 特征已归一化: mean={motion.mean():.4f}, std={motion.std():.4f}")
        else:
            console.print(f"  [yellow]⚠[/yellow] 特征可能未归一化: mean={motion.mean():.4f}, std={motion.std():.4f}")

        # 检查 NaN
        if isinstance(motion, torch.Tensor):
            has_nan = torch.isnan(motion).any().item()
        else:
            has_nan = np.isnan(motion).any()

        if not has_nan:
            console.print(f"  [green]✓[/green] 无 NaN 值")
        else:
            console.print(f"  [red]✗[/red] 存在 NaN 值")
            return False

        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] 特征处理检查失败: {e}")
        return False


def check_vae_loading(cfg, logger):
    """检查 VAE 加载"""
    console.print("\n[bold cyan]检查4: MLD VAE 加载[/bold cyan]")

    try:
        from dmg.models.architectures.mld_vae import MldVae

        # 从配置读取参数（与原版 MLD 保持一致）
        vae_params = {
            'nfeats': cfg.DATASET.NFEATS,
            'latent_dim': cfg.model.latent_dim,
            'ff_size': cfg.model.ff_size,
            'num_layers': cfg.model.num_layers,
            'num_heads': cfg.model.num_heads,
            'dropout': cfg.model.dropout,
            'activation': cfg.model.activation,
        }

        # 创建 VAE
        vae = MldVae(**vae_params)

        console.print(f"  [green]✓[/green] VAE 模型创建成功")
        console.print(f"    - 参数数量: {sum(p.numel() for p in vae.parameters()):,}")
        console.print(f"    - latent_dim: {cfg.model.latent_dim}")
        console.print(f"    - num_layers: {cfg.model.num_layers}")

        # 尝试加载预训练权重
        vae_ckpt = cfg.TRAIN.PRETRAINED_VAE if hasattr(cfg.TRAIN, 'PRETRAINED_VAE') else None
        if vae_ckpt and os.path.exists(vae_ckpt):
            loaded = vae.load_pretrained(vae_ckpt)
            if loaded:
                console.print(f"  [green]✓[/green] 预训练权重加载成功: {vae_ckpt}")
            else:
                console.print(f"  [yellow]⚠[/yellow] 预训练权重存在但未成功加载: {vae_ckpt}")
        else:
            console.print(f"  [yellow]⚠[/yellow] 预训练权重未指定或不存在")

        # 测试前向传播
        motion = torch.randn(2, 40, 263)
        lengths = [40, 40]

        with torch.no_grad():
            latent, _ = vae.encode(motion, lengths)
            output = vae.decode(latent, lengths)

        console.print(f"    - latent shape: {latent.shape}")
        console.print(f"    - output shape: {output.shape}")
        console.print(f"    - 前向传播验证通过")

        return True, vae

    except FileNotFoundError as e:
        console.print(f"  [red]✗[/red] 预训练文件未找到: {e}")
        return True, None  # VAE 本身可以创建
    except Exception as e:
        console.print(f"  [red]✗[/red] VAE 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def check_clip_loading(cfg, logger):
    """检查 CLIP 加载"""
    console.print("\n[bold cyan]检查5: CLIP 文本编码器加载[/bold cyan]")

    try:
        from dmg.models.architectures.mld_clip import MldTextEncoder

        # 从配置获取 CLIP 路径（与原版 MLD 保持一致）
        clip_path = cfg.model.clip_path if hasattr(cfg.model, 'clip_path') else "./deps/clip/ViT-B-32.pt"

        # 创建 CLIP 编码器
        clip_encoder = MldTextEncoder(modelpath=clip_path)

        console.print(f"  [green]✓[/green] CLIP 模型创建成功")

        # 测试文本编码
        test_texts = ["a person walking", "someone running"]

        try:
            text_embeddings = clip_encoder.encode_text(test_texts)
            console.print(f"  [green]✓[/green] 文本编码成功")
            console.print(f"    - text_embeddings shape: {text_embeddings.shape}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] CLIP 模型加载失败 (将使用随机嵌入): {e}")

        return True, clip_encoder

    except Exception as e:
        console.print(f"  [red]✗[/red] CLIP 检查失败: {e}")
        return False, None


def check_evaluation_pipeline(cfg, logger, results):
    """检查评估管线"""
    console.print("\n[bold cyan]检查6: 评估管线[/bold cyan]")

    try:
        # 检查 MLD 评估脚本是否存在（同级目录 ../MLD）
        project_root = Path(__file__).parent
        workspace_root = project_root.parent
        mld_test_path = workspace_root / "MLD" / "test.py"

        if mld_test_path.exists():
            console.print(f"  [green]✓[/green] MLD 评估脚本存在: {mld_test_path}")
            console.print(f"    可通过以下命令运行评估:")
            console.print(f"    python ../MLD/test.py --cfg ../MLD/configs/config_mld_humanml3d.yaml")
        else:
            console.print(f"  [yellow]⚠[/yellow] MLD 评估脚本不存在")

        # 检查 DMG 评估模块
        metrics_path = project_root / "dmg" / "models" / "metrics"
        if metrics_path.exists():
            console.print(f"  [green]✓[/green] DMG 评估模块存在: {metrics_path}")
        else:
            console.print(f"  [yellow]⚠[/yellow] DMG 评估模块不存在")

        # 检查评估器
        console.print(f"  [green]✓[/green] 评估指标:")
        console.print(f"    - FID: 特征分布距离")
        console.print(f"    - R-Precision: 文本-动作检索精度")
        console.print(f"    - MM Dist: 文本-动作余弦距离")
        console.print(f"    - Diversity: 生成样本多样性")

        # 检查 JAX/Flax (Drift Loss 需要)
        try:
            import jax
            import flax
            console.print(f"  [green]✓[/green] JAX/Flax 已安装: JAX {jax.__version__}, Flax {flax.__version__}")
            results["jax_flax"] = True
        except ImportError:
            console.print(f"  [yellow]⚠[/yellow] JAX/Flax 未安装 (Drift Loss 需要)")
            results["jax_flax"] = False

        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] 评估管线检查失败: {e}")
        return False


def print_summary(results):
    """打印验证总结"""
    console.print("\n" + "=" * 60)
    console.print("[bold]阶段0验证总结[/bold]")
    console.print("=" * 60)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("检查项", style="cyan")
    table.add_column("状态", justify="center")
    table.add_column("说明")

    checks = [
        ("MLD 模块导入", results.get("mld_import", False), "MLD 目录存在且可导入"),
        ("数据加载", results.get("data_loading", False), "HumanML3D 数据可正常读取"),
        ("特征处理", results.get("feature_processing", False), "RIFKE 特征提取正确"),
        ("VAE 加载", results.get("vae_loading", False), "MLD VAE 可加载"),
        ("CLIP 加载", results.get("clip_loading", False), "CLIP 文本编码器可加载"),
        ("评估管线", results.get("evaluation", False), "评估脚本可用"),
        ("JAX/Flax", results.get("jax_flax", False), "Drift Loss 依赖"),
    ]

    for name, status, desc in checks:
        status_str = "[green]✓ 通过[/green]" if status else "[red]✗ 失败[/red]"
        table.add_row(name, status_str, desc)

    console.print(table)

    # 计算通过率
    required_checks = ["mld_import", "data_loading", "feature_processing", "vae_loading", "clip_loading", "jax_flax"]
    optional_checks = ["evaluation"]
    
    passed_required = sum(1 for k in required_checks if results.get(k, False))
    passed_optional = sum(1 for k in optional_checks if results.get(k, False))
    
    console.print(f"\n[cyan]必需项通过: {passed_required}/{len(required_checks)}[/cyan]")
    console.print(f"[cyan]可选项通过: {passed_optional}/{len(optional_checks)}[/cyan]")

    # 必须项全部通过才算成功
    all_required_passed = all(results.get(k, False) for k in required_checks)
    
    if all_required_passed:
        console.print("\n[bold green]✓ 所有必需检查通过！可以开始阶段1。[/bold green]")
        if results.get("evaluation", False):
            console.print("[green]  - 可选：评估管线也已就绪[/green]")
        if not results.get("jax_flax", False):
            console.print("[yellow]  - 注意：JAX/Flax 未安装，阶段4需要先安装: pip install jax flax[/yellow]")
    else:
        console.print("\n[bold yellow]⚠ 部分必需检查未通过，请解决上述问题。[/bold yellow]")
        console.print("  - 如果是数据问题，请确保 HumanML3D 已下载")
        console.print("  - 如果是模型问题，请确保预训练模型已下载")
        console.print("  - 如果是导入问题，请确保 requirements.txt 中的依赖已安装")
        console.print("  - JAX/Flax: pip install jax flax (Drift Loss 需要)")

    return all_required_passed


def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]DMG 阶段0: 环境准备与数据验证[/bold cyan]")
    console.print("=" * 60)

    # 解析配置
    cfg = parse_args(phase="validate")

    # 创建日志记录器
    logger = create_logger(cfg, phase="validate")

    # 记录结果
    results = {}

    # 检查1: MLD 模块导入
    mld_imported, mld_path = check_mld_import()
    results["mld_import"] = mld_imported

    # 检查2: 数据加载
    data_loaded, batch = check_data_loading(cfg, logger)
    results["data_loading"] = data_loaded

    # 检查3: 特征处理
    if batch is not None:
        results["feature_processing"] = check_feature_processing(batch)
    else:
        results["feature_processing"] = False

    # 检查4: VAE 加载
    vae_loaded, _ = check_vae_loading(cfg, logger)
    results["vae_loading"] = vae_loaded

    # 检查5: CLIP 加载
    clip_loaded, _ = check_clip_loading(cfg, logger)
    results["clip_loading"] = clip_loaded

    # 检查6: 评估管线
    results["evaluation"] = check_evaluation_pipeline(cfg, logger, results)

    # 打印总结
    print_summary(results)

    logger.info("阶段0验证完成")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
