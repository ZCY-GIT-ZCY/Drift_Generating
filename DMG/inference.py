"""
DMG Inference Script

阶段6：单序列推理与可视化

给定历史动作 + 文本描述 + cfg_scale：
1. VAE encode his → z_h
2. CLIP encode text → text_emb
3. Sample z_noise
4. Single-step forward: z_T = model(z_noise, z_h, text_emb, cfg_scale)
5. VAE decode: future_motion = vae.decode(z_T)
6. 反标准化 → 输出 .npy 用于可视化

支持两种用法：
  1. 单序列推理：给定历史动作 .npy 和文本
  2. 批量推理：从测试集采样多个序列

用法：
  # 单序列推理
  python inference.py \
      --config configs/config_dmg_humanml3d.yaml \
      --checkpoint ./experiments/xxx/checkpoints/last.ckpt \
      --text "a person is walking" \
      --motion data/humanml3d/test/xxx_000.npy \
      --his_len 20 \
      --future_len 25 \
      --cfg_scale 2.0 \
      --output ./results/inference/

  # 批量推理（使用测试集）
  python inference.py \
      --config configs/config_dmg_humanml3d.yaml \
      --checkpoint ./experiments/xxx/checkpoints/last.ckpt \
      --batch_mode \
      --num_samples 100 \
      --cfg_scale 2.0 \
      --output ./results/inference/batch/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dmg.config import parse_args
from dmg.data.sliding_window import SlidingWindowDataModule
from dmg.models.get_model import get_model
from dmg.utils.logger import create_logger

console = Console()


# ============================================================================
# 阶段6：可视化辅助函数
# ============================================================================

try:
    from dmg.transforms.feats2joints import feats2joints, feats2joints_simple
    HAS_VISUALIZATION = True
except ImportError as e:
    HAS_VISUALIZATION = False
    print(f"Warning: Visualization module not available: {e}")

try:
    from mld.utils.joints import smplh_to_mmm_scaling_factor
    SCALE_FACTOR = smplh_to_mmm_scaling_factor
except ImportError:
    SCALE_FACTOR = 1.0
    print("Warning: smplh_to_mmm_scaling_factor not found, using 1.0")


def feats_to_visualization(
    motion_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    scale: float = 1.3,
    joints_num: int = 22,
) -> np.ndarray:
    """
    将归一化的 RIFKE 特征转换为可视化格式

    Args:
        motion_norm: [T, 263] 归一化特征
        mean, std: 归一化参数
        scale: 缩放因子
        joints_num: 关节数

    Returns:
        joints: [T, joints_num, 3] 关节坐标
    """
    if not HAS_VISUALIZATION:
        raise ImportError("Visualization module not available")

    joints = feats2joints(motion_norm, mean, std, joints_num=joints_num, scale=scale)
    return joints


def save_for_blender(
    joints: np.ndarray,
    output_path: str,
) -> str:
    """
    保存关节坐标为 Blender 可用格式

    Args:
        joints: [T, 22, 3] 关节坐标
        output_path: 输出路径

    Returns:
        保存的路径
    """
    # 转换为 MMM 格式
    blender_data = joints * SCALE_FACTOR
    np.save(output_path, blender_data)
    return output_path


def load_motion(
    motion_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    his_len: int,
) -> tuple[np.ndarray, int]:
    """
    加载并预处理动作序列

    Args:
        motion_path: .npy 文件路径
        mean, std: 归一化参数
        his_len: 截取的历史帧数

    Returns:
        his_tensor: [1, his_len, 263] 已归一化
        T_orig: 原始帧数
    """
    motion = np.load(motion_path)
    if motion.ndim == 1:
        raise ValueError(f"Motion must be 2D [T, 263], got 1D from {motion_path}")
    if motion.shape[-1] != 263:
        raise ValueError(f"Expected 263 dim, got {motion.shape[-1]}")

    T_orig = motion.shape[0]

    # Z-score 归一化
    motion_norm = (motion - mean) / std

    # 截取 his_len 帧（从末尾向前取）
    his_len_eff = min(his_len, T_orig)
    his_motion = motion_norm[-his_len_eff:]  # [his_len_eff, 263]

    # Padding（不足 his_len 时补零）
    if his_motion.shape[0] < his_len:
        pad_len = his_len - his_motion.shape[0]
        his_motion = np.concatenate([
            np.zeros((pad_len, 263)) + mean,
            his_motion,
        ], axis=0)

    his_tensor = torch.from_numpy(his_motion).float().unsqueeze(0)  # [1, his_len, 263]
    return his_tensor, T_orig


def infer_single(
    model,
    his_tensor: torch.Tensor,
    text: str,
    future_len: int,
    cfg_scale: float,
    lengths_his: list[int],
    lengths_future: list[int],
) -> np.ndarray:
    """
    单序列推理

    Returns:
        future_np: [future_len, 263] 归一化的生成未来帧
    """
    device = his_tensor.device

    with torch.no_grad():
        # CLIP encode
        text_emb = model.clip.encode_text([text])

        # VAE encode his
        z_h, _ = model.vae.encode(his_tensor, lengths_his)

        # Sample noise
        z_noise = torch.randn(1, 1, 256, device=device)

        # CFG forward
        from dmg.models.architectures.dit_gen_motion import apply_cfg
        z_T = apply_cfg(
            model.generator, z_noise, z_h, text_emb,
            cfg_scale=cfg_scale, cfg_dropout=0.0,
        )  # [1, 1, 256]

        # VAE decode
        future_motion = model.vae.decode(z_T, lengths_future)  # [1, future_len, 263]

    future_np = future_motion[0].cpu().numpy()
    return future_np


def infer_batch(
    model,
    datamodule: SlidingWindowDataModule,
    num_samples: int,
    cfg_scale: float,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> list[dict]:
    """
    批量推理（从测试集采样）

    Returns:
        results: list of {'text': str, 'his': np, 'gen': np, 'gt': np}
        注意：his/gen/gt 都是归一化数据，调用方负责反归一化
    """
    test_loader = datamodule.test_dataloader()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Batch inference", total=num_samples)

        for batch in test_loader:
            his = batch['his'].to(device)
            future = batch['future'].to(device)
            texts = batch['text']

            for i in range(len(texts)):
                if len(results) >= num_samples:
                    break

                his_i = his[i:i + 1]
                future_i = future[i:i + 1]
                text_i = texts[i]

                with torch.no_grad():
                    z_h, _ = model.vae.encode(his_i, [his_i.shape[1]])
                    z_noise = torch.randn(1, 1, 256, device=device)
                    text_emb = model.clip.encode_text([text_i])

                    from dmg.models.architectures.dit_gen_motion import apply_cfg
                    z_T = apply_cfg(
                        model.generator, z_noise, z_h, text_emb,
                        cfg_scale=cfg_scale, cfg_dropout=0.0,
                    )
                    future_gen = model.vae.decode(z_T, [future_i.shape[1]])

                # DataLoader 返回的 his_i/future_i 是已归一化数据
                # VAE decode 输出的 future_gen 也是归一化数据
                # 与 infer_single 保持一致，返回归一化数据
                results.append({
                    'text': text_i,
                    'his': his_i[0].cpu().numpy(),  # 归一化数据
                    'gen': future_gen[0].cpu().numpy(),  # 归一化数据
                    'gt': future_i[0].cpu().numpy(),  # 归一化数据
                })

                progress.update(task, advance=1)
                if len(results) >= num_samples:
                    break

            if len(results) >= num_samples:
                break

    return results


def main():
    parser = argparse.ArgumentParser(description="DMG Inference (Stage 6)")
    parser.add_argument("--config", type=str,
                        default="configs/config_dmg_humanml3d.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None,
                        help="Text description (single mode)")
    parser.add_argument("--motion", type=str, default=None,
                        help="History motion .npy path (single mode)")
    parser.add_argument("--his_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="./results/inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Use batch mode (from test set)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples in batch mode")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = parse_args(phase="test")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载归一化参数
    from dmg.data.get_data import get_global_mean_std
    mean, std = get_global_mean_std('humanml3d', cfg)
    mean_np = mean.astype(np.float32)
    std_np = std.astype(np.float32)

    console.print(f"\n[bold cyan]DMG Inference[/bold cyan]")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Device: {device}")
    console.print(f"  CFG scale: {args.cfg_scale}")

    # 加载模型
    dm = SlidingWindowDataModule(
        cfg=cfg, batch_size=1, num_workers=0,
        phase="test", split="test", dataset_name="humanml3d",
    )

    model = get_model(cfg, dm)
    state_dict = torch.load(args.checkpoint, map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    console.print(f"  Model loaded: {args.checkpoint}")

    if args.batch_mode:
        # ---- 批量推理模式 ----
        console.print(f"\n[bold]Batch Mode[/bold]")
        console.print(f"  Test dataset: {args.num_samples} samples")

        results = infer_batch(
            model=model,
            datamodule=dm,
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            mean=mean_np,
            std=std_np,
            device=str(device),
        )

        # 保存结果
        # results 中的数据是归一化数据，feats2joints 期望归一化数据
        # 所以可视化时可以直接传给 feats_to_visualization
        for idx, r in enumerate(results):
            np.save(output_dir / f"sample_{idx:04d}_his.npy", r['his'])
            np.save(output_dir / f"sample_{idx:04d}_gen.npy", r['gen'])
            np.save(output_dir / f"sample_{idx:04d}_gt.npy", r['gt'])

        # 保存元信息
        meta = [
            {'idx': idx, 'text': r['text']}
            for idx, r in enumerate(results)
        ]
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]✓[/green] Batch inference done!")
        console.print(f"  {len(results)} samples saved to {output_dir}")

        # ---- 阶段6：批量生成可视化文件 ----
        # results 中保存的是原始尺度数据
        # feats_to_visualization 期望归一化数据，需要先归一化回来
        if HAS_VISUALIZATION:
            console.print(f"\n[cyan]Generating visualization files...[/cyan]")
            for idx, r in enumerate(results):
                try:
                    # r['gen'] 已经是归一化数据，直接传给 feats_to_visualization
                    # feats2joints 内部会做反归一化: motion * std + mean
                    gen_joints = feats_to_visualization(r['gen'], mean_np, std_np, scale=1.3)
                    np.save(output_dir / f"sample_{idx:04d}_gen_joints.npy", gen_joints)

                    # Blender 格式
                    blender_path = output_dir / f"sample_{idx:04d}_blender.npy"
                    save_for_blender(gen_joints, str(blender_path))
                except Exception as e:
                    print(f"Warning: Failed to generate visualization for sample {idx}: {e}")

            console.print(f"  [green]✓[/green] Visualization files generated")
            console.print(f"  可视化命令:")
            console.print(f"    python visualize.py --input_dir {output_dir} --mode video --max_files 10")
        else:
            console.print(f"\n[yellow]可视化提示：[/yellow]")
            console.print(f"  运行 python visualize.py --input_dir {output_dir} --mode video 生成视频")

    else:
        # ---- 单序列推理模式 ----
        if not args.text or not args.motion:
            console.print("[red]Error: --text and --motion are required in single mode[/red]")
            return 1

        console.print(f"\n[bold]Single Mode[/bold]")
        console.print(f"  Text: {args.text}")
        console.print(f"  Motion: {args.motion}")

        if not os.path.exists(args.motion):
            console.print(f"[red]Error: motion file not found: {args.motion}[/red]")
            return 1

        his_tensor, T_orig = load_motion(args.motion, mean_np, std_np, args.his_len)
        his_tensor = his_tensor.to(device)
        lengths_his = [his_tensor.shape[1]]
        lengths_future = [args.future_len]

        future_np = infer_single(
            model, his_tensor, args.text,
            args.future_len, args.cfg_scale,
            lengths_his, lengths_future,
        )

        # 反归一化
        future_denorm = future_np * std_np + mean_np

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(args.motion).stem

        gen_path = output_dir / f"{base_name}_gen_{timestamp}.npy"
        np.save(gen_path, future_denorm)

        # 拼接完整序列（his + gen）
        his_denorm = his_tensor[0].cpu().numpy() * std_np + mean_np
        full_seq = np.concatenate([his_denorm, future_denorm], axis=0)
        full_path = output_dir / f"{base_name}_full_{timestamp}.npy"
        np.save(full_path, full_seq)

        # 保存元信息
        meta = {
            'text': args.text,
            'his_motion': args.motion,
            'his_frames': his_tensor.shape[1],
            'gen_frames': args.future_len,
            'cfg_scale': args.cfg_scale,
            'gen_path': str(gen_path),
            'full_path': str(full_path),
        }
        with open(output_dir / f"meta_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]✓[/green] Inference done!")
        console.print(f"  Generated motion: {gen_path}")
        console.print(f"  Full sequence (his+gen): {full_path}")
        console.print(f"  His frames: {his_tensor.shape[1]} | Gen frames: {args.future_len}")
        console.print(f"  Text: {args.text}")

        # ---- 阶段6：生成可视化文件 ----
        if HAS_VISUALIZATION:
            try:
                # 注意：feats2joints 内部会做反归一化，所以这里传入归一化后的数据
                # future_np 是归一化后的数据
                gen_joints = feats_to_visualization(future_np, mean_np, std_np, scale=1.3)
                gen_joints_path = output_dir / f"{base_name}_gen_joints_{timestamp}.npy"
                np.save(gen_joints_path, gen_joints)
                console.print(f"  [cyan]Joints (for visualization): {gen_joints_path}")

                # 完整序列（his + gen）：需要拼接归一化后的数据
                his_norm = his_tensor[0].cpu().numpy()  # 已归一化
                full_norm = np.concatenate([his_norm, future_np], axis=0)
                full_joints = feats_to_visualization(full_norm, mean_np, std_np, scale=1.3)
                full_joints_path = output_dir / f"{base_name}_full_joints_{timestamp}.npy"
                np.save(full_joints_path, full_joints)
                console.print(f"  [cyan]Full joints (his+gen): {full_joints_path}")

                # 保存 Blender 可用格式（已缩放）
                blender_path = output_dir / f"{base_name}_blender_{timestamp}.npy"
                save_for_blender(full_joints, str(blender_path))
                console.print(f"  [cyan]Blender format: {blender_path}")

                console.print(f"\n[yellow]可视化方法：[/yellow]")
                console.print(f"  1. 使用 DMG 可视化脚本（matplotlib 3D）：")
                console.print(f"     python visualize.py --input {full_joints_path} --mode anim")
                console.print(f"  2. 使用 Blender 渲染（需安装 Blender）：")
                console.print(f"     python visualize.py --input {blender_path} --mode video")
                console.print(f"  3. 使用 MLD render.py：")
                console.print(f"     cd ../MLD && blender --background --python render.py -- <args>")
            except ImportError as e:
                console.print(f"\n[yellow]可视化模块不可用: {e}[/yellow]")
                console.print(f"  请手动运行 visualize.py 或使用 MLD/render.py")
        else:
            console.print(f"\n[yellow]可视化：[/yellow]")
            console.print(f"  使用 visualize.py 将 .npy 渲染为视频")
            console.print(f"  python visualize.py --input {full_path} --mode video")

    return 0


if __name__ == "__main__":
    sys.exit(main())
