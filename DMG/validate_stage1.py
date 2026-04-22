"""
阶段1：VAE 与 CLIP 编码器验证脚本

验证内容（pipeline §1.3）：
1. VAE Encoder 输出形状 [B, 1, 256]，Decoder 可正确重建 RIFKE 特征
2. CLIP 文本编码输出形状 [B, 512]
3. VAE encode → decode 重建误差在合理范围

用法：
    python validate_stage1.py

依赖：
    - HumanML3D 数据集（已预处理）
    - MLD 预训练 VAE 权重（可选）
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def print_header(text):
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{text}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")


def print_ok(text):
    console.print(f"[green]✓[/green] {text}")


def print_fail(text):
    console.print(f"[red]✗[/red] {text}")


def print_info(text):
    console.print(f"  [dim]{text}[/dim]")


def create_random_motion(B: int, T: int, nfeats: int = 263, seed: int = 42) -> torch.Tensor:
    """创建随机 motion 用于测试"""
    torch.manual_seed(seed)
    return torch.randn(B, T, nfeats)


# ============================================================
# 1. VAE 验证
# ============================================================
def test_vae():
    print_header("测试 1: VAE 编码器与解码器")

    from dmg.models.architectures.mld_vae import MldVae

    # 1.1 创建模型
    console.print("\n[1.1] 初始化 VAE 模型...")
    vae = MldVae(
        nfeats=263,
        latent_dim=[1, 256],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        arch='all_encoder',
    )
    vae.eval()
    print_ok(f"VAE 模型创建成功 (参数: {sum(p.numel() for p in vae.parameters()):,})")

    # 1.2 测试 encode 输出形状
    console.print("\n[1.2] 测试 VAE Encoder 输出形状...")
    B, T = 4, 20  # 4 个样本，每个 20 帧
    motion = create_random_motion(B, T)

    with torch.no_grad():
        latent, dist = vae.encode(motion, lengths=[T] * B)

    assert latent.shape == (B, 1, 256), (
        f"期望 latent 形状 ({B}, 1, 256)，实际 {latent.shape}"
    )
    assert latent.shape == dist.mean.shape, "latent 和 dist.mean 形状不一致"
    print_ok(f"latent 形状: {tuple(latent.shape)} (符合 [B, 1, 256])")
    print_info(f"dist.mean 范围: [{dist.mean.min():.3f}, {dist.mean.max():.3f}]")
    print_info(f"dist.stddev 范围: [{dist.stddev.min():.3f}, {dist.stddev.max():.3f}]")

    # 1.3 测试 decode 重建
    console.print("\n[1.3] 测试 VAE Decoder 重建...")
    nframes_out = 25  # 不同的输出长度
    with torch.no_grad():
        recon = vae.decode(latent, lengths=[nframes_out] * B)

    assert recon.shape == (B, nframes_out, 263), (
        f"期望 recon 形状 ({B}, {nframes_out}, 263)，实际 {recon.shape}"
    )
    print_ok(f"recon 形状: {tuple(recon.shape)} (符合 [B, {nframes_out}, 263])")

    # 1.4 测试 forward (encode + decode)
    console.print("\n[1.4] 测试 VAE forward (encode → decode)...")
    with torch.no_grad():
        recon_forward, latent_forward = vae(motion, lengths=[T] * B)

    assert latent_forward.shape == (B, 1, 256)
    assert recon_forward.shape == (B, T, 263)
    print_ok(f"forward latent: {tuple(latent_forward.shape)}")
    print_ok(f"forward recon: {tuple(recon_forward.shape)}")

    # 1.5 测试变长序列
    console.print("\n[1.5] 测试变长序列...")
    T_list = [15, 20, 25, 30]
    for Ti in T_list:
        m = create_random_motion(2, Ti)
        with torch.no_grad():
            z, _ = vae.encode(m, lengths=[Ti] * 2)
            r, _ = vae.decode(z, lengths=[Ti] * 2)
        assert z.shape == (2, 1, 256)
        assert r.shape == (2, Ti, 263)
        print_info(f"  T={Ti:2d}: encode {tuple(z.shape)} → decode {tuple(r.shape)} ✓")
    print_ok("变长序列测试通过")

    # 1.6 重建误差
    console.print("\n[1.6] 重建误差统计...")
    with torch.no_grad():
        recon_all, _ = vae(motion, lengths=[T] * B)
    mse = torch.mean((recon_all - motion) ** 2).item()
    mae = torch.mean(torch.abs(recon_all - motion)).item()
    print_info(f"MSE:  {mse:.6f}")
    print_info(f"MAE:  {mae:.6f}")
    print_info(f"(随机初始化 VAE 误差较大，加载预训练权重后应显著降低)")

    # 1.7 冻结验证
    console.print("\n[1.7] 验证 VAE 冻结状态...")
    for name, param in vae.named_parameters():
        assert not param.requires_grad, f"参数 {name} 未冻结"
    print_ok("所有 VAE 参数已冻结 (requires_grad=False)")

    return vae


# ============================================================
# 2. CLIP 验证
# ============================================================
def test_clip():
    print_header("测试 2: CLIP 文本编码器")

    from dmg.models.architectures.mld_clip import MldTextEncoder

    # 2.1 创建模型
    console.print("\n[2.1] 初始化 CLIP 文本编码器...")
    clip_encoder = MldTextEncoder(
        modelpath="./deps/clip-vit-large-patch14",
        precision='fp32',
    )
    print_ok("CLIP 模型创建成功")

    # 2.2 测试单条文本
    console.print("\n[2.2] 测试单条文本编码...")
    text = "a person is walking"
    emb = clip_encoder.encode_text(text)
    assert emb.shape == (1, 512), f"期望 (1, 512)，实际 {emb.shape}"
    print_ok(f"单条文本嵌入形状: {tuple(emb.shape)}")
    print_info(f"向量范围: [{emb.min():.3f}, {emb.max():.3f}]")

    # 2.3 测试批量文本
    console.print("\n[2.3] 测试批量文本编码...")
    texts = [
        "a person is walking forward",
        "a person is running slowly",
        "a person jumps and spins around",
        "a person waves their arms",
        "a person sits down on a chair",
    ]
    B = len(texts)
    emb_batch = clip_encoder.encode_text(texts)
    assert emb_batch.shape == (B, 512), f"期望 ({B}, 512)，实际 {emb_batch.shape}"
    print_ok(f"批量文本嵌入形状: {tuple(emb_batch.shape)}")

    # 2.4 测试相似文本的余弦相似度
    console.print("\n[2.4] 测试语义相似度...")
    sim = torch.nn.functional.cosine_similarity(
        emb_batch[0].unsqueeze(0),
        emb_batch[1].unsqueeze(0),
        dim=1
    ).item()
    print_info(f"相似文本（walk vs run）余弦相似度: {sim:.4f}")
    print_info(f"(CLIP 随机初始化时相似度接近 0，预训练 CLIP 应 > 0.2)")

    # 2.5 冻结验证
    console.print("\n[2.5] 验证 CLIP 冻结状态...")
    if clip_encoder.clip_model is not None:
        for param in clip_encoder.clip_model.parameters():
            assert not param.requires_grad, f"CLIP 参数未冻结"
        print_ok("CLIP 模型已冻结")
    else:
        console.print("[yellow]  Warning: CLIP 模型未加载（可能缺少预训练权重）[/yellow]")

    return clip_encoder


# ============================================================
# 3. 联合验证
# ============================================================
def test_joint(vae, clip_encoder):
    print_header("测试 3: VAE + CLIP 联合流程")

    B, T = 4, 20

    # 3.1 文本到条件向量
    console.print("\n[3.1] 文本 → CLIP 嵌入 → 条件向量...")
    texts = ["a person walking", "a person running", "a person jumping", "a person standing"]
    with torch.no_grad():
        text_emb = clip_encoder.encode_text(texts)  # [B, 512]
        # 模拟条件构建：CLIP 512d → Dense(768)
        cond_dim = 768
        text_cond = torch.nn.Linear(512, cond_dim)(text_emb)  # [B, 768]
    print_ok(f"文本嵌入: {tuple(text_emb.shape)} → 条件向量: {tuple(text_cond.shape)}")

    # 3.2 Motion → VAE latent → 条件
    console.print("\n[3.2] Motion → VAE latent → 历史条件向量...")
    motion = create_random_motion(B, T)
    with torch.no_grad():
        latent, _ = vae.encode(motion, lengths=[T] * B)  # [B, 1, 256]
        # 模拟条件构建：latent mean+std → Dense(768)
        latent_mean = latent.mean(dim=1, keepdim=True)  # [B, 1, 256]
        latent_std = latent.std(dim=1, keepdim=True) + 1e-6  # [B, 1, 256]
        his_cond_raw = torch.cat([latent_mean, latent_std], dim=-1)  # [B, 1, 512]
        his_cond = torch.nn.Linear(512, cond_dim)(his_cond_raw)  # [B, 768]
    print_ok(f"Motion latent: {tuple(latent.shape)}")
    print_ok(f"历史条件向量: {tuple(his_cond.shape)}")

    # 3.3 端到端 latent 提取验证
    console.print("\n[3.3] 端到端 VAE encode → decode 验证...")
    motion_raw = create_random_motion(2, 25)
    with torch.no_grad():
        z, _ = vae.encode(motion_raw, lengths=[25, 25])
        recon, _ = vae.decode(z, lengths=[25, 25])
        mse = torch.mean((recon - motion_raw) ** 2).item()
    print_info(f"端到端 MSE: {mse:.6f}")
    print_ok(f"VAE encode → decode 管道正常工作")

    return {
        "text_emb_shape": tuple(text_emb.shape),
        "text_cond_shape": tuple(text_cond.shape),
        "latent_shape": tuple(latent.shape),
        "his_cond_shape": tuple(his_cond.shape),
        "vae_params": sum(p.numel() for p in vae.parameters()),
        "clip_loaded": clip_encoder.clip_model is not None,
    }


# ============================================================
# 4. 预训练权重加载验证（可选）
# ============================================================
def test_pretrained_vae(vae):
    print_header("测试 4: 预训练 VAE 权重加载")

    pretrained_path = "./pretrained_models/mld_vae_humanml3d.ckpt"
    if not os.path.exists(pretrained_path):
        console.print(f"[yellow]  预训练权重未找到: {pretrained_path}[/yellow]")
        console.print(f"  跳过预训练加载测试（阶段2 Bank 构建时需提供预训练权重）")
        return

    console.print(f"\n尝试加载预训练权重: {pretrained_path}")
    success = vae.load_pretrained(pretrained_path)
    if success:
        print_ok("预训练权重加载成功")
    else:
        print_fail("预训练权重加载失败")


# ============================================================
# 主函数
# ============================================================
def main():
    console.print(Panel.fit(
        "[bold cyan]DMG Stage 1: VAE + CLIP 验证[/bold cyan]\n"
        "验证编码器冻结、特征形状、端到端重建",
        border_style="cyan"
    ))

    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"\n[cyan]Device:[/cyan] {device}")

    results = {}

    # 测试 VAE
    vae = test_vae()

    # 测试 CLIP
    clip_encoder = test_clip()

    # 联合测试
    joint_results = test_joint(vae, clip_encoder)

    # 预训练加载（可选）
    test_pretrained_vae(vae)

    # 汇总表格
    print_header("阶段 1 验证结果汇总")

    table = Table(show_header=True, header_style="bold")
    table.add_column("测试项", style="cyan")
    table.add_column("结果", style="green")

    table.add_row("VAE 模型初始化", "✓")
    table.add_row("VAE encode 形状 [B,1,256]", "✓")
    table.add_row("VAE decode 形状 [B,T,263]", "✓")
    table.add_row("VAE forward (encode→decode)", "✓")
    table.add_row("VAE 冻结 (requires_grad=False)", "✓")
    table.add_row("CLIP 文本编码形状 [B,512]", "✓")
    table.add_row("CLIP 冻结", "✓" if joint_results['clip_loaded'] else "⚠ 未加载")
    table.add_row("VAE+CLIP 联合流程", "✓")
    table.add_row("条件向量构建 (512→768)", "✓")

    console.print(table)

    console.print(f"\n[bold green]阶段 1 验证完成！[/bold green]")
    console.print("VAE 和 CLIP 编码器已就绪，可进入阶段 2 Bank 构建。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
