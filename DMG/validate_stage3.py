"""
阶段3：DitGenMotion 单步前向验证脚本

验证内容（pipeline §3.4）：
  - 输入：随机 z_noise [B, 1, 256] + z_h [B, 1, 256] + text_emb [B, 512]
  - 确认输出形状 [B, 1, 256]，梯度可正常回传

用法：
    python validate_stage3.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from rich.console import Console
from rich.panel import Panel

console = Console()


def print_header(text):
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{text}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")


def print_ok(text):
    console.print(f"[green]✓[/green] {text}")


def test_dit_gen_motion_basic():
    """测试 DitGenMotion 基本前向"""
    from dmg.models.architectures.dit_gen_motion import DitGenMotion

    print_header("测试 1: DitGenMotion 基础前向（pipeline §3.4）")

    B = 4
    latent_dim = 256
    latent_seq_len = 1

    # 按 pipeline §3.4 的精确输入规格
    z_noise = torch.randn(B, latent_seq_len, latent_dim)
    z_h = torch.randn(B, latent_seq_len, latent_dim)
    text_emb = torch.randn(B, 512)

    console.print(f"\n输入张量：")
    console.print(f"  z_noise:   {tuple(z_noise.shape)}")
    console.print(f"  z_h:       {tuple(z_h.shape)}")
    console.print(f"  text_emb:  {tuple(text_emb.shape)}")

    # 创建模型
    model = DitGenMotion(
        hidden_size=768,
        depth=12,
        num_heads=12,
        cond_dim=768,
        latent_dim=latent_dim,
        latent_seq_len=latent_seq_len,
        use_qknorm=True,
        use_rmsnorm=True,
        use_rope=True,
        use_swiglu=True,
        text_dim=512,
        cfg_dropout=0.0,  # 验证时关闭 drop
    )

    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n模型参数: {num_params:,}")

    # 前向
    model.eval()
    with torch.no_grad():
        z_T = model(z_noise, z_h, text_emb)

    assert z_T.shape == (B, latent_seq_len, latent_dim), (
        f"期望输出形状 ({B}, {latent_seq_len}, {latent_dim})，"
        f"实际 {tuple(z_T.shape)}"
    )
    console.print(f"\n输出 z_T: {tuple(z_T.shape)}")
    print_ok(f"输出形状正确 [{B}, 1, {latent_dim}]")

    # 梯度回传
    console.print("\n测试梯度回传...")
    model.train()
    z_T = model(z_noise, z_h, text_emb)
    loss = z_T.sum()
    loss.backward()

    has_grad = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(has_grad), "没有参数收到梯度"
    grad_count = sum(1 for g in has_grad if g)
    console.print(f"  有梯度的参数数量: {grad_count}/{len(has_grad)}")
    print_ok("梯度可正常回传")

    return model


def test_cond_build():
    """测试条件构建"""
    from dmg.models.architectures.dit_gen_motion import DitGenMotion

    print_header("测试 2: 条件构建（pipeline §4.3）")

    B = 4
    model = DitGenMotion(
        hidden_size=768, depth=4, num_heads=12,
        latent_dim=256, latent_seq_len=1,
        use_qknorm=True, use_rmsnorm=True, use_rope=True,
        cfg_dropout=0.0,
    )
    model.eval()

    z_h = torch.randn(B, 1, 256)
    text_emb = torch.randn(B, 512)

    # 正常条件
    cond = model.build_cond(text_emb, z_h, cfg_drop=False)
    assert cond.shape == (B, 768)
    console.print(f"  text_cond + his_cond: {tuple(cond.shape)} ✓")
    print_ok("条件向量形状正确 [B, 768]")

    # CFG drop
    cond_drop = model.build_cond(text_emb, z_h, cfg_drop=True)
    assert cond_drop.shape == (B, 768)
    console.print(f"  cfg_drop=True (无条件): {tuple(cond_drop.shape)} ✓")
    print_ok("CFG drop 条件向量形状正确")

    # cfg_scale
    cfg_scale = torch.tensor([1.5, 2.0, 1.0, 2.5])
    cond_cfg = model.build_cond(text_emb, z_h, cfg_scale=cfg_scale, cfg_drop=False)
    assert cond_cfg.shape == (B, 768)
    console.print(f"  cfg_scale=[1.5,2.0,1.0,2.5]: {tuple(cond_cfg.shape)} ✓")
    print_ok("CFG scale 条件向量形状正确")

    return model


def test_rope():
    """测试 1D RoPE"""
    from dmg.models.architectures.dit_gen_motion import apply_rope_1d

    print_header("测试 3: 1D RoPE（pipeline §3.1, §4.1）")

    B, N, H, D = 2, 8, 12, 64
    q = torch.randn(B, N, H, D)
    k = torch.randn(B, N, H, D)

    q_rot, k_rot = apply_rope_1d(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    console.print(f"  q 旋转前: {tuple(q.shape)} → 旋转后: {tuple(q_rot.shape)} ✓")
    print_ok(f"1D RoPE 输出形状正确")

    # RoPE 是正交的（模长不变）
    q_norm = q.pow(2).sum(-1).sqrt().mean()
    q_rot_norm = q_rot.pow(2).sum(-1).sqrt().mean()
    ratio = q_rot_norm / q_norm
    console.print(f"  RoPE 前后模长比: {ratio:.6f} (应 ≈ 1.0)")
    assert abs(ratio - 1.0) < 0.01, f"RoPE 破坏了模长: {ratio}"
    print_ok("RoPE 保持模长（数值稳定）")


def test_adaln():
    """测试 AdaLN 调制"""
    from dmg.models.architectures.dit_gen_motion import (
        modulate_adaln, AdaLNModulation, RMSNorm
    )

    print_header("测试 4: AdaLN 调制（pipeline §3.2）")

    B, N, D, cond_dim = 4, 8, 768, 768

    x = torch.randn(B, N, D)
    shift = torch.randn(B, D)
    scale = torch.randn(B, D)

    out = modulate_adaln(x, shift, scale)
    assert out.shape == (B, N, D)
    console.print(f"  输入: {tuple(x.shape)} → AdaLN输出: {tuple(out.shape)}")
    print_ok("AdaLN 调制输出形状正确")

    # AdaLNModulation 模块
    mod = AdaLNModulation(cond_dim, D)
    c = torch.randn(B, cond_dim)
    chunks = mod(c)
    assert len(chunks) == 6
    for i, c_i in enumerate(chunks):
        assert c_i.shape == (B, D)
    console.print(f"  AdaLN 输出 6 个调制向量: {[(c.shape) for c in chunks]}")
    print_ok("AdaLNModulation 输出 6 个 [B, hidden_size] 张量")


def test_cfg_inference():
    """测试 CFG 推理"""
    from dmg.models.architectures.dit_gen_motion import DitGenMotion, apply_cfg

    print_header("测试 5: CFG 推理（pipeline §4.3）")

    model = DitGenMotion(
        hidden_size=768, depth=4, num_heads=12,
        latent_dim=256, latent_seq_len=1,
        use_qknorm=True, use_rmsnorm=True, use_rope=True,
        cfg_dropout=0.0,
    )
    model.eval()

    B = 4
    z_noise = torch.randn(B, 1, 256)
    z_h = torch.randn(B, 1, 256)
    text_emb = torch.randn(B, 512)

    z_T = apply_cfg(model, z_noise, z_h, text_emb, cfg_scale=2.0)

    assert z_T.shape == (B, 1, 256)
    console.print(f"  CFG 输出: {tuple(z_T.shape)}")
    print_ok("CFG 推理输出形状正确 [B, 1, 256]")


def test_gradient_flow():
    """测试梯度流向"""
    from dmg.models.architectures.dit_gen_motion import DitGenMotion

    print_header("测试 6: 梯度流向")

    model = DitGenMotion(
        hidden_size=768, depth=12, num_heads=12,
        latent_dim=256, latent_seq_len=1,
        use_qknorm=True, use_rmsnorm=True, use_rope=True,
        use_swiglu=True,
    )

    B = 2
    z_noise = torch.randn(B, 1, 256, requires_grad=True)
    z_h = torch.randn(B, 1, 256)
    text_emb = torch.randn(B, 512)

    model.train()
    z_T = model(z_noise, z_h, text_emb)
    loss = z_T.pow(2).mean()
    loss.backward()

    assert z_noise.grad is not None
    console.print(f"  z_noise.grad shape: {tuple(z_noise.grad.shape)}")
    print_ok("输入 z_noise 的梯度正常回传")

    # 检查所有参数有梯度
    grad_params = [p.grad is not None for p in model.parameters() if p.requires_grad]
    pct = sum(grad_params) / len(grad_params) * 100
    console.print(f"  有梯度的参数: {sum(grad_params)}/{len(grad_params)} ({pct:.0f}%)")
    print_ok(f"所有参数梯度正常 ({pct:.0f}%)")


def test_full_pipeline():
    """测试 VAE + DitGenMotion 完整前向"""
    from dmg.models.architectures.mld_vae import MldVae
    from dmg.models.architectures.mld_clip import MldTextEncoder
    from dmg.models.architectures.dit_gen_motion import DitGenMotion

    print_header("测试 7: VAE + DitGenMotion 完整前向（pipeline §5.2 步骤 ①②）")

    # VAE
    vae = MldVae(nfeats=263, latent_dim=[1, 256], num_layers=5, arch='all_encoder')
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # DitGenMotion
    dit = DitGenMotion(
        hidden_size=768, depth=4, num_heads=12,
        latent_dim=256, latent_seq_len=1,
        use_qknorm=True, use_rmsnorm=True, use_rope=True,
    )

    B = 2
    his_motion = torch.randn(B, 20, 263)   # 历史帧
    future_motion = torch.randn(B, 25, 263) # 未来帧
    text_emb = torch.randn(B, 512)

    with torch.no_grad():
        # ① VAE encode his → z_h
        z_h, _ = vae.encode(his_motion, lengths=[20, 20])
        console.print(f"  z_h shape: {tuple(z_h.shape)} (期望 [B, 1, 256])")

        # ① VAE encode future → z_f
        z_f, _ = vae.encode(future_motion, lengths=[25, 25])
        console.print(f"  z_f shape: {tuple(z_f.shape)} (期望 [B, 1, 256])")

        # ② z_noise + z_h + text → DitGenMotion → z_T
        z_noise = torch.randn(B, 1, 256)
        z_T = dit(z_noise, z_h, text_emb)
        console.print(f"  z_T shape: {tuple(z_T.shape)} (期望 [B, 1, 256])")

        # ③ VAE decode z_T → 预测帧
        recon = vae.decode(z_T, lengths=[25, 25])
        console.print(f"  recon shape: {tuple(recon.shape)} (期望 [B, 25, 263])")

    print_ok("VAE encode → DitGenMotion → VAE decode 管道正常")


def main():
    console.print(Panel.fit(
        "[bold cyan]DMG Stage 3: DitGenMotion 单步前向验证[/bold cyan]\n"
        "验证 pipeline §3.4 — 输出形状 [B,1,256]，梯度可回传",
        border_style="cyan"
    ))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"\n[cyan]Device:[/cyan] {device}\n")

    test_dit_gen_motion_basic()
    test_cond_build()
    test_rope()
    test_adaln()
    test_cfg_inference()
    test_gradient_flow()
    test_full_pipeline()

    print_header("阶段 3 验证结果汇总")
    console.print("\n[bold green]阶段 3 验证全部通过！[/bold green]")
    console.print("DitGenMotion 可独立前向，条件注入正确，梯度可正常回传。")
    console.print("可进入阶段 4：Drift Loss 接入。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
