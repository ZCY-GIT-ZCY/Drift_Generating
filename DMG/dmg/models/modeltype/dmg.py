"""
DMG 主模型（阶段3 + 阶段4 + 阶段5 完整实现）

整合：
  1. 冻结 VAE（阶段1）
  2. 冻结 CLIP（阶段1）
  3. 可训练 DitGenMotion（阶段3）
  4. drift_loss 接入（阶段4）
  5. 完整训练循环（阶段5，manual_optimization 模式）

完整前向路径（pipeline §5.2）：
  ① his/future → VAE encode → z_h, z_f
  ② z_noise + z_h + text → DitGenMotion → z_T
  ③ z_T → VAE decode → motion → VAE encode → feat_gen
  ④ feat_gen + Bank.sample() → drift_loss → backward → generator 更新

梯度桥接机制（阶段4）：
  使用 Lightning manual_optimization，在每个 batch 内：
  1. PyTorch 前向 → 计算 PyTorch 梯度
  2. 额外调用 JAX drift_loss → 获取 grad_JAX
  3. 将 grad_JAX 合并到 PyTorch 参数梯度 → optimizer.step()
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from ..architectures.mld_vae import MldVae
from ..architectures.mld_clip import MldTextEncoder
from ..architectures.dit_gen_motion import DitGenMotion, apply_cfg
from ..data.bank import MotionDriftBank
from .losses.drift_loss_bridge import compute_drift_loss_and_gradients


class DMG(pl.LightningModule):
    """
    DMG 主模型类

    使用 manual_optimization 实现完全手动的训练循环，
    从而精确控制 PyTorch 梯度和 JAX drift_loss 梯度的合并时机。
    """

    def __init__(self, cfg, datamodule=None):
        super().__init__()
        self.cfg = cfg

        # ========== 阶段1：冻结编码器 ==========
        vae_cfg = cfg.get('vae', {})
        clip_cfg = cfg.get('clip', {})

        self.vae = MldVae(
            nfeats=cfg.DATASET.NFEATS,
            latent_dim=cfg.model.latent_dim,
            ff_size=cfg.model.ff_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            arch='all_encoder',
        )
        self._load_vae_weights(vae_cfg.get('pretrained_path', ''))
        self._freeze_vae()

        self.clip = MldTextEncoder(
            modelpath=clip_cfg.get('model_path', './deps/clip-vit-large-patch14'),
            precision=clip_cfg.get('precision', 'fp32'),
        )
        self._freeze_clip()

        # ========== 阶段3：DitGenMotion 生成器（可训练）==========
        gen_cfg = cfg.model.generator
        self.generator = DitGenMotion(
            hidden_size=gen_cfg.hidden_size,
            depth=gen_cfg.depth,
            num_heads=gen_cfg.num_heads,
            cond_dim=gen_cfg.cond_dim,
            mlp_ratio=4.0,
            latent_dim=cfg.model.latent_dim[1],
            latent_seq_len=cfg.model.latent_dim[0],
            patch_size=1,
            use_qknorm=gen_cfg.use_qknorm,
            use_rmsnorm=gen_cfg.use_rmsnorm,
            use_rope=gen_cfg.use_rope,
            use_swiglu=gen_cfg.use_swiglu,
            text_dim=512,
            cfg_dropout=cfg.get('CFG', {}).get('no_cfg_frac', 0.1),
        )

        # ========== 阶段4：Bank 加载（只读）==========
        bank_cfg = cfg.get('BANK', {})
        bank_path = bank_cfg.get('bank_path', None)
        self.bank: Optional[MotionDriftBank] = None
        if bank_path:
            self.bank = MotionDriftBank.load(bank_path)
            print(f"[DMG] Bank loaded: {self.bank.num_classes} classes, "
                  f"{self.bank.num_windows} windows")

        # ========== 阶段5：训练配置 ==========
        # CFG
        self.cfg_min = cfg.get('CFG', {}).get('cfg_min', 1.0)
        self.cfg_max = cfg.get('CFG', {}).get('cfg_max', 2.0)

        # Loss 权重
        self.lambda_drift = cfg.get('LOSS', {}).get('LAMBDA_DRIFT', 1.0)
        self.lambda_mse = cfg.get('LOSS', {}).get('LAMBDA_MSE', 0.0)

        # EMA
        self.use_ema = cfg.get('TRAIN', {}).get('EMA', False)
        self.ema_decay = cfg.get('TRAIN', {}).get('EMA_DECAY', 0.999)

        # Drift Loss 配置
        drift_cfg = cfg.get('DRIFT_LOSS', {})
        self.R_list = tuple(drift_cfg.get('R_list', [0.1, 0.5, 1.0]))
        self.num_pos = drift_cfg.get('pos_per_sample', 16)
        self.num_neg = drift_cfg.get('neg_per_sample', 32)
        self.kernel_sigma = drift_cfg.get('kernel_sigma', 0.1)
        self.hard_neg = drift_cfg.get('hard_neg', True)

        # 优化器引用（manual optimization 需要）
        self.automatic_optimization = False
        self._optimizer = None
        self._lr_scheduler = None

        # 存储输出用于 epoch 汇总
        self.training_step_outputs = []
        self.validation_step_outputs = []

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _load_vae_weights(self, path: str):
        if not path:
            return
        try:
            self.vae.load_pretrained(path)
            print(f"[DMG] VAE 权重已加载: {path}")
        except Exception as e:
            print(f"[DMG] VAE 权重加载失败: {e}（将使用随机初始化）")

    def _freeze_vae(self):
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    def _freeze_clip(self):
        if hasattr(self.clip, 'clip_model') and self.clip.clip_model is not None:
            for p in self.clip.clip_model.parameters():
                p.requires_grad = False
            self.clip.clip_model.eval()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return self.clip.encode_text(texts)

    def vae_encode(
        self, motion: torch.Tensor,
        lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if lengths is None:
            lengths = [motion.shape[1]] * motion.shape[0]
        latent, _ = self.vae.encode(motion, lengths)
        return latent

    def vae_decode(
        self, latent: torch.Tensor,
        lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if lengths is None:
            raise ValueError("vae_decode: lengths 不能为 None")
        return self.vae.decode(latent, lengths)

    # ------------------------------------------------------------------
    # 核心前向路径（pipeline §5.2）
    # ------------------------------------------------------------------

    def forward(
        self,
        his: torch.Tensor,
        future: torch.Tensor,
        texts: List[str],
        lengths_his: Optional[List[int]] = None,
        lengths_future: Optional[List[int]] = None,
        z_noise: Optional[torch.Tensor] = None,
        use_cfg: bool = False,
        cfg_scale: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向路径

        Returns:
            dict: z_h, z_T, feat_gen, z_f, recon, text_emb, z_h_squeezed
        """
        B = his.shape[0]
        T_his = his.shape[1]
        T_fut = future.shape[1] if future is not None else 25
        latent_seq_len = self.cfg.model.latent_dim[0]
        latent_dim = self.cfg.model.latent_dim[1]

        if lengths_his is None:
            lengths_his = [T_his] * B
        if lengths_future is None:
            lengths_future = [T_fut] * B

        z_h = self.vae_encode(his, lengths_his)
        text_emb = self.encode_text(texts)

        if z_noise is None:
            z_noise = torch.randn(B, latent_seq_len, latent_dim, device=his.device)

        if use_cfg:
            z_T = apply_cfg(
                self.generator, z_noise, z_h, text_emb,
                cfg_scale=cfg_scale, cfg_dropout=0.0,
            )
        else:
            z_T = self.generator(z_noise, z_h, text_emb)

        # feat_gen：z_T → VAE decode → VAE encode → [B, 256]
        if future is not None:
            motion_from_zT = self.vae_decode(z_T, lengths_future)
            feat_gen, _ = self.vae.encode(motion_from_zT, lengths_future)
            feat_gen = feat_gen.squeeze(1)  # [B, 256]
        else:
            feat_gen = z_T.squeeze(1)

        outputs = {
            'z_h': z_h,
            'z_T': z_T,
            'feat_gen': feat_gen,
            'text_emb': text_emb,
        }

        if future is not None:
            z_f, _ = self.vae.encode(future, lengths_future)
            outputs['z_f'] = z_f
            outputs['z_h_squeezed'] = z_h.squeeze(1)  # [B, 256]
            outputs['recon'] = self.vae_decode(z_T, lengths_future)

        return outputs

    # ------------------------------------------------------------------
    # 阶段5：完整训练步骤（manual optimization）
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        手动优化训练步骤

        控制流：
          1. 前向 → feat_gen
          2. Bank 采样 + drift_loss → loss_drift
          3. PyTorch backward(loss_drift) → 梯度注入 feat_gen → 回传到 generator
          4. optimizer.step()
        """
        his = batch['his']
        future = batch['future']
        texts = batch['text']
        opt = self._optimizer

        # 1. 清零 PyTorch 梯度
        opt.zero_grad()

        # 2. CFG 尺度采样（pipeline §5.3）
        if torch.rand(1).item() < self.cfg.get('CFG', {}).get('no_cfg_frac', 0.1):
            cfg_scale = 1.0  # 无条件生成
        else:
            cfg_scale = torch.empty(1).uniform_(self.cfg_min, self.cfg_max).item()

        # 3. 前向（包含 feat_gen）
        outputs = self.forward(his, future, texts, use_cfg=True, cfg_scale=cfg_scale)
        feat_gen = outputs['feat_gen']  # [B, 256]
        z_h = outputs.get('z_h_squeezed', outputs['z_h'].squeeze(1))

        # 4. drift_loss（阶段4接入）
        info = {}
        loss_total = torch.tensor(0.0, device=his.device)
        drift_loss_defined = False

        if self.bank is not None:
            try:
                # 获取 text_class
                with torch.no_grad():
                    text_emb_np = outputs['text_emb'].cpu().numpy()
                    class_ids = self.bank.predict_text_class(text_emb_np)
                    if len(class_ids) == 1:
                        class_ids = [class_ids[0]] * len(texts)

                all_drift_losses = []
                all_info = []
                all_grad_gen = []

                for i in range(len(texts)):
                    class_id = class_ids[i]
                    feat_i = feat_gen[i:i + 1]          # [1, 256]
                    his_i = z_h[i:i + 1]               # [1, 256]

                    # Bank 采样
                    future_pos, future_neg, weight_pos = self.bank.sample(
                        class_id=class_id,
                        num_pos=self.num_pos,
                        num_neg=self.num_neg,
                        hard_neg=self.hard_neg,
                        his_reference=his_i.detach().cpu().numpy(),
                        sigma=self.kernel_sigma,
                    )
                    weight_neg = torch.ones(self.num_neg, device=his.device)

                    # JAX drift_loss + 梯度
                    loss_val, info_i, grad_gen_np = compute_drift_loss_and_gradients(
                        feat_gen=feat_i,
                        future_pos=future_pos,
                        future_neg=future_neg,
                        weight_pos=weight_pos,
                        weight_neg=weight_neg,
                        R_list=self.R_list,
                    )
                    all_drift_losses.append(loss_val)
                    all_info.append(info_i)
                    all_grad_gen.append(grad_gen_np)

                # batch 内均值
                drift_loss = torch.tensor(np.mean(all_drift_losses), device=his.device, requires_grad=True)
                info = all_info[0] if all_info else {}

                # grad_JAX: [B, 256] 均值
                grad_gen_np_mean = np.mean(all_grad_gen, axis=0)  # [B, 256]

                # 梯度注入：JAX drift_loss 的梯度注入到 feat_gen，
                # 通过 manual_backward 让梯度回传到 generator 参数
                self._inject_jax_grad(feat_gen, grad_gen_np_mean)
                self.manual_backward(self.lambda_drift * drift_loss)
                loss_total = self.lambda_drift * drift_loss
                drift_loss_defined = True

            except Exception as e:
                print(f"[DMG] Warning: drift_loss failed ({e}). Skipping for this batch.")
                self.bank = None
                info = {}

        # 5. MSE 重建损失（可选）
        if self.lambda_mse > 0 and 'recon' in outputs:
            mse_loss = torch.mean((outputs['recon'] - future) ** 2)
            self.manual_backward(self.lambda_mse * mse_loss)
            loss_total = loss_total + self.lambda_mse * mse_loss

        # 7. 梯度裁剪（可选）
        max_norm = self.cfg.get('TRAIN', {}).get('GRAD_CLIP', 0.0)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm)

        # 8. optimizer step
        opt.step()

        # 9. EMA 更新
        if self.use_ema and hasattr(self, '_ema'):
            self._ema.update(self.generator.parameters())

        # 10. 学习率调度
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # 日志
        self.log("train/loss_total", loss_total.item(), on_step=True, prog_bar=True)
        self.log("train/lr", opt.param_groups[0]['lr'], on_step=True)
        if 'scale' in info:
            self.log("train/scale", info['scale'], on_step=True)
        for k, v in info.items():
            if str(k).startswith('loss_'):
                self.log(f"train/{k}", v, on_step=True)

        self.training_step_outputs.append({'loss': loss_total.item()})
        return {'loss': loss_total}

    def _inject_jax_grad(
        self,
        feat_gen: torch.Tensor,
        grad_np: np.ndarray,
    ):
        """
        将 JAX drift_loss 的梯度注入 PyTorch 计算图

        机制：
          feat_gen 来自：generator → z_T → VAE encode → feat_gen
          它的梯度路径指向 generator 的参数。

          我们手动创建一个梯度注入：
          1. 创建 grad_tensor [B, 256]
          2. 调用 feat_gen.backward(grad_tensor)
          3. PyTorch 会把这个梯度回传到 generator 的参数中
        """
        grad_tensor = torch.from_numpy(grad_np).to(feat_gen.device).float()
        # PyTorch backward：将 grad_tensor 作为上游梯度注入 feat_gen
        feat_gen.backward(grad_tensor)

    def on_training_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = np.mean([o['loss'] for o in self.training_step_outputs])
            self.log("train/epoch_loss", avg_loss, on_epoch=True)
        self.training_step_outputs.clear()

    # ------------------------------------------------------------------
    # 验证步骤
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        his = batch['his']
        future = batch['future']
        texts = batch['text']

        with torch.no_grad():
            outputs = self.forward(his, future, texts, use_cfg=True, cfg_scale=2.0)

        feat_gen = outputs['feat_gen']
        z_h = outputs.get('z_h_squeezed', outputs['z_h'].squeeze(1))
        loss_total = 0.0
        info = {}

        if self.bank is not None:
            try:
                text_emb_np = outputs['text_emb'].cpu().numpy()
                class_ids = self.bank.predict_text_class(text_emb_np)
                if len(class_ids) == 1:
                    class_ids = [class_ids[0]] * len(texts)

                drift_losses = []
                for i in range(len(texts)):
                    class_id = class_ids[i]
                    feat_i = feat_gen[i:i + 1]
                    his_i = z_h[i:i + 1]

                    future_pos, future_neg, weight_pos = self.bank.sample(
                        class_id=class_id,
                        num_pos=self.num_pos,
                        num_neg=self.num_neg,
                        hard_neg=self.hard_neg,
                        his_reference=his_i.detach().cpu().numpy(),
                        sigma=self.kernel_sigma,
                    )
                    weight_neg = torch.ones(self.num_neg, device=his.device)

                    loss_val, info_i, _ = compute_drift_loss_and_gradients(
                        feat_gen=feat_i,
                        future_pos=future_pos,
                        future_neg=future_neg,
                        weight_pos=weight_pos,
                        weight_neg=weight_neg,
                        R_list=self.R_list,
                    )
                    drift_losses.append(loss_val)

                drift_loss = np.mean(drift_losses)
                info = info_i
                loss_total = self.lambda_drift * drift_loss
            except Exception as e:
                print(f"[DMG] val drift_loss warning: {e}")

        self.validation_step_outputs.append({'loss': loss_total, 'info': info})
        return loss_total

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            losses = [o['loss'] for o in self.validation_step_outputs
                      if isinstance(o['loss'], (int, float)) or o['loss'].numel() == 1]
            if losses:
                self.log("val/loss_total", np.mean(losses), on_epoch=True, prog_bar=True)

            all_info = self.validation_step_outputs
            for R in self.R_list:
                r_vals = [o['info'].get(f'loss_{R}', 0.0) for o in all_info if o.get('info')]
                if r_vals:
                    self.log(f"val/loss_{R}", np.mean(r_vals), on_epoch=True)

            if all_info and all_info[0].get('info', {}).get('scale'):
                scales = [o['info']['scale'] for o in all_info if o.get('info')]
                self.log("val/scale", np.mean(scales), on_epoch=True)

        self.validation_step_outputs.clear()

    # ------------------------------------------------------------------
    # 测试步骤
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        his = batch['his']
        future = batch['future']
        texts = batch['text']

        with torch.no_grad():
            outputs = self.forward(his, future, texts, use_cfg=True, cfg_scale=2.0)

        return outputs

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.use_ema:
            try:
                from torch_ema import ExponentialMovingAverage
                self._ema = ExponentialMovingAverage(
                    self.generator.parameters(),
                    decay=self.ema_decay,
                )
            except ImportError:
                print("[DMG] Warning: torch_ema not found. EMA disabled.")
                self.use_ema = False
                self._ema = None

    # ------------------------------------------------------------------
    # 优化器和学习率调度（阶段5）
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg = self.cfg.get('TRAIN', {}).get('OPTIM', {})
        lr = opt_cfg.get('LR', 2e-4)
        weight_decay = opt_cfg.get('WEIGHT_DECAY', 0.01)

        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self._optimizer = optimizer

        sched_cfg = self.cfg.get('TRAIN', {}).get('SCHEDULER', {})
        if sched_cfg.get('enabled', True):
            max_epochs = self.cfg.get('TRAIN', {}).get('END_EPOCH', 300)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=lr * 0.01,
            )
            self._lr_scheduler = scheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
