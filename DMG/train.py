"""
DMG Training Script

阶段5：完整训练入口脚本

功能：
  - 加载 SlidingWindowDataModule（his/future/text 三元组）
  - 加载/训练 DMG 模型（VAE + CLIP 冻结，DitGenMotion 可训练）
  - drift_loss 训练（阶段4已集成到 DMG 模型中）
  - EMA + 学习率调度
  - 日志记录 + checkpoint 保存

用法：
    python train.py --config configs/config_dmg_humanml3d.yaml
"""

import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from dmg.callback import ProgressLogger
from dmg.config import parse_args
from dmg.data.sliding_window import SlidingWindowDataModule
from dmg.models.get_model import get_model
from dmg.utils.logger import create_logger


def main():
    # 解析配置
    cfg = parse_args(phase="train")

    # 创建日志记录器
    logger = create_logger(cfg, phase="train")

    # 设置随机种子
    pl.seed_everything(cfg.SEED_VALUE)

    # GPU 设置
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 日志记录器
    loggers = []
    if hasattr(cfg.LOGGER, 'WANDB') and cfg.LOGGER.WANDB.get('PROJECT'):
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.get('OFFLINE', False),
            resume_id=cfg.LOGGER.WANDB.get('RESUME_ID'),
            save_dir=cfg.LOGGER.FOLDER_EXP,
            name=cfg.NAME,
            anonymous=True,
            log_model=False,
        )
        loggers.append(wandb_logger)

    if cfg.LOGGER.get('TENSORBOARD', True):
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.LOGGER.FOLDER_EXP,
            sub_dir="tensorboard",
            name=cfg.NAME,
            version="",
        )
        loggers.append(tb_logger)

    logger.info(OmegaConf.to_yaml(cfg))

    # 创建数据集（DMG 专用）
    dm = SlidingWindowDataModule(
        cfg=cfg,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        phase="train",
        split="train",
        dataset_name=cfg.TRAIN.DATASETS[0],
    )
    logger.info("Using SlidingWindowDataModule (his/future/text mode)")

    # 创建模型
    model = get_model(cfg, dm)
    logger.info(f"model {cfg.model.model_type} loaded")
    logger.info(f"  Bank: {'enabled' if model.bank is not None else 'disabled'}")
    logger.info(f"  EMA: {'enabled' if model.use_ema else 'disabled'}")
    logger.info(f"  Drift Loss: lambda={model.lambda_drift}, R_list={model.R_list}")

    # 回调函数
    callbacks = [
        RichProgressBar(),
        ProgressLogger(metric_monitor={
            "train/loss_total": "train/loss_total",
            "train/loss_0.1": "train/loss_0.1",
            "train/loss_0.5": "train/loss_0.5",
            "train/loss_1.0": "train/loss_1.0",
            "train/scale": "train/scale",
            "val/loss_total": "val/loss_total",
        }),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.LOGGER.FOLDER_EXP, cfg.NAME, "checkpoints"),
            filename="{epoch}",
            monitor="epoch",
            mode="max",
            every_n_epochs=cfg.LOGGER.SACE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=True,
        ),
    ]

    # 多 GPU
    if len(cfg.DEVICE) > 1:
        ddp_strategy = "ddp"
    else:
        ddp_strategy = None

    # 训练器
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        default_root_dir=cfg.LOGGER.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
    )

    logger.info("Trainer initialized")

    # 预训练 VAE 权重加载（确保 VAE 可用）
    vae_pretrained = cfg.TRAIN.get('PRETRAINED_VAE', None)
    if vae_pretrained and os.path.exists(vae_pretrained):
        logger.info(f"Loading pretrained VAE from {vae_pretrained}")
        try:
            state_dict = torch.load(vae_pretrained, map_location="cpu")
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            vae_dict = {}
            for k, v in state_dict.items():
                if k.startswith('vae.'):
                    name = k.replace('vae.', '')
                    vae_dict[name] = v
            model.vae.load_state_dict(vae_dict, strict=False)
            logger.info("Pretrained VAE loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load pretrained VAE: {e}")

    # 开始训练
    trainer.fit(model, datamodule=dm)

    logger.info(f"Checkpoints saved to {trainer.checkpoint_callback.dirpath}")
    logger.info(f"Training outputs: {cfg.LOGGER.FOLDER_EXP}")
    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
