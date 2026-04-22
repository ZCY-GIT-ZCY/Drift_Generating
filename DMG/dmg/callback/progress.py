"""
DMG Progress Logger Callback
"""

import pytorch_lightning as pl
import torch


class ProgressLogger(pl.callbacks.Callback):
    """
    进度日志回调

    在训练过程中记录关键指标
    """

    def __init__(self, metric_monitor=None):
        """
        初始化进度日志回调

        Args:
            metric_monitor: 指标监控字典，格式为 {显示名称: 指标路径}
        """
        self.metric_monitor = metric_monitor or {}

    def _log_metrics(self, trainer, pl_module, metric_dict, step=None):
        """安全地记录指标（处理 logger 为 None 的情况）"""
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return
        if not hasattr(trainer.logger, 'experiment') or trainer.logger.experiment is None:
            return

        current_step = step if step is not None else trainer.global_step
        try:
            trainer.logger.experiment.log(metric_dict, step=current_step)
        except Exception:
            # 静默忽略日志记录错误
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练 batch 结束后记录"""
        if batch_idx % trainer.log_every_n_steps == 0:
            metric_dict = {}
            for name, metric in self.metric_monitor.items():
                if metric in outputs:
                    value = outputs[metric]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metric_dict[name] = value
            if metric_dict:
                self._log_metrics(trainer, pl_module, metric_dict)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """每个验证 batch 结束后记录"""
        if batch_idx % trainer.log_every_n_steps == 0:
            metric_dict = {}
            for name, metric in self.metric_monitor.items():
                if metric in outputs:
                    value = outputs[metric]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metric_dict[f"val_{name}"] = value
            if metric_dict:
                self._log_metrics(trainer, pl_module, metric_dict)

    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练 epoch 结束后记录"""
        if not self.metric_monitor:
            return

        metric_dict = {}
        for name, metric in self.metric_monitor.items():
            prefixed_metric = metric
            if hasattr(trainer, 'callback_metrics') and prefixed_metric in trainer.callback_metrics:
                value = trainer.callback_metrics[prefixed_metric]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_dict[name] = value
        if metric_dict:
            self._log_metrics(trainer, pl_module, metric_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        """每个验证 epoch 结束后记录"""
        if not self.metric_monitor:
            return

        metric_dict = {}
        for name, metric in self.metric_monitor.items():
            prefixed_metric = f"val_{metric}"
            if hasattr(trainer, 'callback_metrics') and prefixed_metric in trainer.callback_metrics:
                value = trainer.callback_metrics[prefixed_metric]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metric_dict[name] = value
        if metric_dict:
            self._log_metrics(trainer, pl_module, metric_dict)
