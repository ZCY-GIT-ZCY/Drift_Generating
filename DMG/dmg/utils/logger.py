"""
DMG Logger Utilities
"""

import os
import sys
import logging
from datetime import datetime


def create_logger(cfg, phase="train"):
    """
    创建日志记录器

    Args:
        cfg: 配置对象
        phase: 阶段 ('train', 'test', 'validate')

    Returns:
        logger: 日志记录器
    """
    # 创建日志目录
    log_dir = os.path.join(cfg.get('FOLDER_EXP', './experiments'), cfg.NAME)
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(log_dir, f"{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建日志记录器
    logger = logging.getLogger('DMG')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 记录配置
    from omegaconf import OmegaConf
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 50)

    return logger
