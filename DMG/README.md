# DMG: Drifting Motion Generation

Drifting Motion Generation (DMG) - 将 Drifting 范式迁移到动作生成领域，以 HumanML3D 为数据集，MLD 的数据处理和 VAE 为基础。

## 项目概述

本项目实现单步动作生成，通过 Drift Loss 在特征空间中驱动生成样本向真实数据分布漂移，无需迭代采样即可生成高质量动作序列。

## 核心特性

- **单步生成**：相比 MLD 的多步迭代（50-100步），推理速度提升两个数量级
- **Drift Loss**：在特征空间中定义漂移力场，引力拉向正样本，斥力推离负样本
- **多条件注入**：文本描述 + 历史帧 + CFG尺度 → AdaLN 条件注入
- **复用 MLD**：直接复用 MLD 的 VAE、CLIP、数据处理和评估管线

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载预训练模型和数据
bash prepare/download_pretrained_models.sh
bash prepare/download_t2m_evaluators.sh

# 阶段0：验证数据加载和评估管线
python validate.py --cfg configs/config_dmg_humanml3d.yaml

# 阶段1-5：训练（待实现）
python train.py --cfg configs/config_dmg_humanml3d.yaml

# 测试
python test.py --cfg configs/config_dmg_humanml3d.yaml
```

## 项目结构

```
DMG/
├── configs/                 # 配置文件
├── prepare/                # 数据和模型下载脚本
├── dmg/
│   ├── __init__.py
│   ├── config.py          # 配置解析
│   ├── data/              # 数据处理
│   │   ├── __init__.py
│   │   ├── get_data.py    # 数据集加载工厂
│   │   ├── base.py        # 数据模块基类
│   │   └── humanml/       # HumanML3D 数据集
│   │       ├── __init__.py
│   │       └── dataset.py
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── get_model.py   # 模型加载工厂
│   │   ├── architectures/ # 模型架构
│   │   │   ├── __init__.py
│   │   │   ├── mld_vae.py     # 复用 MLD VAE
│   │   │   └── mld_clip.py    # 复用 MLD CLIP
│   │   ├── modeltype/     # 模型类型
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── dmg.py     # DMG 主模型
│   │   └── losses/        # 损失函数
│   │       ├── __init__.py
│   │       └── drift_loss_wrapper.py  # drift_loss PyTorch 包装
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   ├── temos_utils.py # 序列处理工具
│   │   ├── fixseed.py     # 随机种子
│   │   └── logger.py      # 日志工具
│   ├── transforms/        # 特征变换
│   │   ├── __init__.py
│   │   └── feats2smpl.py  # 复用 MLD 可视化
│   └── callback/          # 训练回调
│       ├── __init__.py
│       └── progress.py
├── scripts/               # 辅助脚本
│   └── visualize_motion.sh
├── train.py              # 训练入口
├── test.py               # 测试入口
├── validate.py           # 验证入口（阶段0）
└── requirements.txt
```

## 实施阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段0 | 环境准备与数据验证 | 🔄 进行中 |
| 阶段1 | VAE 与 CLIP 编码器复用（冻结） | ⏳ 待开始 |
| 阶段2 | 滑动窗口采样与 MotionDriftBank 离线构建 | ⏳ 待开始 |
| 阶段3 | 1D DiT 模型 DitGenMotion 搭建 | ⏳ 待开始 |
| 阶段4 | Drift Loss 接入（PyTorch-JAX 桥接） | ⏳ 待开始 |
| 阶段5 | 完整训练循环 | ⏳ 待开始 |
| 阶段6 | 推理与可视化 | ⏳ 待开始 |
| 阶段7 | 评估与消融实验 | ⏳ 待开始 |

## 参考

- [MLD: Motion Latent Diffusion Models](https://arxiv.org/abs/2212.04048) (CVPR 2023)
- [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (arXiv 2026)
