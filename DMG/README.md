# DMG: Drifting Motion Generation

**Drifting Motion Generation (DMG)** — 将 Drifting 范式（arXiv:2602.04770）迁移到动作生成领域，以 HumanML3D 为数据集，MLD 的数据处理和 VAE 为基础。

## 项目概述

本项目实现**单步动作生成**，通过 Drift Loss 在特征空间中驱动生成样本向真实数据分布漂移，无需迭代采样即可生成高质量动作序列。

### 核心特性

- **单步生成**：相比 MLD 的多步迭代（50-100步），推理速度提升两个数量级
- **Drift Loss**：在特征空间中定义漂移力场，引力拉向正样本，斥力推离负样本
- **多条件注入**：文本描述 + 历史帧 + CFG尺度 → AdaLN 条件注入
- **复用 MLD**：直接复用 MLD 的 VAE、CLIP、数据处理和评估管线

---

## 环境准备

### 1. 创建 Conda 环境

```bash
conda create python=3.9 --name dmg
conda activate dmg
```

### 2. 安装依赖

```bash
cd DMG
pip install -r requirements.txt
```

### 3. 下载预训练模型和数据集

#### 3.1 HumanML3D 数据集

**重要**：HumanML3D 数据集需要从 [HumanML3D GitHub](https://github.com/EricGuo5513/HumanML3D) 下载。

下载后，将数据集放置在以下目录结构：

```
Motion_Generating/
└── DMG/
    └── datasets/
        └── humanml3d/
            ├── Mean.npy           # 全局均值
            ├── Std.npy            # 全局标准差
            ├── train.txt          # 训练集序列ID列表
            ├── val.txt            # 验证集序列ID列表
            ├── test.txt           # 测试集序列ID列表
            ├── new_joint_vecs/    # RIFKE 特征 (.npy 文件)
            │   ├── 000000.npy
            │   ├── 000001.npy
            │   └── ...
            └── texts/             # 文本描述
                ├── 000000.txt
                ├── 000001.txt
                └── ...
```

**数据集结构说明**：
- `new_joint_vecs/*.npy`：每条序列的 RIFKE 特征，形状为 `[T, 263]`，T 为帧数
- `texts/*.txt`：每条序列对应多个文本描述，格式为 `caption#tokens#f_tag#to_tag`
- `Mean.npy` / `Std.npy`：用于 Z-score 归一化的全局统计量

#### 3.2 预训练模型和依赖

运行下载脚本：

```bash
cd DMG/prepare

# 下载 MLD 预训练 VAE 模型
bash download_pretrained_models.sh   # Windows: run as admin or use Git Bash

# 下载 CLIP 模型（首次使用时会自动下载）
bash download_clip.sh

# 下载 T2M 评估器
bash download_t2m_evaluators.sh
```

**手动下载**（如自动下载失败）：

1. **MLD HumanML3D 预训练模型（包含 VAE 权重）**
    - 下载地址：https://drive.google.com/file/d/1hplrnQwUK_cZFHirZIOuVP0RSyZEC1YM/view
    - 放置位置：`DMG/pretrained_models/mld_vae_humanml3d.ckpt`

2. **CLIP 模型**
    - 下载地址：https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin
    - 放置位置：`DMG/deps/clip/ViT-B-32.pt`
    - 注意：将下载文件 `pytorch_model.bin` 重命名为 `ViT-B-32.pt`

3. **T2M 评估器**
    - 下载地址：https://drive.google.com/file/d/1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8/view
    - 放置位置：`DMG/deps/t2m/`

---

## 使用流程

按照 `pipeline.md` 中的设计，完整流程分为以下阶段：

### 阶段0：环境验证

验证数据加载、VAE/CLIP 加载等基础设施是否正常：

```bash
cd DMG

# 完整环境验证（数据加载 + 模型加载 + 评估管线）
python validate.py --cfg configs/config_dmg_humanml3d.yaml

# 或者分步验证：
# 阶段1：VAE + CLIP 编码器验证
python validate_stage1.py

# 阶段3：DitGenMotion 单步前向验证
python validate_stage3.py
```

### 阶段2：构建 MotionDriftBank（一次性离线操作）

Bank 构建将原始序列转换为滑动窗口样本，并按文本聚类组织：

```bash
cd DMG

# 构建 Bank（训练集）
python build_bank.py --config configs/config_bank.yaml --split train

# 调试模式（少量样本）
python build_bank.py --config configs/config_bank.yaml --split train --tiny
```

**输出**：`DMG/data/motion_drift_bank.pkl` — MotionDriftBank 文件

**Bank 构建说明**：
- 输入：HumanML3D 训练集
- 处理：滑动窗口采样（his=20帧, future=25帧, stride=5）→ CLIP 编码 → K-means 聚类 → VAE 编码
- 输出：按 512 个 text_class 分组的 latent bank

### 阶段5：训练 DMG 模型

```bash
cd DMG

# 完整训练
python train.py --cfg configs/config_dmg_humanml3d.yaml

# 指定检查点恢复训练
python train.py --cfg configs/config_dmg_humanml3d.yaml --resume ./experiments/xxx/checkpoints/last.ckpt
```

**训练配置**（见 `configs/config_dmg_humanml3d.yaml`）：
- 批量大小：64
- 学习率：2e-4
- 训练轮数：2000
- Drift Loss 权重：1.0
- EMA：启用

**检查点输出**：`experiments/dmg_humanml3d_baseline/checkpoints/`

### 阶段6：推理与可视化

#### 单序列推理

给定历史动作和文本描述，生成未来动作：

```bash
cd DMG

python inference.py \
    --config configs/config_dmg_humanml3d.yaml \
    --checkpoint ./experiments/dmg_humanml3d_baseline/checkpoints/last.ckpt \
    --text "a person is walking" \
    --motion ./datasets/humanml3d/new_joint_vecs/000000.npy \
    --his_len 20 \
    --future_len 25 \
    --cfg_scale 2.0 \
    --output ./results/inference/
```

#### 批量推理（从测试集采样）

```bash
cd DMG

python inference.py \
    --config configs/config_dmg_humanml3d.yaml \
    --checkpoint ./experiments/dmg_humanml3d_baseline/checkpoints/last.ckpt \
    --batch_mode \
    --num_samples 100 \
    --cfg_scale 2.0 \
    --output ./results/inference/batch/
```

### 阶段7：评估

使用 MLD 的评估指标计算 FID、R-Precision、MM Dist 等：

```bash
cd DMG

python test.py --cfg configs/config_dmg_humanml3d.yaml
```

### 可视化

#### 3D 动画预览（matplotlib）

```bash
cd DMG

# 单文件可视化
python visualize.py \
    --input ./results/inference/sample_0000_full.npy \
    --mode anim

# 批量可视化
python visualize.py \
    --input_dir ./results/inference/batch/ \
    --mode anim \
    --max_files 10
```

#### Blender 渲染（高质量视频）

```bash
cd DMG

python visualize.py \
    --input_dir ./results/inference/batch/ \
    --mode video \
    --output ./results/videos/ \
    --max_files 10
```

---

## 项目结构

```
DMG/
├── configs/                 # 配置文件
│   ├── config_dmg_humanml3d.yaml   # DMG 主配置
│   ├── config_bank.yaml             # Bank 构建配置
│   ├── base.yaml                    # 基础配置
│   └── assets.yaml                  # 资源路径配置
├── prepare/                # 数据和模型下载脚本
│   ├── download_pretrained_models.sh
│   ├── download_clip.sh
│   ├── download_t2m_evaluators.sh
│   └── download_humanml3d.sh
├── dmg/
│   ├── __init__.py
│   ├── config.py          # 配置解析
│   ├── data/              # 数据处理
│   │   ├── get_data.py    # 数据集加载工具
│   │   ├── sliding_window.py  # 滑动窗口数据集
│   │   ├── bank.py        # MotionDriftBank
│   │   ├── utils.py       # 工具函数
│   │   ├── base.py        # 数据基类
│   │   └── humanml/       # HumanML3D 数据集
│   │       ├── datamodule.py
│   │       ├── dataset.py
│   │       └── utils/
│   ├── models/            # 模型定义
│   │   ├── get_model.py   # 模型加载工厂
│   │   ├── architectures/
│   │   │   ├── mld_vae.py     # MLD VAE（复用）
│   │   │   ├── mld_clip.py    # CLIP 文本编码器（复用）
│   │   │   ├── dit_gen_motion.py  # 1D DiT 生成器
│   │   │   └── dmg_motion_encoder.py  # 评估特征提取器（VAE/T2M 模式）
│   │   ├── modeltype/
│   │   │   └── dmg.py     # DMG 主模型
│   │   ├── losses/
│   │   │   └── drift_loss_bridge.py  # PyTorch-JAX 桥接
│   │   └── metrics/
│   │       └── evaluator.py
│   ├── transforms/
│   │   └── feats2joints.py  # RIFKE → 3D 关节坐标
│   ├── utils/             # 工具函数
│   │   ├── logger.py
│   │   ├── fixseed.py
│   │   ├── position_encoding.py
│   │   ├── motion_process.py
│   │   ├── operator.py
│   │   └── temos_utils.py
│   └── callback/           # 训练回调
│       └── progress.py
├── data/                   # 运行时数据
│   └── motion_drift_bank.pkl  # MotionDriftBank 输出
├── datasets/              # 数据集（需单独下载）
│   └── humanml3d/
├── deps/                  # 依赖模型
│   ├── clip/
│   │   └── ViT-B-32.pt
│   ├── smpl/
│   ├── transforms/
│   ├── glove/
│   └── t2m/
├── pretrained_models/     # 预训练模型
│   └── mld_vae_humanml3d.ckpt
├── results/               # 推理结果
├── experiments/           # 训练实验记录
├── build_bank.py          # 阶段2：Bank 构建脚本
├── train.py               # 阶段5：训练入口
├── test.py                # 阶段7：测试入口
├── inference.py           # 阶段6：推理脚本
├── visualize.py           # 可视化脚本
├── validate.py            # 阶段0：完整验证脚本
├── validate_stage1.py     # 阶段1：VAE+CLIP 验证
├── validate_stage3.py     # 阶段3：DitGenMotion 验证
├── requirements.txt
└── README.md
```

---

## 实施阶段状态

| 阶段 | 内容 | 脚本 | 状态 |
|------|------|------|------|
| 阶段0 | 环境准备与数据验证 | `validate.py` | ✅ 完成 |
| 阶段1 | VAE 与 CLIP 编码器复用（冻结） | `validate_stage1.py` | ✅ 完成 |
| 阶段2 | 滑动窗口采样与 MotionDriftBank 离线构建 | `build_bank.py` | ✅ 完成 |
| 阶段3 | 1D DiT 模型 DitGenMotion 搭建 | `validate_stage3.py` | ✅ 完成 |
| 阶段4 | Drift Loss 接入（PyTorch-JAX 桥接） | 集成于 `dmg.py` | ✅ 完成 |
| 阶段5 | 完整训练循环 | `train.py` | ✅ 完成 |
| 阶段6 | 推理与可视化 | `inference.py`, `visualize.py` | ✅ 完成 |
| 阶段7 | 评估与消融实验 | `test.py` | ✅ 完成 |

---

## 参考

- [MLD: Motion Latent Diffusion Models](https://arxiv.org/abs/2212.04048) (CVPR 2023)
- [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (arXiv 2026)
- [HumanML3D Dataset](https://github.com/EricGuo5513/HumanML3D)
