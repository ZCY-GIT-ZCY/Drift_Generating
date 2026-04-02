# MDM 使用指南

本文档完整描述了 **MDM (Human Motion Diffusion Model)** 项目的工作内容、工具链架构，以及所有功能的使用方法。

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心架构与原理](#2-核心架构与原理)
3. [目录结构](#3-目录结构)
4. [环境配置与依赖安装](#4-环境配置与依赖安装)
5. [数据集准备](#5-数据集准备)
6. [预训练模型下载](#6-预训练模型下载)
7. [核心功能使用指南](#7-核心功能使用指南)
   - 7.1 [动作生成 (Motion Synthesis)](#71-动作生成-motion-synthesis)
   - 7.2 [动作编辑 (Motion Editing)](#72-动作编辑-motion-editing)
   - 7.3 [模型训练 (Training)](#73-模型训练-training)
   - 7.4 [模型评估 (Evaluation)](#74-模型评估-evaluation)
   - 7.5 [骨骼可视化与SMPL网格渲染](#75-骨骼可视化与smpl网格渲染)
8. [DiP: 极速文本人体运动生成](#8-dip-极速文本人体运动生成)
9. [命令行参数详解](#9-命令行参数详解)
10. [工具链完整依赖关系](#10-工具链完整依赖关系)
11. [输出文件说明](#11-输出文件说明)
12. [常见问题与注意事项](#12-常见问题与注意事项)

---

## 1. 项目概述

MDM（Human Motion Diffusion Model）是论文 **"Human Motion Diffusion Model"**（ICLR 2023）的官方 PyTorch 实现。项目主页: https://guytevet.github.io/mdm-page/

MDM 是一种基于去噪扩散概率模型（DDPM）的 3D 人体运动生成模型，能够根据文本描述或动作类别标签生成自然流畅的人体动作序列。

该项目同时集成了 **DiP (Diffusion Planner)**，这是一种极速的文本人体运动生成模型，作为 CLoSD 论文（ICLR 2025 Spotlight）的一部分发布。相比原版 MDM，DiP 速度提升约 40 倍（~0.4 秒/样本）。

### 主要特点

| 特性 | 说明 |
|------|------|
| **生成模式** | 文本驱动、动作类别驱动、无条件生成 |
| **支持的架构** | Transformer Encoder (`trans_enc`)、Transformer Decoder (`trans_dec`)、GRU |
| **文本编码器** | CLIP (默认) 或 DistilBERT |
| **扩散步数** | 默认 1000 步；DiP 仅需 10 步 |
| **加速技术** | 50 步快速模型 + CLIP 结果缓存，比原版快 40 倍 |
| **输出格式** | 骨骼动画 (MP4)、SMPL 参数、OBJ 网格文件 |
| **支持的数据集** | HumanML3D、KIT-ML、HumanAct12、UESTC |

### 相关论文引用

```bibtex
# MDM
@inproceedings{tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet et al.},
  booktitle={ICLR 2023},
  year={2023},
  url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}

# DiP and CLoSD
@article{tevet2024closd,
  title={CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control},
  author={Tevet, Guy et al.},
  journal={arXiv preprint arXiv:2410.03441},
  year={2024}
}
```

---

## 2. 核心架构与原理

### 2.1 扩散模型基础

MDM 基于去噪扩散概率模型（DDPM），工作流程分为两个阶段：

1. **前向扩散过程（Forward Process）**：对真实运动数据逐步添加高斯噪声，最终变为纯噪声。
2. **反向去噪过程（Reverse Process）**：从纯噪声开始，通过学习逐步去除噪声，恢复出真实的运动数据。

MDM 使用 **预测 x₀（predict start）** 策略，即模型直接预测原始干净的运动数据而非噪声。

### 2.2 模型架构

```
文本/动作嵌入
     ↓
时间步嵌入 (Timestep Embedding)
     ↓
Transformer / GRU 核心网络
  ├── trans_enc: Transformer Encoder（默认架构）
  ├── trans_dec: Transformer Decoder（支持 DiP）
  └── gru: GRU 循环网络
     ↓
输出投影 (Output Process)
     ↓
运动数据 (rot6d / hml_vec / xyz)
```

**关键参数**：
- `latent_dim`: 默认 512，Transformer/GRU 的隐藏层维度
- `num_layers`: 默认 8，Transformer 层数或 GRU 层数
- `num_heads`: 默认 4，注意力头数
- `ff_size`: 默认 1024，前馈网络维度

### 2.3 条件机制

MDM 支持三种条件模式：

| 条件模式 | 条件输入 | 适用数据集 |
|----------|----------|------------|
| `text` | 文本描述（通过 CLIP 或 DistilBERT 编码） | HumanML3D、KIT-ML |
| `action` | 动作类别标签 | HumanAct12、UESTC |
| `no_cond` | 无条件（完全由噪声驱动生成） | HumanAct12 |

**Classifier-Free Guidance（无分类器引导）**：在训练时随机丢弃条件信息（约 10%），推理时通过 `guidance_param` 参数（默认 2.5）调整条件强度。

### 2.4 DiP 架构

DiP 是 MDM 的极速版本，使用以下优化：

- **自回归生成**：每次预测未来 2 秒（40 帧）的运动
- **仅 10 个扩散步骤**：相比 1000 步大幅加速
- **Transformer Decoder + DistilBERT**：专用文本编码器
- **前缀补全机制**：接收已有的前缀运动，生成后续运动

---

## 3. 目录结构

```
MDM/
├── assets/                         # 示例资源文件
│   ├── example_text_prompts.txt   # 文本提示示例
│   ├── example_action_names_*.txt  # 动作名称示例
│   ├── example_dynamic_text_prompts.txt  # 动态文本提示（用于 DiP）
│   ├── *.png, *.gif               # 演示图像和动画
│   └── fixed_results.png/tex      # 论文评估结果
│
├── dataset/                        # 数据集元数据
│   ├── humanml_opt.txt / kit_opt.txt  # 数据集配置
│   ├── t2m_mean.npy / t2m_std.npy    # HumanML3D 归一化参数
│   └── kit_mean.npy / kit_std.npy     # KIT 归一化参数
│
├── data_loaders/                  # 数据加载模块
│   ├── get_data.py                # 数据集加载器入口
│   ├── tensors.py                 # 张量处理工具
│   ├── humanml_utils.py           # HumanML 工具函数
│   ├── a2m/                       # 动作转运动数据加载
│   │   ├── dataset.py
│   │   ├── humanact12poses.py
│   │   └── uestc.py
│   └── humanml/                   # 文本人体运动数据加载
│       ├── data/dataset.py        # 核心数据集类
│       ├── motion_loaders/         # 运动数据加载器
│       ├── networks/               # 评估网络（FID、R-precision）
│       └── utils/                  # 参数工具、骨骼定义等
│
├── diffusion/                     # 扩散模型核心
│   ├── gaussian_diffusion.py      # 高斯扩散过程（核心）
│   ├── respace.py                 # 时间步采样策略
│   ├── nn.py                      # Transformer 模块
│   ├── losses.py                  # 损失函数
│   ├── resample.py                # 重采样策略
│   ├── fp16_util.py               # 混合精度训练
│   └── logger.py                  # 日志记录
│
├── eval/                          # 评估模块
│   ├── eval_humanml.py            # HumanML3D/KIT 评估（文本驱动）
│   ├── eval_humanact12_uestc.py   # Action-to-Motion 评估
│   ├── a2m/                       # 动作转运动评估
│   │   ├── gru_eval.py            # GRU 模型评估
│   │   ├── stgcn_eval.py          # ST-GCN 模型评估
│   │   ├── action2motion/         # 动作识别评估
│   │   └── recognition/           # 姿态识别网络
│   └── unconstrained/             # 无条件评估
│
├── model/                         # 模型定义
│   ├── mdm.py                     # MDM 主模型类
│   ├── rotation2xyz.py            # 旋转到 XYZ 坐标转换
│   ├── smpl.py                    # SMPL 人体模型封装
│   ├── cfg_sampler.py             # CFG 采样器
│   └── BERT/
│       └── BERT_encoder.py        # DistilBERT 编码器
│
├── prepare/                       # 资源下载脚本
│   ├── download_smpl_files.sh     # 下载 SMPL 模型文件
│   ├── download_glove.sh           # 下载 GloVe 词向量
│   ├── download_t2m_evaluators.sh # 下载 T2M 评估器
│   ├── download_a2m_datasets.sh   # 下载 Action 数据集
│   ├── download_recognition_models.sh      # 下载识别模型
│   └── download_recognition_unconstrained_models.sh  # 下载无条件识别模型
│
├── sample/                        # 推理采样模块
│   ├── generate.py                # 运动生成入口
│   ├── edit.py                    # 运动编辑入口
│   └── predict.py                 # Cog 在线推理接口
│
├── train/                         # 训练模块
│   ├── training_loop.py           # 训练主循环
│   ├── train_mdm.py               # 训练入口脚本
│   └── train_platforms.py          # 训练平台支持（TensorBoard/WandB/ClearML）
│
├── utils/                         # 通用工具
│   ├── parser_util.py             # 命令行参数解析
│   ├── model_util.py              # 模型创建与加载
│   ├── dist_util.py               # 分布式训练工具
│   ├── fixseed.py                 # 随机种子固定
│   ├── loss_util.py               # 损失函数工具
│   ├── sampler_util.py            # 采样器工具（CFG、自回归）
│   └── rotation_conversions.py    # 旋转表示转换
│
├── visualize/                     # 可视化模块
│   ├── render_mesh.py             # SMPL 网格渲染
│   ├── motions2hik.py             # 运动转 HIK 格式
│   ├── simplify_loc2rot.py        # 简化局部到旋转
│   ├── vis_utils.py               # 可视化工具
│   └── joints2smpl/               # 关节到 SMPL 拟合
│       ├── fit_seq.py             # 序列拟合
│       ├── smplify.py             # SMPLify 优化
│       ├── src/                   # 源代码
│       └── smpl_models/           # SMPL 模型文件
│
├── environment.yml                # Conda 环境配置
├── cog.yaml                       # Cog 部署配置
├── README.md                      # 项目说明
├── DiP.md                         # DiP 使用说明
└── MDM_use.md                     # 本文档
```

---

## 4. 环境配置与依赖安装

### 4.1 系统要求

- **操作系统**: Ubuntu 18.04.5 LTS（已在 Windows 上测试部分功能）
- **Python**: 3.7+
- **硬件**: CUDA 可用的 GPU（单卡即可）

### 4.2 安装步骤

#### 步骤 1: 安装 ffmpeg

```shell
# Linux
sudo apt update
sudo apt install ffmpeg

# Windows: 从 https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/ 下载并配置
```

#### 步骤 2: 创建 Conda 环境

```shell
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

#### 步骤 3: 下载依赖资源

根据任务类型选择下载：

```shell
# 文本转动作（T2M）— 必需
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh

# 动作转动作（A2M）
bash prepare/download_smpl_files.sh
bash prepare/download_recognition_models.sh

# 无条件生成
bash prepare/download_smpl_files.sh
bash prepare/download_recognition_models.sh
bash prepare/download_recognition_unconstrained_models.sh
```

> **注意**：Windows 系统需要将 `.sh` 脚本中的 `bash` 命令替换为 PowerShell 等效命令，或使用 WSL（Windows Subsystem for Linux）。

---

## 5. 数据集准备

### 5.1 HumanML3D（文本转动作）

从 [Google Drive](https://drive.google.com/drive/folders/1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu?usp=drive_link) 下载完整数据，放置到 `dataset/HumanML3D/` 目录。

**简易方式（仅文本）**：
```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D dataset/HumanML3D
```

### 5.2 KIT-ML（文本转动作）

下载同上，放置到 `dataset/KIT-ML/` 目录。

### 5.3 HumanAct12 和 UESTC（动作转动作）

```shell
bash prepare/download_a2m_datasets.sh
```

### 5.4 无条件生成数据集

```shell
bash prepare/download_unconstrained_datasets.sh
```

---

## 6. 预训练模型下载

所有模型下载后解压放置到 `./save/` 目录。

### 6.1 文本转动作模型（HumanML3D）

| 模型名称 | 链接 | 说明 |
|----------|------|------|
| `humanml_trans_dec_512_bert-50steps` | [Google Drive](https://drive.google.com/file/d/1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC/view?usp=sharing) | **推荐** 速度最快（20倍加速），精度更优 |
| `humanml-encoder-512-50steps` | [Google Drive](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing) | 50步快速模型，与原论文效果相当 |
| `humanml-encoder-512` | [Google Drive](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing) | 原论文最佳模型（1000步） |
| `humanml-decoder-512` | [Google Drive](https://drive.google.com/file/d/1q3soLadvVh7kJuJPd2cegMNY2xVuVudj/view?usp=sharing) | Decoder 架构 |
| `humanml-decoder-with-emb-512` | [Google Drive](https://drive.google.com/file/d/1GnsW0K3UjuOkNkAWmjrGIUmeDDZrmPE5/view?usp=sharing) | 带嵌入的 Decoder |

### 6.2 文本转动作模型（KIT）

| 模型名称 | 链接 |
|----------|------|
| `kit-encoder-512` | [Google Drive](https://drive.google.com/file/d/1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL/view?usp=sharing) |

### 6.3 动作转动作模型

| 模型/数据集 | 链接 |
|-------------|------|
| UESTC | [uestc](https://drive.google.com/file/d/1goB2DJK4B-fLu2QmqGWKAqWGMTAO6wQ6/view?usp=sharing) |
| UESTC (no_fc) | [uestc_no_fc](https://drive.google.com/file/d/1fpv3mR-qP9CYCsi9CrQhFqlLavcSQky6/view?usp=sharing) |
| HumanAct12 | [humanact12](https://drive.google.com/file/d/154X8_Lgpec6Xj0glEGql7FVKqPYCdBFO/view?usp=sharing) |
| HumanAct12 (no_fc) | [humanact12_no_fc](https://drive.google.com/file/d/1frKVMBYNiN5Mlq7zsnhDBzs9vGJvFeiQ/view?usp=sharing) |

### 6.4 无条件生成模型

| 模型 | 链接 |
|------|------|
| HumanAct12 Unconstrained | [humanact12_unconstrained](https://drive.google.com/file/d/1uG68m200pZK3pD-zTmPXu5XkgNpx_mEx/view?usp=share_link) |

### 6.5 DiP 模型

| 模型 | 链接 | 说明 |
|------|------|------|
| DiP (无目标) | [HuggingFace](https://huggingface.co/guytevet/CLoSD/tree/main/checkpoints/dip/DiP_no-target_10steps_context20_predict40) | 纯文本驱动 |
| DiP (带目标) | [Google Drive](https://drive.google.com/file/d/1PsilP2xhcOHHXkmtxtOwNbWeI0njU2ic/view?usp=sharing) | 支持目标位置条件（用于 CLoSD 应用） |

---

## 7. 核心功能使用指南

### 7.1 动作生成 (Motion Synthesis)

#### 7.1.1 文本转动作 — 使用测试集提示词

```shell
# 使用 50 步快速模型（推荐）
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt \
  --num_samples 10 \
  --num_repetitions 3
```

#### 7.1.2 文本转动作 — 使用自定义文本

```shell
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --input_text ./assets/example_text_prompts.txt
```

`example_text_prompts.txt` 示例内容：
```
the person walked forward and is picking up his toolbox.
a person waves their arms and jumps.
someone is running in place.
```

#### 7.1.3 文本转动作 — 单条文本提示

```shell
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --text_prompt "the person walked forward and is picking up his toolbox."
```

#### 7.1.4 动作转动作 — 使用动作标签

```shell
# 使用数据集自带的测试集动作
python -m sample.generate \
  --model_path ./save/humanact12/model000350000.pt \
  --num_samples 10 \
  --num_repetitions 3

# 使用自定义动作文件
python -m sample.generate \
  --model_path ./save/humanact12/model000350000.pt \
  --action_file ./assets/example_action_names_humanact12.txt

# 指定单个动作
python -m sample.generate \
  --model_path ./save/humanact12/model000350000.pt \
  --action_name "drink"
```

HumanAct12 支持的动作：`warm_up, walk, run, jump, drink, lift_dumbbell, sit, eat, turn steering wheel, phone, boxing, throw`

UESTC 支持的动作见 `dataset/uestc/info/action_classes.txt`。

#### 7.1.5 无条件生成

```shell
python -m sample.generate \
  --model_path ./save/unconstrained/model000450000.pt \
  --num_samples 10 \
  --num_repetitions 3
```

#### 7.1.6 常用可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | GPU 设备编号 | 0 |
| `--seed` | 随机种子（不同种子产生不同结果） | 10 |
| `--motion_length` | 运动时长（秒），最大 9.8 秒 | 6.0 |
| `--guidance_param` | Classifier-Free Guidance 参数（越高越符合文本，越低越多样） | 2.5 |

#### 7.1.7 动作长度控制

```shell
# 生成 3 秒的运动
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt \
  --motion_length 3.0

# 生成最大长度运动（约 9.8 秒）
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt \
  --motion_length 9.8
```

---

### 7.2 动作编辑 (Motion Editing)

> 仅支持文本转动作数据集（HumanML3D 和 KIT），且需要完整运动捕获数据。

编辑功能基于运动修复（Motion Inpainting）技术，根据现有运动片段补全或修改部分运动。

#### 7.2.1 无条件编辑（仅编辑，不加文本条件）

**In-Between 编辑**：保留输入运动的前缀和后缀，生成中间部分。

```shell
python -m sample.edit \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --edit_mode in_between
```

输出中蓝色帧来自输入运动，橙色帧由模型生成。

**Upper-Body 编辑**：保留下半身运动，仅生成上半身。

```shell
python -m sample.edit \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --edit_mode upper_body
```

#### 7.2.2 文本条件编辑

在编辑时加入文本条件，使生成部分符合指定文本描述：

```shell
python -m sample.edit \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --edit_mode upper_body \
  --text_condition "A person throws a ball"
```

#### 7.2.3 编辑参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--edit_mode` | 编辑模式：`in_between` 或 `upper_body` | `in_between` |
| `--num_samples` | 采样数量 | 10 |
| `--num_repetitions` | 每个采样的重复次数 | 3 |
| `--prefix_end` | In-between 模式下，前缀结束位置（占总帧数的比例） | 0.25 |
| `--suffix_start` | In-between 模式下，后缀起始位置（占总帧数的比例） | 0.75 |
| `--text_condition` | 编辑时的文本条件（可选） | 空 |

---

### 7.3 模型训练 (Training)

#### 7.3.1 文本转动作 — HumanML3D

**复现原论文模型**：
```shell
python -m train.train_mdm \
  --save_dir save/my_humanml_trans_enc_512 \
  --dataset humanml
```

**训练 50 步快速模型**：
```shell
python -m train.train_mdm \
  --save_dir save/my_humanml_trans_enc_512_50steps \
  --dataset humanml \
  --diffusion_steps 50 \
  --mask_frames \
  --use_ema
```

**训练 MDM + DistilBERT**：
```shell
python -m train.train_mdm \
  --save_dir save/my_humanml_trans_dec_bert_512 \
  --dataset humanml \
  --diffusion_steps 50 \
  --arch trans_dec \
  --text_encoder_type bert \
  --mask_frames \
  --use_ema
```

#### 7.3.2 文本转动作 — KIT

```shell
python -m train.train_mdm \
  --save_dir save/my_kit_trans_enc_512 \
  --dataset kit
```

#### 7.3.3 动作转动作

```shell
python -m train.train_mdm \
  --save_dir save/my_name \
  --dataset humanact12 \
  --cond_mask_prob 0 \
  --lambda_rcxyz 1 \
  --lambda_vel 1 \
  --lambda_fc 1
```

#### 7.3.4 无条件训练

```shell
python -m train.train_mdm \
  --save_dir save/my_name \
  --dataset humanact12 \
  --cond_mask_prob 0 \
  --lambda_rcxyz 1 \
  --lambda_vel 1 \
  --lambda_fc 1 \
  --unconstrained
```

#### 7.3.5 训练推荐参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--eval_during_training` | 每个 checkpoint 评估 | 推荐开启 |
| `--gen_during_training` | 每个 checkpoint 生成示例动画 | 推荐开启 |
| `--use_ema` | 指数移动平均 | 推荐开启 |
| `--mask_frames` | 修复帧掩码 bug | 推荐开启 |
| `--diffusion_steps 50` | 训练快速模型 | 可选 |
| `--train_platform_type WandBPlatform` | WandB 日志 | 可选 |
| `--device` | GPU 编号 | 默认 0 |
| `--arch` | 架构：`trans_enc`（默认）、`trans_dec`、`gru` | 可选 |
| `--text_encoder_type` | 文本编码器：`clip`（默认）、`bert` | 可选 |

#### 7.3.6 训练输出

训练过程中会保存：
- `model#########.pt` — 模型 checkpoint（每隔 50,000 步）
- `opt#########.pt` — 优化器状态
- `args.json` — 训练参数（用于后续推理加载）
- `eval_*.log` — 评估日志
- `samples_*.pt.samples/` — 生成的动作样本

---

### 7.4 模型评估 (Evaluation)

#### 7.4.1 文本转动作评估（HumanML3D）

```shell
# 完整评估（推荐，耗时约 12 小时）
python -m eval.eval_humanml \
  --model_path ./save/humanml_trans_enc_512/model000475000.pt

# 带 WanDB 记录
python -m eval.eval_humanml \
  --model_path ./save/humanml_trans_enc_512/model000475000.pt \
  --train_platform_type WandBPlatform

# 快速评估（较少重复）
python -m eval.eval_humanml \
  --model_path ./save/humanml_trans_enc_512/model000475000.pt \
  --eval_mode mm_short
```

**评估模式**：
| 模式 | 说明 | 耗时 |
|------|------|------|
| `wo_mm` | 20 次重复，不计算多模态性 | ~12 小时 |
| `mm_short` | 5 次重复，计算多模态性 | ~15 小时 |
| `debug` | 快速测试（1000 样本） | ~3 小时 |

#### 7.4.2 文本转动作评估（KIT）

```shell
python -m eval.eval_humanml \
  --model_path ./save/kit_trans_enc_512/model000400000.pt
```

#### 7.4.3 动作转动作评估

```shell
# 完整评估（20 次重复）
python -m eval.eval_humanact12_uestc \
  --model ./save/humanact12/model000350000.pt \
  --eval_mode full
```

#### 7.4.4 无条件生成评估

```shell
python -m eval.eval_humanact12_uestc \
  --model ./save/unconstrained/model000450000.pt \
  --eval_mode full
```

> 注意：精确率和召回率默认不计算以节省时间。如需计算，修改 `eval/a2m/gru_eval.py` 中的 `fast=True` 为 `fast=False`。

#### 7.4.5 评估指标说明

| 指标 | 全称 | 说明 | 越低/越高越好 |
|------|------|------|---------------|
| FID | Fréchet Inception Distance | 生成动作与真实动作分布的差异 | 越低越好 |
| R-precision | Retrieval Precision | 动作与对应文本的匹配准确率 | 越高越好 |
| Diversity | 多样性 | 生成动作的多样性程度 | 适中最好 |
| Multimodality | 多模态性 | 同一文本多次生成的多样性 | 适中最好 |

---

### 7.5 骨骼可视化与 SMPL 网格渲染

#### 7.5.1 骨骼动画生成

运动生成脚本已自动生成 MP4 格式的骨骼动画，保存于输出目录。

#### 7.5.2 SMPL 网格渲染

将骨骼动画转换为 SMPL 人体网格：

```shell
python -m visualize.render_mesh \
  --input_path /path/to/sample##_rep##.mp4
```

此脚本输出：
- `sample##_rep##_smpl_params.npy` — SMPL 参数（theta 旋转、根节点平移、顶点、面）
- `sample##_rep##_obj/` — 每帧的 `.obj` 网格文件

**前置条件**：需要 GPU，可通过 `--device` 参数指定。

**重要**：不要在运行脚本前修改原始 MP4 文件路径。

#### 7.5.3 使用 SMPL 网格的三种方式

1. **Blender/Maya/3DS Max**：导入生成的 `.obj` 文件序列
2. **SMPL Add-on**：使用 Blender 的 SMPL 插件，读取 `smpl_params.npy` 中的 theta 参数（性别中性模型，beta=0）
3. **顶点动画**：所有网格共享相同拓扑（SMPL），直接关键帧化顶点位置。顶点顺序也保存在 `smpl_params.npy` 中。

#### 7.5.4 运动数据导出为 HIK 格式

```shell
python -m visualize.motions2hik
```

将运动导出为 HumanIK 格式，供 Unity/UE4 等引擎使用。

---

## 8. DiP: 极速文本人体运动生成

DiP（Diffusion Planner）是 MDM 的极速版本，速度提升约 40 倍。

### 8.1 工作原理

- **自回归**：每次生成未来 2 秒（40 帧）运动
- **极少扩散步数**：仅需 10 步（甚至 5 步效果也不错）
- **前缀补全**：接收初始前缀运动片段，依次生成后续动作

### 8.2 生成示例

```shell
python -m sample.generate \
  --model_path save/DiP_no-target_10steps_context20_predict40/model000200000.pt \
  --autoregressive \
  --guidance_param 7.5
```

### 8.3 自定义文本提示

```shell
python -m sample.generate \
  --model_path save/DiP_no-target_10steps_context20_predict40/model000200000.pt \
  --autoregressive \
  --guidance_param 7.5 \
  --text_prompt "A person throws a ball"
```

### 8.4 动态文本（DiP 专属）

在 DiP 模式下，每行文本对应一次预测（2 秒运动），适合生成长序列复杂动作：

```shell
python -m sample.generate \
  --model_path save/DiP_no-target_10steps_context20_predict40/model000200000.pt \
  --autoregressive \
  --guidance_param 7.5 \
  --dynamic_text_path assets/example_dynamic_text_prompts.txt
```

### 8.5 训练 DiP

```shell
python -m train.train_mdm \
  --save_dir save/my_humanml_DiP \
  --dataset humanml \
  --arch trans_dec \
  --text_encoder_type bert \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --autoregressive \
  --gen_guidance_param 7.5
```

### 8.6 评估 DiP

```shell
python -m eval.eval_humanml \
  --model_path save/DiP_no-target_10steps_context20_predict40/model000600343.pt \
  --autoregressive \
  --guidance_param 7.5
```

---

## 9. 命令行参数详解

### 9.1 基础参数（Base Options）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cuda` | bool | True | 是否使用 CUDA |
| `--device` | int | 0 | GPU 设备编号 |
| `--seed` | int | 10 | 随机种子 |
| `--batch_size` | int | 64 | 训练批大小 |
| `--train_platform_type` | str | NoPlatform | 训练平台：NoPlatform / ClearmlPlatform / TensorboardPlatform / WandBPlatform |

### 9.2 模型参数（Model Options）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `trans_enc` | 架构：`trans_enc` / `trans_dec` / `gru` |
| `--text_encoder_type` | `clip` | 文本编码器：`clip` / `bert` |
| `--latent_dim` | 512 | Transformer/GRU 隐藏维度 |
| `--layers` | 8 | 网络层数 |
| `--cond_mask_prob` | 0.1 | 条件掩码概率（用于 Classifier-Free Guidance） |
| `--mask_frames` | False | 是否掩码无效帧 |
| `--use_ema` | False | 是否使用指数移动平均 |
| `--unconstrained` | False | 是否为无条件模型 |
| `--emb_trans_dec` | False | Decoder 架构下是否注入条件作为类令牌 |

### 9.3 扩散参数（Diffusion Options）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--noise_schedule` | `cosine` | 噪声调度：`linear` / `cosine` |
| `--diffusion_steps` | 1000 | 扩散步数 |
| `--sigma_small` | True | 是否使用较小 sigma 值 |

### 9.4 几何损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_rcxyz` | 0.0 | 关节位置损失权重 |
| `--lambda_vel` | 0.0 | 速度损失权重 |
| `--lambda_fc` | 0.0 | 足部接触损失权重 |
| `--lambda_target_loc` | 0.0 | 目标位置损失权重 |

### 9.5 前缀补全参数

| 参数 | 说明 |
|------|------|
| `--context_len` | 前缀帧数 |
| `--pred_len` | 预测帧数（默认与 context_len 相同） |
| `--autoregressive` | 启用自回归生成（DiP） |

### 9.6 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必需 | 模型 checkpoint 路径 |
| `--output_dir` | 自动生成 | 输出目录 |
| `--num_samples` | 6 | 采样数量 |
| `--num_repetitions` | 3 | 每个采样的重复次数 |
| `--guidance_param` | 2.5 | Classifier-Free Guidance 参数 |
| `--motion_length` | 6.0 | 运动时长（秒） |
| `--input_text` | 空 | 输入文本文件路径 |
| `--text_prompt` | 空 | 单条文本提示 |
| `--action_file` | 空 | 动作标签文件路径 |
| `--action_name` | 空 | 单个动作名称 |

---

## 10. 工具链完整依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户入口脚本                              │
│  sample.generate / sample.edit / train.train_mdm / eval.*      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  utils.parser   │  │  data_loaders    │  │   diffusion       │
│  - 参数解析     │  │  - 数据加载      │  │  - 扩散过程       │
│  - 参数加载     │  │  - 数据集定义     │  │  - 损失计算       │
└────────┬────────┘  └────────┬────────┘  └────────┬─────────┘
         │                    │                     │
         └────────────┬───────┘─────────────────────┘
                       ▼
              ┌──────────────────┐
              │   utils.model   │
              │  - 模型创建      │
              │  - 模型加载      │
              └────────┬────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   model.mdm     │
              │  - MDM 主类     │
              │  - Transformer   │
              │  - CLIP/BERT     │
              └────────┬────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  model.rotation2xyz │
              │  - SMPL 渲染     │
              │  - 旋转转坐标    │
              └──────────────────┘
```

**核心数据流**：

```
文本/动作 → CLIP/BERT/GloVe → 嵌入向量
    ↓
扩散过程 (T 步去噪)
    ↓
MDM (Transformer/GRU) → 预测 x₀
    ↓
旋转表示转换 (rot6d / hml_vec → XYZ)
    ↓
骨骼可视化 / SMPL 网格
```

---

## 11. 输出文件说明

### 11.1 生成输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `results.npy` | NumPy | 包含 motion、text、lengths 的字典 |
| `results.txt` | 文本 | 所有文本提示 |
| `results_len.txt` | 文本 | 每个运动的长度 |
| `sample##_rep##.mp4` | MP4 | 单个骨骼动画 |
| `samples_00_to_02.mp4` | MP4 | 多行合成动画（无约束模式） |

### 11.2 编辑输出文件

| 文件 | 说明 |
|------|------|
| `input_motion##.mp4` | 输入运动（蓝色关节） |
| `sample##_rep##.mp4` | 编辑后的运动（橙色=生成部分） |
| `sample##.mp4` | 所有重复版本的水平拼接 |

### 11.3 训练输出文件

| 文件 | 说明 |
|------|------|
| `model#########.pt` | 模型权重 |
| `opt#########.pt` | 优化器状态 |
| `args.json` | 训练超参数 |
| `eval_humanml_#########.log` | 评估日志 |
| `*.samples/` | 生成的动作样本 |

### 11.4 SMPL 渲染输出文件

| 文件 | 说明 |
|------|------|
| `frame###.obj` | 每帧的 OBJ 网格文件 |
| `*_smpl_params.npy` | SMPL 参数（thetas, root_trans, vertices） |

---

## 12. 常见问题与注意事项

### 12.1 Windows 系统注意事项

- `.sh` 下载脚本需要在 WSL 或 Git Bash 中运行
- 部分路径处理可能存在兼容性问题，建议使用 WSL
- `conda env create -f environment.yml` 在 Windows 上可能需要手动调整

### 12.2 常见错误

**Q: 提示 "Arguments json file was not found"**
> A: 使用的 checkpoint 目录下缺少 `args.json` 文件。需要从原始 zip 包解压，确保 `args.json` 和模型权重在同一目录。

**Q: 评估时显存不足**
> A: 减小 `--eval_batch_size` 参数（默认 32，不建议修改）或 `--eval_mode debug`。

**Q: CLIP 加载失败**
> A: 确保已安装 CLIP：`pip install git+https://github.com/openai/CLIP.git`

**Q: 渲染 SMPL 网格失败**
> A: 确保 SMPLify 依赖正确安装且 GPU 可用。

### 12.3 性能优化建议

1. **使用 50 步模型**：速度提升 20 倍，精度相当或更优
2. **CLIP 缓存**：最新版代码已自动缓存 CLIP 编码结果
3. **自回归采样**：DiP 使用 `--autoregressive` 模式可大幅加速
4. **混合精度**：代码已弃用 FP16，直接使用默认精度即可

### 12.4 数据集选择建议

| 需求 | 推荐数据集 |
|------|-----------|
| 日常动作文本生成 | HumanML3D |
| 精细动作 | KIT-ML |
| 简单动作分类 | HumanAct12 |
| 学术对比 | UESTC |
| 完全开放生成 | 无条件模式 |

---

## 附录：快速命令参考

```shell
# 1. 快速生成（推荐 50 步模型）
python -m sample.generate --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt --num_samples 6 --num_repetitions 3

# 2. 自定义文本生成
python -m sample.generate --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt --text_prompt "the person walked forward and is picking up his toolbox."

# 3. 动作编辑（In-Between）
python -m sample.edit --model_path ./save/humanml_trans_enc_512/model000200000.pt --edit_mode in_between

# 4. 动作编辑（Upper-Body + 文本条件）
python -m sample.edit --model_path ./save/humanml_trans_enc_512/model000200000.pt --edit_mode upper_body --text_condition "A person throws a ball"

# 5. 训练 HumanML3D 模型
python -m train.train_mdm --save_dir save/my_model --dataset humanml --use_ema --mask_frames --eval_during_training

# 6. 评估模型
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512_50steps/model000200000.pt --eval_mode wo_mm

# 7. DiP 极速生成
python -m sample.generate --model_path save/DiP/model.pt --autoregressive --guidance_param 7.5

# 8. SMPL 网格渲染
python -m visualize.render_mesh --input_path /path/to/sample00_rep00.mp4

# 9. 训练 DiP
python -m train.train_mdm --save_dir save/my_DiP --dataset humanml --arch trans_dec --text_encoder_type bert --diffusion_steps 10 --context_len 20 --pred_len 40 --autoregressive --use_ema
```
