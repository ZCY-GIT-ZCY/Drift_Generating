# MLD (Motion Latent Diffusion) 工作文档

## 一、项目概述

MLD (Motion Latent Diffusion Models) 是一个用于**文本到动作生成**（Text-to-Motion）和**动作到动作生成**（Action-to-Motion）的扩散模型。该项目通过在隐空间中执行扩散过程，实现了高质量的动作生成，并且比直接在原始动作数据上进行扩散的模型快两个数量级。

### 核心特性
- **两阶段训练**：首先训练 Motion VAE 进行动作压缩，然后训练 Latent Diffusion Model
- **多数据集支持**：HumanML3D、Kit、HumanAct12、UESTC
- **文本条件生成**：使用 CLIP 或 BERT 进行文本编码
- **动作条件生成**：支持动作类别作为条件

---

## 二、整体架构

MLD 采用类似 SD (Stable Diffusion) 的架构，将动作生成分为两个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                      Stage 1: VAE                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Motion  │ →  │ Encoder  │ →  │ Latent z │              │
│  │  Features │    │ (Transformer) │         │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                         ↓                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Motion   │ ← │ Decoder  │ ←  │ Latent z │              │
│  │ Features │    │(Transformer)│          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: Diffusion                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Text     │ →  │ Denoiser │ →  │ Latent z │              │
│  │ Embedding│    │(Transformer)│         │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                         ↓                   │
│                         VAE Decoder → Motion Features       │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心脚本功能

### 3.1 入口脚本

| 脚本 | 功能描述 |
|------|----------|
| `train.py` | 训练入口脚本，支持 VAE、Diffusion、VAE+Diffusion 三种训练模式 |
| `test.py` | 测试/评估脚本，支持多种评估指标计算 |
| `demo.py` | 演示脚本，用于生成动作 |
| `render.py` | Blender 渲染脚本，将生成的动作渲染为视频 |

### 3.2 模型核心 (`mld/models/`)

| 文件 | 功能描述 |
|------|----------|
| `modeltype/mld.py` | 主模型类，整合 VAE 和 Diffusion，支持训练和推理 |
| `modeltype/base.py` | 基础模型类，定义训练/验证/测试流程 |
| `architectures/mld_vae.py` | MLD 专用 VAE（Skip-Connection Transformer） |
| `architectures/actor_vae.py` | Actor 风格 VAE（标准 Transformer） |
| `architectures/mld_denoiser.py` | 去噪器 U-Net/Transformer |
| `architectures/mld_clip.py` | CLIP 文本编码器 |
| `architectures/vposert_vae.py` | VPoser VAE |
| `architectures/t2m_textenc.py` | T2M 评估用文本编码器 |
| `architectures/t2m_motionenc.py` | T2M 评估用动作编码器 |

### 3.3 数据处理 (`mld/data/`)

| 文件 | 功能描述 |
|------|----------|
| `get_data.py` | 数据集加载器工厂 |
| `base.py` | 数据模块基类 (LightningDataModule) |
| `humanml/dataset.py` | HumanML3D 数据集实现 |
| `humanml/scripts/motion_process.py` | 动作数据处理工具 |
| `a2m/dataset.py` | Action-to-Motion 数据集 |

### 3.4 特征变换 (`mld/transforms/`)

| 文件 | 功能描述 |
|------|----------|
| `joints2jfeats/rifke.py` | 将关节坐标转换为旋转不变特征 (RIFKE) |
| `joints2rots/` | 关节坐标转旋转表示 |
| `rots2joints/smplh.py` | SMPL-H 骨骼变换 |
| `rots2joints/base.py` | 旋转到关节基类 |
| `rotation2xyz.py` | 旋转表示转换为 XYZ 坐标 |
| `feats2smpl.py` | 特征转 SMPL 模型 |

### 3.5 算子模块 (`mld/models/operator/`)

| 文件 | 功能描述 |
|------|----------|
| `cross_attention.py` | Transformer 编码器/解码器（含 Skip-Connection） |
| `position_encoding.py` | 位置编码（Sine、Learned、1D、2D） |
| `adain.py` | Adaptive Instance Normalization |
| `blocks.py` | Transformer Block 组件 |

### 3.6 损失函数 (`mld/models/losses/`)

| 文件 | 功能描述 |
|------|----------|
| `mld.py` | MLD 专用损失函数（重建、KL、扩散损失） |
| `utils.py` | 损失工具函数 |

### 3.7 工具函数 (`mld/utils/`)

| 文件 | 功能描述 |
|------|----------|
| `temos_utils.py` | 长度掩码生成、去填充、旋转格式转换 |
| `geometry.py` | 几何变换（旋转矩阵、轴角、四元数、6D旋转） |
| `rotation_conversions.py` | 旋转表示转换 |
| `joints.py` | 关节索引映射、骨骼定义 |

---

## 四、可复用组件详解

### 4.1 数据处理模块

#### 4.1.1 长度掩码生成 (`temos_utils.py`)

```python
def lengths_to_mask(lengths: List[int], device: torch.device, max_len: int = None) -> Tensor:
    """根据序列长度生成布尔掩码
    
    参数:
        lengths: 每个序列的实际长度
        device: 计算设备
        max_len: 最大长度，默认为最长序列长度
    
    返回:
        mask: (batch_size, max_len) 的布尔张量，True 表示有效位置
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
```

**复用方式**：此函数是处理变长序列的标准方式，可直接移植到其他序列生成任务。

#### 4.1.2 填充去除 (`temos_utils.py`)

```python
def remove_padding(tensors, lengths):
    """去除批量张量中的填充部分
    
    参数:
        tensors: (batch_size, max_len, feature_dim) 的张量
        lengths: 每个序列的实际长度列表
    
    返回:
        list: 变长序列列表
    """
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]
```

#### 4.1.3 RIFKE 特征变换 (`joints2jfeats/rifke.py`)

RIFKE (Rotation-Invariant Feature) 是一种将骨骼关节坐标转换为旋转不变表示的方法：

**核心处理流程**：
1. **根部对齐**：将根节点（骨盆）移至原点
2. **轨迹提取**：分离根节点的 XY 平面位移作为轨迹
3. **前向计算**：基于骨骼方向计算前向向量
4. **局部旋转**：将所有关节旋转到局部坐标系
5. **速度特征**：计算轨迹和角度的速度（一阶差分）

```python
class Rifke(Joints2Jfeats):
    """旋转不变特征变换
    
    特点:
        - 对根节点位置和全局旋转保持不变
        - 输出特征包含: [root_y, joint_local, vel_angle, vel_trajectory]
    """
    
    def forward(self, joints: Tensor) -> Tensor:
        # 1. 根节点对齐
        poses[..., 1] -= get_floor(poses)  # 去除地面高度
        translation = poses[..., 0, :]      # 根节点位移
        poses = poses[..., 1:, :]           # 移除根节点
        
        # 2. 计算轨迹速度
        vel_trajectory = torch.diff(trajectory, dim=-2)
        
        # 3. 计算前向方向
        forward = get_forward_direction(poses)
        
        # 4. 旋转到局部坐标系
        # ...
        
        # 5. 拼接特征
        features = torch.cat([root_y, poses_features, vel_angles, vel_trajectory_local], -1)
        return features
```

**可复用性**：
- RIFKE 特征可应用于任何需要旋转不变性的动作相关任务
- `inverse()` 方法支持从特征恢复关节坐标
- 支持 `mmm`、`mmmns`、`humanml3d` 骨骼类型

#### 4.1.4 数据归一化

数据集使用 Mean/Std 进行 Z-Score 归一化：

```python
# 归一化
motion_normalized = (motion - mean) / std

# 反归一化
motion_original = motion_normalized * std + mean
```

---

### 4.2 VAE 架构模块

#### 4.2.1 MLD VAE (`mld_vae.py`)

MLD 采用基于 Skip-Connection Transformer 的 VAE 架构：

**架构特点**：
1. **全局 Motion Token**：使用可学习的 token 聚合序列信息
2. **Skip-Connection 编码器/解码器**：保留多尺度特征
3. **变分推断**：支持 MLP 或 token 分割方式预测 μ/σ

```python
class MldVae(nn.Module):
    def __init__(self,
                 nfeats: int,           # 输入特征维度
                 latent_dim: list,       # [token数, 隐层维度], 如 [1, 256]
                 ff_size: int,          # 前馈网络维度
                 num_layers: int,       # Transformer 层数
                 num_heads: int,        # 注意力头数
                 dropout: float,
                 arch: str,              # "all_encoder" 或 "encoder_decoder"
                 **kwargs):
        
        # 编码器：Skip-Connection Transformer Encoder
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, norm)
        
        # 解码器
        if arch == "all_encoder":
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, norm)
        elif arch == "encoder_decoder":
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, norm)
        
        # 全局 Motion Token
        self.global_motion_token = nn.Parameter(torch.randn(latent_size, latent_dim))
        
        # 分布预测层
        if self.mlp_dist:
            self.dist_layer = nn.Linear(latent_dim, 2 * latent_dim)  # 输出 μ, σ
        else:
            # 前半 token 为 μ，后半为 logvar
            self.global_motion_token = nn.Parameter(torch.randn(latent_size * 2, latent_dim))
        
        # 特征嵌入
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        self.final_layer = nn.Linear(latent_dim, output_feats)
```

**编码器前向**：
```python
def encode(self, features, lengths=None):
    # 1. 生成掩码
    mask = lengths_to_mask(lengths, device)
    
    # 2. 特征嵌入
    x = self.skel_embedding(features)  # [B, T, D]
    x = x.permute(1, 0, 2)            # [T, B, D]
    
    # 3. 拼接 Motion Token
    dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))  # [N, B, D]
    xseq = torch.cat((dist, x), 0)    # [N+T, B, D]
    
    # 4. Transformer 编码
    xseq = self.query_pos_encoder(xseq)
    dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]
    
    # 5. 分布预测
    if self.mlp_dist:
        tokens_dist = self.dist_layer(dist)
        mu, logvar = tokens_dist[:, :, :self.latent_dim], tokens_dist[:, :, self.latent_dim:]
    else:
        mu, logvar = dist[0:self.latent_size], dist[self.latent_size:]
    
    # 6. 重参数化采样
    dist = torch.distributions.Normal(mu, logvar.exp().pow(0.5))
    latent = dist.rsample()
    return latent, dist
```

**解码器前向**：
```python
def decode(self, z, lengths):
    # 1. 生成查询
    mask = lengths_to_mask(lengths, z.device)
    queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
    
    # 2. 拼接隐变量
    xseq = torch.cat((z, queries), axis=0)
    augmask = torch.cat((z_mask, mask), axis=1)
    
    # 3. Transformer 解码
    output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
    
    # 4. 输出投影
    output = self.final_layer(output)
    output[~mask.T] = 0  # 填充置零
    return output.permute(1, 0, 2)  # [B, T, D]
```

**复用建议**：
- 可替换 `nfeats` 适配不同的动作特征维度
- `latent_dim` 可调整隐空间维度
- `arch` 参数选择编码器-解码器架构

#### 4.2.2 Actor VAE (`actor_vae.py`)

Actor 风格的 VAE 使用标准 Transformer，结构更简洁：

```python
class ActorVae(nn.Module):
    """Actor 风格 VAE
    
    与 MLD VAE 的区别:
    - 使用标准 Transformer 而非 Skip-Connection
    - 独立的 Encoder/Decoder 类
    - 单一的 μ/σ token
    """
    
    def __init__(self, ...):
        self.encoder = ActorAgnosticEncoder(nfeats, vae=True, latent_dim=...)
        self.decoder = ActorAgnosticDecoder(nfeats, vae=True, latent_dim=...)
```

**复用建议**：适合需要更简单架构或与 Actor 方法对比的场景。

---

### 4.3 Transformer 算子

#### 4.3.1 Skip-Connection Transformer (`cross_attention.py`)

这是 MLD 的核心创新之一，实现了 U-Net 风格的跳跃连接：

```python
class SkipTransformerEncoder(nn.Module):
    """带跳跃连接的 Transformer 编码器
    
    架构: Input → [Block × N//2] → Middle → [Block × N//2] → Output
                ↑______________|  |______________↑
    
    特点:
    - 中间层作为信息瓶颈
    - 跳跃连接保留多尺度信息
    - 通过线性层融合不同尺度的特征
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        # 假设 num_layers = 9
        num_block = (num_layers - 1) // 2  # = 4
        
        self.input_blocks = _get_clones(encoder_layer, num_block)    # 4 层
        self.middle_block = _get_clone(encoder_layer)                # 1 层
        self.output_blocks = _get_clones(encoder_layer, num_block)   # 4 层
        self.linear_blocks = _get_clones(nn.Linear(2*d_model, d_model), num_block)
    
    def forward(self, src, src_key_padding_mask=None):
        x = src
        xs = []  # 存储中间层输出
        
        # 编码器前半部分
        for module in self.input_blocks:
            x = module(x, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)  # 保留跳跃连接
        
        # 中间层
        x = self.middle_block(x, src_key_padding_mask=src_key_padding_mask)
        
        # 解码器后半部分（带跳跃连接）
        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)  # 拼接特征
            x = linear(x)  # 降维
            x = module(x, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        return x
```

**复用建议**：
- 可应用于任何需要保留多尺度信息的序列建模任务
- 跳跃连接缓解了深层 Transformer 的梯度问题
- 适用于需要同时捕获局部和全局信息的场景

#### 4.3.2 位置编码 (`position_encoding.py`)

支持两种位置编码方式：

```python
class PositionEmbeddingSine1D(nn.Module):
    """正弦位置编码
    
    优点:
    - 可处理任意长度的序列
    - 无需学习，泛化能力强
    """
    
    def __init__(self, d_model, max_len=500):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


class PositionEmbeddingLearned1D(nn.Module):
    """可学习位置编码
    
    优点:
    - 可端到端优化
    - 适合特定任务
    """
    
    def __init__(self, d_model, max_len=500):
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))


def build_position_encoding(N_steps, position_embedding="sine"):
    """工厂函数"""
    if position_embedding in ('v2', 'sine'):
        return PositionEmbeddingSine1D(N_steps)
    elif position_embedding in ('v3', 'learned'):
        return PositionEmbeddingLearned1D(N_steps)
```

**复用建议**：可应用于任何序列到序列的 Transformer 模型。

---

### 4.4 去噪器模块 (`mld_denoiser.py`)

去噪器是扩散模型的核心组件：

```python
class MldDenoiser(nn.Module):
    """MLD 去噪器
    
    接受:
        - 加噪样本 sample
        - 时间步 timestep
        - 条件 embedding (文本/动作)
    
    输出:
        - 预测的噪声或干净样本
    """
    
    def __init__(self,
                 condition: str,           # "text" 或 "action"
                 latent_dim: list,         # [token数, 隐层维度]
                 nfeats: int,              # 特征维度（无VAE时使用）
                 text_encoded_dim: int,    # 文本编码维度
                 **kwargs):
        
        # 时间步嵌入
        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(text_encoded_dim, self.latent_dim)
        
        # 文本/条件投影
        if condition == "text":
            self.emb_proj = nn.Sequential(
                nn.ReLU(), 
                nn.Linear(text_encoded_dim, self.latent_dim)
            )
        
        # Transformer 去噪器
        encoder_layer = TransformerEncoderLayer(...)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, norm)
        
        # 或 Decoder 结构
        if arch == "trans_dec":
            self.decoder = TransformerDecoder(decoder_layer, num_layers, norm)
```

**前向过程**：
```python
def forward(self, sample, timestep, encoder_hidden_states, lengths=None):
    # 1. 维度调整
    sample = sample.permute(1, 0, 2)  # [T, B, D]
    
    # 2. 时间步嵌入
    timesteps = timestep.expand(sample.shape[1])
    time_emb = self.time_embedding(self.time_proj(timesteps))  # [1, B, D]
    
    # 3. 条件嵌入 + 时间嵌入
    text_emb = encoder_hidden_states.permute(1, 0, 2)  # [S, B, D]
    text_emb_latent = self.emb_proj(text_emb)           # [S, B, D]
    emb_latent = torch.cat((time_emb, text_emb_latent), 0)
    
    # 4. Transformer 处理
    xseq = torch.cat((sample, emb_latent), axis=0)
    xseq = self.query_pos(xseq)
    tokens = self.encoder(xseq)
    
    # 5. 取样部分输出
    sample = tokens[:sample.shape[0]]
    return sample.permute(1, 0, 2)  # [B, T, D]
```

---

### 4.5 几何变换工具 (`geometry.py`)

提供了完整的旋转表示转换工具：

```python
# 支持的旋转表示
ROTTYPE = ["rotvec", "axisangle",  # 轴角: (N, 3)
           "rotquat", "quaternion", # 四元数: (N, 4)
           "rot6d", "6drot",        # 6D旋转: (N, 6)
           "rotmat"]                # 旋转矩阵: (N, 9) 或 (N, 3, 3)

# 核心转换函数
def axis_angle_to_matrix(axis_angle):     # 轴角 → 矩阵
def matrix_to_axis_angle(rotation_matrix): # 矩阵 → 轴角
def quaternion_to_matrix(quaternion):      # 四元数 → 矩阵
def matrix_to_quaternion(rotation_matrix): # 矩阵 → 四元数
def rotation_6d_to_matrix(rotation_6d):   # 6D → 矩阵
def matrix_to_rotation_6d(rotation_matrix):# 矩阵 → 6D

# 便捷函数
def axis_angle_to(newtype, rotations):     # 轴角转任意表示
def matrix_to(newtype, rotations):         # 矩阵转任意表示
def to_matrix(oldtype, rotations):         # 任意表示转矩阵
```

**复用建议**：
- 任何涉及 3D 旋转计算的任务
- 人体姿态、刚体变换、机器人学等

---

## 五、配置系统

MLD 使用 OmegaConf + YAML 配置文件：

```yaml
# 关键配置项
model:
  model_type: mld
  motion_vae:           # VAE 配置
    target: mld.models.architectures.mld_vae.MldVae
    params:
      nfeats: 263       # HumanML3D 特征维度
      latent_dim: [1, 256]
      num_layers: 9
      ff_size: 1024
      num_heads: 4
      dropout: 0.1
  
  text_encoder:         # 文本编码器配置
    target: mld.models.architectures.mld_clip.MldTextEncoder
    params:
      modelpath: ./pretrained_models/clip-vit-base-patch32
  
  denoiser:            # 去噪器配置
    target: mld.models.architectures.mld_denoiser.MldDenoiser
    params:
      condition: text
      arch: trans_enc

DATASET:
  NFEATS: 263          # 特征维度
  HUMANML3D:
    ROOT: ./datasets/humanml3d
    UNIT_LEN: 4         # 采样单位长度
```

---

## 六、训练流程

### 6.1 Stage 1: VAE 训练

```bash
python -m train --cfg configs/config_vae_humanml3d.yaml --batch_size 64 --nodebug
```

训练目标：最小化重建损失 + KL 散度

### 6.2 Stage 2: Diffusion 训练

```bash
python -m train --cfg configs/config_mld_humanml3d.yaml \
    --cfg_assets configs/assets.yaml \
    --batch_size 64 --nodebug
```

关键参数：`PRETRAINED_VAE` 指向 Stage 1 的模型

### 6.3 评估

```bash
python -m test --cfg configs/config_mld_humanml3d.yaml \
    --cfg_assets configs/assets.yaml
```

评估指标：FID, R-precision, APE, AVE, Diversity, MM Distance

---

## 七、扩展指南

### 7.1 添加新数据集

1. 在 `mld/data/` 创建数据集类，继承 `BASEDataModule`
2. 实现 `__getitem__` 方法，返回 `(word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens)`
3. 在 `get_data.py` 的 `dataset_module_map` 中注册

### 7.2 替换文本编码器

```yaml
model:
  text_encoder:
    target: mld.models.architectures.mld_clip.MldTextEncoder  # 可替换
    params:
      modelpath: ./pretrained_models/bert-base-uncased  # 换用 BERT
```

### 7.3 自定义条件

修改 `mld_denoiser.py` 中的 `EmbedAction` 类或添加新的条件嵌入层。

---

## 八、总结

MLD 项目提供了以下高度可复用的组件：

| 组件 | 位置 | 可复用场景 |
|------|------|-----------|
| 变长序列掩码处理 | `temos_utils.py` | 任何序列生成任务 |
| RIFKE 特征变换 | `joints2jfeats/rifke.py` | 人体动作识别、生成 |
| Skip-Transformer | `operator/cross_attention.py` | 需要多尺度特征的序列建模 |
| 位置编码 | `operator/position_encoding.py` | Transformer 系列模型 |
| 旋转表示转换 | `geometry.py` | 3D 旋转相关计算 |
| MLD VAE | `architectures/mld_vae.py` | 动作/序列的压缩表示学习 |
| 去噪器 | `architectures/mld_denoiser.py` | 扩散模型实现 |
| SMPL 骨骼变换 | `transforms/rots2joints/smplh.py` | 人体渲染、骨骼可视化 |

这些组件都经过良好封装，可以独立使用或组合到其他项目中。
