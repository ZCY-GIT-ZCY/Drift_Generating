# Drifting_Model 模块详解

## 1. 项目概述

本模块是论文 **Generative Modeling via Drifting**（arXiv:2602.04770）的官方 JAX 实现，专注于 **ImageNet 256×256 的单步（1-NFE）图像生成**。核心思想是通过"漂移"（Drifting）机制，让生成样本在特征空间中持续向真实数据分布移动，无需迭代采样即可生成高质量图像。

---

## 2. 目录结构总览

```
Drifting_Model/
├── main.py                    # 统一入口（选择生成器训练或MAE训练）
├── train.py                   # 生成器训练主循环
├── train_mae.py               # MAE预训练主循环
├── inference.py               # 推理/FID评估入口
├── drift_loss.py              # 核心漂移损失函数（最关键的可复用模块）
├── memory_bank.py             # 类别级环形缓冲区
├── configs/                   # YAML配置文件
│   ├── gen/                   # 生成器配置
│   └── mae/                   # MAE配置
├── dataset/                   # 数据处理
│   ├── dataset.py             # ImageNet数据加载与预处理
│   ├── latent.py              # 潜在空间缓存构建与读取
│   └── vae.py                 # VAE编码/解码工具（SD-VAE）
├── models/                    # 模型定义
│   ├── generator.py           # DitGen生成器（LightningDiT主干）
│   ├── mae_model.py           # MAE-ResNet特征提取器
│   ├── convnext.py            # ConvNeXt V2特征提取器
│   └── hf.py                  # HuggingFace加载工具
└── utils/                     # 工具函数
    ├── hsdp_util.py            # HSDP分片与分布式训练核心
    ├── fid_util.py             # FID/IS/Precision-Recall评估
    ├── model_builder.py        # 模型/优化器/数据集构建
    ├── init_util.py            # 检查点保存/加载
    ├── logging.py              # WandB/离线日志
    ├── misc.py                 # 配置解析、profiling
    └── ckpt_util.py            # Orbax检查点工具
```

---

## 3. 算法核心：Drift Loss（`drift_loss.py`）

### 3.1 设计思想

Drift Loss 是本方法的灵魂。它不需要预测噪声或学习-score matching，而是在**特征空间**中定义了一个"漂移力场"——生成样本受来自正样本（真实样本特征）的引力和来自负样本（其他类/无条件特征）的斥力作用，自动向数据分布漂移。

### 3.2 输入

| 参数 | 形状 | 含义 |
|------|------|------|
| `gen` | `[B, C_g, S]` | 生成样本特征（需要优化的目标） |
| `fixed_pos` | `[B, C_p, S]` | 正样本特征（来自Memory Bank的真实样本） |
| `fixed_neg` | `[B, C_n, S]` | 负样本特征（来自Memory Bank的无条件/其他类样本） |
| `weight_gen/pos/neg` | 各特征的权重 | 用于类别不平衡调整 |
| `R_list` | `[R1, R2, ...]` | 多温度RBF核尺度列表 |

其中 `S` 为特征维度（通过MAE等特征提取器输出的特征展平后的维度），`C_*` 为通道/组数。

### 3.3 算法流程详解

#### 第一步：构造 target 序列

```python
old_gen = jax.lax.stop_gradient(gen)
targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)
```

三类样本拼成一个序列，维度布局为：

```
target 序列 = [gen(自), neg, pos]
             ├── C_g ──┤── C_n ──┤── C_p ──┤
```

其中 `old_gen`（即 `jax.lax.stop_gradient(gen)`）是**前一步生成的特征**，它自己也出现在 target 中，但自 attending 时通过第 85 行的对角线 mask 排除。保留 `old_gen` 的目的是：生成的特征在漂移场中"看到"自己的位置，从而避免对自己产生引力。

#### 第二步：动态尺度归一化

```python
dist = cdist(old_gen, targets)  # [B, C_g, C_g+C_n+C_p]
weighted_dist = dist * targets_w[:, None, :]
scale = weighted_dist.mean() / targets_w.mean()
scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3)
old_gen_scaled = old_gen / scale_inputs
targets_scaled = targets / scale_inputs
```

**目的**：把不同特征通道的欧氏距离归一化到量级 1，避免绝对距离值干扰 RBF 核的尺度。具体做法是：
- 用全体样本的**加权平均距离**作为当前尺度估计
- 除以 `sqrt(S)` 是为了将距离无量纲化（S 是特征维度）
- `jnp.clip(scale / sqrt(S), a_min=1e-3)` 防止除零

注意：这个 scale 是**动态**的（每 batch 重新计算），而非固定值，因此可以自适应不同的特征分布。

对角线 mask：
```python
mask_val = 100.0
diag_mask = jnp.eye(C_g, dtype=jnp.float32)
block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p)))
dist_normed = dist_normed + block_mask * mask_val
```
通过在对角线位置加上大值，使得 `exp(-dist/R) → 0`，从而在 softmax 时自 attending 的贡献趋近于零。

#### 第三步：构建亲和力矩阵（核心）

对每个温度 R：

```python
logits = -dist_normed / R                           # [B, C_g, C_all]
affinity_row = jax.nn.softmax(logits, axis=-1)     # softmax 在 target 维度
affinity_col = jax.nn.softmax(logits, axis=-2)     # softmax 在 gen 维度
affinity = jnp.sqrt(jnp.clip(affinity_row * affinity_col, a_min=1e-6))
affinity = affinity * targets_w[:, None, :]
```

**双向 softmax 的物理含义**：
- `softmax(logits, axis=-1)`：给定某个 gen_i，对所有 target 的注意力权重（归一化到 1）
- `softmax(logits, axis=-2)`：给定某个 target_j，对所有 gen 的注意力权重（归一化到 1）
- `sqrt(A * A.T)` → **对称亲和力**：只有当 gen_i 和 target_j **互相"近"** 时，亲和力才大

这等价于一个 **soft kNN 亲和力矩阵**，而不是简单的高斯核 `exp(-d²/2σ²)`。具体来说：
- 若 gen_i 离 target_j 远但其他 gen 离 target_j 更近，则 `affinity_row[i,j]` 仍可能较大
- 但如果 target_j 离其他 target 更近（即 target_j 本身在 target 集合中是"近邻"的），则 `affinity_col[i,j]` 较小
- 两者相乘，几何平均 → 只有**双向都近**时亲和力才大

`R` 的作用：
- R 越小：`logits = -dist/R` 越大，exp 越尖锐 → 只关注最近的 1-2 个邻居
- R 越大：越平坦 → 关注更远的邻居

#### 第四步：计算漂移力

```python
split_idx = C_g + C_n
aff_neg = affinity[:, :, :split_idx]    # [B, C_g, C_n] — 推离负样本的力
aff_pos = affinity[:, :, split_idx:]    # [B, C_g, C_p] — 拉向正样本的力

sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)  # [B, C_g, 1]
r_coeff_neg = -aff_neg * sum_pos                        # [B, C_g, C_n] — 斥力系数
r_coeff_pos = aff_pos * sum_neg                         # [B, C_g, C_p] — 引力系数

R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)
total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)
total_coeffs = R_coeff.sum(axis=-1)
total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
```

**物理直觉**：将漂移力 `V(x)` 看成各 target 样本产生的"力"的叠加。设 gen 特征为 `x`，target 特征为 `y`，则单样本产生的力为 `(y - x)`。引入亲和力权重后，gen_i 受到的整体力为：

```
F_i = sum_j( w_ij * (y_j - x_i) )
    = sum_j(w_ij * y_j) - x_i * sum_j(w_ij)
```

代码实现正是这两项：
- `einsum("biy,byx->bix", R_coeff, targets_scaled)` → `sum_j(w_ij * y_j)`
- `total_coeffs[..., None] * old_gen_scaled` → `x_i * sum_j(w_ij)`

**引力和斥力的来源**：
- `r_coeff_neg = aff_neg * sum_pos`：负样本的亲和力乘以正样本亲和力之和。直观理解：如果 gen 靠近负样本（`aff_neg` 大），且同时远离正样本（`sum_pos` 小），则斥力强。
- `r_coeff_pos = -aff_pos * sum_neg`：正样本的亲和力乘以负样本亲和力之和。直观理解：如果 gen 靠近正样本（`aff_pos` 大），且同时远离负样本（`sum_neg` 小），则引力强。

这两种力合并 → gen 被"拉向正样本密集区、推离负样本密集区"。

#### 第五步：梯度截断（关键）

```python
goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
    calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
)
gen_scaled = gen / scale_inputs
diff = gen_scaled - goal_scaled
loss = jnp.mean(diff ** 2, axis=(-1, -2))
```

**`stop_gradient` 是整个 drift loss 的灵魂**：
- `goal`（漂移目标）是在**前一步的 gen（old_gen）** 上计算的
- 当前 gen 只通过 MSE 损失向这个固定目标靠近
- 因此漂移方向的计算**不依赖当前 gen 的梯度**

这样做有两个好处：
1. **避免了对自己的梯度**：漂移场是由旧 gen 定义的，新 gen 只需向场指示的方向走
2. **类似 EM 算法**：M-step（计算 goal）和 E-step（梯度更新）分离，训练更稳定

#### 第六步：MSE 损失

漂移方向 `force_across_R` 被归一化后加到 `old_gen_scaled` 上，得到 `goal_scaled`，最后用 MSE 驱动当前 gen 逼近目标：

```
loss = mean( || gen_scaled - goal_scaled ||² )
```

这与标准的 L2 回归损失无异，梯度通过 MSE 反传回生成器。

### 3.4 多温度聚合

Drift Loss 在多个 R 值（通常是 `[0.02, 0.05, 0.2]`）上分别计算漂移力，然后对每个温度的力做 `1/||force_R||` 归一化后求和：

```python
force_across_R = jnp.zeros_like(old_gen_scaled)
for R in R_list:
    # ... 计算当前 R 下的 total_force_R ...
    f_norm_val = (total_force_R ** 2).mean()
    force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8))
    force_across_R = force_across_R + total_force_R / force_scale
```

**物理含义**：
- **小 R（如 0.02）**：核尖锐，只关注最近的 1-2 个邻居 → 精细的局部结构
- **中 R（如 0.05）**：中等尺度 → 捕捉中层语义结构
- **大 R（如 0.2）**：核平坦，关注更远的邻居 → 粗粒度的全局结构

归一化后叠加保证了不同尺度的力有相等的贡献，防止大 R 下的力因为绝对值大而主导训练。

### 3.5 与理论公式的对应关系

Drift Loss 的理论公式（论文中）在特征空间定义了漂移力场：

\[
V_{p,q}(x) = \frac{1}{Z_p Z_q}\cdot\mathbb{E}_{y^+\sim p,\,y^-\sim q}\Big[\,k(x,y^+)k(x,y^-)\,(y^+-y^-)\,\Big]
\]

其中：
- \(p\)：正样本（真实样本）的分布
- \(q\)：负样本（其他类/无条件）的分布
- \(k(x,y)\)：核函数，衡量 x 与 y 的亲和力
- \(y^+-y^-\)：力的方向（拉向正样本、推离负样本）

**代码中各项与公式的对应关系**：

| 理论公式 | 代码实现 | 说明 |
|---------|---------|------|
| 核函数 \(k(x,y^+)k(x,y^-)\) | `affinity`（双向 softmax 亲和力） | 不是简单的高斯核，而是双向 soft kNN 亲和力 |
| \((y^+-y^-)\) 方向向量 | `R_coeff`（引斥力系数） | `aff_neg * sum_pos` → 负向力（推离），`aff_pos * sum_neg` → 正向力（拉向） |
| 期望求和 \(\mathbb{E}\) | `einsum("biy,byx->bix", R_coeff, targets)` | 矩阵乘法一次性完成所有样本对的加权求和，本质是离散蒙特卡洛估计 |
| 归一化常数 \(1/Z_p Z_q\) | `sum_pos`、`sum_neg` 归一化 | 防止空集合时力为零（无引力或无斥力时整体力归零） |
| 多温度 R | `for R in R_list` 循环 + 归一化叠加 | 不同 R 对应不同尺度的核，多尺度聚合保证全局+局部结构同时捕获 |
| stop_gradient | `jax.lax.stop_gradient(goal)` | 保证漂移方向的计算不依赖当前 gen 的梯度，类似 EM 算法的 M-step |

**期望的实际计算方式**：
- 公式中的 \(\mathbb{E}_{y^+\sim p,\,y^-\sim q}\) 对应 **双重循环的有限样本求和**
- 代码中没有显式写两个嵌套循环，而是通过 `einsum` 一次完成所有正负样本对的加权求和
- 每个 batch 中实际参与计算的样本数为 \(C_g \times (C_n + C_p)\) 对（gen 与 target 的所有配对）

### 3.6 完整的数学推导

为更清晰地理解代码与公式的关系，以下给出从理论到代码的完整推导：

**Step 1：理论连续形式**

\[
V_{p,q}(x) = \frac{1}{Z_p Z_q}\int\int k(x,y^+)k(x,y^-)(y^+-y^-)dP(y^+)dQ(y^-)
\]

**Step 2：离散化为有限样本**

用 Memory Bank 中的样本 \(\{y^+_i\}_{i=1}^{N^+}\) 和 \(\{y^-_j\}_{j=1}^{N^-}\) 近似：

\[
\hat V(x) = \frac{1}{Z_p Z_q}\cdot\frac{1}{N^+N^-}\sum_{i=1}^{N^+}\sum_{j=1}^{N^-}k(x,y^+_i)k(x,y^-_j)(y^+_i-y^-_j)
\]

**Step 3：引入亲和力矩阵**

令 `aff_neg[i,j] = k(x_i, y^-_j)`，`aff_pos[i,j] = k(x_i, y^+_j)`，则漂移力可写为矩阵形式：

```
F = aff_neg @ y_neg^T - aff_pos @ y_pos^T    # 第一项：负样本产生的斥力
  + aff_pos @ y_pos^T - aff_neg @ y_neg^T    # 第二项：正样本产生的引力（合并后简化为一项）
```

**Step 4：矩阵形式（代码中的 einsum）**

```python
# 斥力（推离负样本）
r_coeff_neg = -aff_neg * sum_pos
# 引力（拉向正样本）
r_coeff_pos = aff_pos * sum_neg
R_coeff = concat([r_coeff_neg, r_coeff_pos])
# 矩阵乘法：求加权和
total_force = einsum("biy,byx->bix", R_coeff, targets)
# 减去自项
total_force = total_force - sum(R_coeff) * old_gen
```

**Step 5：stop_gradient 截断梯度**

漂移目标定义为：
```
goal = old_gen + normalize(force)
```
通过 `jax.lax.stop_gradient`，确保梯度只沿 MSE 反传，而不经过漂移力场的计算路径。

### 3.7 可复用性分析

**`cdist` 函数**（计算批量欧氏距离）是通用组件，不依赖任何生成模型，可直接移植到其他需要样本级对比学习的任务中。

**动态归一化 + 多温度 RBF 亲和力**的组合是一个通用范式，可用于：
- 自监督学习的特征对齐
- 分布匹配（无需对抗训练）
- 图像/视频生成中的结构保持

---

## 4. 模型结构

### 4.1 DitGen 生成器（`models/generator.py`）

#### 整体架构

`DitGen` 是一个**条件生成模型**，接收类别标签 `c` 和 CFG（Classifier-Free Guidance）尺度，输出对应的生成样本/潜在表示。

```
类别嵌入 ──→ AdaLN Conditioning
                ↓
随机噪声 x ──→ Patchify → Linear Projection → + SinCos位置编码
                ↓
            N × LightningDiTBlock（核心变换）
                ↓
            FinalLayer → Unpatchify → 输出样本
```

#### 关键技术细节

**SinCos 位置编码初始化**：使用 2D 正弦-余弦位置编码作为可学习位置嵌入的初始化值，确保模型在训练初期就能获得良好的空间结构感知能力。

**AdaLN 调制**：与标准 DiT 不同，这里使用 `SiLU(cond) → Linear(6*hidden_dim)` 生成 shift/scale/gate 三个调制向量。调制在 LayerNorm 之后应用，公式为 `x * (1 + scale) + shift`，其中 shift/scale/gate 计算在 fp32 下进行以保证精度。

**LightningDiTBlock 核心结构**：
```
输入 x, 条件 c
    ↓ norm(x)
    ↓ AdaLN调制（shift_msa, scale_msa, gate_msa）
    ↓ Self-Attention (QKV → FP32计算 → softmax → O)
    ↓ 残差连接 + gate_msa * attn_out
    ↓ norm(x)
    ↓ AdaLN调制（shift_mlp, scale_mlp, gate_mlp）
    ↓ MLP（SwiGLU 或 标准GELU）
    ↓ 残差连接 + gate_mlp * mlp_out
```

**可选技术组合**（通过配置开关）：
- `use_qknorm`: QK LayerNorm/RMSNorm（稳定大分辨率注意力）
- `use_rmsnorm`: 用 RMSNorm 替代 LayerNorm（更高效的归一化）
- `use_rope`: RoPE（旋转位置编码，增强相对位置建模）
- `use_swiglu`: SwiGLU FFN（比标准 FFN 表达能力更强）
- `use_remat`: 梯度检查点（以计算换内存）

#### 关键参数配置

以 SOTA Latent-L 模型为例：
| 参数 | 值 | 说明 |
|------|-----|------|
| `cond_dim` | 768 | 条件嵌入维度 |
| `hidden_size` | 1152 | Transformer隐藏维度 |
| `depth` | 28 | Transformer层数 |
| `num_heads` | 16 | 注意力头数 |
| `patch_size` | 2 | 潜在空间16×16→8×8 |
| `input_size` | 32 | 潜在空间边长（256/8） |
| `in_channels` | 4 | 潜在空间通道数 |
| `noise_classes` | 噪声类别数 | 用于噪声条件化 |
| `n_cls_tokens` | 类别token数 | 可选的并行类别条件 |

#### 可复用性分析

`LightningDiTBlock` 是一个完全独立的通用 DiT 块，可直接用于：
- 图像/视频生成（如 MDT、SiT、DiT 等项目）
- 掩码图像建模（MIM）的解码器
- 任意需要类别条件的生成任务

`TimestepEmbedder`（频率位置编码）可以独立使用：
```python
freqs = exp(-log(10000) * arange(0, half) / half)  # 频率
emb = concat([cos(args), sin(args)])                # 正弦余弦拼接
mlp = MLP(hidden, silu, MLP(hidden))                 # 两层MLP
```

`modulate()` 函数（`x * (1 + scale) + shift`）是 AdaLN 的核心公式，已广泛在 Stable Diffusion、DiT 等模型中采用。

---

### 4.2 MAE 特征提取器（`models/mae_model.py`）

#### 结构：MAE-ResNet（JAX 实现）

这是一个 **MAE 风格的 ResNet**，同时进行图像重建和分类预训练，输出多尺度特征用于 drift loss。

```
输入图像 → Patchify → 掩码（随机丢弃75%补丁）
    ↓
ResNet 编码器（4阶段，GroupNorm）
    ↓ 提取特征: conv1, layer1, layer2, layer3, layer4
    ↓ 平均池化 → 分类头
    ↓ UNet 解码器（4级上采样）→ 重建图像
```

#### 特征提取 API

```python
feats = model.get_activations(
    x,                           # 输入 [B, H, W, C]
    patch_mean_size=[2, 4],      # 不同空间尺度的均值池化
    patch_std_size=[2, 4],       # 不同空间尺度的标准差池化
    use_std=True,                # 是否使用std特征
    use_mean=True,               # 是否使用mean特征
    every_k_block=2,              # 每隔k个block提取中间特征
)
# 输出: dict，key如 'layer1', 'layer1_mean', 'layer1_std', 
#      'layer1_mean_2', 'layer1_std_4', 'layer1_blk2', ...
```

#### 可复用性分析

`MAEResNetJAX` 是一个**自包含的特征提取器**，不依赖外部库（TIMM 等），完整代码在单个文件中：
- `_BasicBlock`: 带 GroupNorm 和投影的 ResNet 基本块
- `_ResNetEncoder`: 4阶段 ResNet 编码器，支持多尺度特征输出
- `_UNetDecoder`: 对称上采样解码器（用于 MAE 重建）
- `patch_input()`: einops 实现的高效图像分块（`BHWC → BT(C*P*P)`）
- `safe_std()`: fp32 安全标准差计算（避免 bf16 精度问题）

`get_activations()` 方法是一个**通用多尺度特征提取器**，其输出格式（`{name: (B, T, D)}`）可无缝对接 drift_loss 或其他对比学习/特征匹配损失。

---

### 4.3 ConvNeXt V2 特征提取器（`models/convnext.py`）

可选的替代特征提取器，基于 ConvNeXt V2 架构：

```
输入 → 4×下采样（Conv 4×4 stride=4 + LN）
    ↓
4阶段 ConvNeXtV2Block（深度可分离卷积 + GRN）
    ↓
多尺度归一化特征输出
```

`get_activations()` 返回：`{convenxt_stage_{i}, convenxt_stage_{i}_mean, convenxt_stage_{i}_std}`

**权重转换工具**：`convert_weights_to_jax()` 实现了 PyTorch → JAX 的参数格式转换（处理 permute、路径重命名等），可作为跨框架权重迁移的参考模板。

---

## 5. 数据处理

### 5.1 数据模式（`dataset/dataset.py`）

系统支持三种数据模式，通过 YAML 配置切换：

**模式 1：原生像素（Pixel Mode）**
```
use_latent: false
use_cache: false
```
- 使用 `torchvision.datasets.ImageFolder`
- 图像经过中心裁剪和归一化 `[-1, 1]`
- 模型直接处理 RGB 像素

**模式 2：在线 VAE 编码（Latent Mode）**
```
use_latent: true
use_cache: false
```
- 原始图像 → VAE 编码 → 潜在向量
- 潜在向量经过归一化送入生成器
- 输出后通过 VAE 解码回像素空间

**模式 3：预计算潜在缓存（Cache Mode）**
```
use_latent: false
use_cache: true
```
- 预先生成 `.pt` 缓存文件到 `IMAGENET_CACHE_PATH/{train,val}/`
- 每个文件包含 `moments` 和 `moments_flip`（随机选择）
- 大幅加速训练 IO

### 5.2 潜在空间缓存构建（`dataset/latent.py`）

使用 SD-VAE（`pcuenq/sd-vae-ft-mse-flax`）对 ImageNet 进行编码：
```python
dist = vae.encode(images).latent_dist
latents = dist.sample() * 0.18215  # 潜在向量（4通道，H/8×W/8）
```

缓存构建脚本支持：
- **多进程保存**：异步写文件避免 IO 瓶颈
- **水平翻转增强**：同时保存翻转前后的 latent
- **分布式并行**：利用 TPU 多设备并行编码

### 5.3 关键工具函数

**中心裁剪（ADM 风格）**：
```python
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    # 然后缩放到目标尺寸
```

**DataLoader worker 初始化**：
```python
def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
```

**无限采样器**（支持断点续训练）：
```python
def infinite_sampler(it, start_step=0):
    epoch_idx = start_step // len(it)
    skip_batches = start_step % len(it)
    while True:
        for i, batch in enumerate(it):
            if skip_batches > 0 and i < skip_batches: continue
            yield batch
        skip_batches = 0
        epoch_idx += 1
```

### 5.4 数据增强

训练时支持两种增强策略：
- **强增强**（`use_aug=True`）：`RandomResizedCrop(256, scale=(0.2, 1.0))` + `RandomHorizontalFlip`
- **标准增强**（默认）：中心裁剪 + `RandomHorizontalFlip`

---

## 6. 训练流程（`train.py`）

### 6.1 整体训练循环

```
for step in range(total_steps):
    ① Memory Bank 填充（每步 push_per_step 个样本）
    ② 从 DataLoader 采样当前批次
    ③ Memory Bank 按类别采样（positive + negative）
    ④ 多设备数据合并（merge_data）
    ⑤ drift_loss 计算 + 梯度更新
    ⑥ EMA 参数更新
    ⑦ 定期 FID 评估 + 检查点保存
```

### 6.2 Memory Bank（`memory_bank.py`）

**类别级环形缓冲区**：每个类别维护一个固定大小的缓冲区，新样本覆盖旧样本。

```python
class ArrayMemoryBank:
    def __init__(num_classes=1000, max_size=64):
        self.bank = zeros((num_classes, max_size, *feature_shape))
        self.ptr  = zeros(num_classes, dtype=int32)   # 写指针
        self.count = zeros(num_classes, dtype=int32)   # 有效样本计数

    def add(samples, labels):      # O(1) 插入
    def sample(labels, n_samples): # 按类别随机采样
```

正样本 Bank 和负样本 Bank 分开管理：
- 正样本 Bank：类别数 = 1000，每类最多 64-128 个样本
- 负样本 Bank：类别数 = 1（无条件），包含 512-1000 个样本

### 6.3 CFG（Classifier-Free Guidance）策略

训练时动态采样 CFG 尺度：
```python
# 功率律采样（而非均匀采样）
frac ~ Uniform(0, 1)
cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1/pw)
# 其中 pw = 1 - neg_cfg_pw（默认为0，即均匀分布）
cfg = where(random_uniform < no_cfg_frac, 1.0, cfg)  # 随机drop到CFG=1
```

这确保了模型在训练时同时接触到高 CFG 和无条件样本，避免对 CFG 尺度的过度依赖。

### 6.4 EMA（指数移动平均）

```python
new_ema = ema * 0.999 + params * 0.001
```

EMA 参数在 FID 评估时使用，是最终模型的核心。

### 6.5 FID 评估流程（`utils/fid_util.py`）

```
生成样本（N=50000）
    ↓
Inception V3 特征提取（pmap 并行）
    ↓
计算: FID = ||μ_gen - μ_ref||² + Tr(Σ_gen + Σ_ref - 2√(Σ_gen·Σ_ref))
      IS = exp(mean(KL(p(y|x) || p(y))))
      Precision/Recall（基于特征空间 k-NN）
    ↓
日志记录 + 可视化预览图
```

---

## 7. 分布式训练（HSDP）

### 7.1 分片策略（`utils/hsdp_util.py`）

HSDP（Hybrid Sharding Data Parallel）将设备划分为 `[data, fsdp]` 两个维度：

```python
def set_global_mesh(hsdp_dim=8):
    mesh_shape = (total_devices // hsdp_dim, hsdp_dim)
    # data维度: 数据并行（每个副本有完整模型）
    # fsdp维度: 模型分片（权重沿此维度切分）
```

### 7.2 核心操作

| 函数 | 作用 |
|------|------|
| `set_global_mesh()` | 初始化全局设备网格 |
| `enforce_ddp()` | 确保张量在 DDP 分片上对齐 |
| `merge_data()` | 将本地数据合并到全局分片 |
| `pad_and_merge()` | 填充 + 合并（处理最后一批） |
| `init_state_from_dummy_input()` | 初始化分片状态，直接分配到设备 |
| `init_model_distributed()` | 分布式初始化模型参数 |

### 7.3 分布式 RNG

```python
def prepare_rngs(rng, keys):
    return dict(zip(keys, jax.random.split(rng, len(keys))))

def ddp_rand_func(rand_type="normal", shard="ddp"):
    return jax.jit(jax.random.normal, out_shardings=ddp_shard())
```

---

## 8. 可复用组件清单

以下是按可复用性从高到低排列的核心组件：

### Tier 1：完全独立的通用模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `cdist()` | `drift_loss.py` | 批量欧氏距离计算（JAX 实现） |
| `drift_loss()` | `drift_loss.py` | 多温度 RBF 亲和力损失（通用分布匹配范式） |
| `safe_std()` | `models/mae_model.py` | fp32 安全标准差 |
| `patch_input()` / `make_patch_mask()` | `models/mae_model.py` | 图像分块与掩码生成 |
| `center_crop_arr()` | `dataset/dataset.py` | ADM 风格中心裁剪 |
| `infinite_sampler()` | `dataset/dataset.py` | 支持断点续训的无限数据迭代器 |
| `profile_func()` | `utils/misc.py` | JAX 函数性能分析（FLOPs/带宽/内存） |
| `EasyDict` | `utils/misc.py` | dot-accessible 字典（广泛适用） |
| `WandbLogger` | `utils/logging.py` | 支持 WandB 和离线回退的日志器 |

### Tier 2：需要适配的模型组件

| 模块 | 文件 | 适配方向 |
|------|------|----------|
| `LightningDiTBlock` | `models/generator.py` | 替换为你的 DiT/生成模型的主干 |
| `DitGen` | `models/generator.py` | 替换类别嵌入和噪声条件化机制 |
| `TimestepEmbedder` | `models/generator.py` | 用于时间/条件嵌入 |
| `MAEResNetJAX` | `models/mae_model.py` | 替换为你的特征提取器 |
| `ConvNeXtV2` | `models/convnext.py` | 可选的更强特征提取器 |
| `ArrayMemoryBank` | `memory_bank.py` | 替换为你的样本队列管理策略 |

### Tier 3：框架级集成组件

| 模块 | 文件 | 说明 |
|------|------|------|
| HSDP 分片系统 | `utils/hsdp_util.py` | 完整的 JAX 多设备分布式训练框架 |
| FID 评估流水线 | `utils/fid_util.py` | 基于 Inception V3 的多指标评估 |
| VAE 编码/解码 | `dataset/vae.py` | SD-VAE JAX 实现，支持在线和缓存模式 |
| HF 模型加载 | `models/hf.py` | HuggingFace 分发格式支持 |
| 检查点管理 | `utils/ckpt_util.py` / `init_util.py` | Orbax + HF 混合检查点系统 |

---

## 9. 配置系统

### 9.1 生成器配置示例（`configs/gen/latent_ablation.yaml`）

```yaml
dataset:
  resolution: 256
  use_latent: true       # 使用潜在空间
  use_cache: true        # 使用预计算缓存
  batch_size: 1024

model:                    # DitGen 配置
  cond_dim: 768
  hidden_size: 768
  depth: 12
  num_heads: 12
  patch_size: 2
  use_qknorm: true        # QK归一化
  use_swiglu: true        # SwiGLU FFN
  use_rope: true          # RoPE
  use_rmsnorm: true       # RMSNorm
  noise_classes: 64      # 噪声类别数
  n_cls_tokens: 16       # 类别token数

train:
  pos_per_sample: 64     # 正样本数/类
  neg_per_sample: 16     # 负样本数
  positive_bank_size: 128
  negative_bank_size: 1000
  gen_per_label: 64      # 每标签生成数
  cfg_min: 1.0
  cfg_max: 4.0
  loss_kwargs:
    R_list: [0.2, 0.05, 0.02]  # 多温度RBF尺度

feature:
  mae_path: "hf://mae_latent_256"
  activation_kwargs:
    patch_mean_size: [2, 4]
    patch_std_size: [2, 4]
```

### 9.2 MAE 配置示例（`configs/mae/latent_ablation_256.yaml`）

```yaml
model:
  base_channels: 256
  patch_size: 2
  layers: [3, 4, 6, 3]    # ResNet各阶段block数

optimizer:
  lr_schedule:
    learning_rate: 0.004
    warmup_steps: 4000
    lr_schedule: "const"
```

---

## 10. 使用建议

### 如果你要在自己的生成模型中使用 Drift Loss：

1. **替换特征提取器**：将 `activation_fn` 替换为你自己的 CLIP/ViT/DINOv2 等特征提取器
2. **调整 R_list**：R 值决定了亲和力核的尺度，`[0.02, 0.05, 0.2]` 对应从精细到粗糙的结构匹配
3. **调整正负样本数**：`pos_per_sample` 和 `neg_per_sample` 控制对比的强度和多样性
4. **修改尺度归一化**：如果特征分布差异大，调整 `scale` 的计算方式

### 如果你要在自己的模型中使用 LightningDiTBlock：

1. **保持 BHWC 格式**：该模块内部假设输入为 `[B, H, W, C]`
2. **配置 AdaLN**：确保 `cond_dim` 与条件嵌入维度匹配
3. **精度选择**：`attn_fp32=True` 对大多数任务足够；`use_bf16` 可加速但需注意精度
4. **显存优化**：`use_remat=True` 以约 30% 的重计算换 50% 的显存节省

### 如果你要在自己的训练流程中使用 HSDP：

1. **确保 TPU/JAX 环境**：设置 `JAX_PLATFORMS=tpu,cpu`
2. **初始化网格**：`set_global_mesh(hsdp_dim)` 必须在任何 JAX 计算之前调用
3. **保持分片一致**：所有参与训练的设备必须执行相同的 `set_global_mesh` 调用
4. **处理最后一批**：`pad_and_merge()` 是处理变长批次的标准方案
