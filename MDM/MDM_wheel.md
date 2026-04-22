# MDM 工具链详解 — 可复用轮子与模型结构

本文档面向希望在 HumanML3D 上验证与扩散模型（Diffusion）相似的新型生成范式的研究者。文档分两部分：
1. **核心工具链脚本详解**：按数据流顺序逐一描述每个可复用模块的功能、输入输出及对接方式
2. **MDM 模型结构详解**：完整的网络架构、维度变换与关键设计决策

---

## 目录

1. [数据流总览](#1-数据流总览)
2. [工具链一：数据加载与预处理](#2-工具链一数据加载与预处理)
3. [工具链二：扩散过程（可替换的核心生成范式）](#3-工具链二扩散过程可替换的核心生成范式)
4. [工具链三：评估模块（Evaluation Pipeline）](#4-工具链三评估模块evaluation-pipeline)
5. [工具链四：后处理与坐标转换](#5-工具链四后处理与坐标转换)
6. [MDM 模型结构详解](#6-mdm-模型结构详解)
7. [如何替换扩散模型为新范式](#7-如何替换扩散模型为新范式)

---

## 1. 数据流总览

```
文本输入 ──→ [CLIP/BERT 编码] ──→ 文本嵌入向量
                                    │
运动数据 ──→ [HumanML3D Dataset] ──→ 标准化运动向量 [J×1, 帧数]
         │                          │
         └──→ [高斯扩散前向过程]      │
                                  ↓
                          ┌───────────────┐
                          │   MDM 模型     │  ← 核心去噪网络（可替换）
                          │  (Transformer) │
                          └───────┬───────┘
                                  ↓
              [逆标准化] → [旋转→XYZ 坐标] → SMPL 渲染
                                  │
                    ┌─────────────┼─────────────┐
                    ↓             ↓             ↓
              FID / R-precision  可视化MP4     SMPL网格
```

---

## 2. 工具链一：数据加载与预处理

### 2.1 数据入口：`data_loaders/get_data.py`

**功能**：统一的数据加载器工厂，提供各数据集的统一访问接口。

**关键函数**：

```python
get_dataset_loader(name, batch_size, num_frames, split='train',
                  hml_mode='train', fixed_len=0, pred_len=0,
                  device=None, autoregressive=False)
```

| 参数 | 说明 |
|------|------|
| `name` | 数据集名称：`humanml`、`kit`、`humanact12`、`uestc` |
| `batch_size` | 批大小 |
| `num_frames` | 最大帧数限制 |
| `split` | `train` / `test` / `val` |
| `hml_mode` | `train`（标准训练）、`text_only`（仅文本，推理用）、`eval`（评估）、`gt`（Ground Truth） |
| `fixed_len` | 固定长度（用于 DiP 自回归） |
| `pred_len` | 前缀补全的预测长度 |
| `autoregressive` | 是否启用自回归模式 |

**返回**：`torch.utils.data.DataLoader`，可直接在训练/推理循环中使用。

**内部数据流**：
```
get_dataset_loader
  └── get_dataset(name, ...)
        └── get_dataset_class(name)
              └── HumanML3D / KIT / HumanAct12Poses / UESTC
```

### 2.2 核心数据集类：`data_loaders/humanml/data/dataset.py`

#### `HumanML3D` 类（行 753-818）

这是 MDM 中最重要的数据集封装类，**几乎所有新范式验证都需要使用它**。

**初始化流程**：

```python
class HumanML3D(Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt',
                 split="train", **kwargs):
        # 1. 加载数据集配置
        opt = get_opt(dataset_opt_path, device)
        
        # 2. 加载标准化参数（均值/方差）
        self.mean = np.load('.../Mean.npy')   # shape: (263,)
        self.std  = np.load('.../Std.npy')    # shape: (263,)
        
        # 3. 加载词向量工具
        self.w_vectorizer = WordVectorizer(...)
        
        # 4. 创建底层数据集
        self.t2m_dataset = Text2MotionDatasetV2(...)
```

**关键属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `mean` / `std` | np.array | 归一化参数，shape `(263,)`（HumanML3D）或 `(251,)`（KIT） |
| `opt` | Namespace | 数据集配置，包含 `max_motion_length=196`、`joints_num=22` 等 |
| `t2m_dataset` | Text2MotionDatasetV2 | 底层运动+文本数据 |

**HumanML3D 运动数据的向量表示**（`hml_vec` 格式，shape `(帧数, 263)`）：

| 维度范围 | 内容 | 归一化因子 |
|----------|------|------------|
| `[0:1]` | 根旋转速度 `(B, seq, 1)` | `/ feat_bias` |
| `[1:3]` | 根线速度 `(B, seq, 2)` | `/ feat_bias` |
| `[3:4]` | 根 Y 轴位置 `(B, seq, 1)` | `/ feat_bias` |
| `[4:67]` | 相对位置 `(B, seq, 63)` — 21 个关节×3 坐标 | `/ 1.0` |
| `[67:193]` | 旋转数据 `(B, seq, 126)` — 21 个关节×6D 旋转 | `/ 1.0` |
| `[193:259]` | 局部速度 `(B, seq, 66)` — 22 个关节×3 坐标 | `/ 1.0` |
| `[259:263]` | 足部接触标志 `(B, seq, 4)` | `/ feat_bias` |

> **关键接口 — `inv_transform`**：将标准化后的数据还原为原始尺度。**新范式推理时必须调用**：
> ```python
> motion_original = motion * self.std + self.mean  # shape: (batch, 263, frames)
> ```

#### `Text2MotionDatasetV2` 类（行 207-377）

**数据读取**（行 239-294）：
- 从 `split_file`（`train.txt` / `test.txt`）读取样本 ID
- 每个 ID 对应一个 `.npy` 运动文件 + 一个 `.txt` 文本文件
- 文本文件格式：`caption # tokens # f_tag # to_tag`（每运动可有多个文本描述）

**文本处理**（行 324-341）：
```python
# tokens 格式: "walk/VERB forward/ADP ..."
# 经过 WordVectorizer 转为：
word_embeddings:  (max_text_len+2, 300)  # GloVe 词向量
pos_one_hots:    (max_text_len+2, pos_vocab_size)  # 词性 one-hot
```

**运动长度处理**（行 344-377）：
- 随机选择 `single`（原长）或 `double`（更长）裁剪策略
- 支持 `fixed_len` 模式（DiP 前缀补全）

**关键方法**：

| 方法 | 说明 |
|------|------|
| `inv_transform(data)` | 逆标准化：`data * std + mean` |
| `reset_max_len(length)` | 重置最大运动长度，更新二分查找指针 |

#### `TextOnlyDataset` 类（行 673-749）

用于**仅文本推理**（不加载运动数据），生成时使用：

```python
# __getitem__ 返回：
(None, None, caption, None, np.array([0]), fixed_length, None)
# 用于生成分批时的占位
```

### 2.3 批处理函数：`data_loaders/tensors.py`

**`t2m_collate`**（行 67-79）：将 `Text2MotionDatasetV2` 的元组输出转为 MDM 内部格式。

```python
# 输入 batch 元素: (word_embs, pos_ohot, caption, sent_len, motion, length, tokens_key)
# 输出:
{
    'inp':     torch.tensor(motion.T).float().unsqueeze(1),  # [J, 1, T] — MDM 输入格式
    'text':    caption_str,
    'tokens':  tokens_key,
    'lengths': length_int,
    'key':     db_key_str,
}
```

**`lengths_to_mask`**（行 3-6）：将长度数组转为布尔掩码：

```python
lengths = [40, 60, 55]  # 每条运动的有效帧数
max_len = 196
# → mask shape: (3, 196), True=有效, False=填充
```

**`collate`**（行 22-64）：将 `dict` 列表合并为批次，输出格式：

```python
# 返回:
motion:  (batch, J, 1, max_frames)
cond: {
    'y': {
        'mask':    (batch, 1, 1, max_frames) — True=有效帧
        'lengths': (batch,)
        'text':    [str, str, ...] — 文本列表
        'tokens':  [str, str, ...]
    }
}
```

> **对于新范式**：只需将新模型的输出转为 `motion: (batch, J, 1, T)` 格式，即可复用下游所有工具。

### 2.4 骨骼与几何工具：`data_loaders/humanml/utils/paramUtil.py`

**骨骼定义**（可直接用于新模型的骨架可视化）：

```python
# HumanML3D 骨骼链 (t2m_kinematic_chain)
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],      # 右腿
    [0, 1, 4, 7, 10],      # 左腿
    [0, 3, 6, 9, 12, 15],  # 脊柱+头
    [9, 14, 17, 19, 21],   # 左臂
    [9, 13, 16, 18, 20]    # 右臂
]
# 22 个关节，对应 SMPL 的 22 个关节点
```

**KIT 骨骼链**：`kit_kinematic_chain`，21 个关节。

### 2.5 运动特征提取：`data_loaders/humanml/scripts/motion_process.py`

**`recover_from_ric`**（行 ~400）：将 HumanML3D 向量表示还原为 XYZ 关节坐标。

```python
def recover_from_ric(data, joints_num):
    # data: (batch, 263, time) 或 (batch, 251, time)
    # 输出: (batch, joints_num, 3, time) — XYZ 坐标
```

**`get_target_location`**：在 CLoSD/DiP 中用于提取目标关节位置条件。

---

## 3. 工具链二：扩散过程（可替换的核心生成范式）

### 3.1 扩散核心：`diffusion/gaussian_diffusion.py`

这是 MDM 中**最核心、最值得复用**的模块，包含完整的 DDPM 训练与采样逻辑。

#### 扩散参数初始化（行 122-206）

```python
def __init__(self, betas, model_mean_type, model_var_type, loss_type,
             lambda_rcxyz=0., lambda_vel=0., lambda_fc=0., ...):
    # betas: 噪声调度 (T,)
    # 模型预测目标：EPSILON（预测噪声）或 START_X（预测 x0）
```

**噪声调度**（`diffusion/gaussian_diffusion.py` 行 22-46）：

```python
get_named_beta_schedule('cosine', num_steps)
# 或
get_named_beta_schedule('linear', num_steps)
```

**Cosine 调度**：
\[
\beta_t = 1 - \frac{\cos\left(\frac{t+0.008}{1.008} \cdot \frac{\pi}{2}\right)^2}
                 {\cos\left(\frac{0}{1.008} \cdot \frac{\pi}{2}\right)^2}
\]

#### 前向扩散过程（行 226-244）

```python
def q_sample(self, x_start, t, noise=None):
    # 给定 x_0 和时间步 t，从 q(x_t | x_0) 采样
    # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    return (sqrt_alphas_cumprod[t] * x_start
           + sqrt_one_minus_alphas_cumprod[t] * noise)
```

#### 训练损失（行 1224-1359）

```python
def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
    # 1. 前向加噪
    x_t = self.q_sample(x_start, t, noise)
    
    # 2. 模型预测
    model_output = model(x_t, scaled_timesteps, **model_kwargs)
    
    # 3. 计算损失（取决于 model_mean_type）
    if model_mean_type == EPSILON:
        target = noise                    # 预测噪声
    elif model_mean_type == START_X:
        target = x_start                 # 预测 x0
    
    loss = MSE(model_output, target)    # + 几何辅助损失
```

**几何辅助损失**（MDM 特有，可选择性使用）：

| 损失项 | 参数 | 说明 |
|--------|------|------|
| 旋转 MSE | `lambda_rcxyz` | 在 XYZ 空间的关节位置损失 |
| 速度损失 | `lambda_vel` | 运动速度的 L2 损失 |
| 足部接触损失 | `lambda_fc` | 脚触地时速度为零 |
| 目标位置损失 | `lambda_target_loc` | CLoSD/DiP 目标条件 |

#### 反向采样过程（行 489-658）

**`p_sample`**（单步采样，行 489-541）：

```python
def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
    # 1. 模型预测 p(x_{t-1} | x_t)
    pred = model(x, t, **model_kwargs)  # 输出 x0 或 ε
    
    # 2. 计算后验均值和方差
    posterior_mean, posterior_variance, _ = self.q_posterior_mean_variance(x_start=pred, x_t=x, t=t)
    
    # 3. 添加噪声（t>0 时）
    sample = mean + sqrt(var) * noise
    return sample
```

**`p_sample_loop`**（完整采样循环，行 591-658）：从纯噪声 `x_T` 开始，逐步降噪到 `x_0`。

> **替换策略**：如果要将扩散模型替换为新范式（如 Flow Matching、Score Matching、GAN 等），只需实现一个与 `p_sample_loop` 相同签名的函数，从 `torch.randn(*shape)` 开始，通过新模型的单步 `forward` 完成去噪/采样过程。

#### DDIM 采样（行 729-990）

提供了 DDIM（Denoising Diffusion Implicit Models）和 PLMS（Pseudo Linear Multistep Sampler）加速采样实现。**如果要实现与 DDIM 等价的加速策略**，参考行 729-836。

### 3.2 时间步采样策略：`diffusion/respace.py`

**`SpacedDiffusion`**（行 65-115）：支持减少扩散步数的策略（如从 1000 步降至 50 步）。

```python
# 核心思想：从 T 个时间步中稀疏选择 S 个（等间距或 DDIM 间距）
# _WrappedModel 负责将缩放后的时间步映射回原始时间步
```

### 3.3 采样器工具：`utils/sampler_util.py`

#### `ClassifierFreeSampleModel`（行 10-38）

**无需训练的推理包装器**，实现 CFG（Classifier-Free Guidance）：

```python
class ClassifierFreeSampleModel(nn.Module):
    def forward(self, x, timesteps, y=None):
        # 有条件预测
        out     = self.model(x, timesteps, y)
        # 无条件预测（强制掩码条件）
        out_un  = self.model(x, timesteps, {**y, 'uncond': True})
        # CFG 组合
        return out_un + y['scale'] * (out - out_un)
```

> **CFG 公式**：
> \[
> \tilde{\epsilon} = \epsilon_\emptyset + s \cdot (\epsilon_\text{cond} - \epsilon_\emptyset)
> \]
> 其中 `scale = s`（默认 2.5），`s` 越大越符合文本，`s=1` 时退化为无条件。

**关键**：模型训练时需设置 `--cond_mask_prob 0.1`（默认），使模型同时学习有条件和无条件预测。

#### `AutoRegressiveSampler`（行 41-80）

**DiP 专用自回归采样器**，将长序列生成拆分为多个短片段：

```python
# 每次迭代生成 pred_len 帧（默认 40 帧 = 2 秒）
# 用上一段的后 context_len 帧（默认 20 帧）作为下一段的前缀
for i in range(n_iterations):
    prefix = full_sequence[-context_len:]      # 取前一次的输出尾部
    sample = diffusion.sample(model, prefix)   # 生成下一段
    full_sequence = concat(full_sequence, sample)
```

---

## 4. 工具链三：评估模块（Evaluation Pipeline）

### 4.1 评估入口：`eval/eval_humanml.py`

这是 HumanML3D 上进行完整评估的脚本，包含四个核心指标的计算。

#### 评估指标详解

**1. Matching Score + R-Precision**（`evaluate_matching_score`，行 20-70）

文本-动作联合嵌入空间的匹配质量：

```python
# 1. 用预训练评估器提取文本嵌入和运动嵌入
text_embs, motion_embs = evaluator.get_co_embeddings(word_embs, pos_ohot, ...)

# 2. 计算欧氏距离矩阵
dist_mat = euclidean_distance_matrix(text_embs, motion_embs)

# 3. Matching Score = trace(dist_mat) / N
matching_score = dist_mat.trace() / N

# 4. R-Precision@K = top-K 近邻中匹配正确的比例
argmax = argsort(dist_mat, axis=1)
# top-1/2/3 的准确率
```

**2. FID（Fréchet Inception Distance）**（`evaluate_fid`，行 73-96）

生成动作分布与真实动作分布的差异：

```python
# 1. 计算真实数据统计量
gt_embeddings: (N, D)
mu_gt, cov_gt = mean(gt_embeddings), cov(gt_embeddings)

# 2. 计算生成数据统计量
mu_gen, cov_gen = mean(gen_embeddings), cov(gen_embeddings)

# 3. FID 公式
FID = ||mu_gt - mu_gen||^2 + Tr(cov_gt + cov_gen - 2*sqrt(cov_gt * cov_gen))
```

**3. Diversity**（`evaluate_diversity`，行 99-107）

生成动作的多样性：

```python
# 随机采样 diversity_times 对，计算嵌入空间欧氏距离均值
diversity = mean(||emb[i] - emb[j]|| for i,j in random_pairs)
```

**4. Multimodality**（`evaluate_multimodality`，行 110-129）

同一文本多次生成的多样性：

```python
# 同一文本的多次生成，计算嵌入空间距离均值
mm = mean(||same_text_gen[i] - same_text_gen[j]||)
```

#### 评估加载器：`eval/humanml/motion_loaders/model_motion_loaders.py`

`get_mdm_loader` 函数：生成用于评估的运动数据批次，调用 `diffusion.p_sample_loop` 并转换为评估器所需格式。

### 4.2 评估器封装：`data_loaders/humanml/networks/evaluator_wrapper.py`

**`EvaluatorMDMWrapper`**（行 121-187）：封装了预训练的文本-运动匹配模型（来自 Text2Motion 论文）。

**嵌入提取流程**：

```
原始运动 (batch, 263, T)
       │
       ├── 1. 去除最后 4 维（足部接触标志）
       │    → (batch, 259, T)
       │
       ├── 2. MovementConvEncoder（1D 卷积，捕获局部运动特征）
       │    → (batch, dim_movement_latent, T')
       │
       ├── 3. MotionEncoderBiGRUCo（双向 GRU 聚合时序）
       │    → (batch, dim_coemb=512)
       │
       └── 文本分支（使用 GloVe 词向量 + BiGRU）
            → (batch, dim_coemb=512)

# 文本和运动嵌入在同一 512 维空间，可计算余弦/Euclid 距离
```

> **新范式验证关键**：生成的运动只要转为 `(batch, 263, T)` 格式，即可直接传入 `get_mdm_loader` → `evaluator.get_co_embeddings()` → `evaluate_fid()` 等函数完成全套评估。

### 4.3 评估指标计算工具：`data_loaders/humanml/utils/metrics.py`

| 函数 | 用途 |
|------|------|
| `euclidean_distance_matrix` | 批量欧氏距离矩阵（用于 R-precision） |
| `calculate_top_k` | Top-K 命中统计 |
| `calculate_R_precision` | R-precision@K 计算 |
| `calculate_matching_score` | 匹配分数 = mean(||text_emb - motion_emb||) |
| `calculate_activation_statistics` | 计算均值和协方差（用于 FID） |
| `calculate_frechet_distance` | FID 核心公式 |
| `calculate_diversity` | 多样性度量 |
| `calculate_multimodality` | 多模态性度量 |

### 4.4 动作转运动评估：`eval/eval_humanact12_uestc.py`

对于 Action-to-Motion 任务（不使用文本条件），使用 ST-GCN 或 GRU 作为评估网络：

```python
# HumanAct12 → GRU 评估器
# UESTC → ST-GCN 评估器
# 评估指标：FID + Diversity（通过 `action2motion` 和 `stgcn` 子模块）
```

---

## 5. 工具链四：后处理与坐标转换

### 5.1 旋转→XYZ 坐标：`model/rotation2xyz.py`

**`Rotation2xyz`** 类（行 11-92）：将模型的旋转表示输出为 3D XYZ 关节坐标，是**下游可视化的必经之路**。

```python
class Rotation2xyz:
    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0, ...):
        # 输入 x: (batch, njoints, nfeats, frames)
        # pose_rep: 'rot6d' / 'rotmat' / 'rotquat' / 'rotvec' / 'xyz'
        # jointstype: 'smpl' / 'a2m' / 'vertices'
        
        # 1. 旋转表示转换
        rotations = rot6d_to_matrix(x_rotations)  # 或其他转换
        
        # 2. SMPL 前向计算
        smpl_output = self.smpl_model(body_pose=rotations, ...)
        
        # 3. 提取指定关节点
        joints = smpl_output[jointstype]  # 'smpl' → 22 个关节
        
        # 4. 根部平移对齐
        joints = joints - joints[:, [root_idx]]  # 根部归零
        
        # 5. 添加全局平移
        if translation and vertstrans:
            joints = joints + root_translations
        
        return joints  # (batch, njoints, 3, frames)
```

**支持的关节点类型**：

| `jointstype` | 关节数 | 说明 |
|-------------|--------|------|
| `smpl` | 22 | SMPL 人体模型关节点 |
| `a2m` | 24 | Actor 格式 |
| `vertices` | 6890 | SMPL 网格顶点 |

### 5.2 运动数据可视化：`data_loaders/humanml/utils/plot_script.py`

`plot_3d_motion`：将 XYZ 关节坐标渲染为 MP4 视频（骨骼线条动画）。

```python
plot_3d_motion(save_path, skeleton_chain, motion_xyz,
                dataset='humanml',  # 影响骨骼颜色方案
                title='caption', fps=20,
                gt_frames=[...])   # 高亮指定帧（编辑任务中用于区分 GT/生成）
```

---

## 6. MDM 模型结构详解

### 6.1 整体架构图

```
输入：x_t (B, njoints, nfeats, T) + 时间步 t + 条件 y
     │
     ├── 时间步嵌入
     │    └─ TimestepEmbedder
     │         └─ time_embed(pe[t]) → (B, d)  正弦位置编码 + MLP
     │
     ├── 条件嵌入
     │    ├── 文本分支（CLIP / DistilBERT）
     │    │    └─ encode_text(text) → (B, clip_dim) → Linear → (B, d)
     │    └── 动作分支（Action Embedding）
     │         └─ Linear(num_actions, d)
     │
     ├── 条件与时间步融合
     │    └── emb = text_emb + time_emb  （或 concat，取决于 emb_policy）
     │
     ├── 输入投影
     │    └── InputProcess: (B, J, F, T) → (T, B, J*F) → Linear → (T, B, d)
     │
     ├── 核心去噪网络（3 选 1）
     │    │
     │    ├── 【TransFormer Encoder】（默认）
     │    │    xseq = concat([emb[None], x_proj])  # (T+1, B, d)
     │    │    xseq = PositionalEncoding(xseq)
     │    │    output = TransformerEncoder(xseq, src_key_padding_mask=mask)
     │    │    output = output[1:]  # 去除条件 token
     │    │
     │    ├── 【Transformer Decoder】（DiP / trans_dec）
     │    │    memory = concat([time_emb, text_emb])  # 条件（可选：作为首个 token 注入）
     │    │    tgt = PositionalEncoding(x_proj)
     │    │    output = TransformerDecoder(tgt, memory, ...)
     │    │
     │    └── 【GRU】
     │         x_proj_with_emb = concat([x_proj, time_emb_tile])
     │         output, _ = GRU(x_proj_with_emb)
     │
     ├── 输出投影
     │    └── OutputProcess: (T, B, d) → Linear → (T, B, J*F) → reshape → (B, J, F, T)
     │
     └── 输出：(B, njoints, nfeats, T)  — 预测 x_0 或 ε（取决于 model_mean_type）
```

### 6.2 输入处理：`InputProcess`（`model/mdm.py` 行 333-357）

```python
class InputProcess(nn.Module):
    def forward(self, x):
        # x: (B, njoints, nfeats, nframes)
        # 转换为 (nframes, B, njoints*nfeats)
        x = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints*nfeats)
        # 线性投影
        return self.poseEmbedding(x)  # (nframes, B, d)
```

### 6.3 时间步嵌入：`TimestepEmbedder`（行 316-330）

```python
class TimestepEmbedder(nn.Module):
    # 使用正弦位置编码表（可学习）
    pe: (max_len=5000, d)
    
    def forward(self, timesteps):
        # timesteps: (B,)  — 每个样本的时间步索引 [0, T)
        # pe[t]: (B, d) — 正弦位置编码
        return self.time_embed(pe[timesteps]).permute(1, 0, 2)
        # 输出: (1, B, d) — 复用到 batch 维度
```

### 6.4 文本编码器

#### CLIP 编码（`clip_encode_text`，行 163-178）

```python
def clip_encode_text(self, raw_text):
    # raw_text: [str, str, ...] — batch 个文本
    # 1. Tokenize（限制 max_text_len=20）
    texts = clip.tokenize(raw_text, context_length=22, truncate=True)
    # 2. CLIP 编码
    return clip_model.encode_text(texts).float()  # (1, B, 512)
```

#### DistilBERT 编码（`bert_encode_text`，行 180-187）

```python
def bert_encode_text(self, raw_text):
    enc_text, mask = self.bert_model(raw_text)
    # enc_text: (seq_len, B, 768)
    # mask: (B, seq_len) — True=有效 token
    return enc_text, mask
```

### 6.5 核心网络：Transformer Encoder（`trans_enc`，行 249-253）

```python
if self.arch == 'trans_enc':
    xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d] — 条件作为额外的第一个 token
    xseq = self.sequence_pos_encoder(xseq)
    output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]
    # Transformer Encoder: 所有 token 互相 attend
    # frames_mask: True=无效（填充帧），用于忽略填充位置
```

### 6.6 核心网络：Transformer Decoder（`trans_dec`，行 255-270）

```python
if self.arch == 'trans_dec':
    xseq = x  # 仅运动序列加位置编码
    xseq = self.sequence_pos_encoder(xseq)
    
    if text_encoder_type == 'clip':
        # 交叉注意力：运动 token attend 到文本 token
        output = self.seqTransDecoder(tgt=xseq, memory=emb)
    elif text_encoder_type == 'bert':
        # BERT 模式：额外传入文本掩码
        output = self.seqTransDecoder(tgt=xseq, memory=emb,
                                      memory_key_padding_mask=text_mask)
```

### 6.7 无分类器引导推理

模型训练时：`--cond_mask_prob 0.1` → 10% 的样本强制将条件向量置零，使模型同时学习有条件和无条件预测。

推理时：`ClassifierFreeSampleModel` 包装器在单次前向中计算：
```python
out_cond    = model(x, t, y)           # 有条件
out_uncond = model(x, t, {y, 'uncond': True})  # 无条件
return out_uncond + scale * (out_cond - out_uncond)
```

### 6.8 帧掩码机制（`mask_frames`，行 243-247）

处理 HumanML3D 中不同长度运动带来的填充问题：

```python
if self.mask_frames and is_valid_mask:
    frames_mask = ~mask  # True=无效帧
    if emb_trans_dec or arch == 'trans_enc':
        step_mask = zeros((B, 1))  # 条件 token 不掩码
        frames_mask = concat([step_mask, frames_mask], dim=1)
    # 作为 src_key_padding_mask 传入 Transformer，忽略填充 token 的注意力
```

### 6.9 模型参数汇总

| 组件 | 参数 | 默认值 |
|------|------|--------|
| 隐藏维度 | `latent_dim` | 512 |
| 前馈维度 | `ff_size` | 1024 |
| 层数 | `num_layers` | 8 |
| 注意力头数 | `num_heads` | 4 |
| Dropout | `dropout` | 0.1 |
| 位置编码最大长度 | `pos_embed_max_len` | 5000 |

**HumanML3D 默认配置**（`utils/model_util.py` 行 41-45）：

```python
if args.dataset == 'humanml':
    data_rep = 'hml_vec'     # HumanML3D 专用向量格式
    njoints = 263             # 向量维度（不是关节数）
    nfeats = 1
```

---

## 7. 如何替换扩散模型为新范式

### 7.1 推荐的可复用组件

| 组件 | 文件路径 | 复用价值 |
|------|----------|----------|
| 数据集 + 归一化 | `data_loaders/humanml/data/dataset.py` | **最高**，直接复用 |
| 批处理 Collate | `data_loaders/tensors.py` | **高**，转为 MDM 内部格式 |
| 评估指标 | `data_loaders/humanml/utils/metrics.py` | **高**，可直接计算 FID/R-precision |
| 评估器嵌入网络 | `data_loaders/humanml/networks/evaluator_wrapper.py` | **高**，文本-运动联合嵌入 |
| 旋转→XYZ 坐标 | `model/rotation2xyz.py` | **高**，可视化必备 |
| 骨骼定义 | `data_loaders/humanml/utils/paramUtil.py` | **高**，可视化 |
| 采样器包装 | `utils/sampler_util.py` (CFG) | **中**，CFG 可迁移到其他模型 |
| 扩散基础数学 | `diffusion/gaussian_diffusion.py` | **参考**，其中的数学公式可参考实现 Flow Matching |

### 7.2 最小改造路径（保持下游工具链不变）

只需实现一个与 MDM 模型接口兼容的新模型类：

```python
class MyNewModel(nn.Module):
    def __init__(self, ...):
        # 初始化你的新模型（Flow Matching / GAN / VAE / Transformer 等）
        pass
    
    def forward(self, x, timesteps, y=None):
        """
        x:       (batch, 263, 1, T)    — 噪声或中间状态
        timesteps: (batch,)             — 时间步（如果你的范式不需要，可以忽略）
        y:       dict — 包含 mask, lengths, text 等条件
        
        返回: (batch, 263, 1, T) — 预测的 x_0 或噪声
        """
        # 1. 编码条件（复用 CLIP / BERT / DistilBERT）
        text_emb = self.encode_text(y['text'])
        
        # 2. 用你的新模型去噪/生成
        output = self.new_generator(x, text_emb, timesteps)
        
        # 3. 输出（必须与 MDM 输出格式一致）
        return output  # shape: (B, 263, 1, T)
    
    def encode_text(self, raw_text):
        # 直接复用 MDM 的文本编码器
        return self.clip_encode_text(raw_text)  # 或 BERT
```

### 7.3 训练流程改造

新模型只需实现与 MDM 相同的训练循环模式：

```python
# 1. 加载数据集
data = get_dataset_loader('humanml', batch_size=64, num_frames=196,
                          split='train', hml_mode='train')

# 2. 创建你的模型
model = MyNewModel(...)

# 3. 训练循环（替换 diffusion.training_losses 为你的损失函数）
for motion, cond in data:
    # 你的训练逻辑（如 Flow Matching 的 loss）
    loss = model.compute_loss(motion, cond)
    loss.backward()
    optimizer.step()

# 4. 推理时复用 sample/generate.py 的后处理流程：
#    - model.rot2xyz() → XYZ 坐标
#    - plot_3d_motion() → MP4
#    - EvaluatorMDMWrapper → FID/R-precision
```

### 7.4 评估流程改造

```python
# 复用完整的评估工具链
from eval.eval_humanml import evaluation
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

# 生成用于评估的运动（转为 (B, 263, T) 格式）
generated_motions = your_model.generate(texts, ...)

# 转为评估器所需格式
eval_motion_loaders = {
    'vald': lambda: get_mdm_loader(args, model=your_model, ...)
}

# 直接使用所有评估指标
eval_wrapper = EvaluatorMDMWrapper('humanml', device)
results = evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, ...)
# 返回: {'FID_vald': 0.1, 'R_precision_topk_3_vald': 0.8, ...}
```

### 7.5 关键维度对照表

理解 MDM 的维度约定是复用工具链的关键：

| 变量 | Shape | 说明 |
|------|-------|------|
| `motion` (dataset output) | `(seqlen, 263)` | 原始顺序 |
| `motion.T` → `inp` | `(263, seqlen)` | Transpose |
| `inp.unsqueeze(1)` | `(263, 1, seqlen)` | MDM 输入 |
| `collate` output `motion` | `(B, 263, 1, T)` | Batch |
| `model.output` | `(B, 263, 1, T)` | 模型输出 |
| `rot2xyz` output | `(B, 22, 3, T)` | XYZ 关节坐标 |
| `text_emb` (CLIP) | `(1, B, 512)` | 文本嵌入 |
| `text_emb` (BERT) | `(seq_len, B, 768)` | BERT 输出 |

---

## 附录：快速参考

### A. 数据集配置

```python
# HumanML3D
joints_num = 22
vec_dim = 263  # hml_vec 格式
max_frames = 196

# KIT
joints_num = 21
vec_dim = 251
max_frames = 196
fps = 12.5
```

### B. 评估器配置

```python
evaluator_config = {
    'dim_word': 300,          # GloVe 维度
    'dim_movement_latent': 512,
    'dim_coemb_hidden': 512,  # 联合嵌入维度
    'unit_length': 4,        # 下采样步长
}
```

### C. 扩散默认配置

```python
diffusion_config = {
    'noise_schedule': 'cosine',
    'diffusion_steps': 1000,
    'sigma_small': True,
    'model_mean_type': 'START_X',  # 预测 x0
    'model_var_type': 'FIXED_SMALL',
    'loss_type': 'MSE',
}
```
