"""
DMG MLD CLIP 文本编码器
复用 MLD 的 MldTextEncoder

此模块用于：
1. 阶段0-1: 验证 CLIP 加载和文本编码
2. 阶段1-5: 作为冻结的文本条件提取器
"""

import torch
import torch.nn as nn


class MldTextEncoder(nn.Module):
    """
    MLD CLIP 文本编码器

    使用 CLIP 模型提取文本嵌入，用于条件生成
    """

    def __init__(
        self,
        modelpath: str = "./deps/clip/ViT-B-32.pt",
        precision: str = "fp32",
        **kwargs
    ):
        """
        初始化 CLIP 文本编码器

        Args:
            modelpath: CLIP 模型路径
            precision: 精度 ('fp32', 'fp16', 'bf16')
        """
        super().__init__()

        self.modelpath = modelpath
        self.precision = precision

        # 尝试加载 CLIP 模型
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_device = 'cpu'
        self._load_clip()

        # 文本嵌入维度
        self.text_dim = 512  # CLIP ViT-B/32

    def _load_clip(self):
        """加载 CLIP 模型"""
        try:
            import clip
            import torchvision.transforms as transforms
            from PIL import Image

            # 加载 CLIP 模型和预处理
            self.clip_model, self.clip_preprocess = clip.load(
                self.modelpath,
                device='cpu'
            )

            # 获取设备
            self.clip_device = next(self.clip_model.parameters()).device

            # 冻结所有参数
            for param in self.clip_model.parameters():
                param.requires_grad = False

            self.clip_model.eval()
            print(f"Loaded CLIP model from {self.modelpath}, device: {self.clip_device}")

        except ImportError:
            print(f"Warning: Could not load CLIP from {self.modelpath}")
            print("CLIP will be loaded from MLD dependencies if available")
        except Exception as e:
            print(f"Warning: Error loading CLIP: {e}")
            print("CLIP will be loaded from MLD dependencies if available")

    def load_from_mld(self, mld_path: str):
        """
        从 MLD 加载 CLIP 编码器

        Args:
            mld_path: MLD 模型路径
        """
        try:
            # 尝试从 MLD 导入
            import sys
            import os

            # 添加 MLD 路径
            mld_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'MLD')
            if mld_dir not in sys.path:
                sys.path.insert(0, mld_dir)

            from mld.models.architectures.mld_clip import MldTextEncoder as MLDMldTextEncoder

            mld_clip = MLDMldTextEncoder(modelpath=self.modelpath, precision=self.precision)
            self.clip_model = mld_clip.clip_model
            self.clip_preprocess = mld_clip.clip_preprocess
            self.text_dim = mld_clip.text_dim

            print(f"Loaded CLIP from MLD: {mld_path}")

        except Exception as e:
            print(f"Warning: Could not load CLIP from MLD: {e}")

    @torch.no_grad()
    def encode_text(self, texts):
        """
        编码文本

        Args:
            texts: 文本列表或单个文本

        Returns:
            text_embeddings: [B, 512] 文本嵌入
        """
        if self.clip_model is None:
            # 返回随机嵌入作为占位符
            if isinstance(texts, str):
                texts = [texts]
            return torch.randn(len(texts), self.text_dim)

        import clip
        from PIL import Image

        # 处理输入
        if isinstance(texts, str):
            texts = [texts]

        # CLIP 编码
        text_tokens = clip.tokenize(texts)
        if self.clip_model is not None:
            text_embeddings = self.clip_model.encode_text(text_tokens.to(self.clip_device))
        else:
            text_embeddings = torch.randn(len(texts), self.text_dim)

        return text_embeddings.float().cpu()

    @torch.no_grad()
    def encode_text_with_prompt(self, texts, prompt_template="a person is {}"):
        """
        使用提示模板编码文本

        Args:
            texts: 文本列表
            prompt_template: 提示模板

        Returns:
            text_embeddings: [B, 512] 文本嵌入（与 encode_text 保持一致，返回 CPU tensor）
        """
        if self.clip_model is None:
            # 返回随机嵌入作为占位符
            return torch.randn(len(texts), self.text_dim)

        import clip

        # 构建提示文本
        prompted_texts = [prompt_template.format(t) for t in texts]
        text_tokens = clip.tokenize(prompted_texts)
        text_embeddings = self.clip_model.encode_text(text_tokens.to(self.clip_device))

        return text_embeddings.float().cpu()

    def forward(self, texts):
        """
        前向传播

        Args:
            texts: 文本或文本列表

        Returns:
            text_embeddings: [B, 512] 文本嵌入
        """
        return self.encode_text(texts)


class CLIPTextEncoder(nn.Module):
    """
    CLIP 文本编码器的别名，保持与 MLD 的兼容性
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = MldTextEncoder(*args, **kwargs)

    def forward(self, texts):
        return self.encoder.forward(texts)
