"""
DMG Motion Processing Utilities

**重要说明**：
此模块在 DMG 训练链路中**不参与**，因为：
- DMG 训练使用 SlidingWindowDataset，直接加载 HumanML3D 预处理好的 .npy RIFKE 特征文件
- 这些 .npy 文件已经过 MLD 的 motion_process 转换，pipeline.md §2.1 "数据来源" 说明这一点

此模块仅在以下场景使用：
1. validate.py 验证 HumanML3D 数据集完整性时
2. 需要从原始骨骼坐标重新生成 RIFKE 特征时

当此模块被调用但 MLD 目录不存在时，会抛出明确的 ImportError。
"""

import numpy as np
import torch


def process_file(motions, njoints=22):
    """
    将关节坐标转换为 RIFKE 特征

    **DMG 训练不调用此函数**，仅在 validate.py 或 HumanML3D 数据重处理时使用。

    Args:
        motions: [T, J*3] 或 [T, J, 3] 关节坐标
        njoints: 关节数量

    Returns:
        features: [T, 263] RIFKE 特征
    """
    try:
        from mld.utils.motion_process import process_file as mld_process_file
        return mld_process_file(motions, njoints)
    except ImportError:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        mld_path = project_root / "MLD"
        if mld_path.exists() and str(mld_path) not in sys.path:
            sys.path.insert(0, str(project_root))
            try:
                from mld.utils.motion_process import process_file as mld_process_file
                return mld_process_file(motions, njoints)
            except ImportError:
                pass

        raise ImportError(
            "无法导入 MLD 的 motion_process 模块。"
            "此模块仅在 DMG 验证（validate.py）或数据重处理时使用，不影响 DMG 训练。"
            "如需使用，请确保 MLD 目录位于项目根目录。"
        )


def recover_from_ric(features, njoints=22):
    """
    从 RIFKE 特征恢复关节坐标（用于可视化）

    **DMG 训练不调用此函数**，仅在 validate.py 的 feats2joints 中使用。

    Args:
        features: [*, 263] RIFKE 特征
        njoints: 关节数量

    Returns:
        joints: [*, J, 3] 关节坐标
    """
    try:
        from mld.utils.motion_process import recover_from_ric as mld_recover
        return mld_recover(features, njoints)
    except ImportError:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        mld_path = project_root / "MLD"
        if mld_path.exists() and str(mld_path) not in sys.path:
            sys.path.insert(0, str(project_root))
            try:
                from mld.utils.motion_process import recover_from_ric as mld_recover
                return mld_recover(features, njoints)
            except ImportError:
                pass

        raise ImportError(
            "无法导入 MLD 的 recover_from_ric 模块。"
            "此模块仅在 DMG 验证（validate.py）中使用，不影响 DMG 训练。"
            "如需使用，请确保 MLD 目录位于项目根目录。"
        )
