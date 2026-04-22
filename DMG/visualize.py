"""
DMG 可视化脚本

阶段6：SMPL 可视化集成

支持两种可视化方式：
1. 3D 动画（matplotlib）：直接显示骨架动画，无需 Blender
2. Blender 渲染：高质量视频渲染

用法：
  # 3D 动画预览（无需 Blender）
  python visualize.py --input results/inference/sample_0000_full.npy --mode anim

  # Blender 渲染
  python visualize.py --input results/inference/sample_0000_full.npy --mode video --output results/videos/

  # 批量渲染目录
  python visualize.py --input_dir results/inference/ --mode video --output results/videos/
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dmg.config import parse_args
from dmg.transforms.feats2joints import feats2joints, feats2joints_simple, HUMANML3D_JOINT_NAMES
from dmg.utils.logger import create_logger

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. 3D animation will not be available.")

try:
    from mld.utils.joints import smplh_to_mmm_scaling_factor
    SCALE_FACTOR = smplh_to_mmm_scaling_factor
except ImportError:
    SCALE_FACTOR = 1.0
    print("Warning: smplh_to_mmm_scaling_factor not found, using 1.0")

try:
    import bpy
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    print("Warning: bpy (Blender Python) not found. Video rendering will not be available.")


# ============================================================================
# 3.1 3D 动画可视化（matplotlib）
# ============================================================================

# HumanML3D 骨架连接（用于可视化）
HUMANML3D_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # root -> hips
    (1, 3), (2, 4),  # hips -> knees
    (3, 5), (4, 6),  # knees -> ankles
    (5, 7), (6, 8),  # ankles -> feet
    (0, 9),  # root -> spine
    (9, 10), (10, 11),  # spine -> chest -> neck
    (11, 12), (11, 13), (11, 14),  # neck -> shoulders
    (12, 15),  # neck -> head
    (13, 16), (14, 17),  # shoulders -> elbows
    (16, 18), (17, 19),  # elbows -> wrists
]

# 用于 3D 可视化的颜色
HUMANML3D_COLORS = {
    'root': 'red',
    'spine': 'orange',
    'head': 'green',
    'arms': 'blue',
    'legs': 'purple',
}


def plot_skeleton(joints: np.ndarray, ax: Axes3D, alpha: float = 1.0):
    """在 3D 坐标系中绘制骨架"""
    T = joints.shape[0]
    
    for connection in HUMANML3D_SKELETON_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < T and end_idx < T:
            x = [joints[start_idx, 0], joints[end_idx, 0]]
            y = [joints[start_idx, 2], joints[end_idx, 2]]  # swap y/z for better view
            z = [joints[start_idx, 1], joints[end_idx, 1]]
            ax.plot(x, y, z, 'b-', alpha=alpha)
    
    # 绘制关节点
    ax.scatter(joints[:T, 0], joints[:T, 2], joints[:T, 1], c='red', s=10, alpha=alpha)


def create_animation(
    joints_seq: np.ndarray,
    fps: int = 20,
    title: str = "Motion",
    output_path: str = None,
) -> animation.FuncAnimation:
    """
    创建骨架动画

    Args:
        joints_seq: [T, 22, 3] 关节序列
        fps: 帧率
        title: 标题
        output_path: 保存路径（.mp4 或 .gif）
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for animation")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f"{title} - Frame {frame}/{len(joints_seq)}")
        
        # 绘制当前帧
        joints = joints_seq[frame]
        T = 1
        for connection in HUMANML3D_SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < joints.shape[0] and end_idx < joints.shape[0]:
                x = [joints[start_idx, 0], joints[end_idx, 0]]
                y = [joints[start_idx, 2], joints[end_idx, 2]]
                z = [joints[start_idx, 1], joints[end_idx, 1]]
                ax.plot(x, y, z, 'b-', linewidth=2)
        
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='red', s=30)
        
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(joints_seq), interval=1000/fps, blit=True)
    
    if output_path:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"Animation saved to {output_path}")
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {output_path}")
    
    plt.close()
    return anim


def visualize_3d_animation(
    joints_seq: np.ndarray,
    title: str = "Generated Motion",
    block: bool = True,
):
    """
    显示 3D 骨架动画（交互式窗口）

    Args:
        joints_seq: [T, 22, 3] 关节序列
        title: 窗口标题
        block: 是否阻塞直到窗口关闭
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for 3D animation")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f"{title} - Frame {frame + 1}/{len(joints_seq)}")
        
        joints = joints_seq[frame]
        for connection in HUMANML3D_SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < joints.shape[0] and end_idx < joints.shape[0]:
                x = [joints[start_idx, 0], joints[end_idx, 0]]
                y = [joints[start_idx, 2], joints[end_idx, 2]]
                z = [joints[start_idx, 1], joints[end_idx, 1]]
                ax.plot(x, y, z, 'b-', linewidth=2)
        
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='red', s=30)
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=len(joints_seq), interval=50, blit=True)
    
    plt.show(block=block)


# ============================================================================
# 3.2 Blender 渲染
# ============================================================================

def blender_render_sequence(
    joints_seq: np.ndarray,
    output_dir: str,
    fps: int = 20,
    canonicalize: bool = True,
    joints_num: int = 22,
) -> str:
    """
    使用 Blender 渲染动作序列为视频

    Args:
        joints_seq: [T, joints_num, 3] 关节序列
        output_dir: 输出目录
        fps: 帧率
        canonicalize: 是否规范化朝向
        joints_num: 关节数量

    Returns:
        输出视频路径
    """
    if not HAS_BLENDER:
        raise ImportError("bpy (Blender Python) is required for video rendering")
    
    # 导入 MLD Blender 渲染模块
    try:
        sys.path.insert(0, str(project_root.parent / "MLD"))
        from mld.render.blender import render
        from mld.render.blender.tools import mesh_detect
        from mld.utils.joints import smplh_to_mmm_scaling_factor
    except ImportError as e:
        print(f"Warning: Could not import MLD Blender modules: {e}")
        print("Falling back to simple joint rendering...")
        return blender_render_joints_fallback(joints_seq, output_dir, fps)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_folder = os.path.join(output_dir, f"frames_{timestamp}")
    os.makedirs(frames_folder, exist_ok=True)
    
    # 准备数据
    joints_for_render = joints_seq.copy()
    
    # 检测是否为 mesh 数据
    is_mesh = mesh_detect(joints_for_render)
    if not is_mesh:
        # 转换为 MMM 格式并缩放
        # 注意：joints_seq 可能已包含 scale（如 inference.py 传入的 scale=1.3）
        # 这里只需添加 MMM 缩放因子，使用模块级 SCALE_FACTOR 保持一致
        joints_for_render = joints_for_render * SCALE_FACTOR
    
    # 渲染
    out = render(
        joints_for_render,
        frames_folder,
        canonicalize=canonicalize,
        exact_frame=None,
        num=8,
        mode="video",
        faces_path=None,
        downsample=True,
        always_on_floor=False,
        denoising=True,
        oldrender=True,
        jointstype="humanml3d",
        res="high",
        init=True,
        gt=False,
        accelerator='gpu',
        device=[0],
    )
    
    # 生成视频
    import shutil
    video_path = frames_folder.replace("_frames", ".mp4")
    
    # 合成视频（需要 ffmpeg）
    try:
        import subprocess
        frame_pattern = os.path.join(frames_folder, "*.png")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-pattern_type', 'glob', '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.rmtree(frames_folder)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Warning: ffmpeg not found or failed: {e}")
        print(f"Frames saved to {frames_folder}")
        video_path = frames_folder
    
    return video_path


def blender_render_joints_fallback(
    joints_seq: np.ndarray,
    output_dir: str,
    fps: int = 20,
) -> str:
    """
    Blender 关节渲染降级方案（当 MLD Blender 模块不可用时）

    使用 Blender API 直接渲染关节骨架
    """
    if not HAS_BLENDER:
        raise ImportError("bpy (Blender Python) is required")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_folder = os.path.join(output_dir, f"joints_frames_{timestamp}")
    os.makedirs(frames_folder, exist_ok=True)
    
    # 清除现有场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 设置场景
    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = fps
    
    # 创建骨骼骨架
    def create_bone(name, head, tail, parent=None):
        bpy.ops.object.armature_add(location=(0, 0, 0))
        arm = bpy.context.active_object
        arm.name = name
        
        bpy.ops.object.mode_set(mode='EDIT')
        arm.data.edit_bones.remove(arm.data.edit_bones[0])
        
        bone = arm.data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        if parent and parent in arm.data.edit_bones:
            bone.parent = arm.data.edit_bones[parent]
        
        bpy.ops.object.mode_set(mode='OBJECT')
        return arm
    
    # 渲染每一帧
    for frame_idx in range(len(joints_seq)):
        joints = joints_seq[frame_idx]
        
        # 清空场景
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # 创建关节点
        for joint_idx, joint_pos in enumerate(joints):
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=joint_pos,
                scale=(0.05, 0.05, 0.05)
            )
            obj = bpy.context.active_object
            obj.name = f"joint_{joint_idx}"
        
        # 创建连接线
        for start_idx, end_idx in HUMANML3D_SKELETON_CONNECTIONS:
            if start_idx < len(joints) and end_idx < len(joints):
                bpy.ops.mesh.primitive_cylinder_add(
                    location=[
                        (joints[start_idx, 0] + joints[end_idx, 0]) / 2,
                        (joints[start_idx, 1] + joints[end_idx, 1]) / 2,
                        (joints[start_idx, 2] + joints[end_idx, 2]) / 2,
                    ],
                )
                obj = bpy.context.active_object
                obj.name = f"bone_{start_idx}_{end_idx}"
        
        # 设置当前帧
        scene.frame_set(frame_idx)
        
        # 渲染
        output_path = os.path.join(frames_folder, f"frame_{frame_idx:04d}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
    
    # 生成视频
    video_path = frames_folder.replace("_frames", ".mp4")
    try:
        import subprocess
        frame_pattern = os.path.join(frames_folder, "*.png")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-pattern_type', 'glob', '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Warning: ffmpeg not found or failed: {e}")
        video_path = frames_folder
    
    return video_path


# ============================================================================
# 主函数
# ============================================================================

def load_and_convert(
    npy_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    加载 .npy 文件并转换为关节坐标

    Args:
        npy_path: .npy 文件路径
        mean, std: 归一化参数
        scale: 缩放因子

    Returns:
        joints: [T, 22, 3] 关节序列
    """
    motion = np.load(npy_path)
    
    if motion.ndim == 2 and motion.shape[-1] == 263:
        # [T, 263] RIFKE 特征
        joints = feats2joints(motion, mean, std, scale=scale)
    elif motion.ndim == 3 and motion.shape[-1] == 263:
        # [B, T, 263] - 取第一个 batch
        joints = feats2joints(motion[0], mean, std, scale=scale)
    elif motion.ndim == 3 and motion.shape[-1] == 22:
        # [T, 22, 3] 已经是关节坐标
        joints = motion
    elif motion.ndim == 3 and motion.shape[1] == 22:
        # [T, 22, 3] 已经是关节坐标
        joints = motion
    else:
        raise ValueError(f"Unexpected shape: {motion.shape}")
    
    return joints


def process_single_file(
    npy_path: str,
    output_dir: str,
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    scale: float = 1.0,
    fps: int = 20,
):
    """处理单个文件"""
    print(f"\nProcessing: {npy_path}")
    
    # 加载并转换
    joints = load_and_convert(npy_path, mean, std, scale)
    print(f"  Joints shape: {joints.shape}")
    
    # 3D 动画
    if mode == "anim":
        visualize_3d_animation(joints, title=Path(npy_path).stem)
    
    # Blender 视频
    elif mode == "video":
        video_path = blender_render_sequence(
            joints, output_dir, fps=fps
        )
        print(f"  Video saved to: {video_path}")
    
    # 保存关节坐标
    elif mode == "joints":
        output_path = Path(output_dir) / f"{Path(npy_path).stem}_joints.npy"
        np.save(output_path, joints)
        print(f"  Joints saved to: {output_path}")
    
    # 保存为 Blender 可用格式
    elif mode == "blender":
        output_path = Path(output_dir) / f"{Path(npy_path).stem}_blender.npy"
        # 添加缩放和偏移
        blender_data = joints * smplh_to_mmm_scaling_factor if 'smplh_to_mmm_scaling_factor' in dir() else joints * 0.01
        np.save(output_path, blender_data)
        print(f"  Blender data saved to: {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str,
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    scale: float = 1.0,
    fps: int = 20,
    max_files: int = 10,
):
    """批量处理目录中的所有 .npy 文件"""
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("*.npy"))[:max_files]
    
    print(f"Found {len(npy_files)} .npy files")
    
    for npy_file in npy_files:
        process_single_file(
            str(npy_file), output_dir, mode,
            mean, std, scale, fps
        )


def main():
    parser = argparse.ArgumentParser(description="DMG Motion Visualization")
    parser.add_argument("--input", type=str, default=None,
                        help="Input .npy file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory containing .npy files")
    parser.add_argument("--output", type=str, default="./results/visualization",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="anim",
                        choices=["anim", "video", "joints", "blender"],
                        help="Visualization mode: anim (matplotlib), video (Blender), joints (save coords), blender (Blender format)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for visualization")
    parser.add_argument("--max_files", type=int, default=10,
                        help="Max files to process in batch mode")
    parser.add_argument("--dataset", type=str, default="humanml3d",
                        help="Dataset name (humanml3d/kit)")
    args = parser.parse_args()
    
    # 加载归一化参数
    cfg = parse_args(phase="test")
    from dmg.data.get_data import get_global_mean_std
    mean, std = get_global_mean_std(args.dataset, cfg)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.input:
        process_single_file(
            args.input, args.output, args.mode,
            mean, std, args.scale, args.fps
        )
    elif args.input_dir:
        process_directory(
            args.input_dir, args.output, args.mode,
            mean, std, args.scale, args.fps, args.max_files
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # 3D 动画预览")
        print("  python visualize.py --input results/gen.npy --mode anim")
        print()
        print("  # Blender 渲染")
        print("  python visualize.py --input results/gen.npy --mode video --output results/videos/")
        print()
        print("  # 批量处理")
        print("  python visualize.py --input_dir results/inference/ --mode video --max_files 5")


if __name__ == "__main__":
    main()
