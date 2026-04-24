"""
DMG Configuration Parser
复用 MLD 的配置解析逻辑
"""

import importlib
import os
from argparse import ArgumentParser
from omegaconf import OmegaConf


def get_module_config(cfg_model, path="modules"):
    """
    获取模块配置

    Args:
        cfg_model: 基础配置
        path: 模块配置目录

    Returns:
        合并后的配置
    """
    files = os.listdir(f'./configs/{path}/')
    for file in files:
        if file.endswith('.yaml'):
            with open(f'./configs/{path}/' + file, 'r') as f:
                cfg_model = OmegaConf.merge(cfg_model, OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string, reload=False):
    """
    从字符串获取对象

    Args:
        string: "module.class" 格式的字符串
        reload: 是否重新加载模块

    Returns:
        类对象
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    从配置实例化对象

    Args:
        config: 配置对象（包含 target 和 params）

    Returns:
        实例化的对象
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):
    """
    解析命令行参数

    Args:
        phase: 阶段 ('train', 'test', 'demo', 'validate')

    Returns:
        cfg: 配置对象
    """
    parser = ArgumentParser()

    group = parser.add_argument_group("Training options")
    if phase in ["train", "test", "demo", "validate"]:
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/config_dmg_humanml3d.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        group.add_argument("--batch_size", type=int, required=False, help="training batch size")
        group.add_argument("--device", type=int, nargs="+", required=False, help="training device")
        group.add_argument("--nodebug", action="store_true", default=None, required=False, help="debug or not")

    if phase == "demo":
        group.add_argument("--render", action="store_true", help="render visualized figures")
        group.add_argument("--example", type=str, required=False, help="input text file")

    if phase == "render":
        group.add_argument("--cfg", type=str, default="./configs/render.yaml", help="config file")
        group.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml", help="asset config")
        group.add_argument("--npy", type=str, help="npy motion files")
        group.add_argument("--dir", type=str, help="npy motion folder")

    # 解析参数
    params = parser.parse_args()

    # 更新配置
    cfg_base = OmegaConf.load('./configs/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    # --- 配置重映射：modules/*.yaml → 顶层键 ---
    # get_module_config 将 motion_vae / text_encoder / generator 等 merge 进 cfg.model.modules
    # 但 dmg.py 从 cfg.vae / cfg.clip / model.generator 读取，需显式映射
    if hasattr(cfg.model, 'modules'):
        modules = cfg.model.modules
        if hasattr(modules, 'motion_vae'):
            cfg.vae = OmegaConf.create({'pretrained_path': modules.motion_vae.get('pretrained_path', '')})
        if hasattr(modules, 'text_encoder'):
            cfg.clip = OmegaConf.create({
                'model_path': modules.text_encoder.get('modelpath', './deps/clip/ViT-B-32.pt'),
                'precision': modules.text_encoder.get('precision', 'fp32'),
            })
        # generator.yaml 的格式是 generator: {target, params}，
        # config_dmg_humanml3d.yaml 是 generator: {hidden_size, depth, ...}
        # 两者都 merge 进 cfg.model.generator，后者优先级更高（覆盖 target/params）
        if hasattr(modules, 'generator'):
            # generator.yaml 的 params 展开到 cfg.model.generator 顶层
            gen_params = modules.generator.get('params', OmegaConf.create({}))
            base_gen = cfg.model.get('generator', OmegaConf.create({}))
            cfg.model.generator = OmegaConf.merge(base_gen, gen_params)

    if phase in ["train", "test", "validate"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        # 测试时禁用调试模式
        if phase in ["test", "validate"]:
            cfg.DEBUG = False
            cfg.DEVICE = [0] if cfg.ACCELERATOR == 'gpu' else cfg.DEVICE
            print("Force no debugging and one gpu when testing")

    # 调试模式
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        if hasattr(cfg.LOGGER, 'WANDB'):
            cfg.LOGGER.WANDB.OFFLINE = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    return cfg
