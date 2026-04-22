@echo off
REM DMG: Download Pretrained Models

echo Downloading MLD pretrained models...

REM 创建目录
if not exist ..\deps mkdir ..\deps
if not exist ..\pretrained_models mkdir ..\pretrained_models

REM TODO: 添加实际的下载命令
REM MLD VAE 预训练模型
REM CLIP 模型
REM T2M 评估器

echo.
echo Please manually download the following:
echo 1. MLD VAE: https://drive.google.com/file/d/1G9O5arldtHvB66OPr31oE_rJG1bH_R39/view
echo 2. CLIP ViT: ./deps/clip-vit-large-patch14
echo 3. T2M Evaluators: ./deps/t2m/
echo.
echo Or download from the MLD repository:
echo https://github.com/ChenFengYe/motion-latent-diffusion
echo.

pause
