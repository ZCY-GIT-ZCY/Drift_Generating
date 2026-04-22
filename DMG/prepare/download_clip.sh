@echo off
REM DMG: Download CLIP Model

echo Downloading CLIP model...

REM 创建目录
if not exist ..\deps\clip-vit-large-patch14 mkdir ..\deps\clip-vit-large-patch14

echo.
echo CLIP models will be downloaded automatically when loading.
echo Or you can manually download from:
echo https://github.com/openai/CLIP
echo.

pause
