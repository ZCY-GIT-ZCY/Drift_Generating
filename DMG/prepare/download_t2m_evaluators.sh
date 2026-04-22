@echo off
REM DMG: Download T2M Evaluators

echo Downloading T2M evaluators...

REM 创建目录
if not exist ..\deps\t2m mkdir ..\deps\t2m

echo.
echo Please manually download T2M evaluators from:
echo https://github.com/EricGuo5513/HumanML3D/tree/main/evaluators
echo Or from MLD repository:
echo https://github.com/ChenFengYe/motion-latent-diffusion/tree/main/deps/t2m
echo.

pause
