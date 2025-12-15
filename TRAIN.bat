@echo off
title Hand Gesture Recognition - Training
color 0A
echo.
echo ========================================
echo   Hand Gesture Recognition - Training
echo ========================================
echo.

REM Check if torch is installed
python -c "import torch" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch is not installed!
    echo Please run INSTALL.bat first to install dependencies.
    pause
    exit
)

echo Starting model training...
echo This will take some time. Please wait...
echo.
python train.py --data_dir data/leapGestRecog --epochs 50 --batch_size 32
echo.
echo Training completed!
pause

