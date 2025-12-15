@echo off
title Hand Gesture Recognition - Webcam
color 0B
echo.
echo ========================================
echo   Hand Gesture Recognition - Webcam
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

if not exist "models\best_model.pth" (
    echo ERROR: No trained model found!
    echo Please run TRAIN.bat first to train the model.
    pause
    exit
)
echo Starting webcam recognition...
echo Press 'q' to quit
echo.
python inference.py --model_path models/best_model.pth --camera
pause

