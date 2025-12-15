@echo off
title Hand Gesture Recognition - Test Image
color 0E
echo.
echo ========================================
echo   Hand Gesture Recognition - Test Image
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
set /p img_path="Enter image path: "
if "%img_path%"=="" (
    echo No path provided. Exiting.
    pause
    exit
)
echo.
echo Analyzing image...
python inference.py --model_path models/best_model.pth --image "%img_path%"
pause

