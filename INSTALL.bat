@echo off
title Installing Dependencies
color 0E
echo.
echo ========================================
echo   Installing Required Dependencies
echo ========================================
echo.
echo This will install PyTorch and all required packages.
echo This may take a few minutes...
echo.
pause
echo.
echo Installing packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo You can now run:
echo   - TRAIN.bat to train the model
echo   - WEBCAM.bat for webcam recognition
echo   - TEST_IMAGE.bat to test on images
echo.
pause

