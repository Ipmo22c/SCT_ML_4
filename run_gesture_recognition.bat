@echo off
echo ========================================
echo Hand Gesture Recognition System
echo ========================================
echo.
echo Choose an option:
echo 1. Train the model
echo 2. Test on image
echo 3. Test on video file
echo 4. Real-time webcam recognition
echo 5. Visualize dataset
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting training...
    python train.py --data_dir data/leapGestRecog --epochs 50 --batch_size 32
    pause
) else if "%choice%"=="2" (
    set /p img_path="Enter image path: "
    python inference.py --model_path models/best_model.pth --image "%img_path%"
    pause
) else if "%choice%"=="3" (
    set /p vid_path="Enter video path: "
    python inference.py --model_path models/best_model.pth --video "%vid_path%"
    pause
) else if "%choice%"=="4" (
    echo.
    echo Starting webcam recognition...
    echo Press 'q' to quit
    python inference.py --model_path models/best_model.pth --camera
    pause
) else if "%choice%"=="5" (
    echo.
    echo Visualizing dataset...
    python visualize_dataset.py --data_dir data/leapGestRecog
    pause
) else if "%choice%"=="6" (
    exit
) else (
    echo Invalid choice!
    pause
)

