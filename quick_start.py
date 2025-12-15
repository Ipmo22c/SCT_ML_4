"""
Quick start script - Easy launcher for hand gesture recognition system.
"""

import os
import sys
import subprocess


def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("Hand Gesture Recognition System - Quick Start")
    print("="*60)
    print("\n1. Train the model")
    print("2. Test on image")
    print("3. Test on video file")
    print("4. Real-time webcam recognition")
    print("5. Visualize dataset")
    print("6. Exit")
    print("="*60)


def check_model_exists():
    """Check if trained model exists."""
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        return model_path
    return None


def train_model():
    """Train the model."""
    print("\n" + "="*60)
    print("Training Hand Gesture Recognition Model")
    print("="*60)
    
    data_dir = "data/leapGestRecog"
    if not os.path.exists(data_dir):
        print(f"\nError: Dataset not found at {data_dir}")
        print("Please download the dataset first.")
        return
    
    print("\nStarting training...")
    print("This may take a while depending on your hardware.")
    print("\nPress Ctrl+C to stop training early.\n")
    
    try:
        subprocess.run([
            sys.executable, "train.py",
            "--data_dir", data_dir,
            "--epochs", "50",
            "--batch_size", "32"
        ])
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")


def test_image():
    """Test on a single image."""
    model_path = check_model_exists()
    if not model_path:
        print("\nError: No trained model found!")
        print("Please train the model first (option 1).")
        return
    
    print("\n" + "="*60)
    print("Image Gesture Recognition")
    print("="*60)
    
    image_path = input("\nEnter image path (or press Enter to cancel): ").strip()
    if not image_path:
        return
    
    if not os.path.exists(image_path):
        print(f"\nError: Image not found at {image_path}")
        return
    
    print(f"\nAnalyzing image: {image_path}")
    try:
        subprocess.run([
            sys.executable, "inference.py",
            "--model_path", model_path,
            "--image", image_path
        ])
    except Exception as e:
        print(f"\nError: {e}")


def test_video():
    """Test on a video file."""
    model_path = check_model_exists()
    if not model_path:
        print("\nError: No trained model found!")
        print("Please train the model first (option 1).")
        return
    
    print("\n" + "="*60)
    print("Video Gesture Recognition")
    print("="*60)
    
    video_path = input("\nEnter video path (or press Enter to cancel): ").strip()
    if not video_path:
        return
    
    if not os.path.exists(video_path):
        print(f"\nError: Video not found at {video_path}")
        return
    
    print(f"\nProcessing video: {video_path}")
    print("Press 'q' to quit during playback.")
    try:
        subprocess.run([
            sys.executable, "inference.py",
            "--model_path", model_path,
            "--video", video_path
        ])
    except Exception as e:
        print(f"\nError: {e}")


def webcam_recognition():
    """Real-time webcam recognition."""
    model_path = check_model_exists()
    if not model_path:
        print("\nError: No trained model found!")
        print("Please train the model first (option 1).")
        return
    
    print("\n" + "="*60)
    print("Real-time Webcam Gesture Recognition")
    print("="*60)
    print("\nStarting webcam...")
    print("Press 'q' to quit.")
    
    try:
        subprocess.run([
            sys.executable, "inference.py",
            "--model_path", model_path,
            "--camera"
        ])
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your webcam is connected and accessible.")


def visualize_dataset():
    """Visualize the dataset."""
    data_dir = "data/leapGestRecog"
    if not os.path.exists(data_dir):
        print(f"\nError: Dataset not found at {data_dir}")
        print("Please download the dataset first.")
        return
    
    print("\n" + "="*60)
    print("Dataset Visualization")
    print("="*60)
    
    try:
        subprocess.run([
            sys.executable, "visualize_dataset.py",
            "--data_dir", data_dir
        ])
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Main menu loop."""
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            train_model()
        elif choice == "2":
            test_image()
        elif choice == "3":
            test_video()
        elif choice == "4":
            webcam_recognition()
        elif choice == "5":
            visualize_dataset()
        elif choice == "6":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice! Please enter a number between 1-6.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

