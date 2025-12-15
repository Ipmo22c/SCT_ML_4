"""
Example script demonstrating how to use the hand gesture recognition system.
"""

import torch
from inference import GestureRecognizer
from PIL import Image
import numpy as np


def example_image_prediction():
    """Example: Predict gesture from an image file."""
    print("="*60)
    print("Example 1: Image Prediction")
    print("="*60)
    
    # Initialize recognizer
    model_path = "models/best_model.pth"
    recognizer = GestureRecognizer(model_path)
    
    # Predict from image file
    image_path = "test_image.jpg"  # Replace with your image path
    predicted_class, confidence = recognizer.predict_image(image_path, display=False)
    
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")


def example_programmatic_prediction():
    """Example: Predict gesture programmatically."""
    print("\n" + "="*60)
    print("Example 2: Programmatic Prediction")
    print("="*60)
    
    # Initialize recognizer
    model_path = "models/best_model.pth"
    recognizer = GestureRecognizer(model_path)
    
    # Load image
    image = Image.open("test_image.jpg").convert('RGB')
    
    # Get prediction with probabilities
    predicted_class, confidence, probabilities = recognizer.predict(
        image, return_probabilities=True
    )
    
    print(f"Predicted Gesture: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 3 Predictions:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for gesture, prob in sorted_probs[:3]:
        print(f"  {gesture}: {prob:.2%}")


def example_batch_prediction():
    """Example: Predict gestures for multiple images."""
    print("\n" + "="*60)
    print("Example 3: Batch Prediction")
    print("="*60)
    
    import glob
    
    # Initialize recognizer
    model_path = "models/best_model.pth"
    recognizer = GestureRecognizer(model_path)
    
    # Get list of images
    image_paths = glob.glob("test_images/*.jpg")  # Replace with your path
    
    results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        predicted_class, confidence = recognizer.predict(image)
        results.append({
            'image': image_path,
            'prediction': predicted_class,
            'confidence': confidence
        })
        print(f"{image_path}: {predicted_class} ({confidence:.2%})")
    
    return results


def example_custom_image():
    """Example: Predict from numpy array or custom image."""
    print("\n" + "="*60)
    print("Example 4: Custom Image Input")
    print("="*60)
    
    # Initialize recognizer
    model_path = "models/best_model.pth"
    recognizer = GestureRecognizer(model_path)
    
    # Create a dummy image (in practice, this would be from your source)
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Predict
    predicted_class, confidence = recognizer.predict(dummy_image)
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")


def main():
    """Run all examples."""
    print("\nHand Gesture Recognition - Usage Examples")
    print("="*60)
    print("\nNote: Make sure you have:")
    print("1. Trained a model (run train.py)")
    print("2. Have test images available")
    print("3. Model checkpoint at 'models/best_model.pth'")
    print("\n" + "="*60)
    
    # Uncomment the examples you want to run:
    
    # example_image_prediction()
    # example_programmatic_prediction()
    # example_batch_prediction()
    # example_custom_image()
    
    print("\nUncomment the examples in main() to run them.")


if __name__ == '__main__':
    main()

