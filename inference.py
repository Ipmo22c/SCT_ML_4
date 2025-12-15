"""
Inference script for hand gesture recognition on images and videos.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import os
from model import create_model


class GestureRecognizer:
    """Hand gesture recognizer for images and videos."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_to_name = checkpoint['label_to_name']
        self.num_classes = checkpoint['num_classes']
        self.model_type = checkpoint.get('model_type', 'cnn')
        
        # Create and load model
        self.model = create_model(
            model_type=self.model_type,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.label_to_name.values())}")
    
    def preprocess_image(self, image):
        """Preprocess image for inference."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if OpenCV image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def predict(self, image, return_probabilities=False):
        """
        Predict gesture from an image.
        
        Args:
            image: PIL Image, numpy array, or image path
            return_probabilities: If True, return class probabilities
            
        Returns:
            Predicted class name (and probabilities if requested)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.label_to_name[predicted.item()]
        confidence_score = confidence.item()
        
        if return_probabilities:
            probs = probabilities[0].cpu().numpy()
            prob_dict = {self.label_to_name[i]: float(probs[i]) 
                        for i in range(self.num_classes)}
            return predicted_class, confidence_score, prob_dict
        
        return predicted_class, confidence_score
    
    def predict_image(self, image_path: str, display: bool = True):
        """
        Predict gesture from an image file.
        
        Args:
            image_path: Path to image file
            display: Whether to display the result
        """
        image = Image.open(image_path).convert('RGB')
        predicted_class, confidence, probabilities = self.predict(image, return_probabilities=True)
        
        print(f"\nImage: {image_path}")
        print(f"Predicted Gesture: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nAll Probabilities:")
        for gesture, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {gesture}: {prob:.2%}")
        
        if display:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Display image
            ax1.imshow(image)
            ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
            ax1.axis('off')
            
            # Display probabilities
            gestures = list(probabilities.keys())
            probs = list(probabilities.values())
            colors = ['green' if g == predicted_class else 'gray' for g in gestures]
            ax2.barh(gestures, probs, color=colors)
            ax2.set_xlabel('Probability')
            ax2.set_title('Class Probabilities')
            ax2.set_xlim([0, 1])
            
            plt.tight_layout()
            plt.show()
        
        return predicted_class, confidence
    
    def predict_video(self, video_path: str = None, camera_index: int = 0, 
                     output_path: str = None, display: bool = True):
        """
        Predict gestures from video file or webcam.
        
        Args:
            video_path: Path to video file (None for webcam)
            camera_index: Camera index for webcam (default: 0)
            output_path: Path to save output video (optional)
            display: Whether to display the video
        """
        # Open video source
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
        else:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera: {camera_index}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nPress 'q' to quit")
        frame_count = 0
        predicted_class = "Loading..."
        confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Predict every 5 frames for better performance
                if frame_count % 5 == 0 or frame_count == 1:
                    predicted_class, confidence = self.predict(frame, return_probabilities=False)
                
                # Draw prediction on frame
                text = f"{predicted_class} ({confidence:.1%})"
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if output path provided
                if writer:
                    writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Hand Gesture Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description='Hand gesture recognition inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file for prediction')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file for prediction')
    parser.add_argument('--camera', action='store_true',
                       help='Use webcam for real-time prediction')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Initialize recognizer
    recognizer = GestureRecognizer(args.model_path, device=args.device)
    
    # Run inference
    if args.image:
        recognizer.predict_image(args.image, display=True)
    elif args.video:
        recognizer.predict_video(video_path=args.video, output_path=args.output, display=True)
    elif args.camera:
        recognizer.predict_video(camera_index=args.camera_index, output_path=args.output, display=True)
    else:
        print("Please specify --image, --video, or --camera")
        parser.print_help()


if __name__ == '__main__':
    main()

