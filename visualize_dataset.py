"""
Utility script to visualize and explore the dataset.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
from collections import Counter
import argparse


def visualize_dataset(data_dir: str, num_samples: int = 5):
    """
    Visualize samples from each gesture class.
    
    Args:
        data_dir: Root directory containing gesture folders
        num_samples: Number of samples to show per class
    """
    gesture_folders = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
    
    num_classes = len(gesture_folders)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 3 * num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, gesture_name in enumerate(gesture_folders):
        gesture_path = os.path.join(data_dir, gesture_name)
        
        # Get image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(gesture_path, ext)))
            image_files.extend(glob.glob(os.path.join(gesture_path, ext.upper())))
        
        # Select samples
        samples = image_files[:num_samples]
        
        for j, img_path in enumerate(samples):
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(gesture_name, rotation=0, labelpad=40, fontsize=10)
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                               ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.suptitle('Dataset Samples', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("Dataset visualization saved to 'dataset_samples.png'")
    plt.show()


def analyze_dataset(data_dir: str):
    """
    Analyze dataset statistics.
    
    Args:
        data_dir: Root directory containing gesture folders
    """
    gesture_folders = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
    
    stats = {}
    total_images = 0
    
    print("\n" + "="*60)
    print("Dataset Analysis")
    print("="*60)
    
    for gesture_name in gesture_folders:
        gesture_path = os.path.join(data_dir, gesture_name)
        
        # Count image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(gesture_path, ext)))
            image_files.extend(glob.glob(os.path.join(gesture_path, ext.upper())))
        
        count = len(image_files)
        stats[gesture_name] = count
        total_images += count
        print(f"{gesture_name:20s}: {count:5d} images")
    
    print("="*60)
    print(f"{'Total':20s}: {total_images:5d} images")
    print(f"{'Number of classes':20s}: {len(gesture_folders):5d}")
    print(f"{'Average per class':20s}: {total_images // len(gesture_folders):5d} images")
    print("="*60)
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    gestures = list(stats.keys())
    counts = list(stats.values())
    
    plt.bar(gestures, counts, color='steelblue')
    plt.xlabel('Gesture Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Dataset Distribution by Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("\nDataset distribution plot saved to 'dataset_distribution.png'")
    plt.show()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze dataset')
    parser.add_argument('--data_dir', type=str, default='data/leapGestRecog',
                       help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to show per class')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze, do not visualize')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory not found: {args.data_dir}")
        print("Please download the dataset first.")
        return
    
    # Analyze dataset
    analyze_dataset(args.data_dir)
    
    # Visualize dataset
    if not args.analyze_only:
        print("\nVisualizing dataset samples...")
        visualize_dataset(args.data_dir, args.num_samples)


if __name__ == '__main__':
    main()

