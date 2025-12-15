"""
Data loading and preprocessing module for LeapGestRecog dataset.
Handles dataset organization, augmentation, and data loading.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import glob


class HandGestureDataset(Dataset):
    """Custom Dataset class for hand gesture images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (64, 64), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(image_size: int = 64) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transformation pipelines for training and validation.
    
    Args:
        image_size: Target size for resizing images
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_dataset(data_dir: str) -> Tuple[List[str], List[int], dict]:
    """
    Load dataset from directory structure.
    Supports two structures:
    1. Flat: data_dir/gesture_name/image_files
    2. Nested: data_dir/subject_folder/gesture_name/image_files
    
    Args:
        data_dir: Root directory containing gesture folders or subject folders
        
    Returns:
        Tuple of (image_paths, labels, label_to_name mapping)
    """
    image_paths = []
    labels = []
    label_to_name = {}
    gesture_name_to_label = {}
    label_idx = 0
    
    # Get all folders in data_dir
    top_level_folders = sorted([d for d in os.listdir(data_dir) 
                                if os.path.isdir(os.path.join(data_dir, d))])
    
    # Check if we have nested structure (subject folders containing gesture folders)
    # Look at first folder to determine structure
    first_folder_path = os.path.join(data_dir, top_level_folders[0])
    first_folder_contents = [d for d in os.listdir(first_folder_path) 
                           if os.path.isdir(os.path.join(first_folder_path, d))]
    
    is_nested = len(first_folder_contents) > 0 and any(
        any(c.isdigit() for c in name.split('_')) for name in first_folder_contents[:3]
    )
    
    if is_nested:
        # Nested structure: data_dir/subject/gesture_name/image_files
        print("Detected nested structure (subject/gesture folders)")
        for subject_folder in top_level_folders:
            # Skip if this folder name looks like a subject ID (just digits)
            # Only process folders that contain gesture subfolders
            subject_path = os.path.join(data_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue
                
            gesture_folders = sorted([d for d in os.listdir(subject_path) 
                                     if os.path.isdir(os.path.join(subject_path, d))])
            
            # Skip if no gesture folders found (might be a duplicate structure)
            if len(gesture_folders) == 0:
                continue
            
            for gesture_name in gesture_folders:
                gesture_path = os.path.join(subject_path, gesture_name)
                
                # Get all image files in the folder
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    image_files.extend(glob.glob(os.path.join(gesture_path, ext)))
                    image_files.extend(glob.glob(os.path.join(gesture_path, ext.upper())))
                
                # Skip if no images found
                if len(image_files) == 0:
                    continue
                
                # Map gesture name to label (use clean gesture name without prefix)
                # Gesture names are like "01_palm", "02_l", etc. - extract the name part
                if '_' in gesture_name:
                    # Split on underscore and take the part after the number
                    parts = gesture_name.split('_', 1)
                    clean_gesture_name = parts[-1]  # Take the part after the number
                else:
                    # If no underscore, check if it's just digits (subject folder)
                    if gesture_name.isdigit():
                        continue  # Skip subject folders
                    clean_gesture_name = gesture_name
                
                if clean_gesture_name not in gesture_name_to_label:
                    gesture_name_to_label[clean_gesture_name] = label_idx
                    label_to_name[label_idx] = clean_gesture_name
                    label_idx += 1
                
                # Add to dataset
                label = gesture_name_to_label[clean_gesture_name]
                for img_path in image_files:
                    image_paths.append(img_path)
                    labels.append(label)
    else:
        # Flat structure: data_dir/gesture_name/image_files
        print("Detected flat structure (gesture folders)")
        for gesture_name in top_level_folders:
            gesture_path = os.path.join(data_dir, gesture_name)
            
            # Skip if it's just digits (likely a subject folder)
            if gesture_name.isdigit():
                continue
            
            # Get all image files in the folder
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(gesture_path, ext)))
                image_files.extend(glob.glob(os.path.join(gesture_path, ext.upper())))
            
            # Skip if no images found
            if len(image_files) == 0:
                continue
            
            # Add to dataset
            for img_path in image_files:
                image_paths.append(img_path)
                labels.append(label_idx)
            
            label_to_name[label_idx] = gesture_name
            label_idx += 1
    
    return image_paths, labels, label_to_name


def create_dataloaders(data_dir: str, 
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      test_split: float = 0.1,
                      image_size: int = 64,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing gesture folders
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_to_name)
    """
    # Load dataset
    image_paths, labels, label_to_name = load_dataset(data_dir)
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    val_size = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(image_size)
    
    # Create datasets
    train_dataset = HandGestureDataset(X_train, y_train, transform=train_transform)
    val_dataset = HandGestureDataset(X_val, y_val, transform=val_transform)
    test_dataset = HandGestureDataset(X_test, y_test, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"Dataset loaded successfully!")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Number of classes: {len(label_to_name)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {list(label_to_name.values())}")
    
    return train_loader, val_loader, test_loader, label_to_name

