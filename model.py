"""
CNN model architecture for hand gesture recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandGestureCNN(nn.Module):
    """
    Convolutional Neural Network for hand gesture classification.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of gesture classes to classify
            dropout_rate: Dropout rate for regularization
        """
        super(HandGestureCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # After 4 pooling operations (64 -> 32 -> 16 -> 8 -> 4)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class HandGestureResNet(nn.Module):
    """
    ResNet-inspired architecture for hand gesture classification.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of gesture classes to classify
            dropout_rate: Dropout rate for regularization
        """
        super(HandGestureResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_model(model_type: str = 'cnn', num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        num_classes: Number of classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'cnn':
        return HandGestureCNN(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'resnet':
        return HandGestureResNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'resnet'.")

