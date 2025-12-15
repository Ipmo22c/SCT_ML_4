# Hand Gesture Recognition System

A deep learning-based hand gesture recognition system that can accurately identify and classify different hand gestures from image or video data. This system enables intuitive human-computer interaction and gesture-based control systems.

## Features

- **Deep Learning Models**: CNN and ResNet-inspired architectures for gesture classification
- **Real-time Recognition**: Support for webcam and video file input
- **Image Classification**: Single image gesture prediction with confidence scores
- **Data Augmentation**: Robust training with rotation, flipping, and color jitter
- **Comprehensive Evaluation**: Detailed metrics including confusion matrix and classification report

## Dataset

This project uses the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) from Kaggle, which contains:
- 20,000 infrared images
- 10 distinct hand gestures
- 10 subjects (5 males, 5 females)
- Multiple captures per gesture

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
   - Extract the dataset to a `data/` directory
   - Expected structure:
     ```
     data/
     └── leapGestRecog/
         ├── gesture_01/
         │   ├── image1.png
         │   ├── image2.png
         │   └── ...
         ├── gesture_02/
         └── ...
     ```

## Usage

### Training

Train a model on the dataset:

```bash
python train.py --data_dir data/leapGestRecog --epochs 50 --batch_size 32
```

**Training Arguments**:
- `--data_dir`: Path to dataset directory (default: `data/leapGestRecog`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Input image size (default: 64)
- `--model_type`: Model architecture - `cnn` or `resnet` (default: `cnn`)
- `--save_dir`: Directory to save models (default: `models`)
- `--num_workers`: Number of data loading workers (default: 4)

**Example**:
```bash
python train.py --data_dir data/leapGestRecog --epochs 100 --batch_size 64 --model_type resnet --lr 0.0001
```

### Inference

#### Single Image Prediction

```bash
python inference.py --model_path models/best_model.pth --image path/to/image.jpg
```

#### Video File Prediction

```bash
python inference.py --model_path models/best_model.pth --video path/to/video.mp4
```

#### Real-time Webcam Prediction

```bash
python inference.py --model_path models/best_model.pth --camera
```

**Inference Arguments**:
- `--model_path`: Path to trained model checkpoint (required)
- `--image`: Path to image file for prediction
- `--video`: Path to video file for prediction
- `--camera`: Use webcam for real-time prediction
- `--camera_index`: Camera index (default: 0)
- `--output`: Path to save output video (optional)
- `--device`: Device to run inference on (`cuda` or `cpu`)

## Project Structure

```
.
├── data_loader.py      # Dataset loading and preprocessing
├── model.py            # CNN and ResNet model architectures
├── train.py            # Training script
├── inference.py        # Inference script for images/videos
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── models/            # Saved model checkpoints (created after training)
```

## Model Architectures

### CNN Model
- 4 convolutional blocks with batch normalization
- Max pooling for downsampling
- Dropout for regularization
- Fully connected layers for classification

### ResNet Model
- Residual blocks for deeper networks
- Global average pooling
- Adaptive architecture for better feature extraction

## Training Output

The training script generates:
- **Best model checkpoint**: `models/best_model.pth`
- **Training history plot**: `training_history.png`
- **Confusion matrix**: `confusion_matrix.png`
- **Console output**: Training progress, validation metrics, and test results

## Performance Metrics

The model evaluation includes:
- Accuracy (overall and per-class)
- Precision, Recall, and F1-score
- Confusion matrix visualization
- Classification report

## Example Workflow

1. **Prepare Dataset**:
   ```bash
   # Download and extract dataset to data/leapGestRecog/
   ```

2. **Train Model**:
   ```bash
   python train.py --data_dir data/leapGestRecog --epochs 50
   ```

3. **Test on Image**:
   ```bash
   python inference.py --model_path models/best_model.pth --image test_image.jpg
   ```

4. **Real-time Recognition**:
   ```bash
   python inference.py --model_path models/best_model.pth --camera
   ```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- OpenCV for video processing
- See `requirements.txt` for complete list

## Notes

- The model expects images to be resized to 64x64 pixels
- Data augmentation is applied during training to improve generalization
- The best model is saved based on validation accuracy
- For real-time performance, consider using GPU acceleration

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size with `--batch_size 16` or `--batch_size 8`

**Issue**: Dataset not found
- **Solution**: Ensure dataset is extracted to the correct path and structure matches expected format

**Issue**: Camera not working
- **Solution**: Check camera permissions and try different `--camera_index` values

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Dataset: [LeapGestRecog on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- Built with PyTorch and OpenCV

