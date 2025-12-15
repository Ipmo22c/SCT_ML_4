"""
Helper script to download the LeapGestRecog dataset from Kaggle.
Requires Kaggle API credentials.
"""

import os
import zipfile
import argparse
from pathlib import Path


def download_kaggle_dataset(dataset_name: str = "gti-upm/leapgestrecog", 
                           output_dir: str = "data"):
    """
    Download dataset from Kaggle using Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset name (format: username/dataset)
        output_dir: Directory to save the dataset
    """
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed. Installing...")
        os.system("pip install kaggle")
        import kaggle
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
    
    print(f"\nDataset downloaded successfully to {output_dir}")
    print("\nNote: Make sure your Kaggle API credentials are set up.")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials")


def manual_download_instructions():
    """Print manual download instructions."""
    print("="*60)
    print("Manual Download Instructions")
    print("="*60)
    print("\n1. Go to: https://www.kaggle.com/datasets/gti-upm/leapgestrecog")
    print("2. Click 'Download' button")
    print("3. Extract the zip file to 'data/leapGestRecog/' directory")
    print("4. Ensure the structure is:")
    print("   data/leapGestRecog/")
    print("   ├── gesture_01/")
    print("   ├── gesture_02/")
    print("   └── ...")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Download LeapGestRecog dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save dataset')
    parser.add_argument('--manual', action='store_true',
                       help='Show manual download instructions')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_download_instructions()
    else:
        try:
            download_kaggle_dataset(output_dir=args.output_dir)
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("\nFalling back to manual download instructions...")
            manual_download_instructions()


if __name__ == '__main__':
    main()

