#!/usr/bin/env python3
"""
ML4CV VLM Benchmark - Setup Script for Google Colab
This script sets up the environment and downloads necessary components
"""

import os
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path


def install_packages():
    """Install essential packages with compatibility mode"""
    packages_to_install = [
        "transformers>=4.40.0",  # Use recent version compatible with existing
        "accelerate",  # Usually safe to install
        "datasets",  # For dataset handling
        "pycocotools",  # For COCO dataset
        "tqdm",  # Progress bars
        "requests",  # For downloads
        "nltk",  # For custom BLEU calculation (fallback)
        "scikit-learn",  # For additional metrics
    ]
    
    print("Installing essential packages (compatibility mode)...")
    for package in packages_to_install:
        try:
            print(f"Installing {package}...")
            result = subprocess.run(['pip', 'install', '-q', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {package} installed")
            else:
                print(f"⚠ Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"⚠ Failed to install {package}: {e}")
    
    # Try to install evaluate with specific handling
    print("\nTrying to install evaluate library...")
    try:
        subprocess.run(['pip', 'install', '-q', 'evaluate'], check=True)
        import evaluate
        print("✓ Evaluate library installed and imported successfully")
        return True
    except Exception as e:
        print(f"⚠ Evaluate installation/import failed: {e}")
        print("Will use custom BLEU implementation as fallback")
        return False


def check_environment():
    """Check the current environment"""
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("✗ PyTorch not available")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("✓ BLIP models available")
        return True
    except ImportError:
        print("✗ BLIP models not available")
        return False


def setup_coco_dataset():
    """Set up COCO dataset structure"""
    print("\nSetting up COCO dataset structure...")
    
    # Create directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/coco/images/val2017", exist_ok=True)
    os.makedirs("./data/annotations", exist_ok=True)
    
    # Create sample annotation file for testing
    sample_annotations = {
        "images": [
            {"id": 1, "file_name": "sample_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "sample_002.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "caption": "A sample image for testing"},
            {"id": 2, "image_id": 2, "caption": "Another sample image for demo"}
        ]
    }
    
    # Create sample annotation file
    sample_annotations_path = "./data/annotations/captions_val2017.json"
    with open(sample_annotations_path, 'w') as f:
        json.dump(sample_annotations, f)
    print("✓ Created sample annotations for testing")
    
    # Set paths
    coco_images_dir = "./data/coco/images/val2017"
    coco_annotations_file = "./data/annotations/captions_val2017.json"
    
    print(f"\nPaths configured:")
    print(f"Images: {coco_images_dir}")
    print(f"Annotations: {coco_annotations_file}")
    
    return coco_images_dir, coco_annotations_file


def download_model():
    """Download or configure model path"""
    model_configs = [
        {
            "name": "BLIP Base (Small)",
            "hf_name": "Salesforce/blip-image-captioning-base",
            "local_path": "./models/blip-base",
            "size": "~990MB",
            "type": "blip"
        }
    ]
    
    selected_model = model_configs[0]  # BLIP Base
    print(f"\nUsing {selected_model['name']}...")
    
    # For Colab, use HuggingFace model directly (no local download needed)
    MODEL_PATH = selected_model['hf_name']
    MODEL_TYPE = selected_model['type']
    
    print(f"✓ Model configured: {MODEL_PATH}")
    return MODEL_PATH, MODEL_TYPE


def main():
    """Main setup function"""
    print("="*60)
    print("ML4CV VLM BENCHMARK SETUP")
    print("="*60)
    
    # Install packages
    evaluate_available = install_packages()
    
    # Check environment
    blip_available = check_environment()
    
    # Setup COCO dataset
    coco_images_dir, coco_annotations_file = setup_coco_dataset()
    
    # Download/configure model
    model_path, model_type = download_model()
    
    # Create configuration file
    config = {
        "model_path": model_path,
        "model_type": model_type,
        "coco_images_dir": coco_images_dir,
        "coco_annotations_file": coco_annotations_file,
        "evaluate_available": evaluate_available,
        "blip_available": blip_available
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nConfiguration saved to config.json")
    print("Next steps:")
    print("1. Run benchmark.py for performance testing")
    print("2. Run quantization.py for model optimization")
    print("3. Run analysis.py for result visualization")


if __name__ == "__main__":
    main()
