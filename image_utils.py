#!/usr/bin/env python3
"""
ML4CV VLM Benchmark - Image Utilities
Handles image downloading, creation, and testing
"""

import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import json
import time
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt


def download_sample_image(url, filename, description="", target_dir="./data/coco/images/val2017"):
    """Download a sample image from URL"""
    try:
        print(f"  Downloading {description}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Open and save image
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure RGB format
        
        # Save to target directory
        os.makedirs(target_dir, exist_ok=True)
        image_path = os.path.join(target_dir, filename)
        image.save(image_path, 'JPEG', quality=95)
        
        print(f"    ✓ Saved: {filename} ({image.size[0]}x{image.size[1]})")
        return True
    except Exception as e:
        print(f"    ❌ Failed to download {description}: {e}")
        return False


def create_synthetic_image(filename, description="", target_dir="./data/coco/images/val2017"):
    """Create a synthetic test image"""
    try:
        print(f"  Creating synthetic {description}...")
        
        # Create a more realistic synthetic image
        img_array = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        
        # Add some patterns to make it more interesting
        # Sky gradient (blue)
        for y in range(0, 160):
            intensity = int(100 + (y / 160) * 100)
            img_array[y, :, 0] = np.minimum(intensity - 50, 255)  # Less red
            img_array[y, :, 1] = np.minimum(intensity - 20, 255)  # Less green
            img_array[y, :, 2] = np.minimum(intensity + 20, 255)  # More blue
        
        # Ground (brown/green)
        for y in range(320, 480):
            img_array[y, :, 0] = np.random.randint(60, 100)  # Brown-red
            img_array[y, :, 1] = np.random.randint(80, 120)  # Brown-green
            img_array[y, :, 2] = np.random.randint(40, 80)   # Brown-blue
        
        # Add some objects
        # White rectangle (building)
        img_array[200:300, 100:200] = [240, 240, 240]
        
        # Red rectangle (car)
        img_array[350:380, 300:380] = [200, 50, 50]
        
        # Yellow circle (sun)
        center_x, center_y = 550, 80
        for y in range(max(0, center_y-30), min(480, center_y+30)):
            for x in range(max(0, center_x-30), min(640, center_x+30)):
                if (x - center_x)**2 + (y - center_y)**2 <= 30**2:
                    img_array[y, x] = [255, 255, 100]
        
        # Convert to PIL Image and save
        image = Image.fromarray(img_array)
        os.makedirs(target_dir, exist_ok=True)
        image_path = os.path.join(target_dir, filename)
        image.save(image_path, 'JPEG', quality=95)
        
        print(f"    ✓ Created: {filename} ({image.size[0]}x{image.size[1]})")
        return True
    except Exception as e:
        print(f"    ❌ Failed to create synthetic {description}: {e}")
        return False


def populate_sample_images(target_dir="./data/coco/images/val2017", num_real=4, num_synthetic=3):
    """Populate the COCO directory with sample images"""
    print("=" * 60)
    print("ADDING SAMPLE IMAGES TO COCO DIRECTORY")
    print("=" * 60)
    
    # Create the COCO images directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    print(f"  Created directory: {target_dir}")
    
    # Sample images from Unsplash (free stock photos)
    sample_images = [
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop",
            "filename": "sample_mountain.jpg",
            "description": "mountain landscape"
        },
        {
            "url": "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=640&h=480&fit=crop",
            "filename": "sample_breakfast.jpg",
            "description": "breakfast scene"
        },
        {
            "url": "https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=640&h=480&fit=crop",
            "filename": "sample_dog.jpg",
            "description": "dog portrait"
        },
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop",
            "filename": "sample_nature.jpg",
            "description": "nature scene"
        }
    ]
    
    print("  Attempting to download sample images from Unsplash...")
    
    # Try to download real images first
    downloaded_count = 0
    for img_info in sample_images[:num_real]:
        if download_sample_image(img_info["url"], img_info["filename"], 
                                img_info["description"], target_dir):
            downloaded_count += 1
    
    print(f"\n  Downloaded {downloaded_count} real images")
    
    # Create synthetic images to fill remaining slots
    synthetic_images = [
        {"filename": "synthetic_scene1.jpg", "description": "outdoor scene"},
        {"filename": "synthetic_scene2.jpg", "description": "urban landscape"},
        {"filename": "synthetic_scene3.jpg", "description": "abstract composition"}
    ]
    
    print(f"\n  Creating synthetic images...")
    synthetic_count = 0
    for img_info in synthetic_images[:num_synthetic]:
        if create_synthetic_image(img_info["filename"], img_info["description"], target_dir):
            synthetic_count += 1
    
    print(f"\n  Created {synthetic_count} synthetic images")
    
    # Verify final directory contents
    print(f"\n  Final COCO directory contents:")
    if os.path.exists(target_dir):
        image_files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        
        if total_images > 0:
            print(f"    Total images: {total_images}")
            
            # Show details of each image
            for i, filename in enumerate(sorted(image_files)[:10]):  # Show first 10
                filepath = os.path.join(target_dir, filename)
                try:
                    img = Image.open(filepath)
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"    {i+1}. {filename} - {img.size[0]}x{img.size[1]} ({size_kb:.1f} KB)")
                except Exception as e:
                    print(f"    {i+1}. {filename} - Error reading file: {e}")
            
            if total_images > 10:
                print(f"    ... and {total_images - 10} more images")
            
            print(f"\n  ✓ COCO directory is ready with {total_images} test images!")
            print(f"    Directory path: {target_dir}")
            return target_dir, total_images
        else:
            print("    ❌ No images were successfully added")
            return target_dir, 0
    else:
        print("    ❌ COCO directory was not created")
        return None, 0


def load_model_for_testing(model_path):
    """Load the BLIP model for image description"""
    try:
        print("  Loading BLIP model...")
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"  Model loaded on: {device}")
        
        return processor, model, device
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None, None


def generate_description_with_display(image_path, processor, model, device, custom_prompt=None):
    """Generate description and display image with caption"""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Generate description
        if custom_prompt:
            inputs = processor(image, custom_prompt, return_tensors="pt").to(device)
            prompt_text = f" (with prompt: '{custom_prompt}')"
        else:
            inputs = processor(image, return_tensors="pt").to(device)
            prompt_text = ""
        
        # Time the inference
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=50, num_beams=4)
        end_time = time.time()
        
        description = processor.decode(generated_ids[0], skip_special_tokens=True)
        inference_time = end_time - start_time
        
        # Display image with caption
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        
        # Create title with filename and description
        filename = os.path.basename(image_path)
        title = f"File: {filename}\nDescription: '{description}'{prompt_text}\nInference time: {inference_time:.3f}s"
        plt.title(title, fontsize=11, wrap=True, pad=20)
        plt.tight_layout()
        plt.show()
        
        return description, inference_time
    
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None, None


def test_image_descriptions(model_path, images_dir="./data/coco/images/val2017", max_images=5):
    """Test image description generation with visualization"""
    print("=" * 60)
    print("COMPLETE IMAGE DESCRIPTION TEST")
    print("=" * 60)
    
    # Load model
    processor, model, device = load_model_for_testing(model_path)
    if processor is None:
        print("  ❌ Failed to load model. Cannot proceed with testing.")
        return None
    
    # Check for available images
    print(f"\n  Checking for images in: {images_dir}")
    if not os.path.exists(images_dir):
        print(f"    ❌ COCO images directory not found: {images_dir}")
        print("    Run populate_sample_images() first to create and populate the directory")
        return None
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("    ❌ No images found in COCO directory")
        print("    Run populate_sample_images() first to populate the directory")
        return None
    
    print(f"    Found {len(image_files)} images")
    
    # Test results storage
    test_results = {
        "test_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "device": device,
            "total_images_tested": 0
        },
        "results": []
    }
    
    print(f"\n  TESTING IMAGE DESCRIPTIONS")
    print("=" * 50)
    
    # Test first few images (max to avoid overwhelming output)
    test_count = min(max_images, len(image_files))
    total_inference_time = 0
    
    for i, filename in enumerate(sorted(image_files)[:test_count]):
        image_path = os.path.join(images_dir, filename)
        print(f"\n{i+1}. Testing: {filename}")
        print("-" * 30)
        
        # Generate basic description
        description, inference_time = generate_description_with_display(
            image_path, processor, model, device
        )
        
        if description and inference_time:
            total_inference_time += inference_time
            
            # Store result
            test_results["results"].append({
                "filename": filename,
                "description": description,
                "inference_time": inference_time,
                "image_path": image_path
            })
            
            print(f"  Description: '{description}'")
            print(f"  Inference time: {inference_time:.3f}s")
            
            # Test with custom prompt for first image
            if i == 0:
                print(f"\n  Testing with custom prompt...")
                custom_description, custom_time = generate_description_with_display(
                    image_path, processor, model, device, "A detailed photo of"
                )
                
                if custom_description and custom_time:
                    test_results["results"][-1]["custom_prompt_description"] = custom_description
                    test_results["results"][-1]["custom_prompt_time"] = custom_time
                    print(f"  With prompt: '{custom_description}'")
                    print(f"  Prompt inference time: {custom_time:.3f}s")
    
    # Calculate summary statistics
    test_results["test_info"]["total_images_tested"] = len(test_results["results"])
    test_results["test_info"]["total_inference_time"] = total_inference_time
    test_results["test_info"]["avg_inference_time"] = total_inference_time / max(1, len(test_results["results"]))
    test_results["test_info"]["throughput_images_per_second"] = len(test_results["results"]) / max(0.001, total_inference_time)
    
    # Display summary
    print(f"\n  TEST SUMMARY")
    print("=" * 40)
    print(f"  Images tested: {test_results['test_info']['total_images_tested']}")
    print(f"  Total time: {total_inference_time:.3f}s")
    print(f"  Average time per image: {test_results['test_info']['avg_inference_time']:.3f}s")
    print(f"  Throughput: {test_results['test_info']['throughput_images_per_second']:.2f} images/sec")
    print(f"  Device used: {device}")
    
    # Save detailed results
    results_file = "image_description_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"  Detailed results saved: {results_file}")
    
    # Create summary table
    print(f"\n  RESULTS TABLE")
    print("=" * 80)
    print(f"{'#':<3} {'Filename':<25} {'Inference Time':<15} {'Description':<35}")
    print("-" * 80)
    
    for i, result in enumerate(test_results["results"]):
        filename = result["filename"][:24]  # Truncate long filenames
        inf_time = f"{result['inference_time']:.3f}s"
        description = result["description"][:34]  # Truncate long descriptions
        print(f"{i+1:<3} {filename:<25} {inf_time:<15} {description:<35}")
    
    # Show remaining images available for testing
    if len(image_files) > test_count:
        print(f"\n  {len(image_files) - test_count} more images available for testing:")
        for filename in sorted(image_files)[test_count:test_count+3]:
            print(f"    • {filename}")
        if len(image_files) > test_count + 3:
            print(f"    ... and {len(image_files) - test_count - 3} more")
    
    print(f"\n  ✓ Image description testing complete!")
    print(f"    Test images directory: {images_dir}")
    print(f"    Results file: {results_file}")
    
    return test_results


def main():
    """Main function for image utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Utilities for VLM Benchmark")
    parser.add_argument("--populate-images", action="store_true", 
                       help="Populate sample images in COCO directory")
    parser.add_argument("--test-descriptions", action="store_true", 
                       help="Test image description generation")
    parser.add_argument("--model-path", default="Salesforce/blip-image-captioning-base", 
                       help="Model path for testing")
    parser.add_argument("--images-dir", default="./data/coco/images/val2017", 
                       help="Directory containing test images")
    parser.add_argument("--max-images", type=int, default=5, 
                       help="Maximum number of images to test")
    
    args = parser.parse_args()
    
    if args.populate_images:
        populate_sample_images(args.images_dir)
    
    if args.test_descriptions:
        test_image_descriptions(args.model_path, args.images_dir, args.max_images)
    
    if not args.populate_images and not args.test_descriptions:
        print("Use --populate-images to add sample images or --test-descriptions to test model")


if __name__ == "__main__":
    main()
