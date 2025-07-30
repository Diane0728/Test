# ML4CV Project Quick Start Guide

## Overview
This guide helps you set up and run a Vision-Language Model (VLM) benchmark using BLIP models for image captioning with COCO dataset evaluation.

## Prerequisites
- Python 3.8+
- GPU recommended (CUDA support)
- Internet connection for downloading models and datasets
- 4GB+ free disk space

## Quick Start Steps

### 1. Environment Setup

```bash
# Clone or create project directory
mkdir ml4cv-project
cd ml4cv-project

# Install essential packages
pip install torch torchvision transformers accelerate datasets pycocotools tqdm requests nltk scikit-learn pillow matplotlib pandas
```

### 2. Install Dependencies and Check Environment

```python
# Package installation with compatibility mode
packages_to_install = [
    "transformers>=4.40.0",
    "accelerate", 
    "datasets",
    "pycocotools",
    "tqdm",
    "requests",
    "nltk",
    "scikit-learn",
    "pillow",
    "matplotlib",
    "pandas"
]

print("Installing essential packages...")
for package in packages_to_install:
    try:
        print(f"Installing {package}...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", package])
        print(f"✓ {package} installed")
    except Exception as e:
        print(f"⚠ Failed to install {package}: {e}")

# Verify installations
import torch
import transformers
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### 3. Download Model

```python
import os

# Model configuration
MODEL_CONFIG = {
    "name": "BLIP Base",
    "hf_name": "Salesforce/blip-image-captioning-base", 
    "local_path": "./models/blip-base",
    "size": "~990MB"
}

print(f"Model will be loaded: {MODEL_CONFIG['name']}")
MODEL_PATH = MODEL_CONFIG['hf_name']  # Use HuggingFace directly
MODEL_TYPE = "blip"

# Test model loading
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    print("✓ Model accessible")
    BLIP_AVAILABLE = True
except Exception as e:
    print(f"✗ Model loading error: {e}")
    BLIP_AVAILABLE = False
```

### 4. Setup Dataset Structure

```python
import os
import json

# Create dataset directories
print("Setting up dataset structure...")
os.makedirs("./data/coco/images/val2017", exist_ok=True)
os.makedirs("./data/annotations", exist_ok=True)

# Create sample annotations for testing
sample_annotations = {
    "images": [
        {"id": 1, "file_name": "sample_001.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "sample_002.jpg", "width": 640, "height": 480},
        {"id": 3, "file_name": "sample_003.jpg", "width": 640, "height": 480}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "caption": "A sample image for testing"},
        {"id": 2, "image_id": 2, "caption": "Another sample image for demo"},
        {"id": 3, "image_id": 3, "caption": "Test image with various objects"}
    ]
}

# Save sample annotations
annotations_path = "./data/annotations/captions_val2017.json"
with open(annotations_path, 'w') as f:
    json.dump(sample_annotations, f)

print("✓ Dataset structure created")
print(f"Annotations: {annotations_path}")
print(f"Images directory: ./data/coco/images/val2017")
```

### 5. Add Sample Images

```python
import requests
from PIL import Image
import numpy as np
from io import BytesIO

def create_synthetic_image(filename, description=""):
    """Create a synthetic test image"""
    try:
        print(f"Creating {description}...")
        # Create realistic synthetic image
        img_array = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        
        # Add sky gradient (blue)
        for y in range(0, 160):
            intensity = int(100 + (y / 160) * 100)
            img_array[y, :, 0] = np.minimum(intensity - 50, 255)
            img_array[y, :, 1] = np.minimum(intensity - 20, 255) 
            img_array[y, :, 2] = np.minimum(intensity + 20, 255)
        
        # Add objects
        img_array[200:300, 100:200] = [240, 240, 240]  # White building
        img_array[350:380, 300:380] = [200, 50, 50]    # Red car
        
        # Yellow sun
        center_x, center_y = 550, 80
        for y in range(max(0, center_y-30), min(480, center_y+30)):
            for x in range(max(0, center_x-30), min(640, center_x+30)):
                if (x - center_x)**2 + (y - center_y)**2 <= 30**2:
                    img_array[y, x] = [255, 255, 100]
        
        # Save image
        image = Image.fromarray(img_array)
        image_path = os.path.join("./data/coco/images/val2017", filename)
        image.save(image_path, 'JPEG', quality=95)
        print(f"✓ Created: {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to create {description}: {e}")
        return False

# Create sample images
sample_images = [
    {"filename": "sample_001.jpg", "description": "outdoor scene"},
    {"filename": "sample_002.jpg", "description": "urban landscape"}, 
    {"filename": "sample_003.jpg", "description": "nature scene"}
]

print("Creating sample images...")
for img_info in sample_images:
    create_synthetic_image(img_info["filename"], img_info["description"])

# Verify images
images_dir = "./data/coco/images/val2017"
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"✓ Created {len(image_files)} sample images")
```

### 6. Create Benchmark Script

```python
# Create the main benchmark script
benchmark_code = '''
import torch
import time
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

class VLMBenchmark:
    def __init__(self, model_path, model_type="blip"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load BLIP model"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded on {self.device}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def generate_caption(self, image, max_length=50):
        """Generate image caption"""
        if self.model is None:
            return "Model not loaded"
        
        try:
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    do_sample=False
                )
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Generation failed"
    
    def quantize_model(self):
        """Apply INT8 quantization"""
        if self.model is None:
            return False
        try:
            print("Applying quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✓ Quantization applied")
            return True
        except Exception as e:
            print(f"Quantization failed: {e}")
            return False

def load_test_data(annotations_file, images_dir, max_samples=10):
    """Load test images and annotations"""
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        image_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann['caption'])
        
        image_files = {img['id']: img['file_name'] for img in coco_data['images']}
        
        test_data = []
        for img_id, filename in image_files.items():
            if len(test_data) >= max_samples:
                break
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path) and img_id in image_captions:
                test_data.append({
                    'id': img_id,
                    'path': img_path,
                    'filename': filename,
                    'captions': image_captions[img_id]
                })
        
        print(f"Loaded {len(test_data)} test images")
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def run_benchmark(model_path, annotations_file, images_dir, quantize=False):
    """Run the benchmark"""
    print("="*50)
    print("RUNNING VLM BENCHMARK")
    print("="*50)
    
    # Load model
    benchmark = VLMBenchmark(model_path)
    if benchmark.model is None:
        return None
    
    # Apply quantization if requested
    if quantize:
        benchmark.quantize_model()
    
    # Load test data
    test_data = load_test_data(annotations_file, images_dir, max_samples=5)
    if not test_data:
        return None
    
    # Run inference
    print(f"Processing {len(test_data)} images...")
    predictions = []
    references = []
    inference_times = []
    
    for item in tqdm(test_data, desc="Processing"):
        try:
            image = Image.open(item['path']).convert('RGB')
            
            start_time = time.time()
            prediction = benchmark.generate_caption(image)
            inference_time = time.time() - start_time
            
            predictions.append(prediction)
            references.append(item['captions'])
            inference_times.append(inference_time)
            
        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")
            continue
    
    # Calculate metrics
    if not predictions:
        return None
    
    avg_inference_time = np.mean(inference_times)
    throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    results = {
        'model_path': model_path,
        'quantized': quantize,
        'num_samples': len(predictions),
        'avg_inference_time': avg_inference_time,
        'throughput_images_per_sec': throughput,
        'predictions': predictions[:3],
        'device': str(benchmark.device)
    }
    
    # Print results
    print("\\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Device: {benchmark.device}")
    print(f"Quantized: {quantize}")
    print(f"Samples: {len(predictions)}")
    print(f"Avg inference time: {avg_inference_time:.3f} sec")
    print(f"Throughput: {throughput:.2f} images/sec")
    
    return results

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "Salesforce/blip-image-captioning-base"
    annotations_file = sys.argv[2] if len(sys.argv) > 2 else "./data/annotations/captions_val2017.json"
    images_dir = sys.argv[3] if len(sys.argv) > 3 else "./data/coco/images/val2017"
    quantize = "--quantize" in sys.argv
    
    results = run_benchmark(model_path, annotations_file, images_dir, quantize)
    
    # Save results
    output_file = "quantized_results.json" if quantize else "baseline_results.json"
    if results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nResults saved to {output_file}")
'''

with open("benchmark.py", "w") as f:
    f.write(benchmark_code)

print("✓ Benchmark script created: benchmark.py")
```

### 7. Run Baseline Benchmark

```bash
# Run FP32 baseline benchmark
python benchmark.py Salesforce/blip-image-captioning-base ./data/annotations/captions_val2017.json ./data/coco/images/val2017
```

### 8. Run Quantized Benchmark

```bash
# Run INT8 quantized benchmark
python benchmark.py Salesforce/blip-image-captioning-base ./data/annotations/captions_val2017.json ./data/coco/images/val2017 --quantize
```

### 9. Analysis and Visualization

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

def analyze_results():
    """Analyze benchmark results"""
    print("="*50)
    print("BENCHMARK ANALYSIS")
    print("="*50)
    
    results_data = {}
    
    # Load baseline results
    if os.path.exists("baseline_results.json"):
        with open("baseline_results.json", 'r') as f:
            results_data['baseline'] = json.load(f)
        print("✓ Baseline results loaded")
    
    # Load quantized results  
    if os.path.exists("quantized_results.json"):
        with open("quantized_results.json", 'r') as f:
            results_data['quantized'] = json.load(f)
        print("✓ Quantized results loaded")
    
    if not results_data:
        print("No results found. Run benchmarks first.")
        return
    
    # Extract metrics for comparison
    models = []
    inference_times = []
    throughputs = []
    
    for key, data in results_data.items():
        label = "FP32 Baseline" if key == 'baseline' else "INT8 Quantized"
        models.append(label)
        inference_times.append(data['avg_inference_time'])
        throughputs.append(data['throughput_images_per_sec'])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Inference time comparison
    ax1.bar(models, inference_times, color=['#3498db', '#e74c3c'])
    ax1.set_title('Average Inference Time')
    ax1.set_ylabel('Seconds')
    for i, v in enumerate(inference_times):
        ax1.text(i, v + max(inference_times)*0.01, f'{v:.3f}s', 
                ha='center', va='bottom', fontweight='bold')
    
    # Throughput comparison
    ax2.bar(models, throughputs, color=['#3498db', '#e74c3c'])
    ax2.set_title('Throughput')
    ax2.set_ylabel('Images/sec')
    for i, v in enumerate(throughputs):
        ax2.text(i, v + max(throughputs)*0.01, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Performance summary
    if len(models) == 2:
        print("\\nPerformance Comparison:")
        print(f"FP32 Baseline: {inference_times[0]:.3f}s per image")
        print(f"INT8 Quantized: {inference_times[1]:.3f}s per image")
        speedup = inference_times[0] / inference_times[1] if inference_times[1] > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        print(f"\\nThroughput Comparison:")
        print(f"FP32 Baseline: {throughputs[0]:.2f} images/sec")
        print(f"INT8 Quantized: {throughputs[1]:.2f} images/sec")
        throughput_gain = throughputs[1] / throughputs[0] if throughputs[0] > 0 else 0
        print(f"Throughput gain: {throughput_gain:.2f}x")
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Model': models,
        'Inference Time (s)': [f"{t:.3f}" for t in inference_times],
        'Throughput (img/s)': [f"{t:.2f}" for t in throughputs]
    })
    
    print("\\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('benchmark_summary.csv', index=False)
    print("\\n✓ Analysis complete!")
    print("✓ Visualization saved: benchmark_comparison.png") 
    print("✓ Summary saved: benchmark_summary.csv")

# Run analysis
analyze_results()
```

### 10. Test Image Description (Interactive)

```python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

def test_image_description():
    """Interactive image description test"""
    print("="*50)
    print("IMAGE DESCRIPTION TEST")
    print("="*50)
    
    # Load model
    MODEL_PATH = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded on: {device}")
    
    # Test on sample images
    images_dir = "./data/coco/images/val2017"
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, filename in enumerate(image_files[:3]):  # Test first 3 images
        image_path = os.path.join(images_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        # Generate description
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=50, num_beams=4)
        description = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Display result
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"File: {filename}\\nDescription: '{description}'", fontsize=12, wrap=True)
        plt.tight_layout()
        plt.show()
        
        print(f"{i+1}. {filename}: '{description}'")
    
    print("\\n✓ Image description test complete!")

# Run interactive test
test_image_description()
```

## Project Structure

After running all steps, your project structure will be:

```
ml4cv-project/
├── data/
│   ├── coco/
│   │   └── images/
│   │       └── val2017/
│   │           ├── sample_001.jpg
│   │           ├── sample_002.jpg
│   │           └── sample_003.jpg
│   └── annotations/
│       └── captions_val2017.json
├── models/
│   └── blip-base/ (optional local model)
├── benchmark.py
├── baseline_results.json
├── quantized_results.json
├── benchmark_comparison.png
└── benchmark_summary.csv
```

## Expected Results

- **Baseline FP32**: ~0.5-1.0 sec per image
- **Quantized INT8**: Potential 1.2-2x speedup
- **Generated captions**: Natural language descriptions of images
- **Visualizations**: Performance comparison charts

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection
3. **Import errors**: Verify all packages installed correctly

### Solutions:

```python
# Force CPU usage if GPU issues
device = "cpu"
model = model.to(device)

# Reduce image size for memory
image.thumbnail((256, 256), Image.Resampling.LANCZOS)

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Next Steps

1. **Add real COCO dataset** for full evaluation
2. **Test different models** (BLIP-2, GIT, etc.)
3. **Implement BLEU scoring** for caption quality
4. **Add more quantization methods** (FP16, dynamic quantization)
5. **Optimize for deployment** (TensorRT, ONNX)

## Performance Targets

- **Target throughput**: ~13.48 tokens/sec
- **Acceptable latency**: <2 seconds per image  
- **Memory usage**: <4GB GPU memory
- **Model size**: <1GB for deployment

This guide provides a complete workflow from setup to analysis for vision-language model benchmarking!