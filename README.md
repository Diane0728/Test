# ML4CV VLM Benchmark

A comprehensive benchmarking suite for Vision-Language Models (VLMs) designed for the ML4CV course. This project evaluates model performance with different quantization techniques and provides detailed analysis of inference speed, memory usage, and quality metrics.

## Features

- **Multiple Model Support**: BLIP, GIT, and other Hugging Face VLMs
- **Quantization Benchmarks**: INT8, FP16, and FP32 precision comparisons
- **COCO Dataset Integration**: Automated dataset setup and evaluation
- **Performance Analysis**: Comprehensive metrics and visualizations
- **Google Colab Optimized**: Designed to work seamlessly in Colab environments
- **Robust Error Handling**: Fallback mechanisms for different environments

## Project Structure

```
ml4cv-vlm-benchmark/
├── setup.py                    # Environment setup and initialization
├── benchmark.py                # Main benchmarking script
├── quantization.py            # Quantization-specific benchmarks
├── analysis.py                # Results analysis and visualization
├── image_utils.py             # Image handling and testing utilities
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── README.md                 # This file
└── data/                     # Generated data directory
    ├── coco/
    │   └── images/val2017/   # Sample images
    └── annotations/          # COCO annotations
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ml4cv-vlm-benchmark.git
cd ml4cv-vlm-benchmark

# Install dependencies
pip install -r requirements.txt

# Run initial setup
python setup.py
```

### 2. Run Benchmarks

```bash
# Run all benchmarks (FP32 baseline + INT8 quantization)
python quantization.py --run-all

# Or run specific benchmarks
python quantization.py --run-baseline  # FP32 only
python quantization.py --run-int8      # INT8 only
python quantization.py --run-fp16      # FP16 only
```

### 3. Analyze Results

```bash
# Generate analysis and visualizations
python analysis.py

# Custom analysis
python analysis.py --baseline-file baseline_fp32.json --quantized-file quantized_int8.json
```

### 4. Test with Images

```bash
# Populate sample images
python image_utils.py --populate-images

# Test image descriptions
python image_utils.py --test-descriptions --max-images 5
```

## Google Colab Usage

For Google Colab, you can run the entire pipeline in a single notebook:

```python
# Install and setup
!git clone https://github.com/yourusername/ml4cv-vlm-benchmark.git
%cd ml4cv-vlm-benchmark
!pip install -r requirements.txt
!python setup.py

# Run benchmarks
!python quantization.py --run-all --num-samples 10

# Analyze results
!python analysis.py

# Test with images
!python image_utils.py --populate-images --test-descriptions
```

## Command Line Interface

### Setup Script
```bash
python setup.py
```
- Downloads and configures BLIP model
- Sets up COCO dataset structure
- Creates sample annotations
- Generates configuration files

### Benchmark Script
```bash
python benchmark.py --model-path MODEL --annotations ANNOTATIONS --images IMAGES [OPTIONS]
```

Options:
- `--model-path`: Path to model (default: Salesforce/blip-image-captioning-base)
- `--model-type`: Model type (blip, git)
- `--annotations`: Path to COCO annotations file
- `--images`: Path to images directory
- `--num-samples`: Number of samples to test (default: 30)
- `--quantize`: Apply INT8 quantization
- `--output`: Output JSON file

### Quantization Script
```bash
python quantization.py [OPTIONS]
```

Options:
- `--model-name`: Model name or path
- `--num-samples`: Number of test samples (default: 5)
- `--run-baseline`: Run FP32 baseline
- `--run-int8`: Run INT8 quantization
- `--run-fp16`: Run FP16 precision
- `--run-all`: Run all benchmarks

### Analysis Script
```bash
python analysis.py [OPTIONS]
```

Options:
- `--baseline-file`: Baseline results file (default: baseline_fp32.json)
- `--quantized-file`: Quantized results file (default: quantized_int8.json)
- `--output-viz`: Visualization output file
- `--output-csv`: CSV summary output file

### Image Utils Script
```bash
python image_utils.py [OPTIONS]
```

Options:
- `--populate-images`: Download/create sample images
- `--test-descriptions`: Test image description generation
- `--model-path`: Model path for testing
- `--images-dir`: Images directory
- `--max-images`: Maximum images to test

## Output Files

The benchmark generates several output files:

- `baseline_fp32.json`: FP32 baseline results
- `quantized_int8.json`: INT8 quantization results
- `quantized_fp16.json`: FP16 precision results
- `benchmark_results_analysis.png`: Performance visualization
- `benchmark_summary.csv`: Summary table
- `config.json`: Configuration file
- `image_description_detailed_results.json`: Image testing results

## Performance Metrics

The benchmark evaluates:

- **Inference Speed**: Images per second, tokens per second
- **Memory Usage**: Peak memory consumption
- **Model Size**: Quantized vs original model size
- **Quality**: BLEU score for caption quality
- **Latency**: Average inference time per image

## Model Support

Currently supported models:
- **BLIP**: Salesforce/blip-image-captioning-base
- **BLIP Large**: Salesforce/blip-image-captioning-large
- **GIT**: microsoft/git-base, microsoft/git-large

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM
- 2GB+ disk space for models and data

## Quantization Methods

### INT8 Quantization (CPU)
- Uses `torch.quantization.quantize_dynamic`
- Reduces model size by ~75%
- Runs on CPU for compatibility

### FP16 Precision (GPU)
- Half-precision floating point
- Reduces memory usage by ~50%
- Faster inference on modern GPUs

## Performance Targets

The benchmark compares against a target of **13.48 tokens/second**, equivalent to approximately **0.90 images/second** for average captions.

## Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# If model download fails, try:
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
python setup.py
```

**CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
python quantization.py --run-int8  # Uses CPU
```

**Missing Dependencies**
```bash
# Install specific packages
pip install torch torchvision transformers accelerate
```

**COCO Dataset Issues**
```bash
# Recreate sample data
python image_utils.py --populate-images
```

### Google Colab Specific

**Runtime Disconnection**
- Save intermediate results frequently
- Use smaller sample sizes for testing
- Enable GPU runtime for better performance

**Memory Management**
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{ml4cv-vlm-benchmark,
  title={ML4CV VLM Benchmark: A Comprehensive Evaluation Suite for Vision-Language Models},
  author={ML4CV Course Contributors},
  year={2024},
  url={https://github.com/yourusername/ml4cv-vlm-benchmark}
}
```

## Acknowledgments

- Hugging Face Transformers library
- COCO Dataset
- Salesforce BLIP model
- Google Colab platform

## Contact

For questions or issues, please open a GitHub issue or contact the course instructors.

---

## Example Usage

Complete example workflow:

```python
# 1. Setup
!python setup.py

# 2. Populate images
!python image_utils.py --populate-images

# 3. Run benchmarks
!python quantization.py --run-all --num-samples 10

# 4. Analyze results
!python analysis.py

# 5. Test specific images
!python image_utils.py --test-descriptions --max-images 3
```

This will generate a complete performance analysis with visualizations and detailed metrics for your VLM evaluation.
