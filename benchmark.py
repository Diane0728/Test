#!/usr/bin/env python3
"""
ML4CV VLM Benchmark - Main Benchmark Script
Runs performance benchmarks on Vision-Language Models
"""

import torch
import time
import json
import os
import argparse
import gc
from PIL import Image
import numpy as np
from tqdm import tqdm

# Handle different model types
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    from transformers import GitProcessor, GitForCausalLM
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


def custom_bleu_score(predictions, references, max_n=4):
    """Custom BLEU score implementation as fallback for evaluate library"""
    import math
    from collections import Counter
    
    if not predictions or not references:
        return 0.0
    
    # Handle case where references is list of lists
    if isinstance(references[0], list):
        # Use first reference for each prediction
        ref_texts = [refs[0] if refs else "" for refs in references]
    else:
        ref_texts = references
    
    if len(predictions) != len(ref_texts):
        print(f"Warning: predictions ({len(predictions)}) and references ({len(ref_texts)}) length mismatch")
        min_len = min(len(predictions), len(ref_texts))
        predictions = predictions[:min_len]
        ref_texts = ref_texts[:min_len]
    
    def tokenize(text):
        """Simple tokenization"""
        return text.lower().split()
    
    def compute_ngrams(tokens, n):
        """Generate n-grams from tokens"""
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    # Calculate BLEU components
    total_precision = []
    for n in range(1, max_n + 1):
        matches = 0
        total_pred_ngrams = 0
        
        for pred, ref in zip(predictions, ref_texts):
            pred_tokens = tokenize(str(pred))
            ref_tokens = tokenize(str(ref))
            
            pred_ngrams = compute_ngrams(pred_tokens, n)
            ref_ngrams = compute_ngrams(ref_tokens, n)
            
            # Count matches
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            total_pred_ngrams += sum(pred_ngrams.values())
        
        if total_pred_ngrams == 0:
            precision = 0.0
        else:
            precision = matches / total_pred_ngrams
        total_precision.append(precision)
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in total_precision):
        return 0.0
    
    # Geometric mean
    bleu = math.exp(sum(math.log(p) for p in total_precision) / len(total_precision))
    
    # Apply brevity penalty (simplified)
    pred_length = sum(len(tokenize(str(pred))) for pred in predictions)
    ref_length = sum(len(tokenize(str(ref))) for ref in ref_texts)
    
    if pred_length == 0:
        return 0.0
    
    if pred_length >= ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_length / pred_length)
    
    return bleu * brevity_penalty


def simple_similarity_score(predictions, references):
    """Simple word overlap similarity as ultimate fallback"""
    if not predictions or not references:
        return 0.0
    
    total_similarity = 0
    count = 0
    
    for pred, refs in zip(predictions, references):
        if isinstance(refs, list):
            ref = refs[0] if refs else ""
        else:
            ref = refs
        
        pred_words = set(str(pred).lower().split())
        ref_words = set(str(ref).lower().split())
        
        if len(pred_words) == 0 and len(ref_words) == 0:
            similarity = 1.0
        elif len(pred_words) == 0 or len(ref_words) == 0:
            similarity = 0.0
        else:
            intersection = len(pred_words.intersection(ref_words))
            union = len(pred_words.union(ref_words))
            similarity = intersection / union if union > 0 else 0.0
        
        total_similarity += similarity
        count += 1
    
    return total_similarity / count if count > 0 else 0.0


class ColabVLMBenchmark:
    def __init__(self, model_path, model_type="blip"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_type} model...")
        print(f"Device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load model with error handling"""
        try:
            if self.model_type == "blip" and BLIP_AVAILABLE:
                self.processor = BlipProcessor.from_pretrained(self.model_path)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.model.to(self.device)
            elif self.model_type == "git" and GIT_AVAILABLE:
                self.processor = GitProcessor.from_pretrained(self.model_path)
                self.model = GitForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.model.to(self.device)
            else:
                raise ValueError(f"Model type {self.model_type} not supported or not available")
            
            self.model.eval()
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
            self.processor = None
    
    def generate_caption(self, image, max_length=50):
        """Generate caption with robust error handling"""
        if self.model is None:
            return "Model not loaded"
        
        try:
            # Resize image if too large (memory optimization)
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            if self.model_type == "blip":
                # BLIP uses conditional generation
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=3,
                        temperature=1.0,
                        do_sample=False
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            elif self.model_type == "git":
                # Git model approach
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        max_length=max_length,
                        num_beams=3
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return caption.strip()
        
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Caption generation failed"
    
    def quantize_model(self):
        """Apply quantization if possible"""
        if self.model is None:
            return False
        
        try:
            print("Applying dynamic quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✓ Quantization applied")
            return True
        except Exception as e:
            print(f"Quantization failed: {e}")
            return False
    
    def get_model_size(self):
        """Calculate model size in MB"""
        if self.model is None:
            return 0
        total_params = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_params / (1024 ** 2)


def load_sample_data(annotations_file, images_dir, max_samples=50):
    """Load sample data with robust error handling"""
    try:
        print(f"Loading annotations from {annotations_file}")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image to captions mapping
        image_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann['caption'])
        
        # Get image filenames
        image_files = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Select available images
        sample_data = []
        for img_id, filename in image_files.items():
            if len(sample_data) >= max_samples:
                break
            
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path) and img_id in image_captions:
                sample_data.append({
                    'id': img_id,
                    'path': img_path,
                    'filename': filename,
                    'captions': image_captions[img_id]
                })
        
        print(f"Loaded {len(sample_data)} images for evaluation")
        return sample_data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def calculate_bleu_score(predictions, references):
    """Calculate BLEU score with multiple fallback options"""
    # First try: HuggingFace evaluate library
    if EVALUATE_AVAILABLE:
        try:
            bleu = evaluate.load("bleu")
            result = bleu.compute(predictions=predictions, references=references)
            print("✓ Using HuggingFace evaluate BLEU")
            return result['bleu']
        except Exception as e:
            print(f"HuggingFace evaluate failed: {e}")
    
    # Second try: NLTK BLEU
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu
        
        # Convert to NLTK format
        references_nltk = []
        predictions_nltk = []
        
        for pred, refs in zip(predictions, references):
            if isinstance(refs, list):
                # Multiple references per prediction
                refs_tokens = [ref.split() for ref in refs]
            else:
                # Single reference
                refs_tokens = [refs.split()]
            
            references_nltk.append(refs_tokens)
            predictions_nltk.append(pred.split())
        
        bleu_score = corpus_bleu(references_nltk, predictions_nltk)
        print("✓ Using NLTK BLEU")
        return bleu_score
    except Exception as e:
        print(f"NLTK BLEU failed: {e}")
    
    # Third try: Custom BLEU implementation
    try:
        score = custom_bleu_score(predictions, references)
        print("✓ Using custom BLEU implementation")
        return score
    except Exception as e:
        print(f"Custom BLEU failed: {e}")
    
    # Final fallback: Simple similarity
    try:
        score = simple_similarity_score(predictions, references)
        print("✓ Using simple similarity metric")
        return score
    except Exception as e:
        print(f"All evaluation methods failed: {e}")
        return 0.0


def run_benchmark(args):
    """Main benchmark function"""
    print("="*60)
    print("STARTING VLM BENCHMARK")
    print("="*60)
    
    # Load model
    benchmark = ColabVLMBenchmark(args.model_path, args.model_type)
    if benchmark.model is None:
        print("Failed to load model, exiting")
        return
    
    # Apply quantization if requested
    if args.quantize:
        benchmark.quantize_model()
    
    # Load test data
    test_data = load_sample_data(args.annotations, args.images, args.num_samples)
    if not test_data:
        print("No test data available, exiting")
        return
    
    # Run inference
    print(f"Running inference on {len(test_data)} images...")
    predictions = []
    references = []
    inference_times = []
    
    for i, item in enumerate(tqdm(test_data, desc="Processing images")):
        try:
            # Load image
            image = Image.open(item['path']).convert('RGB')
            
            # Generate caption with timing
            start_time = time.time()
            prediction = benchmark.generate_caption(image)
            inference_time = time.time() - start_time
            
            predictions.append(prediction)
            references.append(item['captions'])
            inference_times.append(inference_time)
            
            # Memory cleanup every 10 images
            if (i + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Show progress
            if len(inference_times) >= 10:
                avg_time = np.mean(inference_times[-10:])
                print(f"Processed {i+1}/{len(test_data)}, avg time: {avg_time:.3f}s")
        
        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")
            continue
    
    # Calculate metrics
    if not predictions:
        print("No predictions generated")
        return
    
    avg_inference_time = np.mean(inference_times)
    throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    model_size_mb = benchmark.get_model_size()
    bleu_score = calculate_bleu_score(predictions, references)
    
    # Results
    results = {
        'model_path': args.model_path,
        'model_type': args.model_type,
        'quantized': args.quantize,
        'num_samples': len(predictions),
        'avg_inference_time': avg_inference_time,
        'throughput_images_per_sec': throughput,
        'model_size_mb': model_size_mb,
        'bleu_score': bleu_score,
        'sample_predictions': predictions[:3],
        'device': str(benchmark.device)
    }
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Path: {args.model_path}")
    print(f"Device: {benchmark.device}")
    print(f"Quantized: {args.quantize}")
    print(f"Samples: {len(predictions)}")
    print(f"Avg inference time: {avg_inference_time:.3f} sec/image")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"BLEU score: {bleu_score:.4f}")
    
    # Performance analysis
    TARGET_TOKENS_PER_SEC = 13.48
    AVG_TOKENS_PER_CAPTION = 15  # Typical caption length
    target_images_per_sec = TARGET_TOKENS_PER_SEC / AVG_TOKENS_PER_CAPTION
    performance_ratio = throughput / target_images_per_sec
    
    print(f"\nPerformance Analysis:")
    print(f"Target performance: {target_images_per_sec:.3f} images/sec")
    print(f"Actual performance: {throughput:.3f} images/sec")
    print(f"Performance ratio: {performance_ratio:.2f}x")
    print(f"Meets target: {'✓' if performance_ratio >= 1.0 else '✗'}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Colab VLM Benchmark")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--model-type", default="blip", help="Model type (blip, git)")
    parser.add_argument("--annotations", required=True, help="Path to annotations file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of samples to test")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
