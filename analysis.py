#!/usr/bin/env python3
"""
ML4CV VLM Benchmark - Analysis Module
Analyzes and visualizes benchmark results
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def analyze_single_result(filename, label):
    """Analyze a single benchmark result"""
    if os.path.exists(filename):
        print(f"\n  Analyzing {filename}...")
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"  {filename} loaded successfully")
        
        # Extract and display metrics
        if 'performance_metrics' in data:
            metrics = data['performance_metrics']
            print(f"  {label} Performance Metrics:")
            print(f"    • Tokens per second: {metrics.get('tokens_per_second', 'N/A'):.2f}")
            print(f"    • Images per second: {metrics.get('images_per_second', 'N/A'):.2f}")
            print(f"    • Average inference time: {metrics.get('avg_inference_time', 'N/A'):.3f} seconds")
            print(f"    • Total samples processed: {metrics.get('total_samples', 'N/A')}")
            print(f"    • Memory usage: {metrics.get('memory_usage_mb', 'N/A'):.1f} MB")
        
        if 'model_info' in data:
            model_info = data['model_info']
            print(f"  Model Information:")
            print(f"    • Model used: {model_info.get('model_name', 'N/A')}")
            print(f"    • Device: {model_info.get('device', 'N/A')}")
            print(f"    • Quantization: {model_info.get('quantized', 'N/A')}")
        
        if 'sample_outputs' in data:
            samples = data['sample_outputs']
            print(f"  Sample Captions (first 2):")
            for i, caption in enumerate(samples[:2]):
                print(f"    {i+1}. '{caption}'")
        
        return data
    else:
        print(f"  ❌ {filename} not found")
        return None


def create_performance_visualization(results_data, output_file="benchmark_results_analysis.png"):
    """Create comparison visualization"""
    if len(results_data) == 0:
        print("No data available for visualization")
        return
    
    # Prepare data for plotting
    models = []
    tokens_per_sec = []
    images_per_sec = []
    inference_times = []
    memory_usage = []
    devices = []
    
    for key, data in results_data.items():
        if data and 'performance_metrics' in data:
            metrics = data['performance_metrics']
            model_info = data.get('model_info', {})
            
            label = "FP32 Baseline" if key == 'baseline' else "Quantized"
            models.append(label)
            tokens_per_sec.append(metrics.get('tokens_per_second', 0))
            images_per_sec.append(metrics.get('images_per_second', 0))
            inference_times.append(metrics.get('avg_inference_time', 0))
            memory_usage.append(metrics.get('memory_usage_mb', 0))
            devices.append(model_info.get('device', 'unknown'))
    
    if len(models) > 0:
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('VLM Benchmark Results Comparison', fontsize=16, fontweight='bold')
        
        # Colors for different models
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # Subplot 1: Tokens per second
        if max(tokens_per_sec) > 0:
            axes[0, 0].bar(models, tokens_per_sec, color=colors[:len(models)])
            axes[0, 0].set_title('Tokens per Second')
            axes[0, 0].set_ylabel('Tokens/sec')
            for i, (model, value) in enumerate(zip(models, tokens_per_sec)):
                axes[0, 0].text(i, value + max(tokens_per_sec)*0.01, f'{value:.1f}',
                               ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Images per second
        if max(images_per_sec) > 0:
            axes[0, 1].bar(models, images_per_sec, color=colors[:len(models)])
            axes[0, 1].set_title('Images per Second')
            axes[0, 1].set_ylabel('Images/sec')
            for i, (model, value) in enumerate(zip(models, images_per_sec)):
                axes[0, 1].text(i, value + max(images_per_sec)*0.01, f'{value:.2f}',
                               ha='center', va='bottom', fontweight='bold')
        
        # Subplot 3: Inference time
        if max(inference_times) > 0:
            axes[1, 0].bar(models, inference_times, color=colors[:len(models)])
            axes[1, 0].set_title('Average Inference Time')
            axes[1, 0].set_ylabel('Seconds')
            for i, (model, value) in enumerate(zip(models, inference_times)):
                axes[1, 0].text(i, value + max(inference_times)*0.01, f'{value:.3f}',
                               ha='center', va='bottom', fontweight='bold')
        
        # Subplot 4: Memory usage
        if max(memory_usage) > 0:
            axes[1, 1].bar(models, memory_usage, color=colors[:len(models)])
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_ylabel('MB')
            for i, (model, value) in enumerate(zip(models, memory_usage)):
                axes[1, 1].text(i, value + max(memory_usage)*0.01, f'{value:.1f}',
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Visualization saved as '{output_file}'")
        
        return models, tokens_per_sec, images_per_sec, inference_times, memory_usage, devices
    
    return None, None, None, None, None, None


def create_comparison_table(results_data):
    """Create performance comparison table"""
    if len(results_data) < 2:
        return None
    
    models = []
    devices = []
    tokens_per_sec = []
    images_per_sec = []
    inference_times = []
    memory_usage = []
    
    for key, data in results_data.items():
        if data and 'performance_metrics' in data:
            metrics = data['performance_metrics']
            model_info = data.get('model_info', {})
            
            label = "FP32 Baseline" if key == 'baseline' else "Quantized"
            models.append(label)
            devices.append(model_info.get('device', 'unknown'))
            tokens_per_sec.append(metrics.get('tokens_per_second', 0))
            images_per_sec.append(metrics.get('images_per_second', 0))
            inference_times.append(metrics.get('avg_inference_time', 0))
            memory_usage.append(metrics.get('memory_usage_mb', 0))
    
    comparison_df = pd.DataFrame({
        'Model': models,
        'Device': devices,
        'Tokens/sec': [f"{t:.2f}" for t in tokens_per_sec],
        'Images/sec': [f"{i:.2f}" for i in images_per_sec],
        'Avg Time (s)': [f"{t:.3f}" for t in inference_times],
        'Memory (MB)': [f"{m:.1f}" for m in memory_usage]
    })
    
    return comparison_df


def calculate_improvement_ratios(results_data):
    """Calculate improvement ratios between baseline and quantized"""
    if 'baseline' not in results_data or 'quantized' not in results_data:
        return None
    
    baseline_metrics = results_data['baseline']['performance_metrics']
    quantized_metrics = results_data['quantized']['performance_metrics']
    
    baseline_tokens = baseline_metrics.get('tokens_per_second', 0)
    quantized_tokens = quantized_metrics.get('tokens_per_second', 0)
    baseline_time = baseline_metrics.get('avg_inference_time', 0)
    quantized_time = quantized_metrics.get('avg_inference_time', 0)
    
    improvements = {}
    
    if baseline_tokens > 0 and quantized_tokens > 0:
        improvements['token_ratio'] = quantized_tokens / baseline_tokens
    
    if baseline_time > 0 and quantized_time > 0:
        improvements['time_ratio'] = baseline_time / quantized_time
    
    # Device comparison
    baseline_device = results_data['baseline']['model_info'].get('device', 'unknown')
    quantized_device = results_data['quantized']['model_info'].get('device', 'unknown')
    improvements['baseline_device'] = baseline_device
    improvements['quantized_device'] = quantized_device
    
    return improvements


def create_summary_table(results_data):
    """Create summary table of all results"""
    summary_data = []
    
    for key, data in results_data.items():
        if data:
            metrics = data.get('performance_metrics', {})
            model_info = data.get('model_info', {})
            benchmark_info = data.get('benchmark_info', {})
            
            summary_data.append({
                'Benchmark': key.title(),
                'Device': model_info.get('device', 'N/A'),
                'Quantized': str(model_info.get('quantized', 'N/A')),
                'Samples': metrics.get('total_samples', 'N/A'),
                'Tokens/sec': f"{metrics.get('tokens_per_second', 0):.2f}",
                'Images/sec': f"{metrics.get('images_per_second', 0):.2f}",
                'Timestamp': benchmark_info.get('timestamp', 'N/A')[:19]  # Remove microseconds
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    return None


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="VLM Benchmark Results Analysis")
    parser.add_argument("--baseline-file", default="baseline_fp32.json", 
                       help="Path to baseline benchmark results")
    parser.add_argument("--quantized-file", default="quantized_int8.json", 
                       help="Path to quantized benchmark results")
    parser.add_argument("--output-viz", default="benchmark_results_analysis.png", 
                       help="Output file for visualization")
    parser.add_argument("--output-csv", default="benchmark_summary.csv", 
                       help="Output CSV file for summary")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 60)
    
    # Check available files
    baseline_file = args.baseline_file
    quantized_file = args.quantized_file
    
    print("  Checking available benchmark files...")
    available_files = []
    for file in [baseline_file, quantized_file]:
        if os.path.exists(file):
            available_files.append(file)
            print(f"    ✓ {file}")
        else:
            print(f"    ❌ {file}")
    
    if not available_files:
        print("\n❌ No benchmark files found. Please run the benchmark first.")
        return
    
    # Analyze each available file
    results_data = {}
    if os.path.exists(baseline_file):
        results_data['baseline'] = analyze_single_result(baseline_file, "FP32 Baseline")
    
    if os.path.exists(quantized_file):
        results_data['quantized'] = analyze_single_result(quantized_file, "Quantized Model")
    
    # Create visualizations with available data
    print(f"\n" + "=" * 60)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    visualization_data = create_performance_visualization(results_data, args.output_viz)
    models, tokens_per_sec, images_per_sec, inference_times, memory_usage, devices = visualization_data
    
    # Performance comparison table
    if len(results_data) >= 2:
        print(f"\n  PERFORMANCE COMPARISON TABLE")
        print("=" * 50)
        comparison_df = create_comparison_table(results_data)
        if comparison_df is not None:
            print(comparison_df.to_string(index=False))
        
        # Calculate improvement ratios if we have both baseline and quantized
        improvements = calculate_improvement_ratios(results_data)
        if improvements:
            print(f"\n  PERFORMANCE IMPROVEMENT ANALYSIS")
            print("=" * 50)
            
            if 'token_ratio' in improvements:
                print(f"    Token throughput change: {improvements['token_ratio']:.2f}x")
            
            if 'time_ratio' in improvements:
                print(f"    Inference speed change: {improvements['time_ratio']:.2f}x")
            
            print(f"    Baseline runs on: {improvements['baseline_device']}")
            print(f"    Quantized runs on: {improvements['quantized_device']}")
            
            if improvements['baseline_device'] != improvements['quantized_device']:
                print("    Note: Models run on different devices, so comparison shows device + quantization effects")
    
    # Summary table of all available results
    print(f"\n  SUMMARY OF ALL RESULTS")
    print("=" * 60)
    summary_df = create_summary_table(results_data)
    if summary_df is not None:
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv(args.output_csv, index=False)
        print(f"\n  Summary saved as '{args.output_csv}'")
    
    print(f"\n  Analysis complete!")
    
    # List all generated files
    print(f"\n  Generated analysis files:")
    analysis_files = [args.output_viz, args.output_csv]
    for file in analysis_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"    ✓ {file} ({size:.1f} KB)")
        else:
            print(f"    ❌ {file} (not generated)")
    
    # Performance target analysis
    if models and tokens_per_sec:
        print(f"\n  PERFORMANCE TARGET ANALYSIS")
        print("=" * 50)
        
        TARGET_TOKENS_PER_SEC = 13.48
        AVG_TOKENS_PER_CAPTION = 15  # Typical caption length
        target_images_per_sec = TARGET_TOKENS_PER_SEC / AVG_TOKENS_PER_CAPTION
        
        print(f"    Target performance: {target_images_per_sec:.3f} images/sec")
        
        for i, (model, throughput) in enumerate(zip(models, images_per_sec)):
            performance_ratio = throughput / target_images_per_sec
            status = "✓" if performance_ratio >= 1.0 else "❌"
            print(f"    {model}: {throughput:.3f} images/sec ({performance_ratio:.2f}x) {status}")


if __name__ == "__main__":
    main()