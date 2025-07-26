#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for enhanced batch inference and model caching features.

This script demonstrates the enhanced features of the RealtimeInferenceEngine,
including intelligent model caching, dynamic batch sizing, and priority-based processing.
"""

import os
import sys
import time
import asyncio
import argparse
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protein_drug_discovery.core.realtime_inference import (
    get_inference_engine, PredictionRequest, RealtimeInferenceEngine
)


def test_dynamic_batch_sizing(engine: RealtimeInferenceEngine, num_requests: int = 100):
    """
    Test dynamic batch sizing by sending a large number of requests and monitoring batch sizes.
    """
    print("\n=== Testing Dynamic Batch Sizing ===")
    
    # Create dummy requests
    protein_sequences = [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS" 
        for _ in range(num_requests)
    ]
    
    drug_smiles = [
        "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)C4=CC=C(C=C4)C(=O)NC5=CC=C(C=C5)C(F)(F)F" 
        for _ in range(num_requests)
    ]
    
    # Process in smaller chunks to observe dynamic batch sizing
    batch_sizes = [5, 10, 20, 30, 40]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nProcessing with fixed batch size: {batch_size}")
        start_time = time.time()
        
        # Process a subset of requests with this batch size
        subset_size = min(batch_size * 3, num_requests)
        requests = [
            PredictionRequest(
                protein_sequence=protein_sequences[i],
                drug_smiles=drug_smiles[i],
                request_id=f"batch_{batch_size}_{i}"
            ) for i in range(subset_size)
        ]
        
        # Run batch prediction with fixed batch size (disable dynamic batching)
        engine.batch_processor.enable_dynamic_batching = False
        asyncio.run(engine.predict_batch(requests, batch_size=batch_size))
        
        elapsed = time.time() - start_time
        throughput = subset_size / elapsed
        results[f"fixed_{batch_size}"] = {
            "batch_size": batch_size,
            "time": elapsed,
            "throughput": throughput,
            "requests": subset_size
        }
        print(f"Fixed batch size {batch_size}: {elapsed:.2f}s, Throughput: {throughput:.2f} req/s")
    
    # Now test with dynamic batch sizing
    print("\nProcessing with dynamic batch sizing")
    engine.batch_processor.enable_dynamic_batching = True
    
    start_time = time.time()
    requests = [
        PredictionRequest(
            protein_sequence=protein_sequences[i],
            drug_smiles=drug_smiles[i],
            request_id=f"dynamic_{i}"
        ) for i in range(num_requests)
    ]
    
    asyncio.run(engine.predict_batch(requests))
    
    elapsed = time.time() - start_time
    throughput = num_requests / elapsed
    results["dynamic"] = {
        "time": elapsed,
        "throughput": throughput,
        "requests": num_requests
    }
    print(f"Dynamic batch sizing: {elapsed:.2f}s, Throughput: {throughput:.2f} req/s")
    
    # Get stats to see what batch sizes were used
    stats = engine.get_stats()
    print(f"\nBatch processor stats: {stats['batch_processor']}")
    
    return results


def test_priority_based_processing(engine: RealtimeInferenceEngine, num_requests: int = 30):
    """
    Test priority-based processing by sending requests with different priorities.
    """
    print("\n=== Testing Priority-Based Processing ===")
    
    # Create dummy requests with different priorities
    protein_base = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    drug_base = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)C4=CC=C(C=C4)C(=O)NC5=CC=C(C=C5)C(F)(F)F"
    
    # Create requests with different priorities
    low_priority_requests = [
        PredictionRequest(
            protein_sequence=protein_base + str(i),
            drug_smiles=drug_base,
            request_id=f"low_{i}",
            priority=0
        ) for i in range(num_requests)
    ]
    
    high_priority_requests = [
        PredictionRequest(
            protein_sequence=protein_base + str(i + 100),
            drug_smiles=drug_base,
            request_id=f"high_{i}",
            priority=2
        ) for i in range(num_requests // 3)  # Fewer high priority requests
    ]
    
    # Mix the requests, with high priority ones in the middle
    all_requests = low_priority_requests[:num_requests//2] + high_priority_requests + low_priority_requests[num_requests//2:]
    
    # Process the mixed batch
    print(f"Processing {len(all_requests)} requests with mixed priorities")
    start_time = time.time()
    results = asyncio.run(engine.predict_batch(all_requests))
    elapsed = time.time() - start_time
    
    # Check the order of results
    result_ids = [r.request_id for r in results]
    high_priority_indices = [i for i, id in enumerate(result_ids) if id.startswith("high")]
    
    print(f"Processed in {elapsed:.2f}s")
    print(f"High priority request indices: {high_priority_indices}")
    
    # Calculate average position of high priority requests
    avg_high_priority_pos = sum(high_priority_indices) / len(high_priority_indices)
    expected_avg_pos = len(all_requests) / 2
    
    print(f"Average position of high priority requests: {avg_high_priority_pos:.2f}")
    print(f"Expected average position without priority: {expected_avg_pos:.2f}")
    
    if avg_high_priority_pos < expected_avg_pos:
        print("✅ Priority-based processing is working correctly!")
    else:
        print("❌ Priority-based processing may not be working as expected.")
    
    return {
        "high_priority_indices": high_priority_indices,
        "avg_high_priority_pos": avg_high_priority_pos,
        "expected_avg_pos": expected_avg_pos
    }


def test_model_cache_performance(engine: RealtimeInferenceEngine, num_iterations: int = 5):
    """
    Test model cache performance by loading different models and observing cache behavior.
    """
    print("\n=== Testing Model Cache Performance ===")
    
    # We'll simulate different models by using different paths
    # In a real scenario, these would be different actual models
    model_paths = [
        "models/unsloth_finetuned_model",  # Default model
        "models/unsloth_finetuned_model_v2",  # Simulated model 2
        "models/unsloth_finetuned_model_v3",  # Simulated model 3
        "models/unsloth_finetuned_model_v4",  # Simulated model 4
    ]
    
    # For this test, we'll just use the same model but pretend they're different
    # In a real scenario, these would be different models with different performance characteristics
    
    results = {}
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        for model_idx, model_path in enumerate(model_paths):
            # Create a simple request
            request = PredictionRequest(
                protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQ",
                drug_smiles="CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)C4=CC=C(C=C4)C(=O)NC5=CC=C(C=C5)C(F)(F)F",
                request_id=f"model_cache_{iteration}_{model_idx}"
            )
            
            # Override the model path for this request
            original_model_path = engine.model_path
            engine.model_path = model_path
            
            # Process the request and measure time
            start_time = time.time()
            asyncio.run(engine.predict_single(request))
            elapsed = time.time() - start_time
            
            # Restore original model path
            engine.model_path = original_model_path
            
            print(f"Model {model_path}: {elapsed:.4f}s")
            
            # Store the result
            if model_path not in results:
                results[model_path] = []
            results[model_path].append(elapsed)
    
    # Get cache stats
    stats = engine.get_stats()
    print(f"\nModel cache stats: {stats['cache']}")
    
    # Calculate average times
    avg_times = {model: sum(times) / len(times) for model, times in results.items()}
    print("\nAverage processing times:")
    for model, avg_time in avg_times.items():
        print(f"{model}: {avg_time:.4f}s")
    
    return {
        "detailed_results": results,
        "avg_times": avg_times,
        "cache_stats": stats['cache']
    }


def plot_results(batch_results, priority_results, cache_results, output_dir: str = "."):
    """
    Plot the results of the tests.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Batch Size vs Throughput
    plt.figure(figsize=(10, 6))
    
    # Extract fixed batch size results
    fixed_sizes = []
    fixed_throughputs = []
    
    for key, value in batch_results.items():
        if key.startswith("fixed_"):
            fixed_sizes.append(value["batch_size"])
            fixed_throughputs.append(value["throughput"])
    
    # Sort by batch size
    fixed_data = sorted(zip(fixed_sizes, fixed_throughputs))
    fixed_sizes, fixed_throughputs = zip(*fixed_data)
    
    # Plot fixed batch sizes
    plt.plot(fixed_sizes, fixed_throughputs, 'o-', label="Fixed Batch Sizes")
    
    # Add dynamic batch size result
    if "dynamic" in batch_results:
        dynamic_throughput = batch_results["dynamic"]["throughput"]
        plt.axhline(y=dynamic_throughput, color='r', linestyle='--', label=f"Dynamic Batching ({dynamic_throughput:.2f} req/s)")
    
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (requests/second)")
    plt.title("Batch Size vs Throughput")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "batch_size_throughput.png"))
    
    # Plot 2: Priority-based processing
    if priority_results and "high_priority_indices" in priority_results:
        plt.figure(figsize=(12, 6))
        
        # Create a list of all positions
        all_positions = list(range(len(priority_results["high_priority_indices"]) + 
                               priority_results["high_priority_indices"][-1] + 1))
        
        # Create a list of priorities (1 for high priority, 0 for low priority)
        priorities = [1 if i in priority_results["high_priority_indices"] else 0 for i in all_positions]
        
        # Plot the priorities
        plt.bar(all_positions, priorities, color=['red' if p == 1 else 'blue' for p in priorities])
        plt.axvline(x=priority_results["avg_high_priority_pos"], color='r', linestyle='--', 
                   label=f"Avg High Priority Position ({priority_results['avg_high_priority_pos']:.2f})")
        plt.axvline(x=priority_results["expected_avg_pos"], color='k', linestyle='--',
                   label=f"Expected Avg Position ({priority_results['expected_avg_pos']:.2f})")
        
        plt.xlabel("Request Processing Order")
        plt.ylabel("Priority (1=High, 0=Low)")
        plt.title("Priority-Based Processing Results")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "priority_processing.png"))
    
    # Plot 3: Model Cache Performance
    if cache_results and "detailed_results" in cache_results:
        plt.figure(figsize=(12, 6))
        
        # Plot processing times for each model across iterations
        for model, times in cache_results["detailed_results"].items():
            model_name = os.path.basename(model)
            plt.plot(range(1, len(times) + 1), times, 'o-', label=model_name)
        
        plt.xlabel("Iteration")
        plt.ylabel("Processing Time (seconds)")
        plt.title("Model Cache Performance")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "model_cache_performance.png"))
    
    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test enhanced batch inference and model caching")
    parser.add_argument("--model-path", type=str, default="models/unsloth_finetuned_model",
                        help="Path to the model directory")
    parser.add_argument("--max-cache-size", type=int, default=10000,
                        help="Maximum size of the prediction cache")
    parser.add_argument("--max-batch-size", type=int, default=32,
                        help="Maximum batch size for batch processing")
    parser.add_argument("--min-batch-size", type=int, default=4,
                        help="Minimum batch size for dynamic batch sizing")
    parser.add_argument("--model-cache-size", type=int, default=3,
                        help="Maximum number of models to keep in cache")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results and plots")
    parser.add_argument("--skip-tests", type=str, nargs="*", choices=["batch", "priority", "cache"],
                        help="Skip specific tests")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing enhanced inference engine...")
    engine = get_inference_engine(
        model_path=args.model_path,
        max_cache_size=args.max_cache_size,
        max_batch_size=args.max_batch_size,
        min_batch_size=args.min_batch_size,
        model_cache_size=args.model_cache_size,
        enable_dynamic_batching=True,
        warmup=True,
        warmup_requests=5
    )
    
    # Initialize results
    results = {}
    
    # Run tests
    if not args.skip_tests or "batch" not in args.skip_tests:
        results["batch"] = test_dynamic_batch_sizing(engine, num_requests=50)
    
    if not args.skip_tests or "priority" not in args.skip_tests:
        results["priority"] = test_priority_based_processing(engine, num_requests=30)
    
    if not args.skip_tests or "cache" not in args.skip_tests:
        results["cache"] = test_model_cache_performance(engine, num_iterations=3)
    
    # Get final stats
    final_stats = engine.get_stats()
    results["final_stats"] = final_stats
    
    # Save results to JSON
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(
        results.get("batch", {}),
        results.get("priority", {}),
        results.get("cache", {}),
        args.output_dir
    )
    
    print(f"\nAll tests completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()