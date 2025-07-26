"""
Performance benchmarking utilities for inference service.
Tests latency, throughput, and optimization effectiveness.
"""

import time
import statistics
import logging
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import string

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark suite for inference service performance testing."""
    
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.results = {}
    
    def generate_test_data(self, num_samples: int = 100) -> Tuple[List[str], List[str]]:
        """Generate synthetic test data for benchmarking."""
        proteins = []
        drugs = []
        
        # Common amino acids for realistic protein sequences
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Common SMILES patterns for realistic drug molecules
        smiles_patterns = [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)O',  # Isopropanol
            'CCOCC',  # Diethyl ether
            'CC(C)(C)O',  # tert-Butanol
            'c1ccc(cc1)O',  # Phenol
            'CC(=O)N',  # Acetamide
            'CCCCCCCCCCCCCCCCCC(=O)O'  # Stearic acid
        ]
        
        for i in range(num_samples):
            # Generate protein sequence (20-500 amino acids)
            length = random.randint(20, 500)
            protein = ''.join(random.choices(amino_acids, k=length))
            proteins.append(protein)
            
            # Generate drug SMILES (use patterns with variations)
            base_smiles = random.choice(smiles_patterns)
            # Add some variation
            if random.random() > 0.5:
                variation = ''.join(random.choices('CN', k=random.randint(1, 3)))
                drug = base_smiles + variation
            else:
                drug = base_smiles
            drugs.append(drug)
        
        return proteins, drugs
    
    def benchmark_single_prediction_latency(self, num_tests: int = 100) -> Dict[str, Any]:
        """Benchmark single prediction latency."""
        logger.info(f"Benchmarking single prediction latency with {num_tests} tests...")
        
        proteins, drugs = self.generate_test_data(num_tests)
        latencies = []
        
        # Warm up
        self.inference_service.warmup(5)
        
        for i in range(num_tests):
            start_time = time.time()
            result = self.inference_service.predict_single(proteins[i], drugs[i])
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to milliseconds
            
            if 'error' in result:
                logger.warning(f"Prediction error at index {i}: {result['error']}")
        
        # Calculate statistics
        stats = {
            'test_name': 'single_prediction_latency',
            'num_tests': num_tests,
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'sub_120ms_rate': sum(1 for l in latencies if l < 120) / len(latencies),
            'latencies': latencies
        }
        
        self.results['single_prediction_latency'] = stats
        
        logger.info(f"Single prediction latency results:")
        logger.info(f"  Average: {stats['avg_latency_ms']:.2f}ms")
        logger.info(f"  Median: {stats['median_latency_ms']:.2f}ms")
        logger.info(f"  P95: {stats['p95_latency_ms']:.2f}ms")
        logger.info(f"  Sub-120ms rate: {stats['sub_120ms_rate']:.2%}")
        
        return stats
    
    def benchmark_batch_prediction_throughput(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark batch prediction throughput."""
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20, 50, 100]
        
        logger.info(f"Benchmarking batch prediction throughput with batch sizes: {batch_sizes}")
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            proteins, drugs = self.generate_test_data(batch_size)
            
            # Warm up
            if batch_size == batch_sizes[0]:
                self.inference_service.warmup(5)
            
            start_time = time.time()
            results = self.inference_service.predict_batch(proteins, drugs)
            total_time = time.time() - start_time
            
            # Calculate throughput
            throughput = batch_size / total_time  # predictions per second
            avg_latency = (total_time / batch_size) * 1000  # ms per prediction
            
            throughput_results[batch_size] = {
                'batch_size': batch_size,
                'total_time_s': total_time,
                'throughput_pps': throughput,  # predictions per second
                'avg_latency_ms': avg_latency,
                'successful_predictions': len([r for r in results if 'error' not in r])
            }
            
            logger.info(f"  Batch size {batch_size}: {throughput:.2f} predictions/sec, {avg_latency:.2f}ms avg latency")
        
        self.results['batch_throughput'] = throughput_results
        return throughput_results
    
    def benchmark_cache_effectiveness(self, num_unique: int = 50, num_total: int = 200) -> Dict[str, Any]:
        """Benchmark cache effectiveness with repeated predictions."""
        logger.info(f"Benchmarking cache effectiveness: {num_unique} unique pairs, {num_total} total predictions")
        
        # Generate unique test data
        proteins, drugs = self.generate_test_data(num_unique)
        
        # Create prediction sequence with repeats
        prediction_sequence = []
        for i in range(num_total):
            idx = random.randint(0, num_unique - 1)
            prediction_sequence.append((proteins[idx], drugs[idx]))
        
        # Clear cache and warm up
        self.inference_service.clear_cache()
        self.inference_service.warmup(5)
        
        # Benchmark with cache
        cache_times = []
        start_time = time.time()
        
        for protein, drug in prediction_sequence:
            pred_start = time.time()
            result = self.inference_service.predict_single(protein, drug, use_cache=True)
            pred_time = time.time() - pred_start
            cache_times.append(pred_time * 1000)
        
        cache_total_time = time.time() - start_time
        
        # Benchmark without cache
        self.inference_service.clear_cache()
        no_cache_times = []
        start_time = time.time()
        
        for protein, drug in prediction_sequence:
            pred_start = time.time()
            result = self.inference_service.predict_single(protein, drug, use_cache=False)
            pred_time = time.time() - pred_start
            no_cache_times.append(pred_time * 1000)
        
        no_cache_total_time = time.time() - start_time
        
        # Calculate statistics
        cache_stats = {
            'test_name': 'cache_effectiveness',
            'num_unique_pairs': num_unique,
            'num_total_predictions': num_total,
            'cache_enabled': {
                'avg_latency_ms': statistics.mean(cache_times),
                'total_time_s': cache_total_time,
                'throughput_pps': num_total / cache_total_time
            },
            'cache_disabled': {
                'avg_latency_ms': statistics.mean(no_cache_times),
                'total_time_s': no_cache_total_time,
                'throughput_pps': num_total / no_cache_total_time
            },
            'speedup_factor': no_cache_total_time / cache_total_time,
            'latency_reduction': (statistics.mean(no_cache_times) - statistics.mean(cache_times)) / statistics.mean(no_cache_times)
        }
        
        self.results['cache_effectiveness'] = cache_stats
        
        logger.info(f"Cache effectiveness results:")
        logger.info(f"  With cache: {cache_stats['cache_enabled']['avg_latency_ms']:.2f}ms avg, {cache_stats['cache_enabled']['throughput_pps']:.2f} pps")
        logger.info(f"  Without cache: {cache_stats['cache_disabled']['avg_latency_ms']:.2f}ms avg, {cache_stats['cache_disabled']['throughput_pps']:.2f} pps")
        logger.info(f"  Speedup factor: {cache_stats['speedup_factor']:.2f}x")
        
        return cache_stats
    
    def benchmark_concurrent_load(self, num_threads: int = 4, requests_per_thread: int = 25) -> Dict[str, Any]:
        """Benchmark performance under concurrent load."""
        logger.info(f"Benchmarking concurrent load: {num_threads} threads, {requests_per_thread} requests each")
        
        proteins, drugs = self.generate_test_data(num_threads * requests_per_thread)
        
        # Warm up
        self.inference_service.warmup(5)
        
        def worker_thread(thread_id: int) -> List[float]:
            """Worker thread for concurrent testing."""
            thread_times = []
            start_idx = thread_id * requests_per_thread
            end_idx = start_idx + requests_per_thread
            
            for i in range(start_idx, end_idx):
                start_time = time.time()
                result = self.inference_service.predict_single(proteins[i], drugs[i])
                latency = time.time() - start_time
                thread_times.append(latency * 1000)
            
            return thread_times
        
        # Run concurrent benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            all_times = []
            
            for future in as_completed(futures):
                thread_times = future.result()
                all_times.extend(thread_times)
        
        total_time = time.time() - start_time
        total_requests = num_threads * requests_per_thread
        
        # Calculate statistics
        concurrent_stats = {
            'test_name': 'concurrent_load',
            'num_threads': num_threads,
            'requests_per_thread': requests_per_thread,
            'total_requests': total_requests,
            'total_time_s': total_time,
            'overall_throughput_pps': total_requests / total_time,
            'avg_latency_ms': statistics.mean(all_times),
            'median_latency_ms': statistics.median(all_times),
            'p95_latency_ms': np.percentile(all_times, 95),
            'p99_latency_ms': np.percentile(all_times, 99),
            'sub_120ms_rate': sum(1 for l in all_times if l < 120) / len(all_times)
        }
        
        self.results['concurrent_load'] = concurrent_stats
        
        logger.info(f"Concurrent load results:")
        logger.info(f"  Overall throughput: {concurrent_stats['overall_throughput_pps']:.2f} predictions/sec")
        logger.info(f"  Average latency: {concurrent_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"  P95 latency: {concurrent_stats['p95_latency_ms']:.2f}ms")
        logger.info(f"  Sub-120ms rate: {concurrent_stats['sub_120ms_rate']:.2%}")
        
        return concurrent_stats
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting full performance benchmark suite...")
        
        # Run all benchmarks
        self.benchmark_single_prediction_latency(100)
        self.benchmark_batch_prediction_throughput([1, 5, 10, 20, 50])
        self.benchmark_cache_effectiveness(50, 200)
        self.benchmark_concurrent_load(4, 25)
        
        # Generate summary
        summary = self.generate_benchmark_summary()
        
        logger.info("Full benchmark suite completed!")
        return summary
    
    def generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            'benchmark_timestamp': time.time(),
            'inference_service_config': {
                'model_path': self.inference_service.model_path,
                'device': self.inference_service.device,
                'max_batch_size': self.inference_service.max_batch_size,
                'cache_size': self.inference_service.cache_size
            },
            'results': self.results,
            'performance_grade': self._calculate_performance_grade()
        }
        
        return summary
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade."""
        if not self.results:
            return 'N/A'
        
        score = 0
        max_score = 0
        
        # Single prediction latency (40% weight)
        if 'single_prediction_latency' in self.results:
            latency_result = self.results['single_prediction_latency']
            sub_120ms_rate = latency_result.get('sub_120ms_rate', 0)
            avg_latency = latency_result.get('avg_latency_ms', 1000)
            
            # Score based on sub-120ms rate and average latency
            latency_score = (sub_120ms_rate * 0.6) + (max(0, (120 - avg_latency) / 120) * 0.4)
            score += latency_score * 40
            max_score += 40
        
        # Cache effectiveness (30% weight)
        if 'cache_effectiveness' in self.results:
            cache_result = self.results['cache_effectiveness']
            speedup = cache_result.get('speedup_factor', 1)
            
            # Score based on speedup factor
            cache_score = min(1.0, speedup / 3.0)  # Max score at 3x speedup
            score += cache_score * 30
            max_score += 30
        
        # Concurrent performance (30% weight)
        if 'concurrent_load' in self.results:
            concurrent_result = self.results['concurrent_load']
            throughput = concurrent_result.get('overall_throughput_pps', 0)
            sub_120ms_rate = concurrent_result.get('sub_120ms_rate', 0)
            
            # Score based on throughput and latency consistency
            concurrent_score = (min(1.0, throughput / 50) * 0.5) + (sub_120ms_rate * 0.5)
            score += concurrent_score * 30
            max_score += 30
        
        if max_score == 0:
            return 'N/A'
        
        final_score = score / max_score
        
        if final_score >= 0.9:
            return 'A+'
        elif final_score >= 0.8:
            return 'A'
        elif final_score >= 0.7:
            return 'B+'
        elif final_score >= 0.6:
            return 'B'
        elif final_score >= 0.5:
            return 'C+'
        elif final_score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        import json
        
        summary = self.generate_benchmark_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def plot_results(self, save_path: str = None):
        """Generate performance visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Inference Service Performance Benchmark Results', fontsize=16)
            
            # Plot 1: Latency distribution
            if 'single_prediction_latency' in self.results:
                latencies = self.results['single_prediction_latency']['latencies']
                axes[0, 0].hist(latencies, bins=30, alpha=0.7, color='blue')
                axes[0, 0].axvline(120, color='red', linestyle='--', label='120ms target')
                axes[0, 0].set_xlabel('Latency (ms)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Single Prediction Latency Distribution')
                axes[0, 0].legend()
            
            # Plot 2: Batch throughput
            if 'batch_throughput' in self.results:
                batch_data = self.results['batch_throughput']
                batch_sizes = list(batch_data.keys())
                throughputs = [batch_data[bs]['throughput_pps'] for bs in batch_sizes]
                
                axes[0, 1].plot(batch_sizes, throughputs, marker='o', color='green')
                axes[0, 1].set_xlabel('Batch Size')
                axes[0, 1].set_ylabel('Throughput (predictions/sec)')
                axes[0, 1].set_title('Batch Processing Throughput')
                axes[0, 1].grid(True)
            
            # Plot 3: Cache effectiveness
            if 'cache_effectiveness' in self.results:
                cache_data = self.results['cache_effectiveness']
                categories = ['With Cache', 'Without Cache']
                latencies = [
                    cache_data['cache_enabled']['avg_latency_ms'],
                    cache_data['cache_disabled']['avg_latency_ms']
                ]
                
                axes[1, 0].bar(categories, latencies, color=['blue', 'orange'])
                axes[1, 0].set_ylabel('Average Latency (ms)')
                axes[1, 0].set_title('Cache Effectiveness')
            
            # Plot 4: Performance summary
            if self.results:
                performance_metrics = []
                metric_names = []
                
                if 'single_prediction_latency' in self.results:
                    performance_metrics.append(self.results['single_prediction_latency']['sub_120ms_rate'])
                    metric_names.append('Sub-120ms Rate')
                
                if 'concurrent_load' in self.results:
                    performance_metrics.append(self.results['concurrent_load']['sub_120ms_rate'])
                    metric_names.append('Concurrent Sub-120ms Rate')
                
                if performance_metrics:
                    axes[1, 1].bar(metric_names, performance_metrics, color='purple')
                    axes[1, 1].set_ylabel('Rate')
                    axes[1, 1].set_title('Performance Metrics')
                    axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plots saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")


def run_benchmark_suite(model_path: str = "models/unsloth_finetuned_model") -> Dict[str, Any]:
    """Run complete benchmark suite for inference service."""
    logger.info("Initializing inference service for benchmarking...")
    
    # Initialize inference service
    inference_service = InferenceService(
        model_path=model_path,
        device="auto",
        max_batch_size=32,
        cache_size=1000
    )
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(inference_service)
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    # Save results
    timestamp = int(time.time())
    results_file = f"benchmark_results_{timestamp}.json"
    benchmark.save_results(results_file)
    
    # Generate plots
    plots_file = f"benchmark_plots_{timestamp}.png"
    benchmark.plot_results(plots_file)
    
    return results


if __name__ == "__main__":
    # Run benchmark if called directly
    results = run_benchmark_suite()
    print(f"Benchmark completed! Performance grade: {results['performance_grade']}")