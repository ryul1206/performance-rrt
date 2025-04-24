#!/usr/bin/env python3

import time
import statistics
import csv
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import argparse

from core.utils import Params, SAMPLE_OBSTACLES, COMPLEX_OBSTACLES
from core.logger import NullLogger

# Python implementations
from core.python_impl.rrt_star_pure import rrt as rrt_pure
from core.python_impl.rrt_star_numpy import rrt as rrt_numpy
from core.python_impl.rrt_star_numpy_opt import rrt as rrt_numpy_opt
from core.python_impl.rrt_star_numba import rrt as rrt_numba
from core.python_impl.rrt_star_numba_np import rrt as rrt_numba_np
from core.python_impl.rrt_star_numba_npopt import rrt as rrt_numba_npopt

# C++ implementation
from core.cpp_impl.rrt_star_cpp_fn import rrt as rrt_cpp_fn
from core.cpp_impl.rrt_star_cpp_full import rrt as rrt_cpp_full
from core.cpp_impl.rrt_star_simd_full import rrt as rrt_cpp_simd

# Benchmark parameters
NUM_RUNS = 100  # Number of runs for each implementation (> 1)

# Simple scenario (original)
SIMPLE_PARAMS = Params(
    max_iterations=1000,
    max_step_size=0.1,
    start_point=(0.2, 0.5),
    end_point=(1.8, 0.5),
    wall_segments=SAMPLE_OBSTACLES,
)


COMPLEX_PARAMS = Params(
    max_iterations=5000,  # More iterations for complex scenario
    max_step_size=0.05,  # Smaller step size for more precise navigation
    start_point=(0.1, 0.1),
    end_point=(1.9, 0.9),
    wall_segments=COMPLEX_OBSTACLES,
)


def run_benchmark(rrt_impl, name: str, num_runs: int, params: Params) -> List[float]:
    """Run benchmark for a specific RRT* implementation."""
    times = []  # Store times in nanoseconds
    logger = NullLogger()

    for i in range(num_runs):
        start_time = time.process_time_ns()  # Get time in nanoseconds
        rrt_impl(params, logger)
        end_time = time.process_time_ns()
        elapsed_time = end_time - start_time  # Keep in nanoseconds
        print(f"{name} - Time taken: {elapsed_time/1e9:.9f} seconds")  # Convert to seconds for display
        times.append(elapsed_time)

        if (i + 1) % 10 == 0:
            print(f"{name}: Completed {i + 1}/{num_runs} runs")

    return times


def remove_outliers(times: List[float]) -> List[float]:
    """Remove outliers from the data using the IQR method."""
    if len(times) < 2:
        return times

    q1 = statistics.quantiles(times, n=4)[0]
    q3 = statistics.quantiles(times, n=4)[2]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return [t for t in times if lower_bound <= t <= upper_bound]


def analyze_results(times: List[float]) -> Dict[str, float]:
    """Analyze benchmark results and return statistics."""
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
    }


def format_time(ns: float) -> str:
    """Format time in nanoseconds to appropriate unit."""
    return f"{ns/1e9:.9f} seconds"


def save_results(results: Dict[str, List[float]], filename: str) -> None:
    """Save benchmark results to CSV file."""
    with open(f"reports/{filename}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Implementation", "Run", "Time (nanoseconds)"])
        for impl_name, times in results.items():
            for i, time_val in enumerate(times):
                writer.writerow([impl_name, i + 1, time_val])


def plot_results(results: Dict[str, List[float]], scenario_name: str) -> None:
    """Create and save plots of benchmark results."""
    _sname = scenario_name.lower().replace(' ', '_')

    # Time scaling
    max_time = max(max(times) for times in results.values())
    scale_factor = 1.0
    unit = "nanoseconds"
    if max_time >= 1e9:
        scale_factor = 1e9
        unit = "seconds"
    elif max_time >= 1e6:
        scale_factor = 1e6
        unit = "milliseconds"
    # scale_factor = 1e9
    # unit = "seconds"

    # Box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([np.array(times)/scale_factor for times in results.values()], tick_labels=results.keys())
    plt.title(f"RRT* Implementation Performance Comparison - {scenario_name}")
    plt.ylabel(f"Execution Time ({unit})")
    plt.savefig(f"reports/benchmark_boxplot_{_sname}.png")
    plt.close()

    # Histogram
    plt.figure(figsize=(12, 6))
    for impl_name, times in results.items():
        plt.hist(np.array(times)/scale_factor, bins=20, alpha=0.5, label=impl_name)
    plt.title(f"RRT* Implementation Performance Distribution - {scenario_name}")
    plt.xlabel(f"Execution Time ({unit})")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"reports/benchmark_histogram_{_sname}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run RRT* benchmark with different scenarios")
    parser.add_argument(
        "--scenario", choices=["simple", "complex"], default="simple", help="Choose benchmark scenario (simple or complex)"
    )
    args = parser.parse_args()

    params = SIMPLE_PARAMS if args.scenario == "simple" else COMPLEX_PARAMS
    scenario_name = "Simple Scenario" if args.scenario == "simple" else "Complex Scenario"

    print(f"Starting RRT* benchmark for {scenario_name}...")

    # Run benchmarks
    results = {
        "Pure Python": run_benchmark(rrt_pure, "Pure Python", NUM_RUNS, params),
        # "NumPy": run_benchmark(rrt_numpy, "NumPy", NUM_RUNS, params),  # VERY SLOW
        # "NumPy Optimized": run_benchmark(rrt_numpy_opt, "NumPy Optimized", NUM_RUNS, params),  # SLOW
        "Numba": run_benchmark(rrt_numba, "Numba", NUM_RUNS, params),
        # "Numba (Numpy)": run_benchmark(rrt_numba_np, "Numba (Numpy)", NUM_RUNS, params),
        # "Numba (Numpy Optimized)": run_benchmark(rrt_numba_npopt, "Numba (Numpy) Optimized", NUM_RUNS, params),
        "C++ Bindings": run_benchmark(rrt_cpp_fn, "C++ Bindings", NUM_RUNS, params),
        "C++ Full": run_benchmark(rrt_cpp_full, "C++ Full", NUM_RUNS, params),
        "C++ SIMD": run_benchmark(rrt_cpp_simd, "C++ SIMD", NUM_RUNS, params),
    }
    # results = {
    #     "C++ Full": run_benchmark(rrt_cpp_full, "C++ Full", NUM_RUNS, params),
    #     "C++ SIMD": run_benchmark(rrt_cpp_simd, "C++ SIMD", NUM_RUNS, params),
    # }

    # Remove outliers from all results
    filtered_results = {name: remove_outliers(times) for name, times in results.items()}

    # Analyze and print results
    print(f"\nBenchmark Results for {scenario_name}:")
    for impl_name, times in filtered_results.items():
        stats = analyze_results(times)
        print(f"\n{impl_name}:")
        print(f"  Mean: {format_time(stats['mean'])}")
        print(f"  Median: {format_time(stats['median'])}")
        print(f"  Std Dev: {format_time(stats['stdev'])}")
        print(f"  Min: {format_time(stats['min'])}")
        print(f"  Max: {format_time(stats['max'])}")
        print(f"  Samples: {len(times)}/{len(results[impl_name])} (after outlier removal)")

    # Save and plot results
    save_results(filtered_results, f"benchmark_results_{args.scenario}.csv")
    plot_results(filtered_results, scenario_name)
    print(f"\nResults saved to reports/benchmark_results_{args.scenario}.csv")
    _sname = scenario_name.lower().replace(' ', '_')
    print(
        f"Plots saved to reports/benchmark_boxplot_{_sname}.png and reports/benchmark_histogram_{_sname}.png"
    )


if __name__ == "__main__":
    main()
