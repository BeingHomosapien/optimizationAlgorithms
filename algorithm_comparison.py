"""
Comprehensive Algorithm Comparison

Compare ALL 18 algorithms on multiple test functions.
Generate performance reports and visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Import all algorithms
from algorithms import *
from test_functions.benchmark_functions import FUNCTIONS


def benchmark_algorithm(algorithm, func, x0, name):
    """Benchmark a single algorithm on a function."""
    start_time = time.time()
    try:
        x_opt, history = algorithm.optimize(func, x0)
        elapsed = time.time() - start_time

        final_value = history[-1][1]
        iterations = len(history)
        distance_to_optimum = np.linalg.norm(x_opt - func.global_minimum)
        success = final_value < 0.01  # Threshold for success

        return {
            'name': name,
            'final_value': final_value,
            'iterations': iterations,
            'time': elapsed,
            'distance': distance_to_optimum,
            'success': success,
            'history': history,
            'error': None
        }
    except Exception as e:
        return {
            'name': name,
            'error': str(e),
            'success': False
        }


def compare_all_algorithms(function_name='rosenbrock'):
    """Compare all 18 algorithms on a given function."""

    func = FUNCTIONS[function_name]()
    bounds = np.array(func.bounds)
    x0 = (bounds[:, 0] + bounds[:, 1]) / 2
    x0 = x0 + 0.3 * (bounds[:, 1] - bounds[:, 0])

    print("=" * 80)
    print(f"COMPREHENSIVE ALGORITHM COMPARISON: {func.name}")
    print("=" * 80)
    print(f"Global minimum: {func.global_minimum}")
    print(f"Global minimum value: {func(func.global_minimum):.6f}")
    print(f"Starting point: {x0}")
    print()

    # Define all algorithms
    algorithms_config = {
        # Basic Gradient
        "Gradient Descent": GradientDescent(learning_rate=0.001, max_iterations=500),
        "Momentum": Momentum(learning_rate=0.001, max_iterations=500),
        "RMSprop": RMSprop(learning_rate=0.01, max_iterations=500),
        "Adam": Adam(learning_rate=0.01, max_iterations=500),

        # Advanced Gradient
        "AdaGrad": AdaGrad(learning_rate=0.1, max_iterations=500),
        "NAdam": NAdam(learning_rate=0.01, max_iterations=500),
        "AdamW": AdamW(learning_rate=0.01, max_iterations=500),
        "L-BFGS": LBFGS(max_iterations=100),
        "Conjugate Gradient": ConjugateGradient(max_iterations=500),
        "Newton": NewtonMethod(max_iterations=50),

        # Basic Metaheuristic
        "Genetic Algorithm": GeneticAlgorithm(population_size=50, generations=100),
        "PSO": ParticleSwarmOptimization(n_particles=30, iterations=100),
        "Simulated Annealing": SimulatedAnnealing(initial_temp=100, iterations=500),

        # Advanced Metaheuristic
        "Differential Evolution": DifferentialEvolution(population_size=50, generations=100),
        "CMA-ES": CMAES(population_size=30, generations=50),
        "Nelder-Mead": NelderMead(max_iterations=500),
        "Bayesian Optimization": BayesianOptimization(iterations=80),
        "Harmony Search": HarmonySearch(harmony_memory_size=30, iterations=200),
    }

    results = {}

    # Run benchmarks
    for name, algorithm in algorithms_config.items():
        print(f"Running {name:25s}...", end=" ", flush=True)
        result = benchmark_algorithm(algorithm, func, x0, name)
        results[name] = result

        if result['error']:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"✓ {result['final_value']:.6f} in {result['iterations']:4d} iters ({result['time']:.3f}s)")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Final Value':<15} {'Iterations':<12} {'Time (s)':<10} {'Success'}")
    print("-" * 80)

    successful = []
    failed = []

    for name, result in results.items():
        if result['error']:
            failed.append(name)
            print(f"{name:<25} {'ERROR':<15} {'-':<12} {'-':<10} ✗")
        else:
            if result['success']:
                successful.append(name)
            status = "✓" if result['success'] else "✗"
            print(f"{name:<25} {result['final_value']:<15.6f} {result['iterations']:<12} "
                  f"{result['time']:<10.3f} {status}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        successful_results = [results[name] for name in successful]
        best_value = min(r['final_value'] for r in successful_results)
        best_alg = [r['name'] for r in successful_results if r['final_value'] == best_value][0]
        print(f"Best final value: {best_value:.6f} ({best_alg})")

        fastest = min(successful_results, key=lambda r: r['time'])
        print(f"Fastest: {fastest['name']} ({fastest['time']:.3f}s)")

        fewest_iters = min(successful_results, key=lambda r: r['iterations'])
        print(f"Fewest iterations: {fewest_iters['name']} ({fewest_iters['iterations']} iters)")

    return results


def visualize_results(results, func_name):
    """Create comprehensive visualizations."""

    # Filter successful results
    successful = {name: r for name, r in results.items() if not r['error']}

    if not successful:
        print("No successful runs to visualize!")
        return

    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Final values
    names = list(successful.keys())
    values = [successful[name]['final_value'] for name in names]
    colors = ['green' if successful[name]['success'] else 'orange' for name in names]

    ax = axes[0, 0]
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Function Value')
    ax.set_title('Final Values (green = success)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    # Iterations
    ax = axes[0, 1]
    iterations = [successful[name]['iterations'] for name in names]
    ax.barh(range(len(names)), iterations, alpha=0.7, color='skyblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Number of Iterations')
    ax.set_title('Iterations to Convergence')
    ax.grid(True, alpha=0.3, axis='x')

    # Computation time
    ax = axes[1, 0]
    times = [successful[name]['time'] for name in names]
    ax.barh(range(len(names)), times, alpha=0.7, color='coral')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.grid(True, alpha=0.3, axis='x')

    # Success vs final value scatter
    ax = axes[1, 1]
    for name in names:
        r = successful[name]
        marker = 'o' if r['success'] else 'x'
        color = 'green' if r['success'] else 'red'
        ax.scatter(r['iterations'], r['final_value'], marker=marker, s=100,
                  alpha=0.7, color=color, label=name if len(names) < 10 else None)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Final Value')
    ax.set_yscale('log')
    ax.set_title('Efficiency: Iterations vs Final Value')
    ax.grid(True, alpha=0.3)
    if len(names) < 10:
        ax.legend(fontsize=6)

    plt.suptitle(f'Algorithm Performance Comparison - {func_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 2. Convergence curves
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, result in successful.items():
        history = result['history']
        values = [h[1] for h in history]
        ax.semilogy(values, label=name, linewidth=2, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Function Value (log scale)', fontsize=12)
    ax.set_title(f'Convergence Curves - {func_name}', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def multi_function_comparison():
    """Compare algorithms across multiple test functions."""
    functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley']

    print("\n" + "=" * 80)
    print("MULTI-FUNCTION COMPARISON")
    print("=" * 80)

    all_results = {}

    for func_name in functions:
        print(f"\n{'='*80}")
        print(f"Testing on {func_name.upper()}")
        print('='*80)
        results = compare_all_algorithms(func_name)
        all_results[func_name] = results

        input(f"\nPress Enter to continue to next function...")

    # Summary across all functions
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY ACROSS ALL FUNCTIONS")
    print("=" * 80)

    # Count successes per algorithm
    algorithm_names = list(next(iter(all_results.values())).keys())
    success_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for func_name, results in all_results.items():
        for alg_name, result in results.items():
            total_counts[alg_name] += 1
            if result.get('success', False):
                success_counts[alg_name] += 1

    print(f"{'Algorithm':<25} {'Success Rate':<15} {'Wins'}")
    print("-" * 80)

    for alg_name in algorithm_names:
        rate = success_counts[alg_name] / total_counts[alg_name] * 100
        print(f"{alg_name:<25} {rate:>6.1f}% ({success_counts[alg_name]}/{total_counts[alg_name]})")

    return all_results


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "COMPREHENSIVE ALGORITHM COMPARISON" + " " * 24 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║  Testing all 18 optimization algorithms on multiple benchmark functions  ║")
    print("╚" + "═" * 78 + "╝")
    print()

    print("Choose comparison mode:")
    print("1. Single function (detailed)")
    print("2. Multiple functions (overview)")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        multi_function_comparison()
    else:
        print("\nAvailable functions:", list(FUNCTIONS.keys()))
        func_name = input("Enter function name (default: rosenbrock): ").strip() or 'rosenbrock'

        if func_name not in FUNCTIONS:
            print(f"Unknown function. Using rosenbrock.")
            func_name = 'rosenbrock'

        results = compare_all_algorithms(func_name)

        print("\nGenerating visualizations...")
        visualize_results(results, func_name)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
