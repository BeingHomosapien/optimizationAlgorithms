"""
Compare Multiple Optimization Algorithms

Run all algorithms on the same function and compare performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

from algorithms.gradient_based import GradientDescent, Momentum, RMSprop, Adam
from algorithms.metaheuristic import GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing
from test_functions.benchmark_functions import FUNCTIONS
from utils.visualization import compare_algorithms_visual, plot_convergence


def run_comparison(function_name='rosenbrock', x0=None):
    """Compare all algorithms on a given function."""

    # Get function
    if function_name not in FUNCTIONS:
        print(f"Unknown function: {function_name}")
        print(f"Available: {list(FUNCTIONS.keys())}")
        return

    func = FUNCTIONS[function_name]()
    print("=" * 60)
    print(f"ALGORITHM COMPARISON: {func.name} Function")
    print("=" * 60)

    # Default starting point
    if x0 is None:
        bounds = np.array(func.bounds)
        x0 = (bounds[:, 0] + bounds[:, 1]) / 2  # Middle of bounds
        x0 = x0 + 0.3 * (bounds[:, 1] - bounds[:, 0])  # Offset slightly

    print(f"Starting point: {x0}")
    print(f"Global minimum: {func.global_minimum}")
    print(f"Global minimum value: {func(func.global_minimum):.6f}")
    print()

    results = {}

    # Gradient-based methods
    print("Running gradient-based methods...")

    algorithms = [
        ("Gradient Descent", GradientDescent(learning_rate=0.001, max_iterations=500)),
        ("Momentum", Momentum(learning_rate=0.001, beta=0.9, max_iterations=500)),
        ("RMSprop", RMSprop(learning_rate=0.01, max_iterations=500)),
        ("Adam", Adam(learning_rate=0.01, max_iterations=500)),
    ]

    for name, alg in algorithms:
        print(f"  {name}...", end=" ")
        try:
            x_opt, history = alg.optimize(func, x0)
            results[name] = (x_opt, history)
            print(f"Final value: {history[-1][1]:.6f}")
        except Exception as e:
            print(f"Failed: {e}")

    # Metaheuristic methods
    print("\nRunning metaheuristic methods...")

    meta_algorithms = [
        ("Genetic Algorithm", GeneticAlgorithm(population_size=50, generations=100)),
        ("PSO", ParticleSwarmOptimization(n_particles=30, iterations=100)),
        ("Simulated Annealing", SimulatedAnnealing(initial_temp=100, iterations=500, step_size=0.2)),
    ]

    for name, alg in meta_algorithms:
        print(f"  {name}...", end=" ")
        try:
            x_opt, history = alg.optimize(func, x0)
            results[name] = (x_opt, history)
            print(f"Final value: {history[-1][1]:.6f}")
        except Exception as e:
            print(f"Failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Final Value':<15} {'Iterations':<12} {'Distance to Opt'}")
    print("-" * 60)

    for name, (x_opt, history) in results.items():
        final_val = history[-1][1]
        iterations = len(history)
        distance = np.linalg.norm(x_opt - func.global_minimum)
        print(f"{name:<25} {final_val:<15.6f} {iterations:<12} {distance:.6f}")

    # Visualizations
    print("\nGenerating visualizations...")

    # Path comparison
    compare_algorithms_visual(func, results)
    plt.show()

    # Convergence comparison
    histories = [results[name][1] for name in results.keys()]
    labels = list(results.keys())
    plot_convergence(histories, labels)
    plt.suptitle(f"Convergence Comparison on {func.name}", fontsize=14)
    plt.show()

    # Final comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    final_values = [results[name][1][-1][1] for name in names]
    iterations = [len(results[name][1]) for name in names]

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, final_values, alpha=0.7)

    # Color bars by performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    sorted_indices = np.argsort(final_values)
    for idx, bar in enumerate(bars):
        bar.set_color(colors[np.where(sorted_indices == idx)[0][0]])

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Final Function Value')
    ax.set_title(f'Final Values Comparison on {func.name}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=func(func.global_minimum), color='r', linestyle='--',
               label=f'Global Minimum ({func(func.global_minimum):.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare optimization algorithms')
    parser.add_argument('--function', type=str, default='rosenbrock',
                       help=f'Function to optimize: {list(FUNCTIONS.keys())}')
    args = parser.parse_args()

    run_comparison(args.function)
