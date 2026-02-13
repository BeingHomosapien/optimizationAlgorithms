"""
Advanced Optimization Algorithms Simulation

Compare state-of-the-art optimization methods.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.advanced_gradient import AdaGrad, NAdam, AdamW, LBFGS, ConjugateGradient, NewtonMethod
from algorithms.advanced_metaheuristic import DifferentialEvolution, CMAES, NelderMead, BayesianOptimization, HarmonySearch
from test_functions.benchmark_functions import Rosenbrock, Rastrigin, Ackley
from utils.visualization import plot_optimization_path, plot_convergence


def compare_advanced_gradient_methods():
    """Compare advanced gradient-based optimizers."""
    print("=" * 70)
    print("ADVANCED GRADIENT-BASED METHODS")
    print("=" * 70)
    print("\nThese are state-of-the-art first and second-order methods.")
    print()

    func = Rosenbrock()
    x0 = np.array([-1.0, 2.0])

    algorithms = {
        "AdaGrad": AdaGrad(learning_rate=0.5, max_iterations=500),
        "NAdam": NAdam(learning_rate=0.01, max_iterations=500),
        "AdamW": AdamW(learning_rate=0.01, max_iterations=500),
        "L-BFGS": LBFGS(max_iterations=100),
        "Conjugate Gradient": ConjugateGradient(max_iterations=500),
        "Newton's Method": NewtonMethod(max_iterations=50),
    }

    results = {}

    print(f"Testing on {func.name} function...")
    print(f"Starting point: {x0}\n")

    for name, alg in algorithms.items():
        print(f"Running {name}...", end=" ")
        try:
            x_opt, history = alg.optimize(func, x0)
            results[name] = (x_opt, history)
            print(f"✓ Final: {history[-1][1]:.6f} in {len(history)} iterations")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Visualize paths
    n_algs = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, (x_opt, history)) in enumerate(results.items()):
        plot_optimization_path(func, history, name, ax=axes[idx])

    plt.suptitle("Advanced Gradient Methods on Rosenbrock", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convergence comparison
    histories = [results[name][1] for name in results.keys()]
    labels = list(results.keys())
    plot_convergence(histories, labels)
    plt.suptitle("Convergence Comparison - Gradient Methods", fontsize=14)
    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. L-BFGS: Very fast convergence (uses second-order info)")
    print("2. Newton's Method: Quadratic convergence near optimum")
    print("3. Conjugate Gradient: Good balance of speed and memory")
    print("4. NAdam: Improved Adam with Nesterov momentum")
    print("5. Second-order methods need fewer iterations but more computation per iteration")
    print("=" * 70)

    return results


def compare_advanced_metaheuristics():
    """Compare advanced derivative-free optimizers."""
    print("\n\n" + "=" * 70)
    print("ADVANCED METAHEURISTIC METHODS")
    print("=" * 70)
    print("\nState-of-the-art derivative-free global optimization.")
    print()

    func = Rastrigin(dim=2)
    x0 = np.array([3.0, 3.0])

    algorithms = {
        "Differential Evolution": DifferentialEvolution(population_size=50, generations=100),
        "CMA-ES": CMAES(population_size=30, generations=50, sigma=1.0),
        "Nelder-Mead": NelderMead(max_iterations=500),
        "Bayesian Optimization": BayesianOptimization(iterations=80),
        "Harmony Search": HarmonySearch(harmony_memory_size=30, iterations=200),
    }

    results = {}

    print(f"Testing on {func.name} function (challenging: many local minima)...")
    print(f"Global minimum: {func(func.global_minimum):.6f}\n")

    for name, alg in algorithms.items():
        print(f"Running {name}...", end=" ")
        try:
            x_opt, history = alg.optimize(func, x0)
            results[name] = (x_opt, history)
            distance = np.linalg.norm(x_opt - func.global_minimum)
            print(f"✓ Final: {history[-1][1]:.6f}, Distance: {distance:.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Visualize paths
    n_algs = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, (x_opt, history)) in enumerate(results.items()):
        if idx < len(axes):
            plot_optimization_path(func, history, name, ax=axes[idx])

    # Hide unused subplot
    if n_algs < len(axes):
        axes[-1].axis('off')

    plt.suptitle("Advanced Metaheuristics on Rastrigin", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convergence comparison
    histories = [results[name][1] for name in results.keys()]
    labels = list(results.keys())
    plot_convergence(histories, labels)
    plt.suptitle("Convergence Comparison - Metaheuristics", fontsize=14)
    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. CMA-ES: State-of-the-art for black-box optimization")
    print("2. Differential Evolution: Very robust, excellent for global optimization")
    print("3. Bayesian Optimization: Sample-efficient for expensive functions")
    print("4. Nelder-Mead: Simple, no tuning needed, good for local search")
    print("5. Harmony Search: Elegant, good balance of exploration/exploitation")
    print("=" * 70)

    return results


def second_order_vs_first_order():
    """Demonstrate power of second-order methods."""
    print("\n\n" + "=" * 70)
    print("SECOND-ORDER vs FIRST-ORDER METHODS")
    print("=" * 70)
    print("\nSecond-order methods use curvature information for faster convergence.")
    print()

    func = Rosenbrock()
    x0 = np.array([-1.0, 2.0])

    # Import first-order for comparison
    from algorithms.gradient_based import GradientDescent, Adam

    algorithms = {
        "Gradient Descent (1st order)": GradientDescent(learning_rate=0.001, max_iterations=500),
        "Adam (1st order)": Adam(learning_rate=0.01, max_iterations=500),
        "Conjugate Gradient (smart 1st)": ConjugateGradient(max_iterations=500),
        "L-BFGS (quasi-2nd)": LBFGS(max_iterations=100),
        "Newton (2nd order)": NewtonMethod(max_iterations=50),
    }

    results = {}

    for name, alg in algorithms.items():
        print(f"Running {name}...")
        x_opt, history = alg.optimize(func, x0)
        results[name] = (x_opt, history)
        print(f"  Iterations: {len(history)}, Final: {history[-1][1]:.8f}")

    # Convergence comparison
    histories = [results[name][1] for name in results.keys()]
    labels = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for history, label in zip(histories, labels):
        values = [h[1] for h in history]
        iterations = range(len(values))
        ax1.semilogy(iterations, values, label=label, linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function Value (log scale)')
    ax1.set_title('Convergence Speed Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show iterations to reach threshold
    threshold = 1e-3
    iterations_to_threshold = []
    method_names = []

    for name, (_, history) in results.items():
        values = [h[1] for h in history]
        try:
            iters = next(i for i, v in enumerate(values) if v < threshold)
        except StopIteration:
            iters = len(values)
        iterations_to_threshold.append(iters)
        method_names.append(name.split('(')[0].strip())

    bars = ax2.bar(range(len(method_names)), iterations_to_threshold, alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel(f'Iterations to reach f(x) < {threshold}')
    ax2.set_title('Convergence Speed')
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("=" * 70)
    print("1. Second-order methods converge in FAR fewer iterations")
    print("2. But: each iteration is more expensive (Hessian computation)")
    print("3. L-BFGS: Best of both worlds (quasi-Newton, limited memory)")
    print("4. For high dimensions: stick with first-order methods")
    print("5. For small dimensions & expensive functions: use second-order")
    print("=" * 70)


def when_to_use_what():
    """Guidelines for algorithm selection."""
    print("\n\n" + "=" * 70)
    print("ALGORITHM SELECTION GUIDE")
    print("=" * 70)
    print()
    print("GRADIENT-BASED METHODS:")
    print("-" * 70)
    print("├─ Adam/NAdam/AdamW:    Deep learning, default choice")
    print("├─ L-BFGS:              Scientific computing, smooth objectives")
    print("├─ Conjugate Gradient:  Large-scale, memory-constrained")
    print("├─ Newton's Method:     Small dimensions, high accuracy needed")
    print("└─ AdaGrad:             Sparse data (NLP tasks)")
    print()
    print("METAHEURISTIC METHODS:")
    print("-" * 70)
    print("├─ CMA-ES:              Black-box, < 100 dimensions, expensive evals")
    print("├─ Differential Evol.:  Robust global optimization, continuous")
    print("├─ Bayesian Opt.:       Very expensive functions (simulations, experiments)")
    print("├─ Nelder-Mead:         Simple, no tuning, derivative-free")
    print("└─ Harmony Search:      Constrained problems, good all-rounder")
    print()
    print("DECISION TREE:")
    print("-" * 70)
    print("Have gradients?")
    print("  Yes → High dimensions (>100)?")
    print("        Yes → Adam, AdamW")
    print("        No  → L-BFGS, Conjugate Gradient")
    print("  No  → Function expensive (<1000 evals budget)?")
    print("        Yes → Bayesian Optimization, CMA-ES")
    print("        No  → Differential Evolution")
    print()
    print("Need global optimum?")
    print("  Yes → Metaheuristics (CMA-ES, DE)")
    print("  No  → Gradient methods (faster to local minimum)")
    print("=" * 70)


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ADVANCED OPTIMIZATION ALGORITHMS" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Part 1: Advanced gradient methods
    grad_results = compare_advanced_gradient_methods()

    input("\nPress Enter to continue to advanced metaheuristics...")

    # Part 2: Advanced metaheuristics
    meta_results = compare_advanced_metaheuristics()

    input("\nPress Enter to see second-order vs first-order comparison...")

    # Part 3: Convergence analysis
    second_order_vs_first_order()

    input("\nPress Enter to see algorithm selection guide...")

    # Part 4: Selection guide
    when_to_use_what()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nYou've now seen:")
    print("  ✓ 6 advanced gradient-based methods")
    print("  ✓ 5 advanced metaheuristic methods")
    print("  ✓ Performance comparisons")
    print("  ✓ When to use each algorithm")
    print()
    print("The toolkit now contains 18 optimization algorithms total!")
    print()
    print("Next steps:")
    print("  - Try these on your own custom functions")
    print("  - Tune hyperparameters for your specific problems")
    print("  - Combine methods (e.g., CMA-ES → L-BFGS for refinement)")
    print("=" * 70)


if __name__ == "__main__":
    main()
