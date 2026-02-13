"""
Quick Start Guide

Run this to get a quick overview of optimization algorithms!
"""
import numpy as np
import matplotlib.pyplot as plt

from algorithms.gradient_based import GradientDescent, Adam
from algorithms.metaheuristic import GeneticAlgorithm, ParticleSwarmOptimization
from test_functions.benchmark_functions import Rosenbrock, Rastrigin
from utils.visualization import plot_3d_surface, compare_algorithms_visual, plot_convergence


def main():
    print("=" * 70)
    print(" " * 15 + "OPTIMIZATION ALGORITHMS - QUICK START")
    print("=" * 70)
    print()
    print("Welcome! This demo will show you:")
    print("  1. What optimization problems look like")
    print("  2. How different algorithms solve them")
    print("  3. When to use which algorithm")
    print()
    print("Close each window to continue to the next demo...")
    print()

    # Part 1: Show what we're optimizing
    print("=" * 70)
    print("PART 1: Understanding the Landscape")
    print("=" * 70)
    print()
    print("Optimization is about finding the lowest point (minimum) on a surface.")
    print("Let's look at two classic test functions:")
    print()

    func1 = Rosenbrock()
    print(f"1. {func1.name}: Has a narrow curved valley - tricky for algorithms!")

    func2 = Rastrigin(dim=2)
    print(f"2. {func2.name}: Has many local minima - easy to get trapped!")
    print()

    input("Press Enter to see the landscapes...")

    # Show 3D surfaces
    plot_3d_surface(func1)
    plt.suptitle("Rosenbrock Function - Narrow Valley Challenge", fontsize=14)
    plt.show()

    plot_3d_surface(func2)
    plt.suptitle("Rastrigin Function - Many Local Minima", fontsize=14)
    plt.show()

    # Part 2: Gradient-based vs Gradient-free
    print("\n" + "=" * 70)
    print("PART 2: Two Families of Algorithms")
    print("=" * 70)
    print()
    print("GRADIENT-BASED (use calculus):")
    print("  - Gradient Descent: Follow the slope downhill")
    print("  - Adam: Smart adaptive learning (popular in AI)")
    print("  - Pros: Fast, precise on smooth functions")
    print("  - Cons: Can get stuck in local minima, need derivatives")
    print()
    print("METAHEURISTIC (inspired by nature):")
    print("  - Genetic Algorithm: Mimics evolution")
    print("  - Particle Swarm: Mimics bird flocking")
    print("  - Pros: Work on any function, escape local minima")
    print("  - Cons: Slower, need more function evaluations")
    print()

    input("Press Enter to see them in action on Rosenbrock...")

    # Compare on Rosenbrock
    func = Rosenbrock()
    x0 = np.array([-1.0, 2.0])

    print("\nRunning algorithms...")
    results = {}

    # Gradient Descent
    gd = GradientDescent(learning_rate=0.001, max_iterations=500)
    x_opt, history = gd.optimize(func, x0)
    results["Gradient Descent"] = (x_opt, history)
    print(f"  Gradient Descent: {history[-1][1]:.6f} in {len(history)} iterations")

    # Adam
    adam = Adam(learning_rate=0.01, max_iterations=500)
    x_opt, history = adam.optimize(func, x0)
    results["Adam"] = (x_opt, history)
    print(f"  Adam: {history[-1][1]:.6f} in {len(history)} iterations")

    # PSO
    pso = ParticleSwarmOptimization(n_particles=30, iterations=100)
    x_opt, history = pso.optimize(func, x0)
    results["PSO"] = (x_opt, history)
    print(f"  PSO: {history[-1][1]:.6f} in {len(history)} iterations")

    # GA
    ga = GeneticAlgorithm(population_size=50, generations=100)
    x_opt, history = ga.optimize(func, x0)
    results["Genetic Algorithm"] = (x_opt, history)
    print(f"  Genetic Algorithm: {history[-1][1]:.6f} in {len(history)} iterations")

    # Visualize paths
    compare_algorithms_visual(func, results)
    plt.show()

    # Convergence comparison
    histories = [results[name][1] for name in results.keys()]
    labels = list(results.keys())
    plot_convergence(histories, labels)
    plt.suptitle("Convergence Speed Comparison", fontsize=14)
    plt.show()

    # Part 3: Multi-modal challenge
    print("\n" + "=" * 70)
    print("PART 3: The Multi-Modal Challenge")
    print("=" * 70)
    print()
    print("Rastrigin has MANY local minima. Gradient methods often get stuck.")
    print("Metaheuristic methods explore more and handle this better.")
    print()

    input("Press Enter to see the comparison on Rastrigin...")

    func = Rastrigin(dim=2)
    x0 = np.array([3.0, 3.0])

    print("\nRunning algorithms on multi-modal function...")
    results = {}

    # Gradient Descent
    gd = GradientDescent(learning_rate=0.01, max_iterations=500)
    x_opt, history = gd.optimize(func, x0)
    results["Gradient Descent"] = (x_opt, history)
    print(f"  Gradient Descent: {history[-1][1]:.6f} (target: 0.0)")

    # PSO
    pso = ParticleSwarmOptimization(n_particles=30, iterations=100)
    x_opt, history = pso.optimize(func, x0)
    results["PSO"] = (x_opt, history)
    print(f"  PSO: {history[-1][1]:.6f} (target: 0.0)")

    # GA
    ga = GeneticAlgorithm(population_size=50, generations=100)
    x_opt, history = ga.optimize(func, x0)
    results["Genetic Algorithm"] = (x_opt, history)
    print(f"  Genetic Algorithm: {history[-1][1]:.6f} (target: 0.0)")

    compare_algorithms_visual(func, results)
    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Different landscapes need different algorithms")
    print("  2. Gradient methods: fast but can get stuck")
    print("  3. Metaheuristics: slower but more robust")
    print("  4. No single 'best' algorithm - it depends on your problem!")
    print()
    print("What to explore next:")
    print("  - Run individual simulations:")
    print("    python simulations/gradient_descent_sim.py")
    print("    python simulations/pso_sim.py")
    print()
    print("  - Compare all algorithms:")
    print("    python compare_algorithms.py --function ackley")
    print()
    print("  - Interactive exploration:")
    print("    python interactive_explorer.py")
    print()
    print("  - Read the code! Each algorithm has detailed comments")
    print("=" * 70)


if __name__ == "__main__":
    main()
