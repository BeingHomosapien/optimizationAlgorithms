"""
Particle Swarm Optimization Simulation

Demonstrates swarm intelligence - particles cooperate to find the optimum.
Watch how particles influence each other's movement.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.metaheuristic import ParticleSwarmOptimization
from test_functions.benchmark_functions import Rastrigin, Rosenbrock, Ackley, Sphere
from utils.visualization import plot_optimization_path, plot_convergence, plot_function_contour


def main():
    print("=" * 60)
    print("PARTICLE SWARM OPTIMIZATION SIMULATION")
    print("=" * 60)
    print("\nPSO is inspired by bird flocking:")
    print("  - Each particle remembers its best position")
    print("  - Particles share the global best position")
    print("  - Movement balances exploration and exploitation")
    print()

    # Test on various functions
    functions = [
        Sphere(dim=2),
        Rosenbrock(),
        Rastrigin(dim=2),
    ]

    fig = plt.figure(figsize=(18, 5))

    all_histories = []
    all_labels = []

    for idx, func in enumerate(functions):
        print(f"\nOptimizing {func.name} function...")
        print(f"  Number of particles: 30")
        print(f"  Iterations: 100")

        # Run optimization
        pso = ParticleSwarmOptimization(
            n_particles=30,
            iterations=100,
            w=0.7,   # Inertia
            c1=1.5,  # Cognitive (personal)
            c2=1.5   # Social (global)
        )
        x_opt, history = pso.optimize(func)

        # Results
        final_value = func(x_opt)
        print(f"  Final position: {x_opt}")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Distance to global minimum: {np.linalg.norm(x_opt - func.global_minimum):.6f}")

        all_histories.append(history)
        all_labels.append(func.name)

        # Visualize
        ax = fig.add_subplot(1, 3, idx + 1)
        plot_optimization_path(func, history, "PSO", ax=ax)

    plt.suptitle("Particle Swarm Optimization on Different Functions", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convergence
    plot_convergence(all_histories, all_labels)
    plt.suptitle("PSO Convergence", fontsize=14)
    plt.show()

    # Detailed: Effect of PSO parameters
    print("\n" + "=" * 60)
    print("Effect of Inertia Weight (w)")
    print("=" * 60)

    func = Rastrigin(dim=2)
    inertia_weights = [0.4, 0.6, 0.8, 0.9]
    histories = []
    labels = []

    for w in inertia_weights:
        print(f"\nInertia weight: {w}")
        pso = ParticleSwarmOptimization(
            n_particles=30,
            iterations=100,
            w=w,
            c1=1.5,
            c2=1.5
        )
        x_opt, history = pso.optimize(func)
        histories.append(history)
        labels.append(f'w = {w}')
        print(f"  Final value: {history[-1][1]:.6f}")

    plot_convergence(histories, labels)
    plt.suptitle("Effect of Inertia Weight on PSO", fontsize=14)
    plt.show()

    # Comparison: Social vs Cognitive influence
    print("\n" + "=" * 60)
    print("Social vs Cognitive Balance")
    print("=" * 60)

    configs = [
        (2.0, 0.5, "High Cognitive, Low Social"),
        (0.5, 2.0, "Low Cognitive, High Social"),
        (1.5, 1.5, "Balanced"),
    ]
    histories = []
    labels = []

    for c1, c2, label in configs:
        print(f"\n{label}: c1={c1}, c2={c2}")
        pso = ParticleSwarmOptimization(
            n_particles=30,
            iterations=100,
            w=0.7,
            c1=c1,
            c2=c2
        )
        x_opt, history = pso.optimize(func)
        histories.append(history)
        labels.append(label)
        print(f"  Final value: {history[-1][1]:.6f}")

    plot_convergence(histories, labels)
    plt.suptitle("Social vs Cognitive Influence in PSO", fontsize=14)
    plt.show()

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. PSO is fast and effective on many landscapes")
    print("2. Inertia weight (w):")
    print("   - High w: more exploration (global search)")
    print("   - Low w: more exploitation (local refinement)")
    print("3. Cognitive weight (c1): personal best attraction")
    print("4. Social weight (c2): global best attraction")
    print("5. Balance is key - too much social causes premature convergence")
    print("6. No gradients needed - works on any function")
    print("=" * 60)


if __name__ == "__main__":
    main()
