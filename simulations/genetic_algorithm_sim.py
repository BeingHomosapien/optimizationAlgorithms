"""
Genetic Algorithm Simulation

Demonstrates evolutionary optimization - no gradients needed!
Watch the population evolve over generations.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.metaheuristic import GeneticAlgorithm
from test_functions.benchmark_functions import Rastrigin, Ackley, Himmelblau
from utils.visualization import plot_optimization_path, plot_convergence


def main():
    print("=" * 60)
    print("GENETIC ALGORITHM SIMULATION")
    print("=" * 60)
    print("\nGenetic Algorithms mimic natural evolution:")
    print("  - Population of candidate solutions")
    print("  - Selection (survival of the fittest)")
    print("  - Crossover (combining good solutions)")
    print("  - Mutation (random exploration)")
    print()

    # Test on multimodal functions (many local minima)
    # GA is good at finding global optimum in these cases
    functions = [
        Rastrigin(dim=2),
        Ackley(dim=2),
        Himmelblau(),
    ]

    fig = plt.figure(figsize=(18, 5))

    all_histories = []
    all_labels = []

    for idx, func in enumerate(functions):
        print(f"\nOptimizing {func.name} function...")
        print(f"  Population size: 50")
        print(f"  Generations: 100")

        # Run optimization
        ga = GeneticAlgorithm(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        x_opt, history = ga.optimize(func)

        # Results
        final_value = func(x_opt)
        print(f"  Final position: {x_opt}")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Global minimum value: {func(func.global_minimum):.6f}")

        all_histories.append(history)
        all_labels.append(func.name)

        # Visualize path
        ax = fig.add_subplot(1, 3, idx + 1)
        plot_optimization_path(func, history, "Genetic Algorithm", ax=ax)

    plt.suptitle("Genetic Algorithm on Multimodal Functions", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convergence comparison
    plot_convergence(all_histories, all_labels)
    plt.suptitle("GA Convergence on Different Landscapes", fontsize=14)
    plt.show()

    # Detailed: show effect of population size
    print("\n" + "=" * 60)
    print("Effect of Population Size")
    print("=" * 60)

    func = Rastrigin(dim=2)
    population_sizes = [10, 30, 50, 100]
    histories = []
    labels = []

    for pop_size in population_sizes:
        print(f"\nPopulation size: {pop_size}")
        ga = GeneticAlgorithm(
            population_size=pop_size,
            generations=50,
            mutation_rate=0.1
        )
        x_opt, history = ga.optimize(func)
        histories.append(history)
        labels.append(f'Pop = {pop_size}')
        print(f"  Final value: {history[-1][1]:.6f}")

    plot_convergence(histories, labels)
    plt.suptitle("Effect of Population Size on GA Performance", fontsize=14)
    plt.show()

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. GA doesn't need gradients - works on any function")
    print("2. Good at escaping local minima (through mutation)")
    print("3. Population diversity is key to exploration")
    print("4. Larger populations:")
    print("   - Better exploration")
    print("   - More computational cost")
    print("   - Better for complex landscapes")
    print("5. Path is not smooth (stochastic jumps)")
    print("=" * 60)


if __name__ == "__main__":
    main()
