"""
Gradient Descent Simulation

Demonstrates basic gradient descent on different test functions.
Watch how it follows the steepest descent direction at each step.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.gradient_based import GradientDescent
from test_functions.benchmark_functions import Sphere, Rosenbrock, Rastrigin
from utils.visualization import plot_optimization_path, plot_convergence, plot_3d_surface


def main():
    print("=" * 60)
    print("GRADIENT DESCENT SIMULATION")
    print("=" * 60)

    # Test on different functions
    functions = [
        (Sphere(), np.array([4.0, 4.0]), 0.1),
        (Rosenbrock(), np.array([-1.0, 2.0]), 0.001),
        (Rastrigin(), np.array([3.0, 3.0]), 0.01),
    ]

    fig = plt.figure(figsize=(18, 5))

    for idx, (func, x0, lr) in enumerate(functions):
        print(f"\nOptimizing {func.name} function...")
        print(f"  Starting point: {x0}")
        print(f"  Learning rate: {lr}")

        # Run optimization
        optimizer = GradientDescent(learning_rate=lr, max_iterations=500)
        x_opt, history = optimizer.optimize(func, x0)

        # Results
        final_value = func(x_opt)
        print(f"  Final position: {x_opt}")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Iterations: {len(history)}")
        print(f"  Distance to global minimum: {np.linalg.norm(x_opt - func.global_minimum):.6f}")

        # Visualize
        ax = fig.add_subplot(1, 3, idx + 1)
        plot_optimization_path(func, history, "Gradient Descent", ax=ax)

    plt.suptitle("Gradient Descent on Different Functions", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Detailed view: Rosenbrock with 3D surface
    print("\n" + "=" * 60)
    print("Detailed Analysis: Rosenbrock Function")
    print("=" * 60)

    func = Rosenbrock()
    x0 = np.array([-1.0, 2.0])

    # Show the function landscape
    plot_3d_surface(func)
    plt.suptitle("Rosenbrock Function Landscape", fontsize=14)
    plt.show()

    # Compare different learning rates
    learning_rates = [0.0001, 0.001, 0.005]
    histories = []
    labels = []

    for lr in learning_rates:
        optimizer = GradientDescent(learning_rate=lr, max_iterations=1000)
        x_opt, history = optimizer.optimize(func, x0)
        histories.append(history)
        labels.append(f'LR = {lr}')
        print(f"LR={lr}: Final value = {history[-1][1]:.6f}, Iterations = {len(history)}")

    plot_convergence(histories, labels)
    plt.suptitle("Effect of Learning Rate on Convergence", fontsize=14)
    plt.show()

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. Sphere: Converges quickly (convex, simple)")
    print("2. Rosenbrock: Slow convergence (narrow valley)")
    print("3. Rastrigin: May get stuck in local minima")
    print("4. Learning rate is critical:")
    print("   - Too small: slow convergence")
    print("   - Too large: oscillations or divergence")
    print("=" * 60)


if __name__ == "__main__":
    main()
