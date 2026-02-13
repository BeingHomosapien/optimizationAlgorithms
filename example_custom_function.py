"""
Example: Optimizing a Custom Function

Shows how to define your own optimization problem and solve it.
"""
import numpy as np
import matplotlib.pyplot as plt

from algorithms import GradientDescent, Adam, ParticleSwarmOptimization
from test_functions.benchmark_functions import OptimizationFunction
from utils.visualization import plot_optimization_path, plot_3d_surface


class CustomFunction(OptimizationFunction):
    """
    Example: A custom quadratic bowl with a twist.

    f(x, y) = (x - 2)^2 + 2*(y + 1)^2 + sin(4*x) + cos(4*y)

    The sine and cosine terms add small ripples to the surface.
    """

    def __init__(self):
        self.dim = 2
        self.bounds = [(-5, 5), (-5, 5)]
        self.global_minimum = np.array([2.0, -1.0])  # Approximate
        self.name = "Custom Function"

    def __call__(self, x):
        """Evaluate the function."""
        return (x[0] - 2)**2 + 2*(x[1] + 1)**2 + np.sin(4*x[0]) + np.cos(4*x[1])

    def gradient(self, x):
        """Analytical gradient for faster gradient-based optimization."""
        dx = 2*(x[0] - 2) + 4*np.cos(4*x[0])
        dy = 4*(x[1] + 1) - 4*np.sin(4*x[1])
        return np.array([dx, dy])


def main():
    print("=" * 60)
    print("CUSTOM FUNCTION OPTIMIZATION EXAMPLE")
    print("=" * 60)
    print()
    print("This example shows how to:")
    print("  1. Define your own optimization problem")
    print("  2. Provide analytical gradients (optional but faster)")
    print("  3. Apply different algorithms to your problem")
    print()

    # Create custom function
    func = CustomFunction()

    # Show the landscape
    print("Visualizing the function landscape...")
    plot_3d_surface(func)
    plt.suptitle("Custom Function with Ripples", fontsize=14)
    plt.show()

    # Starting point
    x0 = np.array([0.0, 3.0])
    print(f"Starting point: {x0}")
    print(f"Starting value: {func(x0):.6f}")
    print()

    # Try different algorithms
    algorithms = {
        "Gradient Descent": GradientDescent(learning_rate=0.01, max_iterations=500),
        "Adam": Adam(learning_rate=0.1, max_iterations=500),
        "PSO": ParticleSwarmOptimization(n_particles=30, iterations=100),
    }

    results = {}

    for name, alg in algorithms.items():
        print(f"Running {name}...")
        x_opt, history = alg.optimize(func, x0)
        results[name] = (x_opt, history)

        final_value = func(x_opt)
        print(f"  Final position: [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Iterations: {len(history)}")
        print()

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, (x_opt, history)) in enumerate(results.items()):
        plot_optimization_path(func, history, name, ax=axes[idx])

    plt.suptitle("Comparison on Custom Function", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("Creating Your Own Function - Tips:")
    print("=" * 60)
    print()
    print("1. Inherit from OptimizationFunction")
    print("2. Set dim, bounds, global_minimum, and name")
    print("3. Implement __call__(self, x) to evaluate f(x)")
    print("4. Optionally implement gradient(self, x) for speed")
    print("5. If no gradient provided, numerical gradient is used")
    print()
    print("Your function can be:")
    print("  - Mathematical formula (like above)")
    print("  - Simulation output")
    print("  - Machine learning loss function")
    print("  - Engineering design objective")
    print("  - Anything you can compute!")
    print("=" * 60)


if __name__ == "__main__":
    main()
