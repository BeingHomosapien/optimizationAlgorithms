"""
Benchmark functions for testing optimization algorithms.
Each function has a known global minimum for validation.
"""
import numpy as np


class OptimizationFunction:
    """Base class for optimization test functions."""

    def __init__(self):
        self.bounds = None
        self.global_minimum = None
        self.name = "Unknown"

    def __call__(self, x):
        """Evaluate function at point x."""
        raise NotImplementedError

    def gradient(self, x):
        """Analytical gradient (if available)."""
        # Numerical gradient by default
        eps = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * eps)
        return grad


class Sphere(OptimizationFunction):
    """Simple convex function: f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0
    Easy to optimize - good for testing basic algorithms.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(-5, 5)] * dim
        self.global_minimum = np.zeros(dim)
        self.name = "Sphere"

    def __call__(self, x):
        return np.sum(x**2)

    def gradient(self, x):
        return 2 * x


class Rosenbrock(OptimizationFunction):
    """Rosenbrock's banana function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Global minimum: f(1, 1) = 0
    Has a narrow curved valley - challenging for gradient methods.
    """

    def __init__(self):
        self.dim = 2
        self.bounds = [(-2, 2), (-1, 3)]
        self.global_minimum = np.array([1.0, 1.0])
        self.name = "Rosenbrock"

    def __call__(self, x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def gradient(self, x):
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])


class Rastrigin(OptimizationFunction):
    """Rastrigin function: highly multimodal
    Global minimum: f(0,...,0) = 0
    Many local minima - tests global search capability.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(-5.12, 5.12)] * dim
        self.global_minimum = np.zeros(dim)
        self.name = "Rastrigin"

    def __call__(self, x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def gradient(self, x):
        A = 10
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


class Ackley(OptimizationFunction):
    """Ackley function: many local minima
    Global minimum: f(0,...,0) = 0
    Nearly flat outer region, central peak.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(-5, 5)] * dim
        self.global_minimum = np.zeros(dim)
        self.name = "Ackley"

    def __call__(self, x):
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


class Beale(OptimizationFunction):
    """Beale function: multimodal in 2D
    Global minimum: f(3, 0.5) = 0
    Multiple valleys.
    """

    def __init__(self):
        self.dim = 2
        self.bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        self.global_minimum = np.array([3.0, 0.5])
        self.name = "Beale"

    def __call__(self, x):
        term1 = (1.5 - x[0] + x[0] * x[1])**2
        term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
        term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
        return term1 + term2 + term3


class Himmelblau(OptimizationFunction):
    """Himmelblau's function: has 4 identical local minima
    Global minima: f(3,2) = f(-2.805,3.131) = f(-3.779,-3.283) = f(3.584,-1.848) = 0
    Tests multi-modal optimization.
    """

    def __init__(self):
        self.dim = 2
        self.bounds = [(-5, 5), (-5, 5)]
        self.global_minimum = np.array([3.0, 2.0])  # One of four
        self.name = "Himmelblau"

    def __call__(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


# Dictionary for easy access
FUNCTIONS = {
    'sphere': Sphere,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'ackley': Ackley,
    'beale': Beale,
    'himmelblau': Himmelblau,
}
