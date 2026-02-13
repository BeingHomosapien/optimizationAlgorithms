"""
Gradient-based optimization algorithms.
These methods use the gradient (derivative) to find descent directions.
"""
import numpy as np


class GradientDescent:
    """
    Vanilla Gradient Descent: x_new = x_old - learning_rate * gradient

    Simple but effective for smooth convex functions.
    Can be slow and oscillate on narrow valleys.
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        """
        Optimize function starting from x0.

        Args:
            func: Function object with __call__ and gradient methods
            x0: Starting point

        Returns:
            x_opt: Optimized point
            history: List of (x, f(x)) at each iteration
        """
        x = np.array(x0, dtype=float)
        self.history = [(x.copy(), func(x))]

        for i in range(self.max_iterations):
            grad = func.gradient(x)

            # Update step
            x_new = x - self.learning_rate * grad

            # Store history
            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            # Check convergence
            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class Momentum:
    """
    Gradient Descent with Momentum: accumulates past gradients

    v_new = beta * v_old + gradient
    x_new = x_old - learning_rate * v_new

    Helps accelerate convergence and dampen oscillations.
    Think of a ball rolling down a hill - it builds up speed.
    """

    def __init__(self, learning_rate=0.01, beta=0.9, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta  # Momentum coefficient (typically 0.9)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        velocity = np.zeros_like(x)
        self.history = [(x.copy(), func(x))]

        for i in range(self.max_iterations):
            grad = func.gradient(x)

            # Update velocity (accumulate momentum)
            velocity = self.beta * velocity + grad

            # Update position
            x_new = x - self.learning_rate * velocity

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class RMSprop:
    """
    RMSprop: Root Mean Square Propagation

    Adapts learning rate for each parameter by dividing by a running average
    of recent gradient magnitudes. Good for non-stationary objectives.
    """

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon  # Small constant for numerical stability
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        sq_grad_avg = np.zeros_like(x)
        self.history = [(x.copy(), func(x))]

        for i in range(self.max_iterations):
            grad = func.gradient(x)

            # Update moving average of squared gradients
            sq_grad_avg = self.beta * sq_grad_avg + (1 - self.beta) * grad**2

            # Adaptive learning rate update
            x_new = x - self.learning_rate * grad / (np.sqrt(sq_grad_avg) + self.epsilon)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class Adam:
    """
    Adam: Adaptive Moment Estimation

    Combines momentum and RMSprop. Maintains both:
    - First moment (mean) of gradients (like momentum)
    - Second moment (variance) of gradients (like RMSprop)

    Very popular in deep learning due to robust performance.
    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # First moment decay
        self.beta2 = beta2  # Second moment decay
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)  # First moment (momentum)
        v = np.zeros_like(x)  # Second moment (RMSprop)
        self.history = [(x.copy(), func(x))]

        for t in range(1, self.max_iterations + 1):
            grad = func.gradient(x)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second moment estimate
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            # Update parameters
            x_new = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history
