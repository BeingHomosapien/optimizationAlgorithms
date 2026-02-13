"""
Advanced Gradient-Based Optimization Algorithms

These are more sophisticated methods used in modern machine learning and scientific computing.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class AdaGrad:
    """
    AdaGrad: Adaptive Gradient Algorithm

    Adapts learning rate for each parameter based on historical gradients.
    Good for sparse data but learning rate can become too small.

    Update: x_new = x - lr * g / sqrt(G + ε)
    where G is sum of squared gradients for each parameter
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-8, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        grad_squared_sum = np.zeros_like(x)
        self.history = [(x.copy(), func(x))]

        for i in range(self.max_iterations):
            grad = func.gradient(x)

            # Accumulate squared gradients
            grad_squared_sum += grad**2

            # Adaptive learning rate update
            adjusted_lr = self.learning_rate / (np.sqrt(grad_squared_sum) + self.epsilon)
            x_new = x - adjusted_lr * grad

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class NAdam:
    """
    NAdam: Nesterov-accelerated Adaptive Moment Estimation

    Combines Adam with Nesterov momentum for better convergence.
    Often faster than Adam on some problems.
    """

    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        self.history = [(x.copy(), func(x))]

        for t in range(1, self.max_iterations + 1):
            grad = func.gradient(x)

            # Update biased first moment
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second moment
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            # Nesterov momentum
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - self.beta1**t)

            # Update
            x_new = x - self.learning_rate * m_nesterov / (np.sqrt(v_hat) + self.epsilon)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay

    Fixes weight decay in Adam. Better generalization in deep learning.
    Weight decay is applied directly to weights, not through gradient.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        self.history = [(x.copy(), func(x))]

        for t in range(1, self.max_iterations + 1):
            grad = func.gradient(x)

            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            # Update with decoupled weight decay
            x_new = x - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) +
                                             self.weight_decay * x)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history


class LBFGS:
    """
    L-BFGS: Limited-memory Broyden-Fletcher-Goldfarb-Shanno

    Quasi-Newton method that approximates the Hessian (second derivatives).
    Uses limited memory - stores only recent update vectors.

    Very popular in scientific computing and machine learning.
    Often converges faster than first-order methods with fewer iterations.
    """

    def __init__(self, max_iterations=100, m=10, tolerance=1e-6, c1=1e-4, c2=0.9):
        self.max_iterations = max_iterations
        self.m = m  # Number of correction pairs to store
        self.tolerance = tolerance
        self.c1 = c1  # Armijo condition parameter
        self.c2 = c2  # Curvature condition parameter
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        self.history = [(x.copy(), func(x))]

        # Storage for updates
        s_history = []  # Steps
        y_history = []  # Gradient differences
        rho_history = []

        grad = func.gradient(x)

        for iteration in range(self.max_iterations):
            # Compute search direction using L-BFGS two-loop recursion
            q = grad.copy()
            alpha_history = []

            # First loop (backward)
            for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
                alpha = rho * np.dot(s, q)
                alpha_history.append(alpha)
                q = q - alpha * y

            # Scaling
            if len(y_history) > 0:
                gamma = np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1])
                q = gamma * q

            # Second loop (forward)
            for s, y, rho, alpha in zip(s_history, y_history, rho_history, reversed(alpha_history)):
                beta = rho * np.dot(y, q)
                q = q + s * (alpha - beta)

            # Search direction (negative for minimization)
            direction = -q

            # Line search (simple backtracking)
            step_size = self._line_search(func, x, direction, grad)

            # Update
            x_new = x + step_size * direction
            grad_new = func.gradient(x_new)

            # Store for L-BFGS update
            s = x_new - x
            y = grad_new - grad
            rho = 1.0 / (np.dot(y, s) + 1e-10)

            s_history.append(s)
            y_history.append(y)
            rho_history.append(rho)

            # Maintain limited memory
            if len(s_history) > self.m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            # Check convergence
            if np.linalg.norm(grad_new) < self.tolerance:
                break

            x = x_new
            grad = grad_new

        return x, self.history

    def _line_search(self, func, x, direction, grad, alpha_init=1.0):
        """Simple backtracking line search."""
        alpha = alpha_init
        rho = 0.5
        c = 1e-4

        f_x = func(x)
        grad_dot_dir = np.dot(grad, direction)

        while func(x + alpha * direction) > f_x + c * alpha * grad_dot_dir:
            alpha *= rho
            if alpha < 1e-10:
                break

        return alpha


class ConjugateGradient:
    """
    Conjugate Gradient Method

    More sophisticated than gradient descent - search directions are conjugate
    to each other, leading to faster convergence on quadratic functions.

    Particularly effective for large-scale problems.
    """

    def __init__(self, max_iterations=1000, tolerance=1e-6, restart_iterations=None):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.restart_iterations = restart_iterations  # Periodic restart
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        self.history = [(x.copy(), func(x))]

        grad = func.gradient(x)
        direction = -grad.copy()

        for iteration in range(self.max_iterations):
            # Line search
            alpha = self._line_search(func, x, direction, grad)

            # Update position
            x_new = x + alpha * direction
            grad_new = func.gradient(x_new)

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            # Check convergence
            if np.linalg.norm(grad_new) < self.tolerance:
                break

            # Polak-Ribière formula for beta
            beta = max(0, np.dot(grad_new, grad_new - grad) / (np.dot(grad, grad) + 1e-10))

            # New direction
            direction = -grad_new + beta * direction

            # Periodic restart (use steepest descent)
            if self.restart_iterations and (iteration + 1) % self.restart_iterations == 0:
                direction = -grad_new

            x = x_new
            grad = grad_new

        return x, self.history

    def _line_search(self, func, x, direction, grad):
        """Backtracking line search."""
        alpha = 1.0
        rho = 0.8
        c = 1e-4

        f_x = func(x)
        grad_dot_dir = np.dot(grad, direction)

        max_backtracks = 30
        for _ in range(max_backtracks):
            if func(x + alpha * direction) <= f_x + c * alpha * grad_dot_dir:
                break
            alpha *= rho

        return alpha


class NewtonMethod:
    """
    Newton's Method

    Uses second-order information (Hessian matrix) for faster convergence.
    Quadratic convergence near the optimum!

    Update: x_new = x - H^(-1) * grad
    where H is the Hessian (matrix of second derivatives)

    Note: Requires computing and inverting the Hessian - expensive!
    """

    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

    def optimize(self, func, x0):
        x = np.array(x0, dtype=float)
        self.history = [(x.copy(), func(x))]

        for iteration in range(self.max_iterations):
            grad = func.gradient(x)

            # Compute Hessian (numerically)
            hessian = self._numerical_hessian(func, x)

            # Solve H * p = -grad for search direction p
            try:
                # Use Cholesky decomposition for stability
                c, low = cho_factor(hessian)
                direction = cho_solve((c, low), -grad)
            except np.linalg.LinAlgError:
                # If Hessian is not positive definite, fall back to gradient descent
                direction = -grad

            # Line search for step size
            alpha = self._line_search(func, x, direction, grad)

            # Update
            x_new = x + alpha * direction

            f_new = func(x_new)
            self.history.append((x_new.copy(), f_new))

            # Check convergence
            if np.linalg.norm(grad) < self.tolerance:
                break

            x = x_new

        return x, self.history

    def _numerical_hessian(self, func, x, eps=1e-5):
        """Compute Hessian numerically using finite differences."""
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += eps
                x_pp[j] += eps

                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm[i] -= eps
                x_mm[j] -= eps

                H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps**2)
                H[j, i] = H[i, j]

        # Regularize to ensure positive definiteness
        H += 1e-4 * np.eye(n)

        return H

    def _line_search(self, func, x, direction, grad):
        """Backtracking line search."""
        alpha = 1.0
        rho = 0.8
        c = 1e-4

        f_x = func(x)
        grad_dot_dir = np.dot(grad, direction)

        for _ in range(30):
            if func(x + alpha * direction) <= f_x + c * alpha * grad_dot_dir:
                break
            alpha *= rho

        return alpha
