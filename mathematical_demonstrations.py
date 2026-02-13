"""
Interactive Mathematical Demonstrations

Visualize the mathematics behind optimization algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

from test_functions.benchmark_functions import Rosenbrock, Sphere
from algorithms import *


def demonstrate_gradient_descent_math():
    """Show Taylor approximation and descent direction."""
    print("=" * 70)
    print("GRADIENT DESCENT: MATHEMATICAL INTUITION")
    print("=" * 70)
    print()
    print("Taylor expansion: f(x + p) ≈ f(x) + ∇f(x)ᵀp")
    print("For p = -α∇f(x):")
    print("  f(x - α∇f(x)) ≈ f(x) - α‖∇f(x)‖²")
    print()
    print("This is ALWAYS a descent (for small α > 0)!")
    print()

    # Create a simple quadratic
    def f(x):
        return x[0]**2 + 4*x[1]**2

    def grad_f(x):
        return np.array([2*x[0], 8*x[1]])

    x = np.array([2.0, 1.0])
    grad = grad_f(x)

    # Show Taylor approximation
    alphas = np.linspace(0, 0.5, 100)
    f_actual = [f(x - alpha * grad) for alpha in alphas]
    f_taylor = [f(x) - alpha * np.dot(grad, grad) for alpha in alphas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Taylor approximation quality
    ax1.plot(alphas, f_actual, 'b-', linewidth=2, label='Actual f(x - α∇f)')
    ax1.plot(alphas, f_taylor, 'r--', linewidth=2, label='Taylor approximation')
    ax1.axhline(y=f(x), color='g', linestyle=':', label='f(x)')
    ax1.set_xlabel('Step size α', fontsize=12)
    ax1.set_ylabel('Function value', fontsize=12)
    ax1.set_title('Taylor Approximation Quality', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient direction on contour
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 4*Y**2

    ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax2.plot(x[0], x[1], 'ro', markersize=12, label='Current x')

    # Show gradient direction
    scale = 0.3
    ax2.arrow(x[0], x[1], -scale*grad[0], -scale*grad[1],
             head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax2.text(x[0]-scale*grad[0]-0.3, x[1]-scale*grad[1], '-∇f(x)',
            fontsize=12, color='red')

    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('Gradient Points to Steepest Descent', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def demonstrate_momentum_math():
    """Show how momentum accumulates gradients."""
    print("\n" + "=" * 70)
    print("MOMENTUM: EXPONENTIAL MOVING AVERAGE")
    print("=" * 70)
    print()
    print("v_k = β·v_{k-1} + ∇f(x_k)")
    print("    = Σᵢ₌₀^∞ βⁱ·∇f(x_{k-i})")
    print()
    print("This is an exponential moving average!")
    print(f"For β=0.9: effective memory ≈ 1/(1-β) = 10 steps")
    print()

    # Simulate gradient sequence with oscillation
    np.random.seed(42)
    n_steps = 50
    gradients = np.zeros((n_steps, 2))

    # Create oscillating gradients in one direction, consistent in another
    for i in range(n_steps):
        gradients[i, 0] = 1.0 + 0.1*np.random.randn()  # Consistent
        gradients[i, 1] = 0.5*(-1)**i + 0.1*np.random.randn()  # Oscillating

    # Compute momentum
    beta = 0.9
    velocity = np.zeros((n_steps, 2))
    for i in range(1, n_steps):
        velocity[i] = beta * velocity[i-1] + gradients[i]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw gradients vs momentum (dimension 1)
    axes[0, 0].plot(gradients[:, 0], 'b-', alpha=0.5, label='Raw gradient')
    axes[0, 0].plot(velocity[:, 0], 'r-', linewidth=2, label='Momentum')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Gradient (dimension 1)', fontsize=12)
    axes[0, 0].set_title('Consistent Direction: Momentum Accelerates', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Raw gradients vs momentum (dimension 2)
    axes[0, 1].plot(gradients[:, 1], 'b-', alpha=0.5, label='Raw gradient')
    axes[0, 1].plot(velocity[:, 1], 'r-', linewidth=2, label='Momentum')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Gradient (dimension 2)', fontsize=12)
    axes[0, 1].set_title('Oscillating Direction: Momentum Dampens', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Exponential weights
    weights = [beta**i for i in range(20)]
    axes[1, 0].bar(range(20), weights, alpha=0.7)
    axes[1, 0].set_xlabel('Steps back in time', fontsize=12)
    axes[1, 0].set_ylabel(f'Weight (β={beta})', fontsize=12)
    axes[1, 0].set_title('Exponential Weighting of Past Gradients', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Effect of different beta values
    betas = [0.5, 0.7, 0.9, 0.95, 0.99]
    for beta in betas:
        v = np.zeros(n_steps)
        for i in range(1, n_steps):
            v[i] = beta * v[i-1] + gradients[i, 0]
        axes[1, 1].plot(v, label=f'β={beta}', linewidth=2)

    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Velocity', fontsize=12)
    axes[1, 1].set_title('Effect of Momentum Coefficient β', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_newton_method_math():
    """Show quadratic approximation and Newton step."""
    print("\n" + "=" * 70)
    print("NEWTON'S METHOD: QUADRATIC APPROXIMATION")
    print("=" * 70)
    print()
    print("Second-order Taylor: f(x+p) ≈ f(x) + ∇f(x)ᵀp + ½pᵀHp")
    print("Minimize w.r.t. p: p* = -H⁻¹∇f(x)")
    print()
    print("For quadratic functions: converges in ONE step!")
    print()

    # 1D example for visualization
    def f(x):
        return 0.5 * x**2 - 2*x + 3

    def df(x):
        return x - 2

    def d2f(x):
        return 1.0

    x_plot = np.linspace(-1, 5, 200)
    f_plot = f(x_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot function
    ax1.plot(x_plot, f_plot, 'b-', linewidth=2, label='f(x)')

    # Starting point
    x0 = 4.0
    ax1.plot(x0, f(x0), 'ro', markersize=12, label='x₀')

    # Quadratic approximation
    p_range = np.linspace(-3, 1, 100)
    quadratic_approx = f(x0) + df(x0)*p_range + 0.5*d2f(x0)*p_range**2
    ax1.plot(x0 + p_range, quadratic_approx, 'r--', linewidth=2,
            label='Quadratic approximation')

    # Newton step
    p_newton = -df(x0) / d2f(x0)
    x1 = x0 + p_newton
    ax1.plot(x1, f(x1), 'g*', markersize=15, label='x₁ (Newton step)')
    ax1.axvline(x=2, color='gray', linestyle=':', label='True minimum')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Newton Step Minimizes Quadratic Approximation', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2D contour showing Newton direction
    def f2d(x):
        return x[0]**2 + 4*x[1]**2

    def grad2d(x):
        return np.array([2*x[0], 8*x[1]])

    def hess2d(x):
        return np.array([[2, 0], [0, 8]])

    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 4*Y**2

    ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

    x_current = np.array([2.0, 1.5])
    ax2.plot(x_current[0], x_current[1], 'ro', markersize=12, label='Current x')

    # Newton direction
    p_newton = -np.linalg.solve(hess2d(x_current), grad2d(x_current))
    x_next = x_current + p_newton

    ax2.arrow(x_current[0], x_current[1], p_newton[0], p_newton[1],
             head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax2.plot(x_next[0], x_next[1], 'g*', markersize=15, label='Next x (Newton)')

    # Compare with gradient descent
    alpha = 0.1
    p_gd = -alpha * grad2d(x_current)
    ax2.arrow(x_current[0], x_current[1], p_gd[0], p_gd[1],
             head_width=0.15, head_length=0.1, fc='red', ec='red',
             linewidth=2, alpha=0.5)

    ax2.plot(0, 0, 'k*', markersize=15, label='Optimum')
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('Newton (green) vs Gradient Descent (red)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def demonstrate_adam_math():
    """Show Adam's moment estimates."""
    print("\n" + "=" * 70)
    print("ADAM: ADAPTIVE MOMENT ESTIMATION")
    print("=" * 70)
    print()
    print("m_k = β₁·m_{k-1} + (1-β₁)·∇f(x_k)     [first moment: mean]")
    print("v_k = β₂·v_{k-1} + (1-β₂)·∇f(x_k)²    [second moment: variance]")
    print()
    print("Bias correction needed because m₀=0, v₀=0")
    print()

    # Simulate gradient sequence
    np.random.seed(42)
    n_steps = 100
    true_gradient = 1.0
    noise_std = 0.5
    gradients = true_gradient + noise_std * np.random.randn(n_steps)

    # Compute moments
    beta1, beta2 = 0.9, 0.999
    m = np.zeros(n_steps)
    v = np.zeros(n_steps)
    m_hat = np.zeros(n_steps)
    v_hat = np.zeros(n_steps)

    for k in range(1, n_steps):
        m[k] = beta1 * m[k-1] + (1 - beta1) * gradients[k]
        v[k] = beta2 * v[k-1] + (1 - beta2) * gradients[k]**2

        m_hat[k] = m[k] / (1 - beta1**(k+1))
        v_hat[k] = v[k] / (1 - beta2**(k+1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw gradients and first moment
    axes[0, 0].plot(gradients, 'b.', alpha=0.3, label='Noisy gradients')
    axes[0, 0].axhline(y=true_gradient, color='g', linestyle='--', label='True gradient')
    axes[0, 0].plot(m, 'r-', linewidth=2, label='m (first moment)')
    axes[0, 0].plot(m_hat, 'orange', linewidth=2, label='m̂ (bias corrected)')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Value', fontsize=12)
    axes[0, 0].set_title('First Moment (Mean Estimate)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Second moment
    true_second_moment = true_gradient**2 + noise_std**2
    axes[0, 1].plot(gradients**2, 'b.', alpha=0.3, label='Squared gradients')
    axes[0, 1].axhline(y=true_second_moment, color='g', linestyle='--',
                      label='True 2nd moment')
    axes[0, 1].plot(v, 'r-', linewidth=2, label='v (second moment)')
    axes[0, 1].plot(v_hat, 'orange', linewidth=2, label='v̂ (bias corrected)')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Value', fontsize=12)
    axes[0, 1].set_title('Second Moment (Variance Estimate)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bias correction effect
    k_range = np.arange(1, 50)
    bias1 = 1 - beta1**k_range
    bias2 = 1 - beta2**k_range

    axes[1, 0].plot(k_range, bias1, 'b-', linewidth=2, label=f'1 - β₁^k (β₁={beta1})')
    axes[1, 0].plot(k_range, bias2, 'r-', linewidth=2, label=f'1 - β₂^k (β₂={beta2})')
    axes[1, 0].set_xlabel('Iteration k', fontsize=12)
    axes[1, 0].set_ylabel('Bias correction factor', fontsize=12)
    axes[1, 0].set_title('Why Bias Correction is Needed', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Effective learning rate
    alpha = 0.001
    epsilon = 1e-8
    effective_lr = alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    axes[1, 1].plot(effective_lr, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Effective learning rate', fontsize=12)
    axes[1, 1].set_title('Adaptive Learning Rate in Adam', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_pso_math():
    """Visualize PSO particle dynamics."""
    print("\n" + "=" * 70)
    print("PARTICLE SWARM OPTIMIZATION: VELOCITY UPDATE")
    print("=" * 70)
    print()
    print("v = w·v + c₁·r₁⊙(p_best - x) + c₂·r₂⊙(g_best - x)")
    print()
    print("Three components:")
    print("  1. Inertia (w·v): current momentum")
    print("  2. Cognitive (c₁): attraction to personal best")
    print("  3. Social (c₂): attraction to global best")
    print()

    # Simulate one particle update
    x = np.array([2.0, 2.0])
    v = np.array([-0.5, 0.3])
    p_best = np.array([1.0, 0.5])
    g_best = np.array([0.0, 0.0])

    w, c1, c2 = 0.7, 1.5, 1.5
    r1, r2 = 0.5, 0.8  # Fixed for visualization

    # Compute components
    inertia = w * v
    cognitive = c1 * r1 * (p_best - x)
    social = c2 * r2 * (g_best - x)
    v_new = inertia + cognitive + social

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Velocity decomposition
    ax1.quiver(0, 0, inertia[0], inertia[1], angles='xy', scale_units='xy',
              scale=1, color='blue', width=0.01, label='Inertia')
    ax1.quiver(inertia[0], inertia[1], cognitive[0], cognitive[1],
              angles='xy', scale_units='xy', scale=1, color='green',
              width=0.01, label='Cognitive')
    ax1.quiver(inertia[0]+cognitive[0], inertia[1]+cognitive[1],
              social[0], social[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='Social')
    ax1.quiver(0, 0, v_new[0], v_new[1], angles='xy', scale_units='xy',
              scale=1, color='black', width=0.015, label='New velocity')

    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('Velocity Update Components', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-3, 1)
    ax1.set_ylim(-2, 2)

    # Particle trajectory
    func = Sphere(dim=2)
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax2.plot(x[0], x[1], 'bo', markersize=12, label='Current position')
    ax2.plot(p_best[0], p_best[1], 'g^', markersize=12, label='Personal best')
    ax2.plot(g_best[0], g_best[1], 'r*', markersize=15, label='Global best')

    # Show velocity
    ax2.arrow(x[0], x[1], v_new[0]*0.5, v_new[1]*0.5,
             head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)

    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('PSO Particle Update', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def demonstrate_convergence_rates():
    """Compare convergence rates visually."""
    print("\n" + "=" * 70)
    print("CONVERGENCE RATES COMPARISON")
    print("=" * 70)
    print()
    print("Linear:      ‖xₖ - x*‖ ≤ c^k·‖x₀ - x*‖")
    print("Superlinear: lim ‖xₖ₊₁ - x*‖/‖xₖ - x*‖ = 0")
    print("Quadratic:   ‖xₖ₊₁ - x*‖ ≤ C·‖xₖ - x*‖²")
    print()

    k = np.arange(0, 20)

    # Linear convergence (different rates)
    linear_slow = 0.9**k
    linear_fast = 0.5**k

    # Superlinear (L-BFGS typical)
    superlinear = 0.9**k * k**(-0.5)

    # Quadratic (Newton)
    quadratic = np.zeros(20)
    quadratic[0] = 1.0
    for i in range(1, 20):
        quadratic[i] = min(quadratic[i-1]**2, 1e-16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.plot(k, linear_slow, 'b-', linewidth=2, label='Linear (c=0.9)')
    ax1.plot(k, linear_fast, 'g-', linewidth=2, label='Linear (c=0.5)')
    ax1.plot(k, superlinear, 'orange', linewidth=2, label='Superlinear')
    ax1.plot(k[:15], quadratic[:15], 'r-', linewidth=2, label='Quadratic')

    ax1.set_xlabel('Iteration k', fontsize=12)
    ax1.set_ylabel('‖xₖ - x*‖', fontsize=12)
    ax1.set_title('Convergence Rates (Linear Scale)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(k, linear_slow, 'b-', linewidth=2, label='Linear (c=0.9)')
    ax2.semilogy(k, linear_fast, 'g-', linewidth=2, label='Linear (c=0.5)')
    ax2.semilogy(k, superlinear, 'orange', linewidth=2, label='Superlinear')
    ax2.semilogy(k[:15], quadratic[:15], 'r-', linewidth=2, label='Quadratic')

    ax2.set_xlabel('Iteration k', fontsize=12)
    ax2.set_ylabel('‖xₖ - x*‖ (log scale)', fontsize=12)
    ax2.set_title('Convergence Rates (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MATHEMATICAL DEMONSTRATIONS" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This interactive demo visualizes the mathematics behind")
    print("optimization algorithms.")
    print()

    demos = [
        ("Gradient Descent Math", demonstrate_gradient_descent_math),
        ("Momentum Math", demonstrate_momentum_math),
        ("Newton's Method Math", demonstrate_newton_method_math),
        ("Adam Math", demonstrate_adam_math),
        ("PSO Math", demonstrate_pso_math),
        ("Convergence Rates", demonstrate_convergence_rates),
    ]

    for i, (name, func) in enumerate(demos, 1):
        print(f"{i}. {name}")

    print(f"{len(demos)+1}. Run all demonstrations")
    print("0. Exit")
    print()

    choice = input("Select demonstration (0-{}): ".format(len(demos)+1)).strip()

    if choice == "0":
        return
    elif choice == str(len(demos)+1):
        for name, func in demos:
            func()
            input("\nPress Enter for next demonstration...")
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice)-1][1]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
