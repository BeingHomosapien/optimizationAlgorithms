"""
Visualization utilities for optimization algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation


def plot_function_contour(func, ax=None, n_points=100, show_minimum=True):
    """
    Plot contour map of a 2D function.

    Args:
        func: Function object with bounds
        ax: Matplotlib axis (creates new if None)
        n_points: Resolution of the grid
        show_minimum: Whether to mark the global minimum
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    bounds = np.array(func.bounds)
    x = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
    y = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # Contour plot
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)

    if show_minimum and func.global_minimum is not None:
        ax.plot(func.global_minimum[0], func.global_minimum[1],
                'r*', markersize=20, label='Global Minimum', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{func.name} Function')

    return ax


def plot_3d_surface(func, n_points=50):
    """Plot 3D surface of a 2D function."""
    fig = plt.figure(figsize=(12, 5))

    bounds = np.array(func.bounds)
    x = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
    y = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.set_title(f'{func.name} - 3D Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour
    ax2 = fig.add_subplot(122)
    plot_function_contour(func, ax=ax2, n_points=n_points)

    plt.tight_layout()
    return fig


def plot_optimization_path(func, history, algorithm_name, ax=None):
    """
    Plot the optimization path on a contour map.

    Args:
        func: Function being optimized
        history: List of (x, f(x)) tuples from optimization
        algorithm_name: Name for the title
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot function contour
    plot_function_contour(func, ax=ax)

    # Extract path
    points = np.array([h[0] for h in history])

    # Plot path
    ax.plot(points[:, 0], points[:, 1], 'r.-', linewidth=2,
            markersize=8, label=f'{algorithm_name} Path', alpha=0.7)
    ax.plot(points[0, 0], points[0, 1], 'go', markersize=15,
            label='Start', zorder=5)
    ax.plot(points[-1, 0], points[-1, 1], 'bs', markersize=15,
            label='End', zorder=5)

    ax.legend()
    ax.set_title(f'{algorithm_name} on {func.name}')

    return ax


def plot_convergence(histories, labels):
    """
    Plot convergence curves for multiple algorithms.

    Args:
        histories: List of optimization histories
        labels: List of algorithm names
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for history, label in zip(histories, labels):
        values = [h[1] for h in history]
        iterations = range(len(values))

        # Linear scale
        ax1.plot(iterations, values, label=label, linewidth=2)

        # Log scale
        ax2.semilogy(iterations, values, label=label, linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function Value')
    ax1.set_title('Convergence (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value (log scale)')
    ax2.set_title('Convergence (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(func, history, algorithm_name, interval=50, filename=None):
    """
    Create an animated visualization of the optimization process.

    Args:
        func: Function being optimized
        history: Optimization history
        algorithm_name: Algorithm name
        interval: Delay between frames in ms
        filename: If provided, save animation to this file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Setup contour plot
    plot_function_contour(func, ax=ax1)
    points = np.array([h[0] for h in history])
    values = [h[1] for h in history]

    # Initialize plots
    path_line, = ax1.plot([], [], 'r.-', linewidth=2, markersize=8, alpha=0.7)
    current_point, = ax1.plot([], [], 'bo', markersize=15)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    convergence_line, = ax2.plot([], [], 'b-', linewidth=2)

    def init():
        path_line.set_data([], [])
        current_point.set_data([], [])
        convergence_line.set_data([], [])
        return path_line, current_point, convergence_line

    def update(frame):
        # Update path
        path_line.set_data(points[:frame+1, 0], points[:frame+1, 1])
        current_point.set_data([points[frame, 0]], [points[frame, 1]])

        # Update convergence
        convergence_line.set_data(range(frame+1), values[:frame+1])
        ax2.relim()
        ax2.autoscale_view()

        ax1.set_title(f'{algorithm_name} - Iteration {frame}, f(x)={values[frame]:.6f}')

        return path_line, current_point, convergence_line

    anim = FuncAnimation(fig, update, init_func=init, frames=len(history),
                        interval=interval, blit=True, repeat=True)

    if filename:
        anim.save(filename, writer='pillow')

    return anim


def compare_algorithms_visual(func, algorithms_results):
    """
    Visual comparison of multiple algorithms.

    Args:
        func: Test function
        algorithms_results: Dict of {algorithm_name: (x_opt, history)}
    """
    n_algs = len(algorithms_results)
    fig, axes = plt.subplots(2, (n_algs + 1) // 2, figsize=(16, 10))
    axes = axes.flatten() if n_algs > 1 else [axes]

    for idx, (name, (x_opt, history)) in enumerate(algorithms_results.items()):
        ax = axes[idx]
        plot_optimization_path(func, history, name, ax=ax)

        final_value = history[-1][1]
        ax.text(0.02, 0.98, f'Final: {final_value:.6f}\nIters: {len(history)}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for idx in range(n_algs, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Algorithm Comparison on {func.name}', fontsize=16)
    plt.tight_layout()

    return fig
