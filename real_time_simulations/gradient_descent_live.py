"""
Real-time Gradient Descent Visualization

Watch gradient descent optimize a function in real-time with live updates!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from test_functions.benchmark_functions import Rosenbrock, Sphere, Rastrigin, Beale


class RealTimeGradientDescent:
    """Real-time visualization of gradient descent optimization."""

    def __init__(self, func, x0, learning_rate=0.01, max_iterations=500):
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        # Optimization state
        self.x_current = self.x0.copy()
        self.history = [self.x0.copy()]
        self.values = [func(self.x0)]
        self.iteration = 0
        self.running = False
        self.speed = 50  # milliseconds per iteration

        # Setup visualization
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(16, 8))

        # Main contour plot
        self.ax_contour = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

        # Convergence plot
        self.ax_convergence = plt.subplot2grid((2, 3), (0, 2))

        # Info panel
        self.ax_info = plt.subplot2grid((2, 3), (1, 2))
        self.ax_info.axis('off')

        # Draw the function landscape
        self.draw_landscape()

        # Initialize plots
        self.path_line, = self.ax_contour.plot([], [], 'r.-', linewidth=2,
                                                markersize=8, label='Path', zorder=5)
        self.current_point, = self.ax_contour.plot([], [], 'bo', markersize=15,
                                                    label='Current', zorder=6)
        self.gradient_arrow = None

        self.convergence_line, = self.ax_convergence.semilogy([], [], 'b-', linewidth=2)
        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('f(x) (log scale)')
        self.ax_convergence.set_title('Convergence')
        self.ax_convergence.grid(True, alpha=0.3)

        # Add controls
        self.add_controls()

        self.ax_contour.legend(loc='upper right')
        plt.tight_layout()

    def draw_landscape(self):
        """Draw the function contour."""
        bounds = np.array(self.func.bounds)
        x = np.linspace(bounds[0, 0], bounds[0, 1], 200)
        y = np.linspace(bounds[1, 0], bounds[1, 1], 200)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))

        # Contour plot
        levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
        self.ax_contour.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        self.ax_contour.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)

        # Mark global minimum
        if self.func.global_minimum is not None:
            self.ax_contour.plot(self.func.global_minimum[0],
                                self.func.global_minimum[1],
                                'r*', markersize=20, label='Global Min', zorder=10)

        self.ax_contour.set_xlabel('x₁')
        self.ax_contour.set_ylabel('x₂')
        self.ax_contour.set_title(f'{self.func.name} - Gradient Descent (α={self.learning_rate})')

    def add_controls(self):
        """Add control buttons and sliders."""
        # Start/Stop button
        ax_button_start = plt.axes([0.7, 0.02, 0.08, 0.04])
        self.btn_start = Button(ax_button_start, 'Start')
        self.btn_start.on_clicked(self.toggle_running)

        # Reset button
        ax_button_reset = plt.axes([0.79, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_button_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        # Speed slider
        ax_slider_speed = plt.axes([0.7, 0.08, 0.17, 0.02])
        self.slider_speed = Slider(ax_slider_speed, 'Speed', 1, 200,
                                   valinit=self.speed, valstep=1)
        self.slider_speed.on_changed(self.update_speed)

    def toggle_running(self, event):
        """Toggle animation on/off."""
        self.running = not self.running
        self.btn_start.label.set_text('Pause' if self.running else 'Start')

    def reset(self, event):
        """Reset the optimization."""
        self.running = False
        self.btn_start.label.set_text('Start')
        self.x_current = self.x0.copy()
        self.history = [self.x0.copy()]
        self.values = [self.func(self.x0)]
        self.iteration = 0
        self.update_plot()

    def update_speed(self, val):
        """Update animation speed."""
        self.speed = int(val)

    def step(self):
        """Perform one gradient descent step."""
        if self.iteration >= self.max_iterations:
            self.running = False
            self.btn_start.label.set_text('Done')
            return

        # Compute gradient
        grad = self.func.gradient(self.x_current)

        # Gradient descent update
        self.x_current = self.x_current - self.learning_rate * grad

        # Store history
        self.history.append(self.x_current.copy())
        self.values.append(self.func(self.x_current))
        self.iteration += 1

    def update_plot(self):
        """Update the visualization."""
        if len(self.history) == 0:
            return

        # Update path
        path = np.array(self.history)
        self.path_line.set_data(path[:, 0], path[:, 1])

        # Update current point
        self.current_point.set_data([self.x_current[0]], [self.x_current[1]])

        # Update gradient arrow
        if self.gradient_arrow:
            self.gradient_arrow.remove()

        if self.iteration > 0 and self.iteration < self.max_iterations:
            grad = self.func.gradient(self.x_current)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                scale = 0.3
                self.gradient_arrow = self.ax_contour.arrow(
                    self.x_current[0], self.x_current[1],
                    -scale * grad[0] / (grad_norm + 1e-10),
                    -scale * grad[1] / (grad_norm + 1e-10),
                    head_width=0.1, head_length=0.08,
                    fc='orange', ec='orange', linewidth=2,
                    alpha=0.7, zorder=7
                )

        # Update convergence plot
        self.convergence_line.set_data(range(len(self.values)), self.values)
        self.ax_convergence.relim()
        self.ax_convergence.autoscale_view()

        # Update info panel
        self.update_info()

    def update_info(self):
        """Update the info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        current_value = self.values[-1] if self.values else 0
        grad_norm = np.linalg.norm(self.func.gradient(self.x_current))
        distance_to_min = np.linalg.norm(self.x_current - self.func.global_minimum)

        info_text = f"""
Iteration: {self.iteration}/{self.max_iterations}

Current Position:
  x = [{self.x_current[0]:.4f}, {self.x_current[1]:.4f}]

Current Value:
  f(x) = {current_value:.6f}

Gradient Norm:
  ‖∇f‖ = {grad_norm:.6f}

Distance to Optimum:
  ‖x - x*‖ = {distance_to_min:.6f}

Learning Rate: {self.learning_rate}
"""

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def animate(self, frame):
        """Animation update function."""
        if self.running:
            self.step()
            self.update_plot()
        return self.path_line, self.current_point, self.convergence_line

    def run(self):
        """Start the real-time visualization."""
        self.anim = FuncAnimation(self.fig, self.animate, interval=self.speed,
                                 blit=False, cache_frame_data=False)
        plt.show()


def main():
    print("=" * 70)
    print("REAL-TIME GRADIENT DESCENT VISUALIZATION")
    print("=" * 70)
    print()
    print("Watch gradient descent optimize in real-time!")
    print()
    print("Controls:")
    print("  - Start/Pause: Start or pause the optimization")
    print("  - Reset: Reset to initial position")
    print("  - Speed slider: Control animation speed")
    print()
    print("=" * 70)
    print()

    # Choose function
    functions = {
        '1': (Sphere(dim=2), np.array([4.0, 4.0]), 0.1),
        '2': (Rosenbrock(), np.array([-1.0, 2.0]), 0.001),
        '3': (Rastrigin(dim=2), np.array([3.0, 3.0]), 0.01),
        '4': (Beale(), np.array([1.0, 1.0]), 0.001),
    }

    print("Choose a function:")
    print("1. Sphere (easy, convex)")
    print("2. Rosenbrock (narrow valley)")
    print("3. Rastrigin (many local minima)")
    print("4. Beale (multiple valleys)")
    print()

    choice = input("Enter choice (1-4, default=2): ").strip() or '2'

    if choice in functions:
        func, x0, lr = functions[choice]
        print(f"\nOptimizing {func.name} function...")
        print(f"Starting point: {x0}")
        print(f"Learning rate: {lr}")
        print()

        viz = RealTimeGradientDescent(func, x0, learning_rate=lr, max_iterations=500)
        viz.run()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
