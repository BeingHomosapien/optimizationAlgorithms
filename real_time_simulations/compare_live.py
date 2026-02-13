"""
Real-time Algorithm Comparison

Watch multiple algorithms compete side-by-side in real-time!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from test_functions.benchmark_functions import Rosenbrock, Sphere, Rastrigin, Beale
from algorithms.gradient_based import GradientDescent, Momentum, Adam


class RealTimeComparison:
    """Real-time comparison of multiple optimization algorithms."""

    def __init__(self, func, x0, algorithms_config):
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        self.algorithms_config = algorithms_config
        self.max_iterations = 500

        # Initialize algorithms
        self.algorithms = {}
        self.states = {}

        for name, (AlgClass, params) in algorithms_config.items():
            self.algorithms[name] = AlgClass(**params)
            self.states[name] = {
                'x': self.x0.copy(),
                'history': [self.x0.copy()],
                'values': [func(self.x0)],
                'iteration': 0,
                'active': True
            }

        # Animation state
        self.iteration = 0
        self.running = False
        self.speed = 50

        # Setup visualization
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure."""
        n_algs = len(self.algorithms)
        self.fig = plt.figure(figsize=(18, 10))

        # Main landscape plot
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)

        # Convergence comparison
        self.ax_convergence = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

        # Info panels
        self.ax_info = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self.ax_info.axis('off')

        # Draw landscape
        self.draw_landscape()

        # Initialize algorithm plots
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        self.markers = ['o', 's', '^', 'D', 'v', 'p']

        self.path_lines = {}
        self.current_points = {}

        for idx, name in enumerate(self.algorithms.keys()):
            color = self.colors[idx % len(self.colors)]
            marker = self.markers[idx % len(self.markers)]

            line, = self.ax_main.plot([], [], '-', color=color, linewidth=1.5,
                                     alpha=0.6, label=name)
            point, = self.ax_main.plot([], [], marker, color=color, markersize=10)

            self.path_lines[name] = line
            self.current_points[name] = point

        # Convergence lines
        self.convergence_lines = {}
        for idx, name in enumerate(self.algorithms.keys()):
            color = self.colors[idx % len(self.colors)]
            line, = self.ax_convergence.semilogy([], [], color=color, linewidth=2,
                                                 label=name)
            self.convergence_lines[name] = line

        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('f(x) (log scale)')
        self.ax_convergence.set_title('Convergence Comparison')
        self.ax_convergence.legend(loc='upper right')
        self.ax_convergence.grid(True, alpha=0.3)

        # Add controls
        self.add_controls()

        self.ax_main.legend(loc='upper right', fontsize=9)
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

        levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
        self.ax_main.contour(X, Y, Z, levels=levels, cmap='gray', alpha=0.4)
        self.ax_main.contourf(X, Y, Z, levels=levels, cmap='gray', alpha=0.2)

        if self.func.global_minimum is not None:
            self.ax_main.plot(self.func.global_minimum[0],
                             self.func.global_minimum[1],
                             'r*', markersize=25, label='Global Min',
                             markeredgecolor='black', markeredgewidth=2, zorder=10)

        self.ax_main.set_xlabel('x₁')
        self.ax_main.set_ylabel('x₂')
        self.ax_main.set_title(f'{self.func.name} - Algorithm Comparison')

    def add_controls(self):
        """Add control buttons."""
        ax_button_start = plt.axes([0.4, 0.02, 0.08, 0.03])
        self.btn_start = Button(ax_button_start, 'Start')
        self.btn_start.on_clicked(self.toggle_running)

        ax_button_reset = plt.axes([0.49, 0.02, 0.08, 0.03])
        self.btn_reset = Button(ax_button_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        ax_slider_speed = plt.axes([0.6, 0.025, 0.15, 0.02])
        self.slider_speed = Slider(ax_slider_speed, 'Speed', 1, 200,
                                   valinit=self.speed, valstep=1)
        self.slider_speed.on_changed(self.update_speed)

    def toggle_running(self, event):
        """Toggle animation."""
        self.running = not self.running
        self.btn_start.label.set_text('Pause' if self.running else 'Start')

    def reset(self, event):
        """Reset all algorithms."""
        self.running = False
        self.btn_start.label.set_text('Start')
        self.iteration = 0

        for name in self.algorithms.keys():
            self.states[name] = {
                'x': self.x0.copy(),
                'history': [self.x0.copy()],
                'values': [self.func(self.x0)],
                'iteration': 0,
                'active': True
            }

        self.update_plot()

    def update_speed(self, val):
        """Update animation speed."""
        self.speed = int(val)

    def step(self):
        """Perform one iteration for all algorithms."""
        if self.iteration >= self.max_iterations:
            self.running = False
            self.btn_start.label.set_text('Done')
            return

        for name, alg in self.algorithms.items():
            state = self.states[name]

            if not state['active']:
                continue

            # Perform one step
            if isinstance(alg, GradientDescent):
                grad = self.func.gradient(state['x'])
                state['x'] = state['x'] - alg.learning_rate * grad
            elif isinstance(alg, Momentum):
                grad = self.func.gradient(state['x'])
                if not hasattr(state, 'velocity'):
                    state['velocity'] = np.zeros_like(state['x'])
                state['velocity'] = alg.beta * state['velocity'] + grad
                state['x'] = state['x'] - alg.learning_rate * state['velocity']
            elif isinstance(alg, Adam):
                grad = self.func.gradient(state['x'])
                if not hasattr(state, 'm'):
                    state['m'] = np.zeros_like(state['x'])
                    state['v'] = np.zeros_like(state['x'])
                    state['t'] = 0

                state['t'] += 1
                state['m'] = alg.beta1 * state['m'] + (1 - alg.beta1) * grad
                state['v'] = alg.beta2 * state['v'] + (1 - alg.beta2) * grad**2

                m_hat = state['m'] / (1 - alg.beta1**state['t'])
                v_hat = state['v'] / (1 - alg.beta2**state['t'])

                state['x'] = state['x'] - alg.learning_rate * m_hat / (np.sqrt(v_hat) + alg.epsilon)

            # Store history
            state['history'].append(state['x'].copy())
            state['values'].append(self.func(state['x']))
            state['iteration'] += 1

            # Check convergence
            if np.linalg.norm(self.func.gradient(state['x'])) < 1e-6:
                state['active'] = False

        self.iteration += 1

    def update_plot(self):
        """Update the visualization."""
        # Update paths and current positions
        for name, state in self.states.items():
            if len(state['history']) > 0:
                path = np.array(state['history'])
                self.path_lines[name].set_data(path[:, 0], path[:, 1])
                self.current_points[name].set_data([state['x'][0]], [state['x'][1]])

        # Update convergence
        for name, state in self.states.items():
            if len(state['values']) > 0:
                self.convergence_lines[name].set_data(range(len(state['values'])),
                                                     state['values'])

        self.ax_convergence.relim()
        self.ax_convergence.autoscale_view()

        # Update info
        self.update_info()

    def update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_lines = [f"Iteration: {self.iteration}/{self.max_iterations}\n"]

        for idx, (name, state) in enumerate(self.states.items()):
            current_val = state['values'][-1] if state['values'] else 0
            distance = np.linalg.norm(state['x'] - self.func.global_minimum)
            status = "✓" if not state['active'] else "→"

            color = self.colors[idx % len(self.colors)]
            info_lines.append(
                f"{status} {name:20s}: f(x)={current_val:10.6f}  dist={distance:8.4f}"
            )

        info_text = '\n'.join(info_lines)

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    def animate(self, frame):
        """Animation function."""
        if self.running:
            self.step()
            self.update_plot()
        return list(self.path_lines.values()) + list(self.current_points.values())

    def run(self):
        """Start the visualization."""
        self.update_plot()
        self.anim = FuncAnimation(self.fig, self.animate, interval=self.speed,
                                 blit=False, cache_frame_data=False)
        plt.show()


def main():
    print("=" * 70)
    print("REAL-TIME ALGORITHM COMPARISON")
    print("=" * 70)
    print()
    print("Watch multiple algorithms compete on the same problem!")
    print()

    # Choose function
    functions = {
        '1': (Sphere(dim=2), np.array([4.0, 4.0])),
        '2': (Rosenbrock(), np.array([-1.0, 2.0])),
        '3': (Rastrigin(dim=2), np.array([3.0, 3.0])),
        '4': (Beale(), np.array([1.0, 1.0])),
    }

    print("Choose a function:")
    print("1. Sphere")
    print("2. Rosenbrock")
    print("3. Rastrigin")
    print("4. Beale")
    print()

    choice = input("Enter choice (1-4, default=2): ").strip() or '2'

    if choice in functions:
        func, x0 = functions[choice]

        # Configure algorithms
        algorithms = {
            'GD (α=0.01)': (GradientDescent, {'learning_rate': 0.01}),
            'GD (α=0.001)': (GradientDescent, {'learning_rate': 0.001}),
            'Momentum': (Momentum, {'learning_rate': 0.001, 'beta': 0.9}),
            'Adam': (Adam, {'learning_rate': 0.01}),
        }

        print(f"\nComparing algorithms on {func.name}...")
        print("Algorithms:")
        for name in algorithms.keys():
            print(f"  - {name}")
        print()

        viz = RealTimeComparison(func, x0, algorithms)
        viz.run()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
