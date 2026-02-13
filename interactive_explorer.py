"""
Interactive Optimization Explorer

Experiment with different algorithms, functions, and parameters interactively.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

from algorithms.gradient_based import GradientDescent, Momentum, Adam
from algorithms.metaheuristic import GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing
from test_functions.benchmark_functions import FUNCTIONS
from utils.visualization import plot_function_contour


class OptimizationExplorer:
    """Interactive visualization for exploring optimization algorithms."""

    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_ui()
        self.current_func = FUNCTIONS['rosenbrock']()
        self.current_algorithm = 'gradient_descent'
        self.learning_rate = 0.001
        self.history = None
        self.start_point = None

    def setup_ui(self):
        """Setup the user interface."""
        # Main plot area
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_convergence = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        self.ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
        self.ax_info.axis('off')

        # Function selector
        ax_func = plt.axes([0.1, 0.02, 0.15, 0.15])
        self.func_radio = RadioButtons(ax_func, list(FUNCTIONS.keys()))
        self.func_radio.on_clicked(self.change_function)

        # Algorithm selector
        ax_alg = plt.axes([0.3, 0.02, 0.15, 0.15])
        algorithms = ['gradient_descent', 'momentum', 'adam', 'pso', 'genetic', 'simulated_annealing']
        self.alg_radio = RadioButtons(ax_alg, algorithms)
        self.alg_radio.on_clicked(self.change_algorithm)

        # Run button
        ax_run = plt.axes([0.5, 0.05, 0.1, 0.04])
        self.btn_run = Button(ax_run, 'Run Optimization')
        self.btn_run.on_clicked(self.run_optimization)

        # Reset button
        ax_reset = plt.axes([0.5, 0.1, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

        # Learning rate slider
        ax_lr = plt.axes([0.65, 0.05, 0.25, 0.03])
        self.slider_lr = Slider(ax_lr, 'Learning Rate', 0.0001, 0.1,
                                valinit=self.learning_rate, valstep=0.0001)
        self.slider_lr.on_changed(self.update_lr)

        self.draw_function()

    def draw_function(self):
        """Draw the current function landscape."""
        self.ax_main.clear()
        plot_function_contour(self.current_func, ax=self.ax_main)
        self.ax_main.set_title(f'{self.current_func.name} - Click to set start point')

        # Enable click to set start point
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.draw()

    def on_click(self, event):
        """Handle click to set starting point."""
        if event.inaxes == self.ax_main:
            self.start_point = np.array([event.xdata, event.ydata])
            self.ax_main.plot(event.xdata, event.ydata, 'go', markersize=15,
                            label='Start', zorder=5)
            plt.draw()

    def change_function(self, label):
        """Change the test function."""
        self.current_func = FUNCTIONS[label]()
        self.history = None
        self.start_point = None
        self.draw_function()
        self.update_info()

    def change_algorithm(self, label):
        """Change the optimization algorithm."""
        self.current_algorithm = label
        self.update_info()

    def update_lr(self, val):
        """Update learning rate."""
        self.learning_rate = val

    def run_optimization(self, event):
        """Run the selected optimization algorithm."""
        if self.start_point is None:
            print("Please click on the plot to set a starting point!")
            return

        # Create algorithm
        if self.current_algorithm == 'gradient_descent':
            alg = GradientDescent(learning_rate=self.learning_rate, max_iterations=500)
        elif self.current_algorithm == 'momentum':
            alg = Momentum(learning_rate=self.learning_rate, max_iterations=500)
        elif self.current_algorithm == 'adam':
            alg = Adam(learning_rate=self.learning_rate, max_iterations=500)
        elif self.current_algorithm == 'pso':
            alg = ParticleSwarmOptimization(n_particles=30, iterations=100)
        elif self.current_algorithm == 'genetic':
            alg = GeneticAlgorithm(population_size=50, generations=100)
        elif self.current_algorithm == 'simulated_annealing':
            alg = SimulatedAnnealing(initial_temp=100, iterations=500)

        # Run optimization
        print(f"Running {self.current_algorithm} on {self.current_func.name}...")
        x_opt, self.history = alg.optimize(self.current_func, self.start_point)

        # Update visualization
        self.draw_results()
        self.update_info()

    def draw_results(self):
        """Draw optimization results."""
        if self.history is None:
            return

        # Redraw function
        self.draw_function()

        # Draw path
        points = np.array([h[0] for h in self.history])
        self.ax_main.plot(points[:, 0], points[:, 1], 'r.-',
                         linewidth=2, markersize=6, alpha=0.7, label='Path')
        self.ax_main.plot(points[0, 0], points[0, 1], 'go',
                         markersize=15, label='Start')
        self.ax_main.plot(points[-1, 0], points[-1, 1], 'bs',
                         markersize=15, label='End')
        self.ax_main.legend()

        # Convergence plot
        self.ax_convergence.clear()
        values = [h[1] for h in self.history]
        self.ax_convergence.semilogy(values, 'b-', linewidth=2)
        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('Function Value (log)')
        self.ax_convergence.set_title('Convergence')
        self.ax_convergence.grid(True, alpha=0.3)

        plt.draw()

    def update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = f"Function: {self.current_func.name}\n"
        info_text += f"Algorithm: {self.current_algorithm}\n"
        info_text += f"Learning Rate: {self.learning_rate:.4f}\n\n"

        if self.start_point is not None:
            info_text += f"Start: [{self.start_point[0]:.2f}, {self.start_point[1]:.2f}]\n"

        if self.history is not None:
            final_point = self.history[-1][0]
            final_value = self.history[-1][1]
            info_text += f"\nResults:\n"
            info_text += f"Final: [{final_point[0]:.4f}, {final_point[1]:.4f}]\n"
            info_text += f"Value: {final_value:.6f}\n"
            info_text += f"Iterations: {len(self.history)}\n"
            info_text += f"Distance to optimum: "
            info_text += f"{np.linalg.norm(final_point - self.current_func.global_minimum):.4f}\n"

        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.draw()

    def reset(self, event):
        """Reset the visualization."""
        self.history = None
        self.start_point = None
        self.draw_function()
        self.ax_convergence.clear()
        self.update_info()
        plt.draw()


def main():
    print("=" * 60)
    print("INTERACTIVE OPTIMIZATION EXPLORER")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Select a function (left panel)")
    print("2. Select an algorithm (middle panel)")
    print("3. Click on the plot to set a starting point")
    print("4. Adjust learning rate if needed (bottom slider)")
    print("5. Click 'Run Optimization' to see the algorithm in action")
    print("6. Experiment with different combinations!")
    print("\nNote: Learning rate only affects gradient-based methods")
    print("=" * 60)

    explorer = OptimizationExplorer()
    plt.show()


if __name__ == "__main__":
    main()
