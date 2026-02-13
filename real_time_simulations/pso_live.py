"""
Real-time Particle Swarm Optimization Visualization

Watch the swarm explore and converge in real-time!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from test_functions.benchmark_functions import Rosenbrock, Sphere, Rastrigin, Ackley


class RealTimePSO:
    """Real-time visualization of Particle Swarm Optimization."""

    def __init__(self, func, n_particles=30, iterations=100, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.n_particles = n_particles
        self.max_iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize swarm
        bounds = np.array(func.bounds)
        self.positions = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                          size=(n_particles, 2))
        self.velocities = np.random.uniform(-1, 1, size=(n_particles, 2))

        # Personal and global bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([func(p) for p in self.positions])

        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

        # History
        self.global_best_history = [self.global_best_score]
        self.iteration = 0
        self.running = False
        self.speed = 50

        # Setup visualization
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(16, 8))

        # Main contour plot
        self.ax_swarm = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

        # Convergence plot
        self.ax_convergence = plt.subplot2grid((2, 3), (0, 2))

        # Info panel
        self.ax_info = plt.subplot2grid((2, 3), (1, 2))
        self.ax_info.axis('off')

        # Draw landscape
        self.draw_landscape()

        # Initialize particle plots
        self.particle_scatter = self.ax_swarm.scatter([], [], c='blue', s=50,
                                                      alpha=0.6, label='Particles', zorder=5)
        self.personal_best_scatter = self.ax_swarm.scatter([], [], c='green', s=30,
                                                           alpha=0.4, marker='^',
                                                           label='Personal Bests', zorder=4)
        self.global_best_plot, = self.ax_swarm.plot([], [], 'r*', markersize=20,
                                                    label='Global Best', zorder=10)

        # Velocity arrows (will be updated)
        self.velocity_arrows = []

        # Convergence plot
        self.convergence_line, = self.ax_convergence.semilogy([], [], 'b-', linewidth=2)
        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('Best f(x) (log scale)')
        self.ax_convergence.set_title('Swarm Convergence')
        self.ax_convergence.grid(True, alpha=0.3)

        # Add controls
        self.add_controls()

        self.ax_swarm.legend(loc='upper right')
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
        self.ax_swarm.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        self.ax_swarm.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)

        # Mark global minimum
        if self.func.global_minimum is not None:
            self.ax_swarm.plot(self.func.global_minimum[0],
                              self.func.global_minimum[1],
                              'y*', markersize=25, label='True Min',
                              markeredgecolor='black', markeredgewidth=2, zorder=9)

        self.ax_swarm.set_xlabel('x₁')
        self.ax_swarm.set_ylabel('x₂')
        self.ax_swarm.set_title(f'{self.func.name} - PSO ({self.n_particles} particles)')

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

        # Reinitialize swarm
        bounds = np.array(self.func.bounds)
        self.positions = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                          size=(self.n_particles, 2))
        self.velocities = np.random.uniform(-1, 1, size=(self.n_particles, 2))

        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([self.func(p) for p in self.positions])

        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

        self.global_best_history = [self.global_best_score]
        self.iteration = 0
        self.update_plot()

    def update_speed(self, val):
        """Update animation speed."""
        self.speed = int(val)

    def step(self):
        """Perform one PSO iteration."""
        if self.iteration >= self.max_iterations:
            self.running = False
            self.btn_start.label.set_text('Done')
            return

        bounds = np.array(self.func.bounds)

        for i in range(self.n_particles):
            # Random factors
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)

            # Update velocity
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social = self.c2 * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Update position
            self.positions[i] = self.positions[i] + self.velocities[i]

            # Enforce bounds
            self.positions[i] = np.clip(self.positions[i], bounds[:, 0], bounds[:, 1])

            # Evaluate
            score = self.func(self.positions[i])

            # Update personal best
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

        self.global_best_history.append(self.global_best_score)
        self.iteration += 1

    def update_plot(self):
        """Update the visualization."""
        # Update particle positions
        self.particle_scatter.set_offsets(self.positions)

        # Update personal bests
        self.personal_best_scatter.set_offsets(self.personal_best_positions)

        # Update global best
        self.global_best_plot.set_data([self.global_best_position[0]],
                                       [self.global_best_position[1]])

        # Clear old velocity arrows
        for arrow in self.velocity_arrows:
            arrow.remove()
        self.velocity_arrows = []

        # Draw velocity arrows (sample a few)
        indices = np.random.choice(self.n_particles, min(10, self.n_particles), replace=False)
        for i in indices:
            if np.linalg.norm(self.velocities[i]) > 1e-10:
                scale = 0.3
                vel_normalized = self.velocities[i] / (np.linalg.norm(self.velocities[i]) + 1e-10)
                arrow = self.ax_swarm.arrow(
                    self.positions[i, 0], self.positions[i, 1],
                    scale * vel_normalized[0], scale * vel_normalized[1],
                    head_width=0.1, head_length=0.08,
                    fc='red', ec='red', alpha=0.4, linewidth=1, zorder=6
                )
                self.velocity_arrows.append(arrow)

        # Update convergence plot
        self.convergence_line.set_data(range(len(self.global_best_history)),
                                       self.global_best_history)
        self.ax_convergence.relim()
        self.ax_convergence.autoscale_view()

        # Update info
        self.update_info()

    def update_info(self):
        """Update the info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        distance_to_min = np.linalg.norm(self.global_best_position - self.func.global_minimum)
        avg_velocity = np.mean(np.linalg.norm(self.velocities, axis=1))

        # Diversity (average distance from global best)
        diversity = np.mean(np.linalg.norm(self.positions - self.global_best_position, axis=1))

        info_text = f"""
Iteration: {self.iteration}/{self.max_iterations}

Global Best:
  x = [{self.global_best_position[0]:.4f},
       {self.global_best_position[1]:.4f}]
  f(x) = {self.global_best_score:.6f}

Distance to Optimum:
  ‖x - x*‖ = {distance_to_min:.6f}

Swarm Statistics:
  Avg Velocity: {avg_velocity:.4f}
  Diversity: {diversity:.4f}

Parameters:
  w={self.w}, c₁={self.c1}, c₂={self.c2}
  Particles: {self.n_particles}
"""

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    def animate(self, frame):
        """Animation update function."""
        if self.running:
            self.step()
            self.update_plot()
        return [self.particle_scatter, self.personal_best_scatter,
                self.global_best_plot, self.convergence_line]

    def run(self):
        """Start the real-time visualization."""
        self.update_plot()  # Initial plot
        self.anim = FuncAnimation(self.fig, self.animate, interval=self.speed,
                                 blit=False, cache_frame_data=False)
        plt.show()


def main():
    print("=" * 70)
    print("REAL-TIME PARTICLE SWARM OPTIMIZATION")
    print("=" * 70)
    print()
    print("Watch the swarm of particles explore and converge!")
    print()
    print("Controls:")
    print("  - Start/Pause: Start or pause the optimization")
    print("  - Reset: Reset swarm to random positions")
    print("  - Speed slider: Control animation speed")
    print()
    print("Visual Legend:")
    print("  - Blue circles: Current particle positions")
    print("  - Green triangles: Personal best positions")
    print("  - Red star: Global best position")
    print("  - Yellow star: True global minimum")
    print("  - Red arrows: Velocity vectors (sample)")
    print()
    print("=" * 70)
    print()

    # Choose function
    functions = {
        '1': Sphere(dim=2),
        '2': Rosenbrock(),
        '3': Rastrigin(dim=2),
        '4': Ackley(dim=2),
    }

    print("Choose a function:")
    print("1. Sphere (easy, convex)")
    print("2. Rosenbrock (narrow valley)")
    print("3. Rastrigin (many local minima)")
    print("4. Ackley (sharp peaks)")
    print()

    choice = input("Enter choice (1-4, default=3): ").strip() or '3'

    if choice in functions:
        func = functions[choice]
        print(f"\nOptimizing {func.name} function...")
        print(f"Particles: 30")
        print(f"Max iterations: 100")
        print()

        viz = RealTimePSO(func, n_particles=30, iterations=100,
                         w=0.7, c1=1.5, c2=1.5)
        viz.run()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
