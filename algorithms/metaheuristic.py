"""
Metaheuristic optimization algorithms.
These don't require gradients and are inspired by natural phenomena.
"""
import numpy as np


class GeneticAlgorithm:
    """
    Genetic Algorithm: inspired by natural evolution

    Process:
    1. Initialize random population
    2. Evaluate fitness (function value)
    3. Select best individuals (parents)
    4. Crossover: combine parents to create offspring
    5. Mutate: random changes for diversity
    6. Repeat

    Good for: discrete problems, multimodal landscapes, no gradient available
    """

    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 crossover_rate=0.8, elite_size=2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size  # Keep best individuals
        self.history = []

    def optimize(self, func, x0=None):
        """
        Optimize function using genetic algorithm.
        x0 is ignored - we use random initialization based on func.bounds
        """
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Initialize random population
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.population_size, dim)
        )

        best_individual = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            # Evaluate fitness (lower is better)
            fitness = np.array([func(ind) for ind in population])

            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            self.history.append((best_individual.copy(), best_fitness))

            # Selection: tournament selection
            selected_indices = self._tournament_selection(fitness)
            selected = population[selected_indices]

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = population[elite_indices]

            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                offspring.extend([child1, child2])

            offspring = np.array(offspring[:self.population_size - self.elite_size])

            # Mutation
            for i in range(len(offspring)):
                if np.random.rand() < self.mutation_rate:
                    offspring[i] = self._mutate(offspring[i], bounds)

            # New population
            population = np.vstack([elite, offspring])

        return best_individual, self.history

    def _tournament_selection(self, fitness, tournament_size=3):
        """Select individuals using tournament selection."""
        selected = []
        for _ in range(len(fitness)):
            tournament = np.random.choice(len(fitness), tournament_size, replace=False)
            winner = tournament[np.argmin(fitness[tournament])]
            selected.append(winner)
        return np.array(selected)

    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual, bounds):
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.rand() < 0.3:  # Gene mutation probability
                mutated[i] += np.random.normal(0, 0.1 * (bounds[i, 1] - bounds[i, 0]))
                mutated[i] = np.clip(mutated[i], bounds[i, 0], bounds[i, 1])
        return mutated


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization (PSO): inspired by bird flocking

    Each particle has:
    - Position (current solution)
    - Velocity (direction and speed of movement)
    - Personal best (best position it has found)

    Particles update velocity based on:
    - Inertia (current velocity)
    - Cognitive component (attraction to personal best)
    - Social component (attraction to global best)

    Good for: continuous optimization, doesn't need gradients
    """

    def __init__(self, n_particles=30, iterations=100, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive (personal) weight
        self.c2 = c2  # Social (global) weight
        self.history = []

    def optimize(self, func, x0=None):
        """Optimize function using PSO."""
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Initialize particles
        positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.n_particles, dim)
        )
        velocities = np.random.uniform(
            -1, 1, size=(self.n_particles, dim)
        )

        # Personal bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])

        # Global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        self.history.append((global_best_position.copy(), global_best_score))

        for iteration in range(self.iterations):
            for i in range(self.n_particles):
                # Random factors
                r1, r2 = np.random.rand(dim), np.random.rand(dim)

                # Update velocity
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social

                # Update position
                positions[i] = positions[i] + velocities[i]

                # Enforce bounds
                positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])

                # Evaluate
                score = func(positions[i])

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

            self.history.append((global_best_position.copy(), global_best_score))

        return global_best_position, self.history


class SimulatedAnnealing:
    """
    Simulated Annealing: inspired by metallurgy

    Mimics the process of slowly cooling metal to reach minimum energy state.

    Process:
    1. Start at high "temperature" - accept worse solutions frequently
    2. Gradually cool - become more selective
    3. At low temperature - only accept improvements (like gradient descent)

    Key idea: accepting worse solutions early helps escape local minima

    Good for: avoiding local minima, combinatorial optimization
    """

    def __init__(self, initial_temp=100, cooling_rate=0.95, iterations=1000, step_size=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.step_size = step_size
        self.history = []

    def optimize(self, func, x0):
        """Optimize function using simulated annealing."""
        bounds = np.array(func.bounds)

        # Current solution
        current = np.array(x0, dtype=float)
        current_energy = func(current)

        # Best solution found
        best = current.copy()
        best_energy = current_energy

        self.history.append((best.copy(), best_energy))

        temperature = self.initial_temp

        for iteration in range(self.iterations):
            # Generate neighbor (random perturbation)
            neighbor = current + np.random.normal(0, self.step_size, size=len(current))
            neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])

            neighbor_energy = func(neighbor)

            # Energy difference
            delta_energy = neighbor_energy - current_energy

            # Accept or reject
            if delta_energy < 0:
                # Better solution - always accept
                current = neighbor
                current_energy = neighbor_energy
            else:
                # Worse solution - accept with probability based on temperature
                acceptance_prob = np.exp(-delta_energy / temperature)
                if np.random.rand() < acceptance_prob:
                    current = neighbor
                    current_energy = neighbor_energy

            # Update best
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy

            self.history.append((best.copy(), best_energy))

            # Cool down
            temperature *= self.cooling_rate

        return best, self.history
