"""
Advanced Metaheuristic Optimization Algorithms

State-of-the-art derivative-free optimization methods.
"""
import numpy as np
from scipy.stats import multivariate_normal


class DifferentialEvolution:
    """
    Differential Evolution (DE)

    Very effective global optimization algorithm for continuous functions.
    Uses differences between population members to create new candidates.

    Mutation: v = x_r1 + F * (x_r2 - x_r3)
    Crossover: mix v with target vector
    Selection: keep better of parent and offspring

    Often outperforms genetic algorithms on continuous optimization.
    """

    def __init__(self, population_size=50, generations=100, F=0.8, CR=0.9, strategy='best1bin'):
        self.population_size = population_size
        self.generations = generations
        self.F = F    # Differential weight (mutation strength)
        self.CR = CR  # Crossover probability
        self.strategy = strategy
        self.history = []

    def optimize(self, func, x0=None):
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Initialize population
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.population_size, dim)
        )

        # Evaluate fitness
        fitness = np.array([func(ind) for ind in population])

        # Track best
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.history.append((best_individual.copy(), best_fitness))

        for gen in range(self.generations):
            for i in range(self.population_size):
                # Mutation: select three random individuals (different from i)
                candidates = [idx for idx in range(self.population_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation strategy
                if self.strategy == 'best1bin':
                    # Use best individual
                    mutant = best_individual + self.F * (population[r2] - population[r3])
                else:  # 'rand1bin'
                    mutant = population[r1] + self.F * (population[r2] - population[r3])

                # Ensure bounds
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                # Crossover (binomial)
                cross_points = np.random.rand(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            self.history.append((best_individual.copy(), best_fitness))

        return best_individual, self.history


class CMAES:
    """
    CMA-ES: Covariance Matrix Adaptation Evolution Strategy

    State-of-the-art for continuous black-box optimization.
    Adapts a multivariate normal distribution to the problem landscape.

    Key ideas:
    - Samples from multivariate Gaussian
    - Updates mean toward better solutions
    - Adapts covariance matrix to capture problem structure
    - Very robust and parameter-free

    Often the best choice for expensive black-box optimization.
    """

    def __init__(self, population_size=None, generations=100, sigma=0.5):
        self.population_size = population_size
        self.generations = generations
        self.sigma = sigma  # Step size
        self.history = []

    def optimize(self, func, x0=None):
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Default population size
        if self.population_size is None:
            self.population_size = 4 + int(3 * np.log(dim))

        # Initialize mean
        if x0 is not None:
            mean = np.array(x0, dtype=float)
        else:
            mean = (bounds[:, 0] + bounds[:, 1]) / 2

        # Covariance matrix and related parameters
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for C
        ps = np.zeros(dim)  # Evolution path for sigma
        sigma = self.sigma

        # Strategy parameters
        mu = self.population_size // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)

        # Adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Expectation of ||N(0,I)||
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

        best_ever = None
        best_fitness_ever = float('inf')

        for gen in range(self.generations):
            # Generate population
            population = []
            for _ in range(self.population_size):
                z = np.random.randn(dim)
                y = np.dot(np.linalg.cholesky(C), z)
                x = mean + sigma * y
                x = np.clip(x, bounds[:, 0], bounds[:, 1])
                population.append(x)

            # Evaluate
            fitness = np.array([func(x) for x in population])

            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            population = [population[i] for i in sorted_indices]
            fitness = fitness[sorted_indices]

            # Track best
            if fitness[0] < best_fitness_ever:
                best_fitness_ever = fitness[0]
                best_ever = population[0].copy()

            self.history.append((best_ever.copy(), best_fitness_ever))

            # Update mean
            old_mean = mean
            mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(
                np.linalg.inv(np.linalg.cholesky(C)),
                (mean - old_mean) / sigma
            )

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN
                   < 1.4 + 2 / (dim + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma

            # Update covariance matrix
            artmp = [(population[i] - old_mean) / sigma for i in range(mu)]
            C = ((1 - c1 - cmu) * C +
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu * sum([weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu)]))

            # Update step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        return best_ever, self.history


class NelderMead:
    """
    Nelder-Mead Simplex Method

    Classic derivative-free optimization using a simplex (geometric shape).
    Reflects, expands, and contracts the simplex toward better regions.

    Operations:
    - Reflection: mirror worst point
    - Expansion: if reflection good, go further
    - Contraction: if reflection bad, shrink
    - Shrink: contract entire simplex

    Simple but effective. Default in many optimization libraries.
    """

    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []

        # Algorithm parameters
        self.alpha = 1.0   # Reflection
        self.gamma = 2.0   # Expansion
        self.rho = 0.5     # Contraction
        self.sigma = 0.5   # Shrink

    def optimize(self, func, x0):
        dim = len(x0)

        # Initialize simplex (dim+1 vertices)
        simplex = [np.array(x0, dtype=float)]
        for i in range(dim):
            vertex = np.array(x0, dtype=float)
            vertex[i] += 0.1 if x0[i] != 0 else 0.1
            simplex.append(vertex)

        # Evaluate
        f_values = [func(x) for x in simplex]

        # Track best
        best_idx = np.argmin(f_values)
        self.history.append((simplex[best_idx].copy(), f_values[best_idx]))

        for iteration in range(self.max_iterations):
            # Sort simplex by function value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]

            # Check convergence
            if np.std(f_values) < self.tolerance:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + self.alpha * (centroid - simplex[-1])
            f_r = func(x_r)

            if f_values[0] <= f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r

            elif f_r < f_values[0]:
                # Try expansion
                x_e = centroid + self.gamma * (x_r - centroid)
                f_e = func(x_e)

                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r

            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + self.rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - self.rho * (centroid - simplex[-1])

                f_c = func(x_c)

                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                        f_values[i] = func(simplex[i])

            # Update history
            best_idx = np.argmin(f_values)
            self.history.append((simplex[best_idx].copy(), f_values[best_idx]))

        best_idx = np.argmin(f_values)
        return simplex[best_idx], self.history


class BayesianOptimization:
    """
    Bayesian Optimization

    Extremely sample-efficient for expensive black-box functions.
    Builds a probabilistic model (Gaussian Process) of the function.
    Uses acquisition function to decide where to sample next.

    Perfect for:
    - Hyperparameter tuning
    - Expensive simulations
    - Physical experiments

    This is a simplified implementation.
    """

    def __init__(self, iterations=50, kappa=2.0):
        self.iterations = iterations
        self.kappa = kappa  # Exploration-exploitation tradeoff
        self.history = []

    def optimize(self, func, x0=None):
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Initial samples
        n_init = min(10, self.iterations // 2)
        X_sample = []
        Y_sample = []

        for _ in range(n_init):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            y = func(x)
            X_sample.append(x)
            Y_sample.append(y)

        X_sample = np.array(X_sample)
        Y_sample = np.array(Y_sample)

        best_idx = np.argmin(Y_sample)
        best_x = X_sample[best_idx]
        best_y = Y_sample[best_idx]
        self.history.append((best_x.copy(), best_y))

        # Bayesian optimization loop
        for iteration in range(n_init, self.iterations):
            # Fit Gaussian Process (simplified - using distance-based covariance)
            def predict(x):
                """Predict mean and std at point x."""
                # Simple kernel: RBF with distance
                distances = np.linalg.norm(X_sample - x, axis=1)
                weights = np.exp(-distances**2)
                weights = weights / (np.sum(weights) + 1e-10)

                mean = np.sum(weights * Y_sample)
                variance = max(0.01, 1.0 - np.sum(weights))  # Uncertainty decreases near samples
                return mean, np.sqrt(variance)

            # Acquisition function: Upper Confidence Bound (UCB)
            def acquisition(x):
                mean, std = predict(x)
                return -(mean - self.kappa * std)  # Negative for maximization

            # Optimize acquisition function
            best_acq = -float('inf')
            best_x_next = None

            # Random search for next point
            for _ in range(1000):
                x_candidate = np.random.uniform(bounds[:, 0], bounds[:, 1])
                acq_value = acquisition(x_candidate)

                if acq_value > best_acq:
                    best_acq = acq_value
                    best_x_next = x_candidate

            # Evaluate at new point
            y_next = func(best_x_next)

            # Update samples
            X_sample = np.vstack([X_sample, best_x_next])
            Y_sample = np.append(Y_sample, y_next)

            # Update best
            if y_next < best_y:
                best_y = y_next
                best_x = best_x_next.copy()

            self.history.append((best_x.copy(), best_y))

        return best_x, self.history


class HarmonySearch:
    """
    Harmony Search

    Inspired by music improvisation. Musicians adjust pitches to achieve harmony.

    Key concepts:
    - Harmony Memory: stores good solutions
    - Pitch adjustment: modify existing solutions
    - Randomization: explore new areas

    Elegant algorithm that works well on many problems.
    """

    def __init__(self, harmony_memory_size=30, iterations=100, HMCR=0.9, PAR=0.3, bandwidth=0.1):
        self.HMS = harmony_memory_size
        self.iterations = iterations
        self.HMCR = HMCR  # Harmony Memory Considering Rate
        self.PAR = PAR    # Pitch Adjusting Rate
        self.bandwidth = bandwidth
        self.history = []

    def optimize(self, func, x0=None):
        bounds = np.array(func.bounds)
        dim = len(bounds)

        # Initialize Harmony Memory
        harmony_memory = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.HMS, dim)
        )
        fitness = np.array([func(h) for h in harmony_memory])

        # Track best
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.history.append((best_harmony.copy(), best_fitness))

        for iteration in range(self.iterations):
            # Improvise new harmony
            new_harmony = np.zeros(dim)

            for i in range(dim):
                if np.random.rand() < self.HMCR:
                    # Use value from harmony memory
                    new_harmony[i] = harmony_memory[np.random.randint(self.HMS), i]

                    # Pitch adjustment
                    if np.random.rand() < self.PAR:
                        new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (bounds[i, 1] - bounds[i, 0])
                else:
                    # Random selection
                    new_harmony[i] = np.random.uniform(bounds[i, 0], bounds[i, 1])

            # Ensure bounds
            new_harmony = np.clip(new_harmony, bounds[:, 0], bounds[:, 1])

            # Evaluate
            new_fitness = func(new_harmony)

            # Update harmony memory if better than worst
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness

                # Update best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_harmony = new_harmony.copy()

            self.history.append((best_harmony.copy(), best_fitness))

        return best_harmony, self.history
