# Optimization Algorithms - Interactive Learning

A hands-on exploration of optimization algorithms with visual simulations.

## What are Optimization Algorithms?

Optimization algorithms find the minimum (or maximum) of a function. They're used everywhere:
- Training neural networks (gradient descent)
- Engineering design (finding optimal parameters)
- Scheduling and logistics
- Portfolio optimization

## Algorithms Implemented

### 1. Basic Gradient-Based Methods
These use derivatives to find the direction of steepest descent:
- **Gradient Descent**: Basic iterative method following the negative gradient
- **Momentum**: Accelerates convergence by accumulating past gradients
- **RMSprop**: Adapts learning rate for each parameter
- **Adam**: Combines momentum and adaptive learning rates (very popular in deep learning)

### 2. Advanced Gradient-Based Methods
State-of-the-art optimization for modern applications:
- **AdaGrad**: Adaptive learning per parameter (good for sparse data)
- **NAdam**: Nesterov-accelerated Adam
- **AdamW**: Adam with proper weight decay (SOTA for deep learning)
- **L-BFGS**: Quasi-Newton method (scientific computing standard)
- **Conjugate Gradient**: Large-scale smooth optimization
- **Newton's Method**: Uses second derivatives for quadratic convergence

### 3. Basic Metaheuristic Algorithms
These are inspired by natural phenomena and don't require derivatives:
- **Genetic Algorithm**: Mimics natural evolution with selection, crossover, and mutation
- **Particle Swarm Optimization (PSO)**: Simulates social behavior of bird flocking
- **Simulated Annealing**: Inspired by metallurgy annealing process

### 4. Advanced Metaheuristic Algorithms
Cutting-edge derivative-free global optimization:
- **Differential Evolution**: Robust global optimizer using population differences
- **CMA-ES**: State-of-the-art black-box optimization (adapts covariance matrix)
- **Nelder-Mead**: Classic simplex method
- **Bayesian Optimization**: Sample-efficient for expensive functions
- **Harmony Search**: Music-inspired elegant optimization

## Test Functions

Classic optimization benchmark functions:
- **Sphere**: Simple convex function (easy)
- **Rosenbrock**: Banana-shaped valley (hard for gradient methods)
- **Rastrigin**: Many local minima (tests global search)
- **Ackley**: Sharp peaks testing exploration vs exploitation
- **Beale**: Multiple valleys

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# 🎬 Real-time simulations (Watch algorithms work live!)
python real_time_simulations/gradient_descent_live.py  # Watch GD step-by-step
python real_time_simulations/pso_live.py              # Watch particle swarm
python real_time_simulations/compare_live.py          # Algorithm race!

# Run individual algorithm simulations
python simulations/gradient_descent_sim.py
python simulations/genetic_algorithm_sim.py
python simulations/pso_sim.py
python simulations/advanced_algorithms_sim.py

# Compare all algorithms on a function
python compare_algorithms.py --function rosenbrock

# Interactive visualization
python interactive_explorer.py

# Quick overview
python quick_start.py
```

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Math Explained Simply](docs/MATH_EXPLAINED_SIMPLY.md)** - ⭐ **START HERE!** Beginner-friendly mathematical explanations with examples
- **[Learning Guide](docs/LEARNING_GUIDE.md)** - Theory and concepts from beginner to advanced
- **[Advanced Guide](docs/ADVANCED_GUIDE.md)** - Deep dive into state-of-the-art methods (L-BFGS, CMA-ES, etc.)
- **[Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md)** - Complete mathematical derivations and proofs
- **[Algorithms Reference](docs/ALGORITHMS_REFERENCE.md)** - Quick reference for all 18 algorithms

## Project Structure

```
optimizationAlgorithms/
├── docs/                          # Comprehensive documentation
├── algorithms/                    # Algorithm implementations
│   ├── gradient_based.py          # 4 basic gradient methods
│   ├── advanced_gradient.py       # 6 advanced gradient methods
│   ├── metaheuristic.py           # 3 basic metaheuristics
│   └── advanced_metaheuristic.py  # 5 advanced metaheuristics
├── test_functions/                # Benchmark functions
├── real_time_simulations/         # 🎬 NEW! Watch algorithms live!
│   ├── gradient_descent_live.py   # Real-time GD visualization
│   ├── pso_live.py                # Real-time particle swarm
│   └── compare_live.py            # Real-time algorithm race
├── simulations/                   # Individual simulations
├── utils/                         # Visualization and helper utilities
├── compare_algorithms.py
├── algorithm_comparison.py        # Compare ALL 18 algorithms
├── interactive_explorer.py        # Interactive GUI
├── mathematical_demonstrations.py # Interactive math visualizations
├── quick_start.py                 # Start here!
└── example_custom_function.py     # Create your own optimization problem
```

## Learning Path

1. Start with `gradient_descent_sim.py` to understand basic optimization
2. Try `momentum_sim.py` to see how momentum helps
3. Explore `genetic_algorithm_sim.py` for a completely different approach
4. Run `compare_algorithms.py` to see how they perform on different landscapes
5. Use `interactive_explorer.py` to experiment with parameters

## Key Concepts

**Convergence**: How quickly an algorithm reaches the optimum
**Local vs Global Minima**: Some algorithms can get stuck in local minima
**Exploration vs Exploitation**: Balance between searching new areas and refining known good solutions
**Learning Rate**: Step size (too large = unstable, too small = slow)
