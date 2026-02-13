# Optimization Algorithms - Learning Guide

## Introduction

Optimization is the process of finding the best solution from all possible solutions. In mathematical terms, we want to find the input `x` that minimizes (or maximizes) a function `f(x)`.

**Example:** Training a neural network is optimization - we search for the weights that minimize prediction error.

## Core Concepts

### 1. The Optimization Problem

```
minimize f(x)
subject to x in some feasible region
```

- **Objective function** `f(x)`: What we're trying to minimize
- **Decision variables** `x`: Parameters we can control
- **Constraints**: Restrictions on valid values of `x` (e.g., bounds)

### 2. Local vs Global Minima

- **Global minimum**: The absolute lowest point on the entire landscape
- **Local minimum**: Lowest point in a neighborhood, but not globally

Many real-world problems have multiple local minima. Finding the global minimum is often challenging.

### 3. Convexity

A function is **convex** if a line segment between any two points on the function lies above the function.

- Convex functions: Have a single minimum (no local minima)
- Non-convex functions: Can have many local minima
- Most real problems are non-convex

## Gradient-Based Methods

These methods use the gradient (derivative) to determine the direction of steepest descent.

### Gradient Descent

**Idea:** Move in the direction opposite to the gradient (steepest descent).

```
x_{k+1} = x_k - α ∇f(x_k)
```

- `α` is the learning rate (step size)
- `∇f(x_k)` is the gradient at current point

**Pros:**
- Simple and intuitive
- Works well on smooth, convex functions
- Guaranteed to converge (with right learning rate)

**Cons:**
- Can be slow on narrow valleys
- Oscillates in directions of high curvature
- Learning rate is critical (too big = divergence, too small = slow)

### Momentum

**Idea:** Accumulate past gradients to build up "speed" in consistent directions.

```
v_{k+1} = β·v_k + ∇f(x_k)
x_{k+1} = x_k - α·v_{k+1}
```

- `β` is the momentum coefficient (typically 0.9)
- Helps accelerate in consistent directions
- Dampens oscillations

**Analogy:** A ball rolling down a hill gains momentum.

**Pros:**
- Faster convergence than vanilla GD
- Smooths out oscillations
- Better on ill-conditioned problems

### RMSprop

**Idea:** Adapt the learning rate for each parameter based on recent gradient magnitudes.

```
E[g²]_k = β·E[g²]_{k-1} + (1-β)·(∇f(x_k))²
x_{k+1} = x_k - α·∇f(x_k) / √(E[g²]_k + ε)
```

- Divides gradient by a running average of its recent magnitude
- Parameters with large gradients get smaller effective learning rates
- Parameters with small gradients get larger effective learning rates

**Pros:**
- Automatically adapts learning rate
- Works well with non-stationary objectives
- Good for neural networks

### Adam (Adaptive Moment Estimation)

**Idea:** Combine momentum and RMSprop - maintain both first moment (mean) and second moment (variance) of gradients.

```
m_k = β₁·m_{k-1} + (1-β₁)·∇f(x_k)     [momentum]
v_k = β₂·v_{k-1} + (1-β₂)·(∇f(x_k))²  [RMSprop]

m̂_k = m_k / (1 - β₁^k)    [bias correction]
v̂_k = v_k / (1 - β₂^k)

x_{k+1} = x_k - α·m̂_k / (√v̂_k + ε)
```

**Pros:**
- Robust to hyperparameter choices
- Very popular in deep learning
- Combines benefits of momentum and adaptive learning rates
- Includes bias correction for early iterations

**Cons:**
- More parameters to tune
- Can sometimes fail to converge to optimal solution

## Metaheuristic Methods

These algorithms don't require gradients and are inspired by natural phenomena.

### Genetic Algorithm

**Inspiration:** Natural evolution and survival of the fittest.

**Process:**
1. **Initialization:** Create random population of solutions
2. **Evaluation:** Compute fitness (function value) for each
3. **Selection:** Choose best individuals to be parents
4. **Crossover:** Combine pairs of parents to create offspring
5. **Mutation:** Random changes to maintain diversity
6. **Replacement:** New population replaces old
7. **Repeat** until convergence

**Key Components:**
- **Population size:** Number of candidate solutions
- **Selection:** Tournament, roulette wheel, rank-based
- **Crossover rate:** Probability of combining parents
- **Mutation rate:** Probability of random changes

**Pros:**
- No gradient needed
- Works on discrete or continuous problems
- Naturally handles multi-objective optimization
- Good at escaping local minima

**Cons:**
- Computationally expensive (many function evaluations)
- Many hyperparameters
- No convergence guarantees
- Slower than gradient methods when gradients available

### Particle Swarm Optimization (PSO)

**Inspiration:** Social behavior of bird flocking or fish schooling.

Each particle has:
- Position (candidate solution)
- Velocity (direction and speed)
- Personal best position
- Knowledge of global best position

**Update rules:**
```
v_{i,k+1} = w·v_{i,k} + c₁·r₁·(p_i - x_{i,k}) + c₂·r₂·(g - x_{i,k})
x_{i,k+1} = x_{i,k} + v_{i,k+1}
```

- `w`: Inertia weight (current velocity)
- `c₁`: Cognitive coefficient (personal best attraction)
- `c₂`: Social coefficient (global best attraction)
- `r₁, r₂`: Random numbers for stochasticity
- `p_i`: Personal best of particle i
- `g`: Global best of swarm

**Pros:**
- Simple to implement
- Few parameters to tune
- Fast convergence
- Works well on continuous problems

**Cons:**
- Can converge prematurely if diversity is lost
- Parameters need tuning for different problems
- No gradient needed but specific to continuous optimization

**Parameter Tuning:**
- High `w`: More exploration (global search)
- Low `w`: More exploitation (local refinement)
- High `c₂`: Swarm converges quickly (risk of premature convergence)
- High `c₁`: More independent search

### Simulated Annealing

**Inspiration:** Metallurgical annealing process (slowly cooling metal to minimize energy).

**Process:**
1. Start at high "temperature"
2. Randomly perturb current solution
3. If new solution is better, accept it
4. If worse, accept with probability `exp(-ΔE/T)`
5. Gradually decrease temperature
6. Repeat until temperature is very low

**Key Idea:** Early on, accept worse solutions to explore broadly. As temperature decreases, become more selective.

**Acceptance Probability:**
```
P(accept) = {
  1                    if ΔE < 0 (improvement)
  exp(-ΔE/T)          if ΔE ≥ 0 (worse)
}
```

**Pros:**
- Simple conceptually
- Can escape local minima
- Proven to converge to global optimum (given infinite time and proper cooling)

**Cons:**
- Cooling schedule is critical
- Can be slow
- No use of population information

## Choosing an Algorithm

### Use Gradient-Based Methods When:
- Function is smooth and differentiable
- You can compute gradients (analytically or automatically)
- You need fast convergence
- Problem is convex or nearly convex
- Examples: Training neural networks, continuous optimization

**Best Choices:**
- Adam: Most robust, good default
- Gradient Descent: Simple problems, educational purposes
- Momentum: When you have oscillation issues

### Use Metaheuristic Methods When:
- No gradient available (black-box function)
- Function is discrete or has discontinuities
- Many local minima (highly multimodal)
- You need global optimization guarantees
- Examples: Combinatorial optimization, hyperparameter tuning, engineering design

**Best Choices:**
- PSO: Continuous optimization, fast results needed
- Genetic Algorithm: Discrete problems, multi-objective optimization
- Simulated Annealing: Simple implementation needed

## Convergence Analysis

### Convergence Criteria

How do we know when to stop?

1. **Gradient magnitude:** `||∇f(x)|| < ε`
   - Only for gradient-based methods
   - Small gradient means we're at a stationary point

2. **Function value change:** `|f(x_{k+1}) - f(x_k)| < ε`
   - Works for any method
   - Small change means we're converging

3. **Parameter change:** `||x_{k+1} - x_k|| < ε`
   - Small movement in parameter space

4. **Maximum iterations:** Computational budget limit

### Convergence Speed

- **Linear convergence:** Error decreases linearly
  - Gradient Descent on strongly convex functions

- **Superlinear convergence:** Error decreases faster than linearly
  - Newton's method, Quasi-Newton methods

- **Stochastic convergence:** Random fluctuations
  - Most metaheuristic methods
  - May never fully converge but get "close enough"

## Common Pitfalls

### 1. Learning Rate Issues
- Too large: Oscillation or divergence
- Too small: Extremely slow convergence
- **Solution:** Learning rate schedules, adaptive methods (Adam, RMSprop)

### 2. Local Minima
- Gradient methods can get stuck
- **Solutions:** Multiple random restarts, metaheuristic methods, better initialization

### 3. Ill-Conditioned Problems
- Functions with very different curvatures in different directions
- **Solutions:** Momentum, second-order methods, preconditioning

### 4. Noisy Functions
- Function evaluations have randomness
- **Solutions:** Larger population sizes, averaging, robust methods

### 5. Expensive Function Evaluations
- Each evaluation takes significant time
- **Solutions:** Gradient-based methods (fewer evaluations), surrogate models, Bayesian optimization

## Advanced Topics (Beyond This Tutorial)

- **Second-order methods:** Newton's method, L-BFGS (use curvature information)
- **Constrained optimization:** Handle inequality/equality constraints
- **Multi-objective optimization:** Optimize multiple objectives simultaneously
- **Bayesian optimization:** Build probabilistic model of function
- **Evolutionary strategies:** More sophisticated than genetic algorithms
- **Reinforcement learning:** Optimization in dynamic environments
- **Stochastic optimization:** Handle randomness in objectives

## Practical Tips

1. **Start simple:** Try gradient descent first if you have gradients
2. **Visualize:** Plot convergence curves, visualize paths
3. **Scale your variables:** Normalize inputs to similar ranges
4. **Use good initialization:** Start from a reasonable point
5. **Try multiple runs:** Random initialization can help find global optima
6. **Monitor convergence:** Watch for plateaus, oscillations, divergence
7. **Tune hyperparameters:** Learning rate, population size, etc. matter
8. **Know your problem:** Convex? Multimodal? Noisy? This guides algorithm choice

## Further Reading

- **Books:**
  - "Numerical Optimization" by Nocedal & Wright
  - "Convex Optimization" by Boyd & Vandenberghe

- **Online:**
  - CS229 Optimization notes (Stanford)
  - Deep Learning Book Chapter 8 (Goodfellow et al.)
  - Distill.pub articles on optimization

- **Papers:**
  - Adam: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
  - PSO: "Particle Swarm Optimization" (Kennedy & Eberhart, 1995)

## Exercises

1. Implement a simple gradient descent from scratch
2. Compare convergence on different learning rates
3. Visualize optimization paths on 2D functions
4. Try optimizing a simple neural network
5. Implement a custom benchmark function
6. Experiment with hybrid methods (e.g., start with PSO, refine with gradient descent)
