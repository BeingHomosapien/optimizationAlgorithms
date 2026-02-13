# Complete Optimization Algorithms Reference

Quick reference for all 18 implemented algorithms.

## Quick Selection Guide

```
┌─ Need GRADIENTS? ────────────────────────────────────────────┐
│                                                               │
│  YES → High Dimensions (>100)?                               │
│        ├─ YES → Adam, AdamW, NAdam (deep learning)          │
│        └─ NO  → L-BFGS (scientific computing)               │
│                                                               │
│  NO  → Expensive Function (<1000 evals)?                     │
│        ├─ YES → Bayesian Optimization, CMA-ES               │
│        └─ NO  → Differential Evolution, Genetic Algorithm    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## All 18 Algorithms at a Glance

### BASIC GRADIENT-BASED (4)

| Algorithm | Learning Rate | When to Use |
|-----------|--------------|-------------|
| **Gradient Descent** | 0.001-0.1 | Learning, simple problems |
| **Momentum** | 0.001-0.1 | Oscillating gradients |
| **RMSprop** | 0.001-0.01 | Non-stationary objectives |
| **Adam** | 0.001-0.01 | Default for deep learning |

### ADVANCED GRADIENT-BASED (6)

| Algorithm | Learning Rate | When to Use |
|-----------|--------------|-------------|
| **AdaGrad** | 0.01-0.5 | Sparse data (NLP) |
| **NAdam** | 0.001-0.01 | Faster than Adam |
| **AdamW** | 0.001-0.01 | SOTA deep learning |
| **L-BFGS** | Auto | Scientific computing |
| **Conjugate Gradient** | Auto | Large-scale smooth |
| **Newton's Method** | Auto | High accuracy needed |

### BASIC METAHEURISTIC (3)

| Algorithm | Population | When to Use |
|-----------|-----------|-------------|
| **Genetic Algorithm** | 50-200 | Discrete, multi-objective |
| **PSO** | 20-50 | Continuous, fast results |
| **Simulated Annealing** | N/A | Simple implementation |

### ADVANCED METAHEURISTIC (5)

| Algorithm | Population | When to Use |
|-----------|-----------|-------------|
| **Differential Evolution** | 50-100 | Robust global search |
| **CMA-ES** | 4+3·ln(n) | Black-box optimization |
| **Nelder-Mead** | n+1 | Quick prototyping |
| **Bayesian Optimization** | N/A | Very expensive functions |
| **Harmony Search** | 20-50 | Elegant alternative |

## Detailed Specifications

### 1. GRADIENT DESCENT
```python
GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-6)
```
- **Type:** First-order gradient
- **Convergence:** Linear (on convex)
- **Memory:** O(n)
- **Best for:** Learning, simple smooth functions
- **Pros:** Simple, guaranteed convergence (convex)
- **Cons:** Slow, sensitive to learning rate

### 2. MOMENTUM
```python
Momentum(learning_rate=0.01, beta=0.9, max_iterations=1000)
```
- **Type:** First-order gradient with momentum
- **Convergence:** Faster than GD
- **Memory:** O(n)
- **Best for:** Oscillating gradients, narrow valleys
- **Pros:** Accelerated convergence, dampens oscillations
- **Cons:** Still first-order

### 3. RMSPROP
```python
RMSprop(learning_rate=0.01, beta=0.9, epsilon=1e-8)
```
- **Type:** Adaptive gradient
- **Convergence:** Linear
- **Memory:** O(n)
- **Best for:** Non-stationary objectives, RNNs
- **Pros:** Adapts learning rate per parameter
- **Cons:** Still requires tuning

### 4. ADAM
```python
Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
```
- **Type:** Adaptive moment estimation
- **Convergence:** Linear
- **Memory:** O(n)
- **Best for:** Deep learning default
- **Pros:** Robust, few hyperparameters
- **Cons:** Can fail to converge in some cases

### 5. ADAGRAD
```python
AdaGrad(learning_rate=0.01, epsilon=1e-8)
```
- **Type:** Adaptive gradient (historical)
- **Convergence:** Linear (but slows)
- **Memory:** O(n)
- **Best for:** Sparse data, NLP
- **Pros:** Automatic per-feature rates
- **Cons:** Learning rate diminishes

### 6. NADAM
```python
NAdam(learning_rate=0.002, beta1=0.9, beta2=0.999)
```
- **Type:** Nesterov + Adam
- **Convergence:** Linear (faster than Adam)
- **Memory:** O(n)
- **Best for:** When Adam works but want faster
- **Pros:** Improved momentum
- **Cons:** Slightly more complex

### 7. ADAMW
```python
AdamW(learning_rate=0.001, weight_decay=0.01)
```
- **Type:** Adam + decoupled weight decay
- **Convergence:** Linear
- **Memory:** O(n)
- **Best for:** Modern deep learning (transformers)
- **Pros:** Better generalization
- **Cons:** Extra hyperparameter

### 8. L-BFGS
```python
LBFGS(max_iterations=100, m=10)
```
- **Type:** Quasi-Newton (second-order approximation)
- **Convergence:** Superlinear
- **Memory:** O(m·n)
- **Best for:** Batch optimization, scientific computing
- **Pros:** Very fast convergence
- **Cons:** Not for stochastic/noisy gradients

### 9. CONJUGATE GRADIENT
```python
ConjugateGradient(max_iterations=1000, restart_iterations=50)
```
- **Type:** Conjugate directions
- **Convergence:** Superlinear (on quadratics)
- **Memory:** O(n)
- **Best for:** Large-scale smooth optimization
- **Pros:** Fast, low memory
- **Cons:** Needs line search

### 10. NEWTON'S METHOD
```python
NewtonMethod(max_iterations=100, tolerance=1e-6)
```
- **Type:** Full second-order
- **Convergence:** Quadratic
- **Memory:** O(n²)
- **Best for:** Small dimensions, high accuracy
- **Pros:** Fastest convergence
- **Cons:** O(n³) per iteration, needs Hessian

### 11. GENETIC ALGORITHM
```python
GeneticAlgorithm(population_size=50, generations=100,
                 mutation_rate=0.1, crossover_rate=0.8)
```
- **Type:** Evolutionary
- **Convergence:** Stochastic
- **Memory:** O(pop_size · n)
- **Best for:** Discrete, multi-objective
- **Pros:** No gradients, versatile
- **Cons:** Many hyperparameters

### 12. PARTICLE SWARM OPTIMIZATION
```python
ParticleSwarmOptimization(n_particles=30, iterations=100,
                          w=0.7, c1=1.5, c2=1.5)
```
- **Type:** Swarm intelligence
- **Convergence:** Stochastic
- **Memory:** O(n_particles · n)
- **Best for:** Continuous optimization
- **Pros:** Simple, fast
- **Cons:** Can converge prematurely

### 13. SIMULATED ANNEALING
```python
SimulatedAnnealing(initial_temp=100, cooling_rate=0.95,
                   iterations=1000)
```
- **Type:** Probabilistic
- **Convergence:** Stochastic
- **Memory:** O(n)
- **Best for:** Escaping local minima
- **Pros:** Simple, accepts worse solutions
- **Cons:** Cooling schedule critical

### 14. DIFFERENTIAL EVOLUTION
```python
DifferentialEvolution(population_size=50, generations=100,
                      F=0.8, CR=0.9)
```
- **Type:** Evolutionary (population differences)
- **Convergence:** Stochastic (robust)
- **Memory:** O(pop_size · n)
- **Best for:** Robust global optimization
- **Pros:** Very reliable, few parameters
- **Cons:** Can be slow

### 15. CMA-ES
```python
CMAES(population_size=None, generations=100, sigma=0.5)
```
- **Type:** Evolution strategy (covariance adaptation)
- **Convergence:** Stochastic (state-of-the-art)
- **Memory:** O(n²)
- **Best for:** Black-box optimization
- **Pros:** Learns problem structure
- **Cons:** O(n²) memory, <1000 dims

### 16. NELDER-MEAD
```python
NelderMead(max_iterations=1000, tolerance=1e-6)
```
- **Type:** Direct search (simplex)
- **Convergence:** None guaranteed
- **Memory:** O(n²)
- **Best for:** Quick prototyping, small problems
- **Pros:** No tuning, simple
- **Cons:** Can stagnate

### 17. BAYESIAN OPTIMIZATION
```python
BayesianOptimization(iterations=50, kappa=2.0)
```
- **Type:** Model-based (Gaussian Process)
- **Convergence:** Sample-efficient
- **Memory:** O(n³) in iterations
- **Best for:** Very expensive functions
- **Pros:** Extremely sample-efficient
- **Cons:** GP fitting expensive, <20 dims

### 18. HARMONY SEARCH
```python
HarmonySearch(harmony_memory_size=30, iterations=100,
              HMCR=0.9, PAR=0.3)
```
- **Type:** Music-inspired metaheuristic
- **Convergence:** Stochastic
- **Memory:** O(HMS · n)
- **Best for:** Mixed discrete/continuous
- **Pros:** Elegant, good balance
- **Cons:** Not as popular as GA/PSO

## Performance Characteristics

### Speed (Iterations to Convergence)
```
Fastest  → Newton, L-BFGS, Conjugate Gradient
Fast     → Adam, NAdam, AdamW, Differential Evolution
Medium   → Momentum, RMSprop, PSO, CMA-ES
Slow     → Gradient Descent, Genetic Algorithm
Variable → Bayesian Opt (depends on function)
```

### Memory Requirements
```
O(n)      → Most gradient methods, Simulated Annealing
O(m·n)    → L-BFGS (m typically 5-20)
O(n²)     → Newton, Nelder-Mead, CMA-ES
O(pop·n)  → Population-based methods
```

### Robustness to Local Minima
```
Best      → CMA-ES, Differential Evolution, Bayesian Opt
Good      → Genetic Algorithm, PSO, Harmony Search
Medium    → Simulated Annealing
Poor      → All gradient-based methods
```

### Sample Efficiency
```
Most Efficient  → Bayesian Optimization
Efficient       → Newton, L-BFGS, Conjugate Gradient
Moderate        → Gradient methods, CMA-ES
Less Efficient  → Population-based methods
```

## Typical Hyperparameter Values

### Learning Rates
- Deep Learning: 0.001 - 0.01 (Adam, AdamW)
- Gradient Descent: 0.001 - 0.1
- RMSprop: 0.001 - 0.01
- AdaGrad: 0.01 - 0.5

### Population Sizes
- Genetic Algorithm: 50-200
- PSO: 20-50
- Differential Evolution: 10·n to 100
- CMA-ES: 4 + 3·ln(n) (automatic)
- Harmony Search: 20-50

### Stopping Criteria
- Gradient norm: < 1e-6
- Function change: < 1e-8
- Max iterations: 100-1000 (gradient), 50-500 (population)

## Implementation Notes

### All algorithms support:
```python
x_optimal, history = algorithm.optimize(func, x0)
```

Where:
- `func`: Function object with `__call__` and `gradient` methods
- `x0`: Initial point (numpy array)
- `x_optimal`: Best found solution
- `history`: List of (x, f(x)) at each iteration

### For metaheuristics:
- `x0` is often ignored (use random initialization)
- `func.bounds` must be defined

## Citation Information

If using these implementations for research:

- **Adam:** Kingma & Ba (2014)
- **L-BFGS:** Liu & Nocedal (1989)
- **CMA-ES:** Hansen & Ostermeier (2001)
- **Differential Evolution:** Storn & Price (1997)
- **PSO:** Kennedy & Eberhart (1995)
- **Genetic Algorithms:** Holland (1975), Goldberg (1989)

## Common Pitfalls

1. **Wrong learning rate** → Divergence or slow convergence
2. **Not scaling inputs** → Poor conditioning
3. **Wrong algorithm choice** → Stuck in local minima
4. **Too small population** → Poor exploration
5. **Premature stopping** → Not reaching optimum

## Tips for Success

1. **Always visualize convergence** → Catch issues early
2. **Try multiple random starts** → Find global optimum
3. **Scale your variables** → Similar ranges (0-1)
4. **Start simple** → Gradient descent first
5. **Know your problem** → Convex? Noisy? Discrete?
6. **Combine methods** → Global search → Local refinement
