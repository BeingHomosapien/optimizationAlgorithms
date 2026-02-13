# Advanced Optimization Algorithms - Deep Dive

This guide covers state-of-the-art optimization algorithms for specialized applications.

## Advanced Gradient-Based Methods

### 1. AdaGrad (Adaptive Gradient)

**Key Idea:** Accumulate all past gradients to adaptively scale learning rate per parameter.

```
G_t = G_{t-1} + g_t²
x_{t+1} = x_t - (lr / √(G_t + ε)) * g_t
```

**Strengths:**
- Automatically adapts learning rates
- Excellent for sparse gradients (NLP, recommender systems)
- No manual learning rate tuning per feature

**Weaknesses:**
- Learning rate monotonically decreases
- Can become too small and stop learning
- RMSprop and Adam address this issue

**When to use:**
- Sparse data (natural language processing)
- Different features have different frequencies
- You want automatic per-parameter learning rates

### 2. NAdam (Nesterov Adam)

**Key Idea:** Combine Adam's adaptive learning rates with Nesterov momentum.

Nesterov momentum "looks ahead" - evaluates gradient at the projected future position rather than current position. This provides better gradient information.

**Strengths:**
- Often faster convergence than Adam
- More stable on some problems
- Good default for many tasks

**Weaknesses:**
- Slightly more complex than Adam
- Marginal gains over Adam on some problems

**When to use:**
- When Adam works but you want faster convergence
- Deep learning training
- Alternative to Adam for experimentation

### 3. AdamW (Adam with Decoupled Weight Decay)

**Key Idea:** Separate weight decay from gradient-based updates.

Standard Adam applies weight decay through the gradient, which interacts poorly with adaptive learning rates. AdamW applies it directly to weights.

**Strengths:**
- Better generalization than Adam
- Proper weight decay implementation
- State-of-the-art for many deep learning tasks

**Weaknesses:**
- Need to tune weight decay parameter
- Not necessary for all problems

**When to use:**
- Training deep neural networks
- When regularization is important
- Modern deep learning (transformers, vision models)

### 4. L-BFGS (Limited-memory BFGS)

**Key Idea:** Approximate the Hessian using only recent gradient differences.

BFGS is a Quasi-Newton method - builds up curvature information without computing the full Hessian. L-BFGS keeps only the last m updates in memory.

**Mathematics:**
```
x_{k+1} = x_k - α_k H_k^{-1} ∇f(x_k)
```
where H_k is approximated using m recent (s, y) pairs:
- s_k = x_{k+1} - x_k
- y_k = ∇f(x_{k+1}) - ∇f(x_k)

**Strengths:**
- Superlinear convergence
- Memory efficient (only stores m vectors)
- Very fast on smooth problems
- Industry standard for scientific computing

**Weaknesses:**
- Requires accurate line search
- Not suitable for stochastic/noisy gradients
- Memory grows with dimensionality (though limited)

**When to use:**
- Batch optimization (full gradients)
- Smooth, deterministic objectives
- Medium-scale problems (up to millions of parameters)
- Neural network training with full batch
- Classic machine learning (logistic regression, etc.)

**Not suitable for:**
- Stochastic gradient descent scenarios
- Mini-batch training
- Very noisy gradients

### 5. Conjugate Gradient

**Key Idea:** Search directions are conjugate (orthogonal in transformed space).

Instead of independent steepest descent steps, CG ensures each search direction is "conjugate" to previous ones, eliminating zig-zagging.

**For quadratic functions:** converges in at most n steps (n = dimension)

**Strengths:**
- No matrix storage needed (unlike Newton)
- Faster than gradient descent
- Good for large-scale problems
- Works well on quadratic and nearly-quadratic functions

**Weaknesses:**
- Performance degrades on non-quadratic functions
- Requires periodic restarts
- Line search is critical

**Variants:**
- Fletcher-Reeves
- Polak-Ribière (implemented here - generally better)
- Hestenes-Stiefel

**When to use:**
- Large-scale smooth optimization
- Quadratic or nearly-quadratic objectives
- Memory is limited
- Alternative to L-BFGS

### 6. Newton's Method

**Key Idea:** Use second-order Taylor expansion for quadratic convergence.

```
x_{k+1} = x_k - H^{-1}∇f(x_k)
```

where H is the Hessian (matrix of second partial derivatives).

**Convergence:** Quadratic near the optimum (error² → error⁴ → error⁸ → ...)

**Strengths:**
- Extremely fast convergence (when close to optimum)
- Automatic step size
- Accounts for curvature

**Weaknesses:**
- Requires computing and inverting n×n Hessian (O(n³) per iteration!)
- Only works when Hessian is positive definite
- Expensive for high dimensions
- Can be unstable far from optimum

**When to use:**
- Small to medium dimensions (< 1000)
- Very smooth functions
- High accuracy requirements
- You can compute Hessian efficiently (closed form or automatic differentiation)

**Practical note:** Rarely used directly. L-BFGS is usually preferred (quasi-Newton).

## Advanced Metaheuristic Methods

### 1. Differential Evolution (DE)

**Key Idea:** Use differences between population members to create new candidates.

**Algorithm:**
1. For each individual x_i:
2. Select three random distinct individuals: x_r1, x_r2, x_r3
3. Mutation: v = x_r1 + F·(x_r2 - x_r3)
4. Crossover: mix v with x_i (binomial)
5. Selection: keep better of x_i and offspring

**Parameters:**
- F (mutation factor): typically 0.5-1.0
  - Higher F: more exploration
  - Lower F: more exploitation
- CR (crossover rate): typically 0.8-0.95
  - Controls how much of mutant vector is used

**Strategies:**
- DE/rand/1/bin: Random base vector
- DE/best/1/bin: Best individual as base (implemented)
- DE/current-to-best/1/bin: Mix current and best

**Strengths:**
- Very robust global optimizer
- Few parameters to tune
- Handles non-differentiable, noisy, multimodal functions
- Often outperforms genetic algorithms

**Weaknesses:**
- Can be slow on large populations
- Requires tuning F and CR for optimal performance
- Not the most sample-efficient

**When to use:**
- Global optimization needed
- Continuous parameters
- Reliable, "works out of the box" solution
- You have budget for 1000s of evaluations

**Success stories:**
- Engineering design optimization
- Training machine learning hyperparameters
- Chemical/biological optimization

### 2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Key Idea:** Learn the shape of the fitness landscape by adapting a multivariate Gaussian.

**What it adapts:**
1. Mean vector (toward better solutions)
2. Covariance matrix (captures problem structure)
3. Step size (global scaling)

**Mathematics:**
Samples: x ~ N(m, σ²C)
- m: mean vector
- σ: step size
- C: covariance matrix

Updates based on weighted best samples.

**Strengths:**
- State-of-the-art for black-box optimization
- Automatically learns problem structure
- Very few parameters to tune
- Invariant to monotonic transformations
- Works on ill-conditioned problems

**Weaknesses:**
- Memory: O(n²) for covariance matrix
- Practical limit: ~100-1000 dimensions
- Requires enough function evaluations (100+ per dimension)

**When to use:**
- You have NO gradients
- Function is expensive (but not extremely)
- < 100 dimensions
- Global optimum needed
- High-quality solution required

**Not suitable for:**
- Very high dimensions (> 1000)
- Extremely expensive functions (< 100 evals total)
- Real-time optimization

**Considered the best general-purpose derivative-free optimizer.**

### 3. Nelder-Mead Simplex

**Key Idea:** Maintain a simplex (n+1 points in n dimensions) and iteratively improve it.

**Operations:**
1. **Reflection:** Mirror worst point through centroid
2. **Expansion:** If reflection is best, go further
3. **Contraction:** If reflection is bad, shrink toward centroid
4. **Shrink:** Contract entire simplex toward best point

**Strengths:**
- Simple, intuitive
- No parameters to tune
- Direct search (no derivatives)
- Works well for small dimensions
- Robust to noise

**Weaknesses:**
- Can stagnate on poorly scaled problems
- No convergence guarantees for general functions
- Slow on high dimensions
- Can converge to non-stationary points

**When to use:**
- Small dimensions (< 20)
- Quick and dirty optimization
- No derivative information
- Prototyping

**Historical note:** One of the oldest optimization algorithms (1965), still widely used as a default in many software packages.

### 4. Bayesian Optimization

**Key Idea:** Build probabilistic model of f(x), use it to decide where to sample next.

**Components:**
1. **Surrogate model:** Gaussian Process (GP) models f(x)
2. **Acquisition function:** Decides next point to sample
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB)
   - Probability of Improvement (PI)

**Algorithm:**
1. Sample a few points randomly
2. Fit GP to observed data
3. Optimize acquisition function → get next point
4. Evaluate objective, update GP
5. Repeat

**Strengths:**
- Extremely sample-efficient
- Perfect for expensive functions
- Principled uncertainty quantification
- Automatic exploration-exploitation tradeoff

**Weaknesses:**
- GP fitting is O(n³) in number of observations
- Typically limited to < 20 dimensions
- Requires tuning GP kernel
- Sequential (hard to parallelize)

**When to use:**
- **Function is expensive** (minutes to hours per evaluation)
- Physical experiments
- Engineering simulations
- Hyperparameter tuning
- Budget of 50-500 evaluations

**Applications:**
- AutoML (hyperparameter optimization)
- Drug discovery
- Materials science
- A/B testing with limited budget

**Note:** The implementation here is simplified. Production use: GPyOpt, Ax, BoTorch.

### 5. Harmony Search

**Key Idea:** Inspired by musical improvisation - combine existing harmonies to create better ones.

**Algorithm:**
1. Initialize Harmony Memory (HM) with random solutions
2. Improvise new harmony:
   - With probability HMCR: pick from HM
   - With probability PAR: adjust pitch
   - Otherwise: random value
3. Replace worst harmony if new one is better

**Parameters:**
- HMCR (Harmony Memory Considering Rate): 0.7-0.95
- PAR (Pitch Adjusting Rate): 0.2-0.5
- Bandwidth: for pitch adjustment

**Strengths:**
- Simple and elegant
- Few parameters
- Good balance of exploration/exploitation
- Works on discrete and continuous

**Weaknesses:**
- Not as well-studied as GA or PSO
- Performance varies by problem
- Can be slow compared to DE or CMA-ES

**When to use:**
- You want something different from GA/PSO
- Discrete or mixed optimization
- Engineering design

**Interesting variant:** Self-adaptive Harmony Search (parameters adapt during search)

## Comparison Table

| Algorithm | Type | Gradients? | Dimensions | Convergence | Best For |
|-----------|------|------------|------------|-------------|----------|
| AdaGrad | Gradient | Yes | High | Linear | Sparse data |
| NAdam | Gradient | Yes | High | Linear | Deep learning |
| AdamW | Gradient | Yes | High | Linear | Deep learning |
| L-BFGS | Gradient | Yes | Medium | Superlinear | Scientific computing |
| Conjugate Gradient | Gradient | Yes | High | Superlinear* | Large-scale smooth |
| Newton | Gradient | Yes | Small | Quadratic | High accuracy needed |
| Differential Evolution | Metaheuristic | No | Medium | Stochastic | Robust global search |
| CMA-ES | Metaheuristic | No | Small-Medium | Stochastic | Black-box optimization |
| Nelder-Mead | Metaheuristic | No | Small | None | Quick prototyping |
| Bayesian Opt. | Metaheuristic | No | Small | N/A | Expensive functions |
| Harmony Search | Metaheuristic | No | Medium | Stochastic | Mixed discrete/continuous |

*On quadratic functions

## Modern Research Directions

### 1. Gradient-Free Methods for Deep Learning
- Evolution Strategies (ES)
- Genetic algorithms for neural architecture search
- Population-based training

### 2. Hybrid Methods
- Start with global search (CMA-ES, DE)
- Refine with local search (L-BFGS, gradient descent)
- Example: CMA-ES → L-BFGS

### 3. Constrained Optimization
- Augmented Lagrangian
- Interior Point Methods
- Penalty methods

### 4. Multi-Objective Optimization
- NSGA-II (genetic algorithm)
- MOEA/D
- Pareto front approximation

### 5. Online/Adaptive Optimization
- Online convex optimization
- Adaptive learning rates
- Non-stationary objectives

### 6. Very High Dimensions
- Random projections
- Coordinate descent
- Sparse optimization

## Practical Recommendations

### For Deep Learning:
1. **Default:** AdamW or NAdam
2. **Fine-tuning:** Lower learning rate + AdamW
3. **Batch optimization:** L-BFGS (if memory allows)

### For Scientific Computing:
1. **Default:** L-BFGS
2. **Large-scale:** Conjugate Gradient
3. **High accuracy:** Newton's method (small scale)

### For Black-Box Optimization:
1. **< 100 dims, moderate budget:** CMA-ES
2. **Expensive function:** Bayesian Optimization
3. **Need robustness:** Differential Evolution
4. **Quick & dirty:** Nelder-Mead

### For Hyperparameter Tuning:
1. **Small budget (< 100):** Bayesian Optimization
2. **Medium budget:** CMA-ES or Differential Evolution
3. **Large budget:** Random search + early stopping

## Implementation Tips

### 1. Scaling
Always normalize inputs to similar ranges:
```python
x_norm = (x - mean) / std
```

### 2. Initialization
- Gradient methods: Start near expected optimum
- Metaheuristics: Random initialization often fine
- Multiple restarts improve global search

### 3. Convergence Criteria
Combine multiple criteria:
- Gradient norm (gradient methods)
- Function value change
- Parameter change
- Maximum iterations

### 4. Hyperparameter Tuning
- Learning rate: Most critical for gradient methods
  - Start with 0.001, adjust by 10×
- Population size: Larger is better (but slower)
  - Start with 5-10× dimension size
- Use adaptive methods when possible (Adam, CMA-ES)

### 5. Debugging
- Visualize convergence curves (always!)
- Check gradient correctness (finite differences)
- Monitor for oscillations, plateaus
- Track best value over time

## Further Resources

### Books
- "Numerical Optimization" - Nocedal & Wright (gradient methods)
- "Introduction to Derivative-Free Optimization" - Conn, Scheinberg, Vicente
- "Evolutionary Computation" - De Jong

### Papers
- L-BFGS: "On the Limited Memory BFGS Method" (Liu & Nocedal, 1989)
- CMA-ES: "Completely Derandomized Self-Adaptation" (Hansen & Ostermeier, 2001)
- Adam: Kingma & Ba (2014)
- Bayesian Optimization: Brochu et al. (2010)

### Software
- SciPy: `scipy.optimize` (L-BFGS, CG, Nelder-Mead, DE)
- JAX: Automatic differentiation + optimizers
- Optuna: Bayesian optimization for hyperparameters
- PyGMO: Multi-objective metaheuristics
- CMA-ES: `cma` package (Hansen's implementation)

## Exercises

1. Implement a hybrid optimizer: CMA-ES for global search → L-BFGS for refinement
2. Compare convergence on ill-conditioned quadratics
3. Add constraints to Differential Evolution (penalty method)
4. Implement adaptive learning rate schedule for Adam
5. Benchmark on CEC2017 competition functions
6. Parallelize population-based methods
