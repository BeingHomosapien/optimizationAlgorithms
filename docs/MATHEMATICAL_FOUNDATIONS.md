# Mathematical Foundations of Optimization Algorithms

A comprehensive mathematical treatment of all 18 implemented algorithms.

## Table of Contents
1. [Mathematical Preliminaries](#mathematical-preliminaries)
2. [Basic Gradient Methods](#basic-gradient-methods)
3. [Advanced Gradient Methods](#advanced-gradient-methods)
4. [Second-Order Methods](#second-order-methods)
5. [Basic Metaheuristics](#basic-metaheuristics)
6. [Advanced Metaheuristics](#advanced-metaheuristics)
7. [Convergence Analysis](#convergence-analysis)

---

## Mathematical Preliminaries

### The Optimization Problem

We want to solve:
```
minimize f(x)
subject to x ∈ ℝⁿ
```

Where:
- `f: ℝⁿ → ℝ` is the objective function
- `x ∈ ℝⁿ` is the decision variable
- `x* ∈ ℝⁿ` is the optimal solution

### Key Concepts

#### Gradient (First Derivative)
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

The gradient points in the direction of steepest **ascent**. Therefore, `-∇f(x)` points toward steepest **descent**.

#### Hessian (Second Derivative Matrix)
```
H(x) = ∇²f(x) = [∂²f/∂xᵢ∂xⱼ]
```

The Hessian describes the curvature of the function. For a quadratic function:
```
f(x) = ½xᵀAx + bᵀx + c
```
The Hessian is simply `H = A`.

#### Taylor Series Expansion

**First-order (linear approximation):**
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp
```

**Second-order (quadratic approximation):**
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp + ½pᵀ∇²f(x)p
```

This is fundamental to understanding optimization algorithms!

#### Optimality Conditions

**Necessary condition (first-order):**
```
∇f(x*) = 0
```
At a minimum, the gradient must be zero (stationary point).

**Sufficient condition (second-order):**
```
∇f(x*) = 0  AND  ∇²f(x*) ≻ 0 (positive definite)
```

#### Lipschitz Continuity

A function has L-Lipschitz continuous gradient if:
```
‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖
```

This bounds how fast the gradient can change, crucial for convergence proofs.

---

## Basic Gradient Methods

### 1. Gradient Descent

#### Algorithm
```
xₖ₊₁ = xₖ - αₖ∇f(xₖ)
```

Where:
- `xₖ` is the current point
- `αₖ` is the step size (learning rate)
- `∇f(xₖ)` is the gradient at xₖ

#### Mathematical Intuition

From Taylor expansion:
```
f(xₖ - α∇f(xₖ)) ≈ f(xₖ) - α‖∇f(xₖ)‖²
```

For small α, this is a **descent** (function value decreases).

#### Step Size Selection

**Fixed step size:**
```
αₖ = α (constant)
```

**Exact line search (optimal α):**
```
αₖ = argmin_α f(xₖ - α∇f(xₖ))
```

**Armijo backtracking:**
```
Find α such that: f(xₖ - α∇f(xₖ)) ≤ f(xₖ) - c·α·‖∇f(xₖ)‖²
```
Where c ∈ (0, 1), typically c = 10⁻⁴.

#### Convergence Rate

For L-smooth, μ-strongly convex functions:
```
f(xₖ) - f(x*) ≤ (1 - μ/L)ᵏ(f(x₀) - f(x*))
```

**Linear convergence** with rate (1 - μ/L).

The **condition number** κ = L/μ determines speed:
- Small κ: fast convergence
- Large κ: slow convergence (ill-conditioned)

#### Why It Can Be Slow

On a quadratic function `f(x) = ½xᵀAx`, the optimal step size is:
```
α* = 2/(λₘᵢₙ + λₘₐₓ)
```

Where λ are eigenvalues of A. If λₘₐₓ >> λₘᵢₙ (ill-conditioned), convergence is slow.

---

### 2. Momentum

#### Algorithm
```
vₖ₊₁ = βvₖ + ∇f(xₖ)
xₖ₊₁ = xₖ - αvₖ₊₁
```

Or equivalently:
```
vₖ₊₁ = βvₖ + (1-β)∇f(xₖ)     [normalized form]
xₖ₊₁ = xₖ - αvₖ₊₁
```

Where:
- `vₖ` is the velocity (momentum term)
- `β ∈ [0, 1)` is the momentum coefficient (typically 0.9)

#### Mathematical Intuition

Expanding the recursion:
```
vₖ = βvₖ₋₁ + ∇f(xₖ₋₁)
   = β(βvₖ₋₂ + ∇f(xₖ₋₂)) + ∇f(xₖ₋₁)
   = β²vₖ₋₂ + β∇f(xₖ₋₂) + ∇f(xₖ₋₁)
   = ...
   = Σᵢ₌₀^∞ βⁱ∇f(xₖ₋ᵢ)
```

Momentum is an **exponentially weighted moving average** of gradients!

The effective "memory" length is:
```
1/(1-β) iterations
```
For β = 0.9: remembers ~10 past gradients.

#### Physical Analogy

Consider a ball rolling down a hill with friction:
```
m·a = -∇f(x) - γ·v
```

Where:
- m: mass (analogous to 1/α)
- γ: friction coefficient (analogous to 1-β)
- ∇f(x): force from potential energy

Momentum helps:
1. **Accelerate** in consistent directions
2. **Dampen** oscillations in inconsistent directions

#### Convergence

For quadratic functions, momentum can achieve:
```
Convergence rate ∝ (1 - 2/√κ)ᵏ
```

Compared to gradient descent: (1 - 1/κ)ᵏ

Much faster when κ is large!

#### Optimal Momentum Coefficient

For a quadratic with condition number κ:
```
β_optimal = (√κ - 1)/(√κ + 1)
```

As κ → ∞, β → 1 (more momentum needed).

---

### 3. RMSprop (Root Mean Square Propagation)

#### Algorithm
```
Eₖ₊₁ = βEₖ + (1-β)∇f(xₖ)²    [element-wise square]
xₖ₊₁ = xₖ - α·∇f(xₖ)/√(Eₖ₊₁ + ε)
```

Where:
- `Eₖ` is the moving average of squared gradients
- `β` is the decay rate (typically 0.9)
- `ε` is a small constant for numerical stability (10⁻⁸)
- `/` denotes element-wise division

#### Mathematical Intuition

Each parameter gets an **adaptive learning rate**:
```
αᵢ,ₖ = α/√(Eᵢ,ₖ + ε)
```

If parameter i has:
- **Large gradients** historically → Eᵢ is large → αᵢ is small (slow down)
- **Small gradients** historically → Eᵢ is small → αᵢ is large (speed up)

This is like **automatic preconditioning**!

#### Motivation: Non-Stationary Objectives

Consider training a neural network where the loss landscape changes as parameters update. RMSprop adapts to these changes by using recent gradient statistics.

#### Connection to Second-Order Methods

RMSprop approximates diagonal preconditioning:
```
xₖ₊₁ = xₖ - α·D⁻¹∇f(xₖ)
```

Where D ≈ diag(∇²f(xₖ)) is approximated by gradient squares.

True Newton would use: `xₖ₊₁ = xₖ - α·H⁻¹∇f(xₖ)`

#### Why the Square Root?

For a quadratic function with Hessian H:
```
∇f(x) ∝ Hx
```

So:
```
∇f(x)² ∝ H²x²
```

Taking the square root gives:
```
√(∇f(x)²) ∝ Hx
```

Dividing gradient by this approximates H⁻¹.

---

### 4. Adam (Adaptive Moment Estimation)

#### Algorithm
```
mₖ₊₁ = β₁mₖ + (1-β₁)∇f(xₖ)           [first moment - mean]
vₖ₊₁ = β₂vₖ + (1-β₂)∇f(xₖ)²          [second moment - variance]

m̂ₖ₊₁ = mₖ₊₁/(1-β₁^(k+1))            [bias correction]
v̂ₖ₊₁ = vₖ₊₁/(1-β₂^(k+1))            [bias correction]

xₖ₊₁ = xₖ - α·m̂ₖ₊₁/√(v̂ₖ₊₁ + ε)
```

Where:
- `mₖ` is the first moment (like momentum)
- `vₖ` is the second moment (like RMSprop)
- `β₁` typically 0.9, `β₂` typically 0.999
- `ε` is 10⁻⁸

#### Mathematical Derivation

**Step 1: Exponential Moving Averages**

First moment:
```
mₖ = (1-β₁)Σᵢ₌₀^(k-1) β₁ⁱ∇f(xₖ₋ᵢ)
```

Second moment:
```
vₖ = (1-β₂)Σᵢ₌₀^(k-1) β₂ⁱ∇f(xₖ₋ᵢ)²
```

**Step 2: Bias Correction**

At initialization (m₀ = 0, v₀ = 0), the estimates are biased toward zero.

Expected value:
```
E[mₖ] = E[∇f(x)]·(1 - β₁ᵏ)
```

Dividing by (1 - β₁ᵏ) corrects this:
```
E[m̂ₖ] = E[mₖ]/(1-β₁ᵏ) = E[∇f(x)]
```

**Step 3: Update Rule**

Combines momentum (m̂) and adaptive learning rate (1/√v̂):
```
xₖ₊₁ = xₖ - α·m̂ₖ/√v̂ₖ
```

#### Why Adam Works

1. **Momentum component (m̂):**
   - Accelerates convergence
   - Reduces variance in stochastic gradients

2. **Adaptive learning rate (1/√v̂):**
   - Different rates per parameter
   - Handles different scales

3. **Bias correction:**
   - Prevents large initial steps
   - Important in early iterations

#### Theoretical Properties

Adam approximates:
```
xₖ₊₁ = xₖ - α·E[∇f(x)]/√(Var[∇f(x)])
```

This is the **signal-to-noise ratio** - move faster when gradient is consistent (high signal, low noise).

#### Default Hyperparameters

Kingma & Ba (2014) recommended:
```
α = 0.001
β₁ = 0.9
β₂ = 0.999
ε = 10⁻⁸
```

These work well across many problems!

---

## Advanced Gradient Methods

### 5. AdaGrad (Adaptive Gradient)

#### Algorithm
```
Gₖ₊₁ = Gₖ + ∇f(xₖ)²                [accumulate all squared gradients]
xₖ₊₁ = xₖ - α·∇f(xₖ)/√(Gₖ₊₁ + ε)
```

Where G₀ = 0.

#### Mathematical Properties

The effective learning rate for parameter i is:
```
αᵢ,ₖ = α/√(Σⱼ₌₀ᵏ gᵢ,ⱼ² + ε)
```

Where gᵢ,ⱼ is the i-th component of gradient at iteration j.

#### Key Insight

**For sparse features:**
- Frequent features: accumulate large G → small learning rate
- Rare features: accumulate small G → large learning rate

This is perfect for sparse data (NLP, recommender systems)!

#### Regret Bound

AdaGrad has a **regret bound** for online convex optimization:
```
Regret = Σₖ(f(xₖ) - f(x*)) ≤ O(√T)
```

Where T is the number of iterations. This is optimal!

#### Problem: Learning Rate Decay

Since G only accumulates:
```
lim_{k→∞} αₖ → 0
```

Eventually, learning stops! This motivates RMSprop and Adam.

---

### 6. NAdam (Nesterov-Accelerated Adam)

#### Algorithm
```
mₖ₊₁ = β₁mₖ + (1-β₁)∇f(xₖ)
vₖ₊₁ = β₂vₖ + (1-β₂)∇f(xₖ)²

m̂ₖ₊₁ = mₖ₊₁/(1-β₁^(k+1))
v̂ₖ₊₁ = vₖ₊₁/(1-β₂^(k+1))

m̃ₖ₊₁ = β₁·m̂ₖ₊₁ + (1-β₁)·∇f(xₖ)/(1-β₁^(k+1))    [Nesterov]

xₖ₊₁ = xₖ - α·m̃ₖ₊₁/√(v̂ₖ₊₁ + ε)
```

#### Nesterov Momentum Intuition

Standard momentum:
```
v = βv + ∇f(x)
x = x - αv
```

Nesterov momentum ("look ahead"):
```
x_lookahead = x - α·βv
v = βv + ∇f(x_lookahead)
x = x - αv
```

Evaluate gradient at where you're **about to be**, not where you **are**!

#### Mathematical Advantage

Nesterov acceleration achieves:
```
f(xₖ) - f(x*) ≤ O(1/k²)
```

Compared to standard momentum: O(1/k)

Much faster convergence!

#### Why It Works

The "look ahead" provides better gradient information:
- Standard: gradient tells you about current position
- Nesterov: gradient tells you about future position (after momentum step)

This reduces overshooting.

---

### 7. AdamW (Adam with Weight Decay)

#### Algorithm
```
mₖ₊₁ = β₁mₖ + (1-β₁)∇f(xₖ)
vₖ₊₁ = β₂vₖ + (1-β₂)∇f(xₖ)²

m̂ₖ₊₁ = mₖ₊₁/(1-β₁^(k+1))
v̂ₖ₊₁ = vₖ₊₁/(1-β₂^(k+1))

xₖ₊₁ = xₖ - α·(m̂ₖ₊₁/√(v̂ₖ₊₁ + ε) + λxₖ)    [decoupled weight decay]
```

Where λ is the weight decay coefficient.

#### Standard Adam with L2 Regularization

Objective with L2 regularization:
```
f̃(x) = f(x) + (λ/2)‖x‖²
```

Gradient:
```
∇f̃(x) = ∇f(x) + λx
```

Standard Adam applies adaptive learning rate to this entire gradient:
```
x ← x - α·(∇f(x) + λx)/√v
```

#### Problem with Standard Approach

The weight decay term `λx` also gets the adaptive learning rate! This couples weight decay with gradient adaptation.

For parameters with large gradients: v is large, so λx/√v is small (weak regularization).

#### AdamW Solution

**Decouple** weight decay from gradient-based update:
```
x ← x - α·∇f(x)/√v - α·λx
```

Now weight decay is applied uniformly, independent of gradient statistics.

#### Why It Matters

In deep learning:
- Better generalization
- More consistent behavior across architectures
- Especially important for transformers

---

## Second-Order Methods

### 8. Newton's Method

#### Algorithm
```
xₖ₊₁ = xₖ - Hₖ⁻¹∇f(xₖ)
```

Where Hₖ = ∇²f(xₖ) is the Hessian matrix.

#### Mathematical Derivation

From second-order Taylor expansion:
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp + ½pᵀHp
```

Minimize this quadratic model with respect to p:
```
∇ₚ[f(x) + ∇f(x)ᵀp + ½pᵀHp] = ∇f(x) + Hp = 0
```

Solve for p:
```
p = -H⁻¹∇f(x)
```

This is the Newton step!

#### Why It's Fast

**For a quadratic function** f(x) = ½xᵀAx + bᵀx + c:
- Gradient: ∇f(x) = Ax + b
- Hessian: H = A (constant!)

Newton's method:
```
xₖ₊₁ = xₖ - A⁻¹(Axₖ + b) = -A⁻¹b = x*
```

Converges in **one step**!

#### Convergence Rate

Near the optimum, Newton has **quadratic convergence**:
```
‖xₖ₊₁ - x*‖ ≤ C·‖xₖ - x*‖²
```

Error squares at each iteration:
- 10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹⁶

Extremely fast!

#### Computational Cost

Each iteration requires:
1. Compute Hessian: O(n²) evaluations
2. Solve Hp = -∇f(x): O(n³) using Cholesky/LU

Total: **O(n³)** per iteration

For n = 10,000: this is 10¹² operations!

#### Modified Newton

If H is not positive definite, add regularization:
```
xₖ₊₁ = xₖ - (Hₖ + λI)⁻¹∇f(xₖ)
```

This is the **Levenberg-Marquardt** modification.

---

### 9. L-BFGS (Limited-Memory BFGS)

#### BFGS (Full)

BFGS maintains an approximation Bₖ ≈ H⁻¹ without computing the Hessian.

**Update formula:**
```
sₖ = xₖ₊₁ - xₖ             [step]
yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)    [gradient difference]

Bₖ₊₁ = (I - ρₖsₖyₖᵀ)Bₖ(I - ρₖyₖsₖᵀ) + ρₖsₖsₖᵀ
```

Where ρₖ = 1/(yₖᵀsₖ).

This is the **BFGS formula** - a rank-2 update.

#### Derivation (Sketch)

We want Bₖ₊₁ to satisfy the **secant equation**:
```
Bₖ₊₁yₖ = sₖ
```

This ensures the approximation is consistent with observed curvature.

Among all matrices satisfying this, BFGS chooses the closest to Bₖ in the weighted Frobenius norm.

#### L-BFGS: The Memory-Efficient Version

Instead of storing full n×n matrix Bₖ, store only:
- Last m pairs {(sᵢ, yᵢ)} where m ≈ 5-20

**Two-loop recursion** computes H⁻¹∇f efficiently:

```python
q = ∇f(xₖ)
for i = k-1, k-2, ..., k-m:
    αᵢ = ρᵢsᵢᵀq
    q = q - αᵢyᵢ

r = H₀q    # Initial Hessian approximation (often I)

for i = k-m, ..., k-2, k-1:
    β = ρᵢyᵢᵀr
    r = r + sᵢ(αᵢ - β)

return r   # This is approximately H⁻¹∇f(xₖ)
```

Memory: O(mn) instead of O(n²)!

#### Initial Hessian Approximation

Typically use:
```
H₀ = γₖI    where γₖ = (sₖ₋₁ᵀyₖ₋₁)/(yₖ₋₁ᵀyₖ₋₁)
```

This scaling dramatically improves performance.

#### Convergence

L-BFGS has **superlinear convergence**:
```
lim_{k→∞} ‖xₖ₊₁ - x*‖/‖xₖ - x*‖ = 0
```

Faster than linear, slower than quadratic.

#### When to Use

- Batch optimization (full gradient)
- Smooth, deterministic objectives
- Medium-scale problems (millions of variables)

**Don't use** for mini-batch training (noisy gradients break the curvature assumptions).

---

### 10. Conjugate Gradient

#### Motivation

For solving:
```
Ax = b    (A positive definite)
```

Equivalent to minimizing:
```
f(x) = ½xᵀAx - bᵀx
```

#### Algorithm
```
r₀ = Ax₀ - b = ∇f(x₀)    [residual = gradient]
p₀ = -r₀                   [first search direction]

for k = 0, 1, 2, ...:
    αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)
    xₖ₊₁ = xₖ + αₖpₖ
    rₖ₊₁ = rₖ + αₖApₖ
    βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
    pₖ₊₁ = -rₖ₊₁ + βₖpₖ
```

#### Conjugate Directions

Directions {p₀, p₁, ..., pₙ₋₁} are **A-conjugate** (orthogonal in the A-norm):
```
pᵢᵀApⱼ = 0    for i ≠ j
```

#### Remarkable Property

For a quadratic function with n variables, conjugate gradient converges in **at most n steps**!

Compare to gradient descent: can take infinite steps.

#### Why It Works

The search directions span a **Krylov subspace**:
```
Kₖ = span{r₀, Ar₀, A²r₀, ..., Aᵏ⁻¹r₀}
```

At iteration k, xₖ is the optimal point within Kₖ. By dimension, Kₙ = ℝⁿ, so xₙ = x*.

#### Non-Quadratic Functions

For general functions, use:
- **Fletcher-Reeves:** βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
- **Polak-Ribière:** βₖ = (rₖ₊₁ᵀ(rₖ₊₁-rₖ))/(rₖᵀrₖ)

Polak-Ribière tends to work better; it's what we implemented.

#### Periodic Restarts

For non-quadratic functions, restart with steepest descent direction every n iterations to maintain convergence.

---

## Basic Metaheuristics

### 11. Genetic Algorithm

#### Mathematical Model

**Population:** P = {x₁, x₂, ..., xₙ}

**Fitness:** fᵢ = f(xᵢ)

#### Selection

**Tournament Selection:**
```
P(individual i wins tournament of size k) =
    (# individuals worse than i choose k-1) / (# total choose k-1)
```

For minimum finding, lower fitness is better.

**Fitness-Proportional Selection:**
```
P(select individual i) = fᵢ/Σⱼfⱼ    (for maximization)
```

For minimization, use rank-based selection instead.

#### Crossover

**Single-point crossover:**
```
Parent 1: [a₁, a₂, a₃ | a₄, a₅, a₆]
Parent 2: [b₁, b₂, b₃ | b₄, b₅, b₆]

Child 1:  [a₁, a₂, a₃ | b₄, b₅, b₆]
Child 2:  [b₁, b₂, b₃ | a₄, a₅, a₆]
```

**Arithmetic crossover (continuous):**
```
Child = λ·Parent1 + (1-λ)·Parent2
```
Where λ ~ Uniform(0, 1).

#### Mutation

**Gaussian mutation:**
```
x' = x + N(0, σ²)
```

Mutation rate controls how often this occurs.

#### Schema Theorem (Holland)

A **schema** is a template: [*, 1, *, 0, *] where * is wildcard.

**Building Block Hypothesis:** GAs work by discovering and combining good "building blocks" (schemas).

Above-average schemas grow exponentially:
```
E[m(H, t+1)] ≥ m(H, t)·f(H)/f̄·(1 - pₘ)
```

Where:
- m(H, t): number of individuals matching schema H at time t
- f(H): average fitness of schema
- f̄: population average fitness
- pₘ: probability schema destroyed by mutation/crossover

#### No Free Lunch Theorem

For any algorithm A, averaged over all possible functions:
```
Σf P(f|A) = Σf P(f|random search)
```

**Implication:** No algorithm is universally better! GA shines on specific problem types.

---

### 12. Particle Swarm Optimization

#### Mathematical Model

**Particle i has:**
- Position: xᵢ,ₖ ∈ ℝⁿ
- Velocity: vᵢ,ₖ ∈ ℝⁿ
- Personal best: pᵢ ∈ ℝⁿ
- Global best: g ∈ ℝⁿ

#### Update Equations
```
vᵢ,ₖ₊₁ = w·vᵢ,ₖ + c₁r₁⊙(pᵢ - xᵢ,ₖ) + c₂r₂⊙(g - xᵢ,ₖ)
xᵢ,ₖ₊₁ = xᵢ,ₖ + vᵢ,ₖ₊₁
```

Where:
- w: inertia weight
- c₁, c₂: cognitive and social coefficients
- r₁, r₂ ~ Uniform(0,1)ⁿ (random vectors)
- ⊙: element-wise multiplication

#### Velocity Components

1. **Inertia:** w·vᵢ,ₖ
   - Keeps moving in current direction
   - w > 1: explosive (diverge)
   - w < 1: damping (converge)

2. **Cognitive:** c₁r₁⊙(pᵢ - xᵢ,ₖ)
   - Attraction to personal best
   - Independent thinking

3. **Social:** c₂r₂⊙(g - xᵢ,ₖ)
   - Attraction to global best
   - Swarm influence

#### Convergence Analysis

Define **constriction factor** χ:
```
χ = 2/|2 - φ - √(φ² - 4φ)|
```
Where φ = c₁ + c₂ > 4.

Modified PSO:
```
vᵢ,ₖ₊₁ = χ[vᵢ,ₖ + c₁r₁⊙(pᵢ - xᵢ,ₖ) + c₂r₂⊙(g - xᵢ,ₖ)]
```

Guarantees convergence to a point (though not necessarily the optimum).

#### Expected Position

Taking expectation (r₁, r₂ random):
```
E[xᵢ,ₖ₊₁] = xᵢ,ₖ + E[vᵢ,ₖ₊₁]
          = xᵢ,ₖ + w·vᵢ,ₖ + (c₁/2)(pᵢ - xᵢ,ₖ) + (c₂/2)(g - xᵢ,ₖ)
```

Particle moves toward weighted average of pᵢ and g.

---

### 13. Simulated Annealing

#### Algorithm
```
x₀ = initial solution
T = T₀ (initial temperature)

for k = 0, 1, 2, ...:
    x' = neighbor(xₖ)
    ΔE = f(x') - f(xₖ)

    if ΔE < 0:
        xₖ₊₁ = x'
    else:
        xₖ₊₁ = x' with probability exp(-ΔE/T)
        xₖ₊₁ = xₖ with probability 1 - exp(-ΔE/T)

    T = cool(T)    [decrease temperature]
```

#### Acceptance Probability

The **Metropolis criterion:**
```
P(accept) = {
    1                if ΔE ≤ 0
    exp(-ΔE/T)       if ΔE > 0
}
```

#### Connection to Statistical Mechanics

At temperature T, probability distribution over states:
```
P(x) = (1/Z)·exp(-f(x)/T)
```

Where Z is the partition function (normalization).

As T → 0:
```
P(x) → {
    uniform over argmin f    if x is optimal
    0                        otherwise
}
```

The system "freezes" into minimum energy state.

#### Cooling Schedule

**Logarithmic (theoretical guarantee):**
```
T(k) = T₀/log(k + 1)
```

Guarantees convergence to global minimum but **very slow**.

**Geometric (practical):**
```
T(k) = α^k·T₀
```
Where α ∈ (0.8, 0.99).

**Adaptive:**
```
T(k+1) = T(k)/(1 + β·T(k)/σ)
```
Where σ is standard deviation of recent cost changes.

#### Convergence Theorem

With logarithmic cooling and infinite time:
```
lim_{k→∞} P(xₖ = x*) = 1
```

But practical cooling schedules don't guarantee global optimum!

---

## Advanced Metaheuristics

### 14. Differential Evolution

#### Algorithm
```
For each individual xᵢ in population:
    1. Mutation:
       v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)

    2. Crossover:
       uⱼ = {
           vⱼ    if rand() < CR or j = jᵣₐₙ
           xᵢ,ⱼ  otherwise
       }

    3. Selection:
       xᵢ = {
           u     if f(u) < f(xᵢ)
           xᵢ    otherwise
       }
```

Where:
- r₁, r₂, r₃ are distinct random indices ≠ i
- F ∈ [0, 2] is the differential weight
- CR ∈ [0, 1] is the crossover probability
- jᵣₐₙ ensures at least one component is from mutant

#### Mutation Strategies

**DE/rand/1/bin:**
```
v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
```

**DE/best/1/bin (implemented):**
```
v = xₛₑₛₜ + F·(xᵣ₁ - xᵣ₂)
```

**DE/rand/2/bin:**
```
v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃) + F·(xᵣ₄ - xᵣ₅)
```

The differences (xᵣ₂ - xᵣ₃) provide **scaled random directions** that adapt to the population distribution.

#### Why It Works

The mutation vector:
```
v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
```

Can be rewritten as:
```
v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
  = (1+F)xᵣ₁ + F·(xᵣ₂ - xᵣ₃ - xᵣ₁)
```

This is a combination of:
1. Exploration: random direction (xᵣ₂ - xᵣ₃)
2. Self-adaptation: scale adapts to population spread

#### Parameter Selection

**Rule of thumb:**
- F ∈ [0.5, 1.0]: good starting range
- CR ∈ [0.8, 0.95]: high crossover usually helps
- Population: 5·n to 10·n

#### Convergence Properties

No formal convergence proof for general functions, but:
- Empirically very robust
- Often finds global optimum on difficult functions
- Used in many engineering applications

---

### 15. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

#### Core Idea

Sample from multivariate Gaussian:
```
xᵢ ~ N(m, σ²C)
```

Adapt:
1. **m** (mean) → toward better solutions
2. **σ** (step size) → global scaling
3. **C** (covariance) → shape of distribution

#### Mathematical Framework

**Sampling:**
```
xᵢ = m + σ·N(0, C)
  = m + σ·B·D·N(0, I)
```

Where C = B·D²·Bᵀ (eigenvalue decomposition):
- B: eigenvectors (principal axes)
- D: eigenvalues (scaling along axes)

#### Update Equations

**1. Mean Update:**
```
m ← Σᵢ₌₁^μ wᵢ·x_{i:λ}
```

Where x_{i:λ} is i-th best sample, and wᵢ are weights:
```
wᵢ = log(μ + 0.5) - log(i)
wᵢ ← wᵢ/Σⱼwⱼ    [normalize]
```

**2. Evolution Paths:**

Cumulative step path:
```
pσ ← (1-cσ)pσ + √(cσ(2-cσ)μₑff)·C^(-1/2)·(m_new - m_old)/σ
```

Covariance path:
```
pc ← (1-cc)pc + h·√(cc(2-cc)μₑff)·(m_new - m_old)/σ
```

Where h is Heaviside function based on ‖pσ‖.

**3. Covariance Update:**
```
C ← (1-c₁-cμ)C + c₁·pc·pc^T + cμ·Σᵢ₌₁^μ wᵢ·yᵢ·yᵢ^T
```

Where yᵢ = (x_{i:λ} - m_old)/σ.

**4. Step Size Update:**
```
σ ← σ·exp((cσ/dσ)·(‖pσ‖/E[‖N(0,I)‖] - 1))
```

#### What Each Component Does

**Evolution paths:** Track successful steps
- pσ: for step size control
- pc: for covariance update

**Covariance matrix:**
- Captures correlations between variables
- Enables efficient search in ill-conditioned landscapes

**Step size:**
- Too large: exploration (slow convergence)
- Too small: premature convergence
- Adapted using path length control

#### Why CMA-ES is State-of-the-Art

1. **Invariance properties:**
   - Rotation invariant
   - Scale invariant
   - Translation invariant

2. **Adapts to problem structure:**
   - Learns correlations
   - Handles ill-conditioning

3. **Few parameters to tune:**
   - Population size (default: 4 + 3·ln(n))
   - Initial σ (rough estimate sufficient)

#### Computational Complexity

- Eigendecomposition: O(n³) every few iterations
- Memory: O(n²) for covariance matrix
- Practical limit: ~1000 dimensions

---

### 16. Nelder-Mead Simplex

#### Simplex

A simplex in ℝⁿ is the convex hull of n+1 points.
- 2D: triangle (3 points)
- 3D: tetrahedron (4 points)

#### Operations

Given simplex vertices {x₁, x₂, ..., xₙ₊₁} ordered by f(x₁) ≤ f(x₂) ≤ ... ≤ f(xₙ₊₁):

**Centroid** (of all but worst):
```
x̄ = (1/n)Σᵢ₌₁ⁿ xᵢ
```

**Reflection:**
```
xᵣ = x̄ + α(x̄ - xₙ₊₁)
```
Default: α = 1 (mirror worst point through centroid)

**Expansion** (if reflection is best so far):
```
xₑ = x̄ + γ(xᵣ - x̄)
```
Default: γ = 2 (go further in good direction)

**Contraction** (if reflection is bad):
```
xc = x̄ + ρ(xₙ₊₁ - x̄)    [inside]
xc = x̄ + ρ(xᵣ - x̄)       [outside]
```
Default: ρ = 0.5

**Shrink** (if contraction fails):
```
xᵢ ← x₁ + σ(xᵢ - x₁)    for i = 2, ..., n+1
```
Default: σ = 0.5

#### Decision Tree
```
if f(xᵣ) < f(x₁):
    try expansion
else if f(xᵣ) < f(xₙ):
    accept reflection
else if f(xᵣ) < f(xₙ₊₁):
    outside contraction
else:
    inside contraction
    if still bad: shrink
```

#### Convergence

**No general convergence theory!**

Can fail on non-smooth functions or converge to non-stationary points.

But works well in practice for:
- Small dimensions (< 20)
- Reasonably smooth functions
- When you don't have gradients

#### Why It Works (Heuristic)

The simplex adapts its:
- **Position:** moves toward better regions
- **Size:** contracts when near optimum
- **Shape:** stretches along valleys

Like a biological organism searching for food!

---

### 17. Bayesian Optimization

#### Problem Setup

We want to minimize f(x), but f is:
- Expensive to evaluate
- Black-box (no gradients)
- Possibly noisy

#### Framework

1. **Surrogate model:** P(f|data)
   - Usually Gaussian Process

2. **Acquisition function:** α(x|P)
   - Determines where to sample next

3. **Optimize acquisition:**
   - xₙₑₓₜ = argmax α(x)

4. **Evaluate:** yₙₑₓₜ = f(xₙₑₓₜ)

5. **Update model** and repeat

#### Gaussian Process Prior

A GP is a distribution over functions:
```
f(x) ~ GP(m(x), k(x, x'))
```

Where:
- m(x): mean function (often 0)
- k(x, x'): covariance kernel

**RBF Kernel (common choice):**
```
k(x, x') = σ²·exp(-‖x - x'‖²/(2ℓ²))
```

Parameters:
- σ²: signal variance
- ℓ: length scale

#### GP Posterior

Given observations D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ:

Posterior at x:
```
f(x)|D ~ N(μ(x), σ²(x))
```

Where:
```
μ(x) = k(x)ᵀ(K + σₙ²I)⁻¹y
σ²(x) = k(x,x) - k(x)ᵀ(K + σₙ²I)⁻¹k(x)
```

And:
- K: Gram matrix [k(xᵢ, xⱼ)]
- k(x): [k(x, x₁), ..., k(x, xₙ)]ᵀ
- σₙ²: noise variance

#### Acquisition Functions

**Expected Improvement (EI):**
```
EI(x) = E[max(f(x⁺) - f(x), 0)]
     = (f(x⁺) - μ(x))Φ(Z) + σ(x)φ(Z)
```

Where:
- x⁺ = current best
- Z = (f(x⁺) - μ(x))/σ(x)
- Φ, φ: CDF and PDF of N(0,1)

**Upper Confidence Bound (UCB):**
```
UCB(x) = μ(x) + κ·σ(x)
```

Where κ controls exploration (typically 2-3).

**Probability of Improvement (PI):**
```
PI(x) = Φ((f(x⁺) - μ(x))/σ(x))
```

#### Why It Works

**Exploration vs Exploitation:**
- High μ(x): exploitation (go to promising area)
- High σ(x): exploration (go to uncertain area)
- Acquisition function balances both

**Sample Efficiency:**
- Each evaluation improves the model
- Smart sequential design
- Typically needs only 50-500 evaluations

#### Regret Bounds

For UCB with appropriate κ:
```
Cumulative regret ≤ O(√(T·γₜ·log T))
```

Where γₜ is the information gain (depends on kernel).

Sub-linear regret → average regret decreases!

---

### 18. Harmony Search

#### Musical Analogy

Musicians improvise to find perfect harmony:
- **Harmony Memory (HM):** past good harmonies
- **Pitch adjustment:** small tweaks
- **Randomization:** try new notes

#### Algorithm

**Initialization:**
Create Harmony Memory:
```
HM = [x₁, x₂, ..., xₕₘₛ]
```

**Improvisation:**
For each variable j:
```
1. With probability HMCR:
     x'ⱼ ← xᵢ,ⱼ for random i ∈ {1,...,HMS}

     With probability PAR:
       x'ⱼ ← x'ⱼ + BW·U(-1, 1)

2. With probability 1-HMCR:
     x'ⱼ ← random in [Lⱼ, Uⱼ]
```

**Update:**
If f(x') < f(worst in HM):
```
Replace worst with x'
```

#### Parameters

- **HMCR** (Harmony Memory Considering Rate): 0.7-0.95
  - Probability of using memory

- **PAR** (Pitch Adjusting Rate): 0.2-0.5
  - Probability of adjusting selected value

- **BW** (Bandwidth): problem-dependent
  - Range of adjustment

#### Mathematical Model

Let pⱼ(value) be probability distribution of variable j in current HM.

Improvisation samples from mixture:
```
P(x'ⱼ) = HMCR·[PAR·G(pⱼ, BW) + (1-PAR)·pⱼ] + (1-HMCR)·U(Lⱼ, Uⱼ)
```

Where G is Gaussian perturbation around pⱼ.

#### Comparison to Genetic Algorithm

**Similarities:**
- Population-based
- Stochastic operators

**Differences:**
- HS: all population members can contribute to offspring
- GA: only selected parents contribute

**Advantages of HS:**
- Simpler (no crossover complexity)
- All members participate
- Easy to implement

---

## Convergence Analysis

### Convergence Rates

**Linear convergence:**
```
‖xₖ - x*‖ ≤ c^k·‖x₀ - x*‖
```
Where c ∈ (0, 1). Error decreases exponentially.

**Superlinear convergence:**
```
lim_{k→∞} ‖xₖ₊₁ - x*‖/‖xₖ - x*‖ = 0
```
Faster than any linear rate.

**Quadratic convergence:**
```
‖xₖ₊₁ - x*‖ ≤ C·‖xₖ - x*‖²
```
Error squares each iteration!

### Algorithm Classification

| Algorithm | Rate | Conditions |
|-----------|------|------------|
| Gradient Descent | Linear | Convex, smooth |
| Momentum | Linear (faster) | Convex, smooth |
| Adam, RMSprop | Linear | Convex (empirical) |
| Conjugate Gradient | Superlinear | Strongly convex |
| L-BFGS | Superlinear | Smooth |
| Newton | Quadratic | Twice differentiable, near optimum |
| Metaheuristics | Stochastic | Problem-dependent |

### Complexity Lower Bounds

**Black-box first-order methods:**

For finding ε-optimal solution:
```
Iterations ≥ Ω(√(L/μ)·log(1/ε))
```

This is achieved by accelerated gradient descent (Nesterov).

**Black-box zero-order methods:**

Need at least:
```
Evaluations ≥ Ω(n·log(1/ε))
```

Where n is dimension.

### Practical Convergence

**Stopping criteria:**

1. **Gradient norm:**
   ```
   ‖∇f(x)‖ < ε
   ```

2. **Function change:**
   ```
   |f(xₖ₊₁) - f(xₖ)| < ε
   ```

3. **Parameter change:**
   ```
   ‖xₖ₊₁ - xₖ‖ < ε
   ```

Use **multiple criteria** for robustness!

---

## Summary

This covers the mathematical foundations of all 18 algorithms:

**Gradient-based methods** use first-order information (∇f) and possibly second-order (∇²f) to iteratively improve solutions.

**Metaheuristic methods** use population-based search, randomization, and problem-specific heuristics without requiring derivatives.

Each algorithm has:
- **Update rule**: how to generate next iterate
- **Parameters**: that need tuning
- **Convergence properties**: how fast it reaches the optimum
- **Use cases**: where it excels

The mathematics shows us **why** these algorithms work and **when** to use each one!
