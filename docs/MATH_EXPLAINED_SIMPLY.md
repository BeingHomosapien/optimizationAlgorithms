# Mathematics of Optimization - Explained Simply

A comprehensive, beginner-friendly guide to understanding the mathematics behind all 18 optimization algorithms.

## Table of Contents
1. [Prerequisites - What You Need to Know](#prerequisites)
2. [The Optimization Problem](#the-optimization-problem)
3. [Gradient Descent - The Foundation](#gradient-descent)
4. [Momentum - Adding Memory](#momentum)
5. [Adam - Smart Adaptive Learning](#adam)
6. [Newton's Method - Using Curvature](#newtons-method)
7. [L-BFGS - Efficient Quasi-Newton](#l-bfgs)
8. [Conjugate Gradient - Clever Directions](#conjugate-gradient)
9. [Advanced Gradient Methods](#advanced-gradient-methods)
10. [Particle Swarm - Swarm Intelligence](#particle-swarm-optimization)
11. [Genetic Algorithms - Evolution](#genetic-algorithms)
12. [CMA-ES - Learning Structure](#cma-es)
13. [Bayesian Optimization - Smart Sampling](#bayesian-optimization)
14. [Other Metaheuristics](#other-metaheuristics)
15. [Convergence Analysis](#convergence-analysis)
16. [Practical Examples](#practical-examples)

---

## Prerequisites

### What You Need to Know

**Calculus:**
- Derivatives: rate of change
- Partial derivatives: change in one variable
- Gradients: vector of partial derivatives

**Linear Algebra:**
- Vectors: ordered lists of numbers
- Matrices: 2D arrays of numbers
- Matrix multiplication
- Dot products

**Basic Probability:**
- Mean (average)
- Variance (spread)
- Normal distribution (bell curve)

### The Essential Math Tools

#### 1. Gradient (∇f)

Think of a hill. The gradient points in the direction you should walk to go **uphill** fastest.

**Mathematically:**
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Example:** For `f(x, y) = x² + 2y²`:
```
∂f/∂x = 2x
∂f/∂y = 4y
∇f(x, y) = [2x, 4y]ᵀ
```

At point (1, 1):
```
∇f(1, 1) = [2, 4]ᵀ
```

This vector [2, 4] points uphill. To go downhill, use -∇f = [-2, -4].

#### 2. Hessian (H)

The Hessian is like a curvature map. It tells you how the gradient itself is changing.

**Mathematically:**
```
H = [∂²f/∂xᵢ∂xⱼ]
```

**Example:** For `f(x, y) = x² + 2y²`:
```
∂²f/∂x² = 2        ∂²f/∂x∂y = 0
∂²f/∂y∂x = 0       ∂²f/∂y² = 4

H = [2  0]
    [0  4]
```

This tells us:
- Curvature in x direction: 2
- Curvature in y direction: 4 (steeper!)
- No coupling between x and y (off-diagonal zeros)

#### 3. Taylor Series

Taylor series lets us approximate complicated functions with simple polynomials.

**First-order (linear):**
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp
```

Think: "If I move by p, the function changes by roughly ∇f·p"

**Second-order (quadratic):**
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp + ½pᵀHp
```

More accurate! Includes curvature effects.

**Example:** f(x) = x² at x = 1, estimating f(1.1):

Actual:
```
f(1.1) = (1.1)² = 1.21
```

First-order Taylor:
```
f(1 + 0.1) ≈ f(1) + f'(1)·0.1
           = 1 + 2·0.1
           = 1.2
```

Second-order Taylor:
```
f(1 + 0.1) ≈ f(1) + f'(1)·0.1 + ½f''(1)·(0.1)²
           = 1 + 2·0.1 + ½·2·0.01
           = 1 + 0.2 + 0.01
           = 1.21  ✓ Exact!
```

For quadratics, second-order Taylor is **exact**!

---

## The Optimization Problem

### What Are We Trying to Do?

Find the point x* where function f(x) is smallest:

```
x* = argmin f(x)
    x ∈ ℝⁿ
```

**Real-world examples:**

**Example 1: Curve Fitting**
```
Data: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
Model: y = ax + b
Find a, b that minimize: f(a, b) = Σ(yᵢ - (axᵢ + b))²
```

**Example 2: Neural Network Training**
```
Network: f(x; θ) where θ = weights
Loss: L(θ) = how wrong the predictions are
Find θ* = argmin L(θ)
```

**Example 3: Portfolio Optimization**
```
Variables: x = [amount in stock 1, stock 2, ...]
Objective: f(x) = -expected return + λ·risk
Find x* to maximize return while controlling risk
```

### Optimality Conditions

**At a minimum x*, we need:**

**First-order (necessary):**
```
∇f(x*) = 0
```
The gradient is zero - no direction to descend!

**Second-order (sufficient):**
```
∇f(x*) = 0  AND  H(x*) ≻ 0 (positive definite)
```
Not only is gradient zero, but we're at a bowl (not a saddle point).

**Example:** f(x) = x²
- f'(x) = 2x → f'(0) = 0 ✓
- f''(x) = 2 > 0 ✓
- Therefore x* = 0 is a minimum

**Counter-example:** f(x) = x³
- f'(x) = 3x² → f'(0) = 0 ✓
- f''(0) = 0 ✗ (not positive!)
- x = 0 is an **inflection point**, not a minimum

---

## Gradient Descent

### The Basic Idea

Imagine you're on a mountain in fog. You can only see your immediate surroundings. How do you get to the valley?

**Strategy:** Feel which way slopes down most steeply, take a small step that direction, repeat.

This is gradient descent!

### The Algorithm

```
Start: x₀ (initial guess)
Repeat:
    1. Compute gradient: g = ∇f(xₖ)
    2. Take step downhill: xₖ₊₁ = xₖ - α·g
Until: converged
```

Where α > 0 is the **step size** (or **learning rate**).

### Why Does It Work?

**Mathematical proof:**

From Taylor expansion:
```
f(xₖ - α∇f(xₖ)) ≈ f(xₖ) + ∇f(xₖ)ᵀ(-α∇f(xₖ))
                 = f(xₖ) - α·‖∇f(xₖ)‖²
```

Since α > 0 and ‖∇f‖² ≥ 0:
```
f(xₖ - α∇f(xₖ)) ≤ f(xₖ)
```

**We always go downhill!** (for small enough α)

### Step-by-Step Example

**Problem:** Minimize f(x) = x²

**Starting point:** x₀ = 4
**Learning rate:** α = 0.5

**Iteration 1:**
```
f(x₀) = 16
∇f(x₀) = 2·4 = 8
x₁ = x₀ - α·∇f(x₀) = 4 - 0.5·8 = 0
```

Done in one step! (Quadratics are special)

**Another example:** f(x, y) = x² + 4y²

Starting at (4, 2), α = 0.2:

```
Iteration 0:
  x = [4, 2]
  f(x) = 16 + 16 = 32
  ∇f = [2·4, 8·2] = [8, 16]

Iteration 1:
  x = [4, 2] - 0.2·[8, 16] = [2.4, -1.2]
  f(x) = 5.76 + 5.76 = 11.52
  ∇f = [4.8, -9.6]

Iteration 2:
  x = [2.4, -1.2] - 0.2·[4.8, -9.6] = [1.44, 0.72]
  f(x) = 2.07 + 2.07 = 4.14
  ...
```

Converging to (0, 0)!

### Choosing the Learning Rate

**Too small:** Slow progress
```
α = 0.01: takes forever
```

**Too large:** Overshoot, diverge
```
α = 2.0: might oscillate or explode
```

**Just right:** Fast convergence
```
α ≈ 1/L where L is Lipschitz constant
```

**Practical strategies:**

1. **Fixed:** α = 0.01 (try different values)

2. **Decaying:** α_k = α₀/(1 + k)
   ```
   Start fast, slow down over time
   ```

3. **Backtracking line search:**
   ```
   Start with α = 1
   While f(x - α∇f) > f(x) - c·α·‖∇f‖²:
       α = 0.5·α
   ```
   Automatically finds good α!

### Convergence Rate

For **strongly convex** functions (bowl-shaped):

```
‖xₖ - x*‖ ≤ (1 - μ/L)ᵏ·‖x₀ - x*‖
```

Where:
- L: Lipschitz constant (how steep function can be)
- μ: strong convexity constant (how bowl-shaped)
- κ = L/μ: **condition number**

**Example:** κ = 100
```
After k steps, error reduced by (0.99)ᵏ
k = 100: error × 0.37
k = 500: error × 7×10⁻³
```

**The problem:** When κ is large (10,000+), convergence is SLOW!

### Why Gradient Descent Can Be Slow

**Ill-conditioned problems:**

Imagine a narrow valley (like f(x, y) = x² + 100y²):
- Steep in y direction
- Gentle in x direction

Gradient descent zig-zags down the valley!

```
x₀ = [10, 10]
∇f = [20, 2000]  (mostly in y direction!)

After step:
  y decreases a lot
  x barely moves

Next gradient points back across valley...
```

**Solution:** Use momentum, Adam, or second-order methods!

### Gradient Descent Variants

**Batch:** Use full gradient
```
g = ∇f(x)  [compute on all data]
```

**Stochastic (SGD):** Use single sample
```
g ≈ ∇fᵢ(x)  [compute on one data point]
Fast but noisy!
```

**Mini-batch:** Middle ground
```
g ≈ (1/m)Σᵢ∇fᵢ(x)  [compute on m samples]
Common: m = 32, 64, 128
```

---

## Momentum

### The Intuition

Imagine rolling a ball down a hill:
- It gains speed going downhill (accelerates)
- It dampens oscillations (friction prevents wild swings)
- It can roll through small bumps (momentum carries it)

This is what momentum does for optimization!

### The Algorithm

```
v₀ = 0  (initial velocity)

Repeat:
    vₖ₊₁ = β·vₖ + ∇f(xₖ)
    xₖ₊₁ = xₖ - α·vₖ₊₁
```

Where β ∈ [0, 1) is the momentum coefficient (typically 0.9).

### The Mathematics

**Expanding the recursion:**
```
v₁ = β·v₀ + ∇f(x₀) = ∇f(x₀)
v₂ = β·v₁ + ∇f(x₁) = β∇f(x₀) + ∇f(x₁)
v₃ = β·v₂ + ∇f(x₂) = β²∇f(x₀) + β∇f(x₁) + ∇f(x₂)
...
vₖ = Σᵢ₌₀^(k-1) βⁱ∇f(xₖ₋ᵢ)
```

This is an **exponentially weighted moving average**!

**Weights decay exponentially:**
```
Current gradient: weight = 1
1 step ago: weight = β
2 steps ago: weight = β²
3 steps ago: weight = β³
```

For β = 0.9:
```
Current: 1.000
1 back: 0.900
2 back: 0.810
5 back: 0.590
10 back: 0.349
20 back: 0.122
```

**Effective memory:**
```
Memory ≈ 1/(1-β) iterations
```

β = 0.9 → remembers ~10 steps
β = 0.99 → remembers ~100 steps

### Why It Works Better

**Case 1: Consistent gradient direction**

Imagine going down a long valley. Gradients all point the same way:
```
∇f(x₀) = [1, 0]
∇f(x₁) = [1, 0]
∇f(x₂) = [1, 0]
...
```

Without momentum:
```
Step size each iteration: α
```

With momentum (β = 0.9):
```
v₁ = [1, 0]
v₂ = 0.9·[1, 0] + [1, 0] = [1.9, 0]
v₃ = 0.9·[1.9, 0] + [1, 0] = [2.71, 0]
...
v_∞ → [10, 0]  (sum of geometric series: 1/(1-0.9))
```

**Effective step size: 10×α!** Massive acceleration!

**Case 2: Oscillating gradients**

Zig-zagging across a valley:
```
∇f(x₀) = [1, 5]
∇f(x₁) = [1, -4]
∇f(x₂) = [1, 4]
∇f(x₃) = [1, -4]
...
```

Without momentum: large oscillations

With momentum:
```
v₁ = [1, 5]
v₂ = 0.9·[1, 5] + [1, -4] = [1.9, 0.5]
v₃ = 0.9·[1.9, 0.5] + [1, 4] = [2.71, 4.45]
v₄ = 0.9·[2.71, 4.45] + [1, -4] = [3.44, 0.01]
...
```

Oscillations in y are dampened! Only consistent x-direction accumulates.

### Physical Analogy

Newton's second law with friction:
```
m·a = -∇f(x) - γ·v
```

Where:
- m: mass (related to 1/α)
- -∇f(x): force from potential energy (gradient)
- -γ·v: friction force (opposes motion)

Discretizing this gives momentum update!

The ball:
- Accelerates down slopes
- Friction prevents runaway
- Momentum carries through small bumps (local minima)

### Optimal Momentum Coefficient

For quadratic f(x) = ½xᵀAx with condition number κ:

**Optimal β:**
```
β* = (√κ - 1)/(√κ + 1)
```

Examples:
```
κ = 4:    β* = 0.33
κ = 100:  β* = 0.82
κ = 10000: β* = 0.98
```

**Key insight:** Worse conditioning → need more momentum!

### Convergence Improvement

Gradient descent: O(κ) iterations
Momentum: O(√κ) iterations

**Example:** κ = 10,000
- Gradient descent: ~10,000 iterations
- Momentum: ~100 iterations

**100× faster!**

### Nesterov Momentum (Preview)

Standard momentum:
```
v = β·v + ∇f(x)
x = x - α·v
```

Nesterov momentum (look ahead):
```
x_ahead = x - α·β·v
v = β·v + ∇f(x_ahead)  [gradient at future position!]
x = x - α·v
```

Even faster! Converges as O(1/k²) vs O(1/k).

---

## Adam

### The Big Idea

**Problem:** Different parameters need different learning rates!

**Example:** Neural network
- Weight in first layer: gradient = 0.001 → needs large learning rate
- Weight in last layer: gradient = 10.0 → needs small learning rate

**Adam's solution:** Adapt the learning rate for each parameter based on:
1. **First moment (m):** Average gradient (like momentum)
2. **Second moment (v):** Average squared gradient (estimates variance)

### The Algorithm

```
Initialize:
  m₀ = 0  (first moment)
  v₀ = 0  (second moment)

For k = 1, 2, 3, ...:
  g = ∇f(xₖ)

  # Update moments
  m = β₁·m + (1-β₁)·g
  v = β₂·v + (1-β₂)·g²  (element-wise square)

  # Bias correction
  m̂ = m/(1 - β₁ᵏ)
  v̂ = v/(1 - β₂ᵏ)

  # Update parameters
  x = x - α·m̂/(√v̂ + ε)
```

Default hyperparameters (work amazingly well!):
```
α = 0.001
β₁ = 0.9
β₂ = 0.999
ε = 10⁻⁸
```

### Step-by-Step Breakdown

**1. First Moment (m):**

Exponential moving average of gradients:
```
m = 0.9·m + 0.1·g
```

This is like momentum! Accumulates past gradients.

**2. Second Moment (v):**

Exponential moving average of **squared** gradients:
```
v = 0.999·v + 0.001·g²
```

This tracks how much the gradient fluctuates.

**3. Bias Correction:**

**Problem:** Starting from m₀ = 0, early estimates are biased toward zero.

After k steps:
```
E[m] = E[g]·(1 - β₁ᵏ)
```

**Solution:** Divide by (1 - β₁ᵏ):
```
m̂ = m/(1 - β₁ᵏ)
```

Now E[m̂] = E[g] ✓

**4. Adaptive Learning Rate:**

Each parameter i gets learning rate:
```
αᵢ = α/√(vᵢ + ε)
```

- If gradients are large and consistent: v is large → small effective learning rate
- If gradients are small and noisy: v is small → large effective learning rate

### Numerical Example

**Problem:** Minimize f(x, y) = x² + 100y²

Starting: x₀ = [10, 10]
Gradient: ∇f = [20, 2000] (note: huge difference!)

**Iteration 1:**
```
g = [20, 2000]
m = 0.9·0 + 0.1·[20, 2000] = [2, 200]
v = 0.999·0 + 0.001·[400, 4000000] = [0.4, 4000]

m̂ = [2, 200]/(1-0.9) = [20, 2000]
v̂ = [0.4, 4000]/(1-0.999) = [400, 4000000]

Update:
  x₁ = [10, 10] - 0.001·[20, 2000]/(√[400, 4000000] + 10⁻⁸)
     = [10, 10] - 0.001·[20/20, 2000/2000]
     = [10, 10] - 0.001·[1, 1]
     = [9.999, 9.999]
```

**Key observation:** Despite gradient being 100× larger in y direction, the adaptive learning rate made the steps equal!

This is the genius of Adam!

### Why Adam Works

**Interpret the update:**
```
x = x - α·m̂/(√v̂ + ε)
  = x - α·E[g]/√Var[g]
```

This is approximately:
```
Signal-to-noise ratio!
```

- High signal (consistent gradient): large update
- High noise (fluctuating gradient): small update

**Perfect for stochastic optimization** (mini-batch training)!

### Bias Correction Importance

**Without bias correction:**

Starting from m = 0, v = 0:
```
Iteration 1:
  m = 0.9·0 + 0.1·g = 0.1·g  (too small!)
  v = 0.999·0 + 0.001·g² = 0.001·g²

  Update: x - α·(0.1·g)/(√0.001·g²)
        = x - α·(0.1·g)/(0.0316·g)
        = x - 3.16·α·sign(g)  (huge step!)
```

Early steps would be way too large!

**With bias correction:**
```
m̂ = 0.1·g/(1-0.9) = g  ✓
v̂ = 0.001·g²/(1-0.999) = g²  ✓

Update: x - α·g/√g² = x - α·sign(g)  (reasonable)
```

Much more stable!

### Adam vs Other Methods

**vs Gradient Descent:**
- Adam adapts learning rate per parameter
- No need to tune learning rate as carefully

**vs Momentum:**
- Adam has adaptive learning rates
- Better for problems with different scales

**vs RMSprop:**
- Adam adds momentum (first moment)
- Bias correction for early iterations

### Variants

**AMSGrad:** Uses max of past v values
```
v̂_max = max(v̂_max, v̂)
x = x - α·m̂/(√v̂_max + ε)
```

Never decreases learning rate → better convergence guarantees

**AdamW:** Decouples weight decay
```
x = x - α·(m̂/(√v̂ + ε) + λ·x)
```

Better for regularization in deep learning

---

## Newton's Method

### The Big Idea

Gradient descent uses a **linear** approximation:
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp
```

Newton uses a **quadratic** approximation:
```
f(x + p) ≈ f(x) + ∇f(x)ᵀp + ½pᵀH(x)p
```

This includes curvature information (Hessian H), allowing much better steps!

### The Algorithm

```
For k = 0, 1, 2, ...:
  Compute Hessian: H = ∇²f(xₖ)
  Compute gradient: g = ∇f(xₖ)
  Solve: Hp = -g  (for search direction p)
  Update: xₖ₊₁ = xₖ + p
```

Equivalent form:
```
xₖ₊₁ = xₖ - H⁻¹∇f(xₖ)
```

### Mathematical Derivation

We want to minimize the quadratic model:
```
m(p) = f(xₖ) + ∇f(xₖ)ᵀp + ½pᵀH(xₖ)p
```

Take derivative with respect to p:
```
∇ₚm(p) = ∇f(xₖ) + H(xₖ)p
```

Set to zero for minimum:
```
∇f(xₖ) + H(xₖ)p = 0
```

Solve for p:
```
p = -H(xₖ)⁻¹∇f(xₖ)
```

This is the Newton step!

### Why It's So Fast

**For quadratic functions:**

Consider f(x) = ½xᵀAx + bᵀx + c

Gradient: ∇f(x) = Ax + b
Hessian: H = A (constant!)

Newton step from xₖ:
```
xₖ₊₁ = xₖ - A⁻¹(Axₖ + b)
     = xₖ - xₖ - A⁻¹b
     = -A⁻¹b
     = x*  (the exact optimum!)
```

**Newton converges in ONE step for quadratics!**

### Quadratic Convergence

Near the optimum x*, Newton has **quadratic convergence**:
```
‖xₖ₊₁ - x*‖ ≤ C·‖xₖ - x*‖²
```

**What this means:**

If you're within 10⁻² of the optimum:
```
k=0: error = 10⁻²
k=1: error ≤ C·(10⁻²)² = C·10⁻⁴
k=2: error ≤ C·(10⁻⁴)² = C·10⁻⁸
k=3: error ≤ C·(10⁻⁸)² = C·10⁻¹⁶
```

Error **squares** each iteration! Incredibly fast!

### Worked Example

**Problem:** f(x, y) = x² + 4y²

**Derivatives:**
```
∂f/∂x = 2x
∂f/∂y = 8y

∂²f/∂x² = 2
∂²f/∂y² = 8
∂²f/∂x∂y = 0

H = [2  0]
    [0  8]

H⁻¹ = [0.5  0  ]
      [0    0.125]
```

**Starting point:** x₀ = [4, 2]

**Newton step:**
```
∇f(x₀) = [8, 16]

p = -H⁻¹∇f = -[0.5  0  ] [8 ] = [-4]
              [0  0.125] [16]   [-2]

x₁ = [4, 2] + [-4, -2] = [0, 0] = x*
```

**Converged in one step!** (Because it's quadratic)

### The Computational Cost

**Per iteration:**

1. **Compute Hessian:** O(n²) function evaluations
   - Need ∂²f/∂xᵢ∂xⱼ for all i, j pairs
   - That's n×n matrix

2. **Invert Hessian:** O(n³) operations
   - Using Cholesky decomposition
   - Or solve Hp = -g directly

**Example:** n = 1,000
- Hessian: 1,000,000 entries
- Inversion: 1,000,000,000 operations

For n = 10,000: 10¹² operations per iteration!

**This is why we need L-BFGS for large problems!**

### Modified Newton Method

**Problem:** H might not be positive definite (could be at saddle point)

**Solution:** Add regularization:
```
p = -(H + λI)⁻¹∇f
```

λ > 0 ensures H + λI is positive definite.

**Choosing λ:**
- If H is positive definite: λ = 0 (pure Newton)
- Otherwise: λ = max eigenvalue of -H + δ

This is called **Levenberg-Marquardt** modification.

### When to Use Newton

**Good for:**
- Small dimensions (n < 1000)
- High-accuracy requirements
- Few iterations acceptable
- Hessian available (or cheap to compute)

**Not good for:**
- Large-scale problems (n > 10,000)
- Hessian expensive
- Need many cheap iterations
- Non-smooth functions

### Trust Region Interpretation

Newton assumes quadratic model is good globally. Not always true!

**Trust region:** Only trust model within radius Δ:
```
min m(p) subject to ‖p‖ ≤ Δ
```

Start with small Δ, increase if model is accurate, decrease if not.

This makes Newton robust far from optimum!

---

## L-BFGS

### The Problem Newton Solves (and Its Cost)

Newton's method is amazing but expensive:
- Computing Hessian: O(n²) storage, O(n²) evaluations
- Inverting Hessian: O(n³) operations

For n = 1,000,000 (common in machine learning):
- Can't even store full Hessian!

### The Big Idea

**Don't compute the Hessian explicitly!**

Instead, build an approximation Bₖ ≈ H⁻¹ using only:
- Gradient information
- Past steps

### The Secant Equation

True Hessian satisfies:
```
H(x)·(x₁ - x₀) ≈ ∇f(x₁) - ∇f(x₀)
```

This says: "Change in gradient ≈ Hessian × change in position"

Define:
```
sₖ = xₖ₊₁ - xₖ           (step taken)
yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)   (gradient change)
```

**Secant equation:**
```
Hₖsₖ ≈ yₖ
```

Or for the inverse:
```
Bₖyₖ ≈ sₖ    where Bₖ ≈ H⁻¹
```

**Key insight:** We can build Bₖ from past (sᵢ, yᵢ) pairs!

### BFGS Update Formula

Start with Bₖ (current approximation of H⁻¹).

After taking step sₖ and observing yₖ, update:
```
Bₖ₊₁ = (I - ρₖsₖyₖᵀ)Bₖ(I - ρₖyₖsₖᵀ) + ρₖsₖsₖᵀ
```

Where:
```
ρₖ = 1/(yₖᵀsₖ)
```

This update:
1. Satisfies secant equation: Bₖ₊₁yₖ = sₖ ✓
2. Maintains symmetry ✓
3. Maintains positive definiteness (if yₖᵀsₖ > 0) ✓
4. Is a rank-2 update (only modifies Bₖ in two directions)

### L-BFGS: Limited Memory Version

**Problem:** Even Bₖ requires O(n²) storage!

**Solution:** Don't store Bₖ explicitly. Store only recent {(sᵢ, yᵢ)} pairs.

**Memory:** Keep last m pairs (typically m = 5-20)
```
{(sₖ₋ₘ, yₖ₋ₘ), ..., (sₖ₋₁, yₖ₋₁)}
```

**Storage:** Only O(mn) instead of O(n²)!

For n = 1,000,000, m = 10:
- L-BFGS: 10 million numbers
- Full BFGS: 1 trillion numbers!

### Two-Loop Recursion

To compute d = H⁻¹∇f without forming H⁻¹ explicitly:

```python
def compute_direction(g, s_list, y_list, m):
    """
    Compute H^{-1} * g using two-loop recursion

    g: current gradient
    s_list: [s_{k-m}, ..., s_{k-1}]
    y_list: [y_{k-m}, ..., y_{k-1}]
    """
    q = g.copy()
    alphas = []

    # First loop (backward)
    for i in reversed(range(m)):
        s, y = s_list[i], y_list[i]
        rho = 1.0 / (y.dot(s))
        alpha = rho * s.dot(q)
        alphas.append(alpha)
        q = q - alpha * y

    # Initial Hessian approximation
    if m > 0:
        s, y = s_list[-1], y_list[-1]
        gamma = s.dot(y) / y.dot(y)
        r = gamma * q
    else:
        r = q

    # Second loop (forward)
    for i in range(m):
        s, y = s_list[i], y_list[i]
        rho = 1.0 / (y.dot(s))
        beta = rho * y.dot(r)
        alpha = alphas[m - 1 - i]
        r = r + s * (alpha - beta)

    return -r  # Negative for descent direction
```

**Complexity:** O(mn) time, O(mn) storage

**Comparison:**
- Newton: O(n³) time, O(n²) storage
- L-BFGS: O(mn) time, O(mn) storage

For n = 1M, m = 10:
- Newton: 10¹⁸ ops, 10¹² storage → impossible!
- L-BFGS: 10⁷ ops, 10⁷ storage → totally doable!

### Numerical Example

**Simple 2D problem:** f(x, y) = x² + 4y²

True Hessian:
```
H = [2  0]
    [0  8]

H⁻¹ = [0.5  0  ]
      [0    0.125]
```

**Build approximation:**

Start at x₀ = [4, 2], step to x₁ = [2, 1]:
```
s₀ = x₁ - x₀ = [-2, -1]
y₀ = ∇f(x₁) - ∇f(x₀) = [4, 8] - [8, 16] = [-4, -8]

ρ₀ = 1/(y₀ᵀs₀) = 1/((-4)(-2) + (-8)(-1)) = 1/16

Initial B₀ = I
```

After one BFGS update:
```
B₁ ≈ [0.5    0.125]
     [0.125  0.156]
```

Compare to true H⁻¹:
```
H⁻¹ = [0.5  0  ]
      [0    0.125]
```

Pretty close! Gets better with more iterations.

### Convergence Properties

**Theorem:** On strongly convex functions, L-BFGS has **superlinear convergence**:
```
lim_{k→∞} ‖xₖ₊₁ - x*‖/‖xₖ - x*‖ = 0
```

Not as fast as Newton (quadratic), but much better than linear!

**Practical behavior:**
- First few iterations: similar to gradient descent
- Near optimum: accelerates dramatically
- Overall: much fewer iterations than gradient descent

### When to Use L-BFGS

**Perfect for:**
- Batch optimization (full gradients available)
- Smooth, deterministic objectives
- Medium to large scale (10³ to 10⁶ variables)
- Need fast convergence

**Examples:**
- Logistic regression training
- Neural network training (full batch)
- Scientific computing
- Parameter estimation

**Don't use for:**
- Stochastic/mini-batch training (noisy gradients)
- Very large scale (>10⁷ variables)
- Non-smooth functions

**Practical tip:** L-BFGS is the default choice for smooth batch optimization!

---

## Conjugate Gradient

### The Motivation

**Problem:** Solve Ax = b where A is n×n positive definite

**Why not just invert:** A⁻¹b?
- Cost: O(n³)
- For n = 1,000,000: impossible!

**Gradient descent:**
- Minimizing f(x) = ½xᵀAx - bᵀx
- Converges slowly (O(κn) iterations)

**Conjugate gradient:**
- Converges in **at most n iterations**
- Each iteration: O(n) (just matrix-vector product)
- Total: O(n²) instead of O(n³)!

### The Key Concept: Conjugate Directions

**Orthogonal directions:** pᵢᵀpⱼ = 0

**A-orthogonal (conjugate) directions:** pᵢᵀApⱼ = 0

**Why this matters:**

When minimizing f(x) = ½xᵀAx - bᵀx, if we minimize along conjugate directions:
```
Each direction is "done" - no need to revisit!
```

**Remarkable theorem:**
For an n-dimensional quadratic, n conjugate directions span the space.
Minimizing along each **once** gives the exact minimum!

### The Algorithm

```
r₀ = Ax₀ - b  (initial residual = gradient)
p₀ = -r₀      (first direction)

for k = 0, 1, 2, ..., n-1:
    αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)  (step size)
    xₖ₊₁ = xₖ + αₖpₖ        (update position)
    rₖ₊₁ = rₖ + αₖApₖ       (update residual)

    if ‖rₖ₊₁‖ < tolerance:
        break

    βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)  (update parameter)
    pₖ₊₁ = -rₖ₊₁ + βₖpₖ        (new direction)
```

**Note:** Only one matrix-vector product per iteration: Apₖ

### Why It Works: Krylov Subspaces

At iteration k, xₖ lies in the **Krylov subspace**:
```
Kₖ = span{r₀, Ar₀, A²r₀, ..., Aᵏ⁻¹r₀}
```

And xₖ is the **optimal** point in Kₖ!

**Proof sketch:**
1. p₀, p₁, ..., pₖ₋₁ span Kₖ
2. They are A-conjugate (algorithm ensures this)
3. Minimizing along conjugate directions in Kₖ gives optimal point in Kₖ

Since dim(Kₙ) = n for n×n matrix:
```
xₙ = argmin_{x ∈ ℝⁿ} f(x) = x*
```

**Converges in n steps (for exact arithmetic)!**

### Numerical Example

**Problem:** Solve Ax = b
```
A = [4  1]    b = [1]
    [1  3]        [2]
```

True solution: x* = A⁻¹b = [1/11, 7/11]

**Starting:** x₀ = [0, 0]

**Iteration 0:**
```
r₀ = Ax₀ - b = [0] - [1] = [-1]
                [0]   [2]   [-2]

p₀ = -r₀ = [1]
           [2]

α₀ = (r₀ᵀr₀)/(p₀ᵀAp₀)
   = (1 + 4)/([1,2]·[4,1; 1,3]·[1,2]ᵀ)
   = 5/(1·6 + 2·7)
   = 5/20 = 0.25

x₁ = x₀ + α₀p₀ = [0] + 0.25[1] = [0.25]
                   [0]       [2]   [0.5 ]

r₁ = r₀ + α₀Ap₀ = [-1] + 0.25[6] = [-1 + 1.5] = [0.5]
                   [-2]       [7]   [-2 + 1.75]  [-0.25]
```

**Iteration 1:**
```
β₀ = (r₁ᵀr₁)/(r₀ᵀr₀) = (0.25 + 0.0625)/5 = 0.0625

p₁ = -r₁ + β₀p₀ = [-0.5  ] + 0.0625[1] = [-0.4375]
                   [0.25]           [2]   [0.375  ]

α₁ = (r₁ᵀr₁)/(p₁ᵀAp₁) = 0.3125/0.3125 = 1

x₂ = x₁ + α₁p₁ = [0.25] + 1[-0.4375] = [1/11]  ≈ x*!
                  [0.5 ]    [0.375  ]   [7/11]
```

**Converged in 2 iterations** (n = 2)!

### For Non-Quadratic Functions

The Fletcher-Reeves formula:
```
βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
```

The Polak-Ribière formula (usually better):
```
βₖ = max(0, (rₖ₊₁ᵀ(rₖ₊₁ - rₖ))/(rₖᵀrₖ))
```

**Periodic restarts:** Every n iterations, restart with steepest descent direction to maintain convergence.

### Convergence Rate

For quadratics with condition number κ:
```
‖xₖ - x*‖_A ≤ 2((√κ - 1)/(√κ + 1))ᵏ‖x₀ - x*‖_A
```

Much better than gradient descent!

**Example:** κ = 100
- Gradient descent: rate = 0.98
- Conjugate gradient: rate = 0.67

CG converges **much faster**!

### Preconditioning

Apply to M⁻¹Ax = M⁻¹b where M ≈ A but easier to invert.

**Effect:** Reduces condition number!

Common preconditioners:
- Diagonal: M = diag(A)
- Incomplete Cholesky
- Multigrid

Can improve convergence dramatically!

---

## Advanced Gradient Methods

### AdaGrad

**Idea:** Accumulate ALL past squared gradients, use for adaptive learning rate.

**Algorithm:**
```
G₀ = 0
for k = 0, 1, 2, ...:
    g = ∇f(xₖ)
    Gₖ₊₁ = Gₖ + g²  (element-wise)
    xₖ₊₁ = xₖ - α·g/√(Gₖ₊₁ + ε)
```

**Effective learning rate:**
```
αᵢ,ₖ = α/√(Σⱼ₌₀ᵏ gᵢ,ⱼ² + ε)
```

**Perfect for sparse data:**
- Frequent features: G large → small learning rate
- Rare features: G small → large learning rate

**Problem:** G only grows → learning rate → 0 eventually!

This motivated RMSprop and Adam.

### NAdam

**Combines:** Nesterov momentum + Adam

**Key difference from Adam:**

Adam momentum:
```
m = β₁·m + (1-β₁)·g
Update with m
```

NAdam (lookahead):
```
m = β₁·m + (1-β₁)·g
m_lookahead = β₁·m + (1-β₁)·g  (one more step of momentum)
Update with m_lookahead
```

**Convergence:** O(1/k²) like Nesterov, faster than Adam's O(1/k)

### AdamW

**Problem with Adam + L2 regularization:**

Objective: f̃(x) = f(x) + (λ/2)‖x‖²
Gradient: ∇f̃(x) = ∇f(x) + λx

Standard Adam:
```
m = β₁m + (1-β₁)(∇f + λx)
v = β₂v + (1-β₂)(∇f + λx)²
x = x - α·m/√v
```

Weight decay gets adaptive learning rate! Couples regularization with optimization.

**AdamW solution (decoupled):**
```
m = β₁m + (1-β₁)∇f
v = β₂v + (1-β₂)∇f²
x = x - α·(m/√v + λx)
```

Weight decay applied uniformly! Better generalization in deep learning.

---

## Particle Swarm Optimization

### The Biological Inspiration

Watch a flock of birds searching for food:
1. Each bird remembers where it found the best food (personal experience)
2. Birds see where others find food (social learning)
3. Birds balance exploring new areas vs exploiting known good spots

This is PSO!

### The Mathematical Model

**Each particle i has:**
- Position: xᵢ,ₖ ∈ ℝⁿ (current location)
- Velocity: vᵢ,ₖ ∈ ℝⁿ (direction and speed)
- Personal best: pᵢ ∈ ℝⁿ (best position it has found)
- Global best: g ∈ ℝⁿ (best position anyone has found)

**Update equations:**
```
vᵢ,ₖ₊₁ = w·vᵢ,ₖ + c₁r₁⊙(pᵢ - xᵢ,ₖ) + c₂r₂⊙(g - xᵢ,ₖ)
xᵢ,ₖ₊₁ = xᵢ,ₖ + vᵢ,ₖ₊₁
```

Where:
- w: inertia weight (0.4 - 0.9)
- c₁, c₂: cognitive and social constants (often 2.0 each)
- r₁, r₂: random vectors ~ U(0,1)ⁿ
- ⊙: element-wise multiplication

### Three Components Explained

**1. Inertia: w·vᵢ,ₖ**

Keep moving in current direction.

w > 1: velocity explodes (bad!)
w = 1: no damping, keeps going
w < 1: gradually slows down

Typical: w = 0.7 (mild damping)

**2. Cognitive: c₁r₁⊙(pᵢ - xᵢ,ₖ)**

Attraction to personal best.

Pulls particle back to where **it** found success.

**3. Social: c₂r₂⊙(g - xᵢ,ₖ)**

Attraction to global best.

Pulls particle toward where **swarm** found success.

### Detailed Example

**Problem:** Minimize f(x, y) = x² + y²

**One particle at position [1, 2]:**
```
Current: x = [1, 2]
Velocity: v = [-0.2, 0.3]
Personal best: p = [0.5, 1]
Global best: g = [0, 0]
```

**Parameters:** w = 0.7, c₁ = c₂ = 2.0

**Random samples:** r₁ = [0.3, 0.8], r₂ = [0.6, 0.4]

**Velocity update:**
```
Inertia = 0.7·[-0.2, 0.3] = [-0.14, 0.21]

Cognitive = 2.0·[0.3, 0.8]⊙([0.5, 1] - [1, 2])
          = 2.0·[0.3, 0.8]⊙[-0.5, -1]
          = 2.0·[-0.15, -0.8]
          = [-0.3, -1.6]

Social = 2.0·[0.6, 0.4]⊙([0, 0] - [1, 2])
       = 2.0·[0.6, 0.4]⊙[-1, -2]
       = 2.0·[-0.6, -0.8]
       = [-1.2, -1.6]

v_new = [-0.14, 0.21] + [-0.3, -1.6] + [-1.2, -1.6]
      = [-1.64, -2.99]
```

**Position update:**
```
x_new = [1, 2] + [-1.64, -2.99]
      = [-0.64, -0.99]
```

Moved toward the optimum!

### Expected Behavior

Taking expectation over random r₁, r₂:
```
E[r₁] = E[r₂] = [0.5, 0.5, ...]

E[v_{k+1}] = w·v_k + (c₁/2)(p - x) + (c₂/2)(g - x)

E[x_{k+1}] = x + E[v_{k+1}]
```

Particle moves toward **weighted average** of p and g!

### Constriction Coefficient

To guarantee convergence, use:
```
φ = c₁ + c₂
χ = 2/|2 - φ - √(φ² - 4φ)|
```

For φ > 4, replace w with χ in velocity update.

Typical: c₁ = c₂ = 2.05 → φ = 4.1 → χ ≈ 0.73

### Variants

**Inertia weight strategies:**

Linear decrease:
```
w(k) = w_max - k·(w_max - w_min)/max_iterations
```
Start exploring (w = 0.9), end exploiting (w = 0.4)

**Neighborhood topologies:**

Global: all particles share same g
Ring: particles only see neighbors
Star: some particles are "leaders"

**Bounds handling:**

Reflect: if x > max, set x = max, v = -v
Clamp: if x > max, set x = max, v = 0
Periodic: wrap around (x mod (max - min))

---

## Genetic Algorithms

### The Biology

**Natural selection:**
1. Population with variation
2. Competition for resources
3. Survival of the fittest
4. Reproduction passes traits to offspring
5. Mutation introduces new variation

Over generations: population adapts!

### The Algorithm

```
Initialize population P of size N
Evaluate fitness f(x) for each x in P

for generation = 1 to max_gen:
    # Selection
    Parents = select_best(P, fitness)

    # Crossover
    Offspring = []
    for i = 1 to N/2:
        parent1, parent2 = random_pair(Parents)
        if rand() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        Offspring.append(child1, child2)

    # Mutation
    for child in Offspring:
        if rand() < mutation_rate:
            mutate(child)

    # Replacement
    P = best_of(P, Offspring)  # or just Offspring
```

### Selection Mechanisms

**Fitness-proportional:**
```
P(select individual i) = f_i/Σⱼf_j
```

Problem: if one individual much better, it dominates → premature convergence

**Rank-based:**
```
Sort by fitness: x₁, x₂, ..., xₙ
P(select xᵢ) = (2-s)/N + 2i(s-1)/(N(N-1))
```
where s ∈ [1, 2] controls selection pressure

**Tournament:**
```
Randomly pick k individuals
Select the best among them
```

k = 2: mild selection pressure
k = 10: strong selection pressure

### Crossover Operations

**Single-point (for binary strings):**
```
Parent1: [1, 1, 0, 1 | 0, 1, 1]
Parent2: [0, 1, 1, 0 | 1, 0, 1]

Child1:  [1, 1, 0, 1 | 1, 0, 1]
Child2:  [0, 1, 1, 0 | 0, 1, 1]
```

**Uniform:**
```
For each position i:
    if rand() < 0.5:
        child1[i] = parent1[i], child2[i] = parent2[i]
    else:
        child1[i] = parent2[i], child2[i] = parent1[i]
```

**Arithmetic (for continuous):**
```
child1 = λ·parent1 + (1-λ)·parent2
child2 = (1-λ)·parent1 + λ·parent2
```
where λ ~ U(0, 1)

### Mutation

**Binary flip:**
```
For each bit i:
    if rand() < mutation_rate:
        flip bit i
```

**Gaussian (continuous):**
```
x' = x + N(0, σ²)
```

**Adaptive mutation:**
```
σ(gen) = σ₀·exp(-gen/τ)
```
Large mutations early (exploration), small later (refinement)

### Schema Theorem (Holland 1975)

A **schema** is a template, e.g., [*, 1, *, 0, *] where * is wildcard.

**Theorem:** Above-average schemas grow exponentially:
```
E[m(H, t+1)] ≥ m(H, t)·(f(H)/f̄)·S(H)
```

Where:
- m(H, t): count of individuals matching H at generation t
- f(H): average fitness of schema H
- f̄: population average fitness
- S(H): survival probability (depending on schema length)

**Building Block Hypothesis:** GA discovers good "building blocks" and combines them.

### No Free Lunch Theorem

**Theorem (Wolpert & Macready, 1997):**

Averaged over all possible objective functions, all algorithms perform equally!

**Implication:** No algorithm is universally best.

GA works well on:
- Multimodal landscapes
- Discrete search spaces
- Problems with building-block structure

GA works poorly on:
- Smooth, convex functions (use gradient methods!)
- Problems where recombination breaks good solutions

---

## CMA-ES

### The Fundamental Idea

Most metaheuristics use fixed search distribution (e.g., uniform sampling).

CMA-ES uses **adaptive** multivariate Gaussian:
```
xᵢ ~ N(m, σ²C)
```

And learns:
1. m: where to search (mean)
2. σ: how far to search (step size)
3. C: what shape to search (covariance)

### Why Covariance Matters

**Example:** f(x, y) = x² + 100y²

This function is **ill-conditioned**:
- Steep in y direction (eigenvalue = 200)
- Gentle in x direction (eigenvalue = 2)
- Ratio κ = 100

**Naive sampling:** N([0,0], I)
```
Most samples far from optimum in y direction
Wastes samples!
```

**Smart sampling:** N([0,0], C) where
```
C = [1    0  ]
    [0  0.01]
```

Samples tightly in y, broadly in x. Much more efficient!

**CMA-ES learns this C automatically!**

### The Algorithm

```
Initialize:
  m = random  (mean)
  σ = 1  (step size)
  C = I  (covariance)
  pc = 0, pσ = 0  (evolution paths)

for gen = 1, 2, 3, ...:
    # Sample population
    for i = 1 to λ:
        xᵢ = m + σ·N(0, C)

    # Evaluate and sort
    f₁, ..., f_λ = [f(x₁), ..., f(x_λ)]
    Sort by fitness: x₁:λ ≤ ... ≤ x_λ:λ

    # Update mean (weighted average of best)
    m_old = m
    m = Σᵢ₌₁^μ wᵢ·xᵢ:λ

    # Update evolution paths
    pσ = (1-cσ)·pσ + √(cσ(2-cσ)μₑff)·C^(-1/2)·(m - m_old)/σ
    pc = (1-cc)·pc + hsig·√(cc(2-cc)μₑff)·(m - m_old)/σ

    # Update covariance
    C = (1-c₁-cμ)·C + c₁·pc·pcᵀ +
        cμ·Σᵢ₌₁^μ wᵢ·(xᵢ:λ - m_old)·(xᵢ:λ - m_old)ᵀ/σ²

    # Update step size
    σ = σ·exp((cσ/dσ)·(‖pσ‖/E[‖N(0,I)‖] - 1))
```

### Evolution Paths

**Idea:** Track cumulative step direction over several generations.

**pσ (for step size control):**
```
pσ = (1-cσ)·pσ + √(cσ(2-cσ))·C^(-1/2)·δm
```

where δm = (m_new - m_old)/σ

**Interpretation:** If consistently moving in same direction, ‖pσ‖ is large → increase σ

**pc (for covariance update):**
```
pc = (1-cc)·pc + hsig·√(cc(2-cc))·δm
```

**Interpretation:** Tracks successful search directions → emphasize in C

### Step Size Adaptation (CSA)

```
σ_new = σ·exp((cσ/dσ)·(‖pσ‖/E[‖N(0,I)‖] - 1))
```

**Logic:**

If ‖pσ‖ > E[‖N(0,I)‖]:
- Evolution path is longer than expected
- We're making progress in consistent direction
- → Increase σ (take bigger steps)

If ‖pσ‖ < E[‖N(0,I)‖]:
- Evolution path is shorter than expected
- We're zigzagging or stuck
- → Decrease σ (take smaller steps)

**For n dimensions:** E[‖N(0,I)‖] = √n·(1 - 1/(4n) + 1/(21n²))

### Covariance Matrix Update

**Three components:**

1. **Old covariance:** (1-c₁-cμ)·C
   - Don't forget everything

2. **Rank-one update:** c₁·pc·pcᵀ
   - Evolution path information
   - Emphasizes direction of cumulative progress

3. **Rank-μ update:** cμ·Σwᵢ·yᵢ·yᵢᵀ
   - Current generation information
   - yᵢ = (xᵢ:λ - m_old)/σ (normalized steps)

### Why CMA-ES is State-of-the-Art

**Invariance properties:**

1. **Translation:** Shifting all points by constant doesn't change behavior
2. **Rotation:** Rotating coordinate system doesn't change behavior
3. **Scaling:** Multiplying objective by constant doesn't change behavior

**Practical implication:** No need to normalize or scale variables!

**Learning:** Captures problem structure that gradient methods miss
- Correlations between variables
- Different scaling in different directions
- Direction of progress

### Parameter Values

**Default population size:**
```
λ = 4 + ⌊3·ln(n)⌋
```

n = 10 → λ = 11
n = 100 → λ = 18
n = 1000 → λ = 25

**Learning rates:**
```
cc ≈ 4/n
cσ ≈ 4/n
c₁ ≈ 2/n²
cμ ≈ μₑff/n²
```

These are carefully tuned for optimal performance!

### Computational Cost

**Per generation:**
- Sample λ points: λ function evaluations
- Eigendecomposition of C: O(n³) every ~1/(c₁+cμ) generations
- Other updates: O(λn²)

**Memory:** O(n²) for covariance matrix

**Practical limit:** n ≈ 1000-10000

---

## Bayesian Optimization

### The Core Problem

We want to minimize f(x), but:
- f is **expensive** (hours per evaluation)
- f is **black-box** (no gradients)
- We have a **limited budget** (say, 100 evaluations)

**Examples:**
- Hyperparameter tuning: train a neural network for each evaluation
- Engineering design: run a finite element simulation
- Drug discovery: synthesize and test a compound

**Question:** How to choose the next x to evaluate?

### The Bayesian Approach

**Idea:** Build a probabilistic model of f(x), use it to decide where to sample next.

**Model:** Gaussian Process (GP)

**Strategy:**
1. Start: Sample a few random points
2. Loop:
   a. Fit GP to observed data
   b. Use GP to find promising next point
   c. Evaluate f at that point
   d. Add to data, repeat

### Gaussian Processes

**Definition:** A GP is a distribution over functions.

```
f ~ GP(m(x), k(x, x'))
```

Where:
- m(x): mean function (often 0)
- k(x, x'): covariance kernel

**Kernel examples:**

**RBF (Radial Basis Function):**
```
k(x, x') = σ²·exp(-‖x - x'‖²/(2ℓ²))
```

Parameters:
- σ²: signal variance (vertical scale)
- ℓ: length scale (horizontal scale)

Large ℓ: smooth functions
Small ℓ: wiggly functions

**Matérn:**
```
k(x, x') = σ²·(1 + √5·r/ℓ + 5r²/(3ℓ²))·exp(-√5·r/ℓ)
```
where r = ‖x - x'‖

**What this means:**

```
For any finite set of points x₁, ..., xₙ:
  f(x₁)              [m(x₁)]   [k(x₁,x₁) ... k(x₁,xₙ)]
  f(x₂)    ~ N (     [m(x₂)] , [   ...   ...    ...  ])
  ...                [  ...  ]   [k(xₙ,x₁) ... k(xₙ,xₙ)]
  f(xₙ)              [m(xₙ)]
```

Function values are jointly Gaussian!

### GP Posterior

**Given observations:** D = {(x₁, y₁), ..., (xₙ, yₙ)}

**Posterior at new point x:**
```
f(x)|D ~ N(μ(x), σ²(x))
```

Where:
```
μ(x) = k(x)ᵀ·(K + σₙ²I)⁻¹·y

σ²(x) = k(x, x) - k(x)ᵀ·(K + σₙ²I)⁻¹·k(x)
```

Notation:
- K: n×n matrix [k(xᵢ, xⱼ)]
- k(x): n×1 vector [k(x, x₁), ..., k(x, xₙ)]ᵀ
- y: n×1 vector [y₁, ..., yₙ]ᵀ
- σₙ²: noise variance

**Interpretation:**

**μ(x):** Our best guess for f(x)
```
Weighted average of observed y values
Weights based on similarity (via kernel)
```

**σ²(x):** Our uncertainty about f(x)
```
Large σ²: far from observations, uncertain
Small σ²: near observations, confident
```

### Acquisition Functions

**Question:** Which x should we evaluate next?

**Trade-off:**
- **Exploitation:** Sample where μ(x) is small (looks good)
- **Exploration:** Sample where σ(x) is large (uncertain, might be great)

**Acquisition function** α(x) balances these.

**1. Expected Improvement (EI):**

```
EI(x) = E[max(f(x⁺) - f(x), 0)]
```

where x⁺ = current best.

**Closed form:**
```
EI(x) = (f(x⁺) - μ(x))·Φ(Z) + σ(x)·φ(Z)
```

where:
```
Z = (f(x⁺) - μ(x))/σ(x)
Φ(Z) = CDF of standard normal
φ(Z) = PDF of standard normal
```

**Interpretation:**
- First term: how much better than current best (exploitation)
- Second term: uncertainty (exploration)

**2. Upper Confidence Bound (UCB):**

```
UCB(x) = μ(x) + κ·σ(x)
```

For **minimization**, use:
```
LCB(x) = μ(x) - κ·σ(x)
```

κ controls exploration (typically 2-3):
- Large κ: more exploration
- Small κ: more exploitation

**3. Probability of Improvement (PI):**

```
PI(x) = P(f(x) < f(x⁺))
      = Φ((f(x⁺) - μ(x))/σ(x))
```

Simpler than EI, but often less effective.

### The Algorithm

```
# Initialize
X_obs = [x₁, ..., xₙ]  (random points)
Y_obs = [f(x₁), ..., f(xₙ)]

for t = 1 to budget:
    # Fit GP
    θ = optimize_hyperparameters(X_obs, Y_obs)
    GP = fit(X_obs, Y_obs, θ)

    # Find next point
    x_next = argmax_x α(x|GP)

    # Evaluate
    y_next = f(x_next)

    # Update data
    X_obs.append(x_next)
    Y_obs.append(y_next)

# Return best observed
return x_obs[argmin(Y_obs)]
```

### Numerical Example

**Problem:** Minimize f(x) = x·sin(x) on [0, 10]

**Initial samples:**
```
x₁ = 2, y₁ = f(2) = 1.82
x₂ = 7, y₂ = f(7) = 4.69
```

**Fit GP with RBF kernel:**
```
k(x, x') = exp(-(x - x')²/2)
```

**Posterior:**
At x = 5:
```
k(5) = [exp(-9/2), exp(-4/2)] = [0.011, 0.135]
K = [1.0   0.006]
    [0.006  1.0 ]

μ(5) ≈ 0.011·1.82 + 0.135·4.69 ≈ 0.65

σ²(5) ≈ 1 - (small term) ≈ 0.98
σ(5) ≈ 0.99
```

High uncertainty! Good exploration target.

**Acquisition (EI):**

Current best: f(x⁺) = 1.82

```
Z = (1.82 - 0.65)/0.99 = 1.18
Φ(1.18) ≈ 0.88
φ(1.18) ≈ 0.20

EI(5) = (1.82 - 0.65)·0.88 + 0.99·0.20
      = 1.03 + 0.20
      = 1.23
```

**Next sample:** xₙₑₓₜ = 5 (high EI)

### Regret Bounds

**Cumulative regret:**
```
R_T = Σₜ₌₁ᵀ (f(xₜ) - f(x*))
```

**Theorem (Srinivas et al., 2010):**

For UCB with appropriate κ(t):
```
R_T ≤ O(√(T·γ_T·log T))
```

where γ_T is the **maximum information gain** (depends on kernel).

**Implication:** Sub-linear regret!
```
Average regret = R_T/T → 0
```

Much better than random search (linear regret).

### Practical Considerations

**Hyperparameter optimization:**

Kernel parameters θ = {σ², ℓ, σₙ²} optimized by maximizing **marginal likelihood**:
```
log p(y|X, θ) = -½yᵀ(K + σₙ²I)⁻¹y - ½log|K + σₙ²I| - (n/2)log(2π)
```

**Computational cost:**
- GP fitting: O(n³) for n observations
- Acquisition optimization: depends on method

**Practical limit:** n ≈ 100-1000 observations

**Scalability:**
- Sparse GPs: approximate with subset of data
- Local models: separate GPs for different regions
- Neural networks: replace GP (Bayesian NNs)

---

## Other Metaheuristics

### Simulated Annealing

**Inspiration:** Annealing in metallurgy - slowly cool metal to minimize energy.

**Algorithm:**
```
x = x₀
T = T₀ (temperature)

while T > T_min:
    x' = neighbor(x)  (random perturbation)
    ΔE = f(x') - f(x)

    if ΔE < 0:
        x = x'  (accept improvement)
    else:
        p = exp(-ΔE/T)  (Boltzmann probability)
        if rand() < p:
            x = x'  (accept with probability)

    T = cool(T)  (decrease temperature)
```

**Metropolis Criterion:**
```
P(accept) = {
    1           if ΔE ≤ 0
    exp(-ΔE/T)  if ΔE > 0
}
```

**Statistical Mechanics:**

Boltzmann distribution at temperature T:
```
P(state x) ∝ exp(-f(x)/T)
```

As T → 0: probability concentrates on minimum energy states.

**Cooling Schedules:**

Logarithmic (theory):
```
T(k) = T₀/log(k + 1)
```
Guarantees global optimum but **very slow**!

Geometric (practice):
```
T(k) = α^k·T₀
```
α ∈ (0.8, 0.99), much faster.

### Differential Evolution

**Key innovation:** Use population differences for mutation.

**Mutation:**
```
v = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
```

where r₁, r₂, r₃ are random distinct indices, F ∈ [0, 2].

**Binomial Crossover:**
```
uⱼ = {
    vⱼ    if rand() < CR or j = j_rand
    xᵢ,ⱼ  otherwise
}
```

**Selection:**
```
xᵢ,ₖ₊₁ = {
    u     if f(u) < f(xᵢ,ₖ)
    xᵢ,ₖ  otherwise
}
```

**Why it works:**

The difference (xᵣ₂ - xᵣ₃) provides:
1. Random direction
2. Scale adapted to population spread
3. Self-organizing search

### Nelder-Mead Simplex

**Simplex:** n+1 points in n dimensions
- 2D: triangle
- 3D: tetrahedron

**Operations:**

Reflection:
```
x_r = x̄ + α(x̄ - x_{n+1})
```

Expansion:
```
x_e = x̄ + γ(x_r - x̄)
```

Contraction:
```
x_c = x̄ + ρ(x_{n+1} - x̄)
```

Shrink:
```
xᵢ → x₁ + σ(xᵢ - x₁)
```

**No guarantees** but works surprisingly well!

### Harmony Search

**Inspiration:** Musical improvisation.

**Algorithm:**
```
Initialize Harmony Memory HM

for iter = 1 to max_iter:
    # Improvise new harmony
    for j = 1 to n:
        if rand() < HMCR:
            new[j] = HM[rand_idx][j]
            if rand() < PAR:
                new[j] += BW·randn()
        else:
            new[j] = random_in_bounds()

    # Update HM if better
    if f(new) < f(worst_in_HM):
        replace worst_in_HM with new
```

Parameters:
- HMCR: 0.7-0.95 (memory consideration)
- PAR: 0.2-0.5 (pitch adjustment)
- BW: bandwidth for adjustment

---

## Convergence Analysis

### Types of Convergence

**1. Linear:**
```
‖xₖ - x*‖ ≤ c^k·‖x₀ - x*‖
```

c ∈ (0, 1) is the convergence rate.

**Example:** c = 0.9
```
k=0: error = 1.0
k=10: error = 0.35
k=20: error = 0.12
k=50: error = 0.005
```

**2. Superlinear:**
```
lim_{k→∞} ‖xₖ₊₁ - x*‖/‖xₖ - x*‖ = 0
```

Faster than any linear rate!

**3. Quadratic:**
```
‖xₖ₊₁ - x*‖ ≤ C·‖xₖ - x*‖²
```

**Example:**
```
k=0: error = 10⁻²
k=1: error = C·10⁻⁴
k=2: error = C·10⁻⁸
k=3: error = C·10⁻¹⁶
```

Error squares each step!

### Algorithm Classification

| Algorithm | Rate | Conditions |
|-----------|------|------------|
| Gradient Descent | Linear | Strongly convex |
| Momentum | Linear (faster) | Strongly convex |
| Adam | Linear | Empirical |
| AdaGrad | O(1/√T) regret | Online convex |
| Conjugate Gradient | Superlinear | Strongly convex |
| L-BFGS | Superlinear | Smooth |
| Newton | Quadratic | Smooth, near optimum |

### Complexity Lower Bounds

**Theorem (Nesterov):**

For L-smooth, μ-strongly convex functions, any first-order method needs:
```
Ω(√κ·log(1/ε)) iterations
```

where κ = L/μ.

**Achieved by:** Accelerated gradient descent (Nesterov 1983)

**For zero-order methods:**
```
Ω(n·ε^(-2)) evaluations
```

Much worse! Need gradients when possible.

### Practical Convergence

**Stopping criteria:**

1. Gradient norm:
```
‖∇f(x)‖ < ε
```

2. Function change:
```
|f(xₖ₊₁) - f(xₖ)| < ε
```

3. Parameter change:
```
‖xₖ₊₁ - xₖ‖ < ε
```

Use multiple criteria!

---

## Practical Examples

### Example 1: Simple Quadratic

**Problem:** f(x, y) = x² + 4y²

**Gradient Descent:**
```python
x = np.array([4.0, 2.0])
alpha = 0.2

for k in range(10):
    grad = np.array([2*x[0], 8*x[1]])
    x = x - alpha * grad
    print(f"k={k}: x={x}, f={x[0]**2 + 4*x[1]**2}")
```

Output:
```
k=0: x=[3.2  0.8], f=12.8
k=1: x=[2.56 0.32], f=7.31
k=2: x=[2.05 0.13], f=4.20
...
```

**Adam:**
```python
m = np.zeros(2)
v = np.zeros(2)
beta1, beta2 = 0.9, 0.999

for k in range(10):
    grad = np.array([2*x[0], 8*x[1]])
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2
    m_hat = m/(1-beta1**(k+1))
    v_hat = v/(1-beta2**(k+1))
    x = x - 0.1*m_hat/(np.sqrt(v_hat) + 1e-8)
    print(f"k={k}: x={x}")
```

### Example 2: Rosenbrock

**Function:** f(x, y) = (1-x)² + 100(y-x²)²

**Minimum:** (1, 1), f* = 0

**Gradient:**
```
∇f = [2(x-1) - 400x(y-x²), 200(y-x²)]
```

**Challenge:** Narrow curved valley

**Code:**
```python
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def grad_rosenbrock(x):
    dx = 2*(x[0]-1) - 400*x[0]*(x[1]-x[0]**2)
    dy = 200*(x[1]-x[0]**2)
    return np.array([dx, dy])

# L-BFGS
from scipy.optimize import minimize
result = minimize(rosenbrock, x0=[0, 0], method='L-BFGS-B',
                  jac=grad_rosenbrock)
print(result.x, result.fun)
```

### Example 3: Neural Network (Conceptual)

**Problem:** Minimize loss L(θ) where θ = network weights

**Data:** {(x₁, y₁), ..., (xₙ, yₙ)}

**Loss:** L(θ) = (1/n)Σᵢ loss(fθ(xᵢ), yᵢ)

**Mini-batch Adam:**
```python
# Hyperparameters
alpha = 0.001
beta1, beta2 = 0.9, 0.999
batch_size = 32

# Initialize
m = zeros_like(theta)
v = zeros_like(theta)

for epoch in range(num_epochs):
    for batch in get_batches(data, batch_size):
        # Compute gradient on batch
        grad = compute_gradient(theta, batch)

        # Adam update
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m/(1-beta1**epoch)
        v_hat = v/(1-beta2**epoch)
        theta = theta - alpha*m_hat/(sqrt(v_hat) + 1e-8)
```

---

## Summary

All 18 algorithms explained mathematically!

**Key takeaways:**

1. **Gradient descent** uses first-order Taylor approximation
2. **Momentum** accumulates gradients exponentially
3. **Adam** adapts learning rates using gradient statistics
4. **Newton** uses second-order approximation (Hessian)
5. **L-BFGS** approximates Hessian from gradient history
6. **Conjugate gradient** uses Krylov subspaces
7. **PSO** balances personal and social knowledge
8. **GA** evolves solutions through selection and recombination
9. **CMA-ES** learns problem structure via covariance
10. **Bayesian optimization** models f(x) for sample efficiency

Each has its place - choose based on:
- Gradient availability
- Problem size
- Smoothness
- Budget
- Desired accuracy

The mathematics shows us **why** they work and **when** to use each!
