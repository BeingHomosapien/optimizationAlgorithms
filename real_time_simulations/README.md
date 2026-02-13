# Real-Time Optimization Simulations

Watch optimization algorithms work in real-time with interactive animations!

## Overview

These simulations provide live visualizations of optimization algorithms as they search for minima. You can start, pause, reset, and control the speed of the animations.

## Simulations

### 1. Gradient Descent Live (`gradient_descent_live.py`)

Watch gradient descent navigate the optimization landscape step-by-step.

**Features:**
- Real-time path visualization on contour plot
- Gradient arrow showing descent direction
- Live convergence curve
- Iteration statistics
- Start/Pause/Reset controls
- Speed control slider

**Run:**
```bash
python real_time_simulations/gradient_descent_live.py
```

**What to observe:**
- How gradient descent follows the steepest descent direction
- Oscillations on ill-conditioned problems (Rosenbrock)
- Fast convergence on convex functions (Sphere)
- Getting stuck in local minima (Rastrigin)

---

### 2. Particle Swarm Live (`pso_live.py`)

Watch a swarm of particles explore and converge together!

**Features:**
- Real-time particle positions (blue circles)
- Personal best positions (green triangles)
- Global best position (red star)
- Velocity vectors (red arrows)
- Swarm statistics (diversity, average velocity)
- Interactive controls

**Run:**
```bash
python real_time_simulations/pso_live.py
```

**What to observe:**
- Particles exploring different regions initially
- Gradual convergence toward global best
- Balance between exploration and exploitation
- How particles influence each other
- Swarm diversity decreasing over time

---

### 3. Algorithm Comparison Live (`compare_live.py`)

Watch multiple algorithms compete side-by-side in real-time!

**Features:**
- Multiple algorithms on same landscape
- Color-coded paths for each algorithm
- Comparative convergence curves
- Real-time performance statistics
- See which algorithm is fastest

**Run:**
```bash
python real_time_simulations/compare_live.py
```

**Algorithms compared:**
- Gradient Descent (two different learning rates)
- Momentum
- Adam

**What to observe:**
- Which algorithm converges fastest
- Different paths taken by different methods
- Effect of learning rate on convergence
- Momentum's acceleration vs standard GD
- Adam's adaptive behavior

---

## Controls

All simulations have these controls:

| Control | Function |
|---------|----------|
| **Start/Pause Button** | Start or pause the optimization |
| **Reset Button** | Reset to initial conditions |
| **Speed Slider** | Control animation speed (1-200 ms/iteration) |

## Understanding the Visualizations

### Contour Plot
- **Background colors/lines**: Function landscape (darker = lower values)
- **Colored paths**: Algorithm trajectory
- **Markers**: Current position(s)
- **Star markers**: Global minimum (if known)

### Convergence Plot
- **Y-axis (log scale)**: Function value f(x)
- **X-axis**: Iteration number
- **Descending line**: Algorithm improving over time
- **Plateau**: Algorithm converged or stuck

### Info Panel
Shows real-time statistics:
- Current iteration
- Current position and function value
- Gradient norm (for gradient methods)
- Distance to known optimum
- Algorithm-specific metrics

## Tips for Learning

### Experiment with:

1. **Different Functions**
   - Easy (Sphere): See clean convergence
   - Hard (Rosenbrock): See struggle with narrow valleys
   - Multimodal (Rastrigin): See local minima trapping

2. **Speed Settings**
   - Slow (100-200 ms): Watch details of each step
   - Fast (1-20 ms): See overall behavior quickly

3. **Multiple Runs**
   - Hit Reset to try different random initializations
   - Compare behavior from different starting points

### What to Look For

**Gradient Descent:**
- Does it zig-zag? (sign of ill-conditioning)
- Does it slow down near the minimum? (gradient → 0)
- Does it get stuck? (local minimum)

**PSO:**
- How quickly does swarm diversity decrease?
- Do particles cluster too early? (premature convergence)
- Are particles still exploring? (high diversity)

**Comparison:**
- Which algorithm takes fewer iterations?
- Which is more stable (less oscillation)?
- Which finds better solution?

## Performance Notes

- Animations run at the specified speed (default: 50ms per iteration)
- Higher speeds may appear choppy on slower computers
- Particle Swarm with many particles may be slower to render
- Comparison mode renders multiple paths simultaneously

## Extending

You can easily add more algorithms or functions by:

1. **Adding algorithms:**
   - Import from `algorithms/` package
   - Add to configuration dictionary
   - Implement step logic

2. **Adding functions:**
   - Import from `test_functions/`
   - Add to functions dictionary
   - Set appropriate learning rates

3. **Customizing visualization:**
   - Modify plot styles in `setup_plot()`
   - Add more statistics to info panel
   - Change colors, markers, line styles

## Examples

**Quick start - Gradient Descent on Rosenbrock:**
```bash
python real_time_simulations/gradient_descent_live.py
# Choose option 2 (Rosenbrock)
# Click "Start"
# Watch it struggle with the narrow valley!
```

**Quick start - PSO on Rastrigin:**
```bash
python real_time_simulations/pso_live.py
# Choose option 3 (Rastrigin)
# Click "Start"
# Watch the swarm avoid local minima!
```

**Quick start - Compare algorithms:**
```bash
python real_time_simulations/compare_live.py
# Choose option 2 (Rosenbrock)
# Click "Start"
# See which algorithm wins!
```

## Educational Value

These real-time simulations help you understand:

✅ **How** algorithms search the space
✅ **Why** some algorithms are faster than others
✅ **When** algorithms struggle (ill-conditioning, local minima)
✅ **What** happens at each iteration
✅ **Where** algorithms spend most of their time

Much more intuitive than just looking at final results!

---

Enjoy watching optimization in action! 🚀
