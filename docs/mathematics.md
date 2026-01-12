# Mathematical Foundations of Aegis

This document explains the mathematical concepts underlying Aegis, written for readers without a strong math background.

---

## Table of Contents

1. [Correlation and Distance](#1-correlation-and-distance)
2. [Graph Theory Basics](#2-graph-theory-basics)
3. [Differential Geometry Concepts](#3-differential-geometry-concepts)
4. [Portfolio Optimization](#4-portfolio-optimization)
5. [Risk Metrics](#5-risk-metrics)
6. [Stochastic Processes](#6-stochastic-processes)
7. [Partial Differential Equations](#7-partial-differential-equations)

---

## 1. Correlation and Distance

### What is Correlation?

**Intuition:** Correlation measures how two things move together.

- **ρ = +1.0:** Perfect positive correlation (when A goes up, B always goes up)
- **ρ = 0.0:** No correlation (A and B move independently)
- **ρ = -1.0:** Perfect negative correlation (when A goes up, B always goes down)

**Example:** 
- Oil stocks and airline stocks often have negative correlation (high oil prices hurt airlines)
- Tech stocks often correlate positively with each other

### Converting Correlation to Distance

**Problem:** Correlation isn't a proper "distance" — we need a true metric for geometry.

**Solution:** The Mantegna distance formula (1999):

```
d(i,j) = √(2 × (1 - ρ(i,j)))
```

**What this means:**

| Correlation (ρ) | Distance (d) | Interpretation |
|-----------------|--------------|----------------|
| +1.0 | 0.0 | Identical assets (no distance) |
| +0.5 | 1.0 | Somewhat similar |
| 0.0 | 1.41 | Uncorrelated |
| -0.5 | 1.73 | Negatively correlated |
| -1.0 | 2.0 | Perfect hedge (maximum distance) |

**Why this works:** This formula satisfies the "triangle inequality" — a requirement for proper geometry. The triangle inequality says: the direct path between two points is never longer than going via a third point.

---

## 2. Graph Theory Basics

### Networks and Graphs

**Intuition:** A graph is just dots connected by lines.

- **Node (vertex):** A dot — in our case, an asset
- **Edge:** A line connecting two nodes — represents a relationship
- **Weight:** A number on an edge — represents the strength/distance

**Example:** If we have 5 stocks, we create a graph where:
- Each stock is a node
- We draw edges between similar stocks (low distance)
- The edge weight is the distance

### Adjacency Matrix

A table showing which nodes are connected:

```
        AAPL  GOOGL  MSFT  JPM  XOM
AAPL    0     0.5    0.4   0    0
GOOGL   0.5   0      0.3   0    0
MSFT    0.4   0.3    0     0    0
JPM     0     0      0     0    0.6
XOM     0     0      0     0.6  0
```

This shows: Tech stocks (AAPL, GOOGL, MSFT) are connected to each other, while JPM (bank) and XOM (oil) are only connected to each other.

### Geodesic Distance

**Intuition:** The shortest path between two nodes.

If AAPL isn't directly connected to JPM, but both connect to some intermediate stock, the geodesic distance is the shortest route through the network.

**Algorithm:** We use Dijkstra's algorithm to find all shortest paths.

---

## 3. Differential Geometry Concepts

### What is Differential Geometry?

**Intuition:** Differential geometry studies curved surfaces and spaces.

- **Flat space:** Like a table — parallel lines stay parallel forever
- **Curved space:** Like Earth's surface — parallel lines eventually meet (at the poles)

**In finance:** We treat the market as a "curved space" where assets live. The curvature tells us about market structure.

### Curvature: The Key Concept

**Intuition:** Curvature measures how space bends.

- **Positive curvature:** Like a sphere — things cluster together
- **Zero curvature:** Flat — like a table
- **Negative curvature:** Like a saddle — things spread apart

**In finance:**
- **Positive curvature asset:** Sits in a cluster (redundant, similar to neighbors)
- **Negative curvature asset:** Acts as a bridge between clusters (diversifier)

### Ollivier-Ricci Curvature

**The Formula:**
```
κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
```

**Breaking it down:**

- **κ(x,y):** Curvature of the edge between assets x and y
- **d(x,y):** Distance between x and y
- **μₓ:** A "probability cloud" around asset x
- **W₁:** Wasserstein distance (explained below)

**Interpretation:**
- If neighbors of x are close to neighbors of y → W₁ is small → κ is positive (clustered)
- If neighbors of x are far from neighbors of y → W₁ is large → κ is negative (bridge)

### Wasserstein Distance (Earth Mover's Distance)

**Intuition:** Imagine you have two piles of sand. How much work does it take to reshape pile A into pile B?

**Example:**
- Pile A: Sand at positions [1, 2, 3]
- Pile B: Sand at positions [2, 3, 4]
- Work needed: Move each grain 1 unit → Wasserstein distance = 1

In our context, we're comparing the "neighborhood clouds" of two assets.

### Sectional Curvature

**Intuition:** How much do paths bend when traveling through a region?

We measure this using **triangle defects**:

1. Pick three connected assets (a triangle)
2. Measure: going directly A→C vs going A→B→C
3. If A→B→C is much longer, the space is "curved"

**Formula:**
```
defect = (d(A,B) + d(B,C) - d(A,C)) / (d(A,B) + d(B,C))
```

---

## 4. Portfolio Optimization

### The Goal

Find the best weights (w₁, w₂, ..., wₙ) for each asset such that:
- Weights sum to 1 (100% invested)
- Risk is minimized for a given return, OR
- Return is maximized for a given risk

### Mean-Variance Optimization (Markowitz)

**The classic approach:**

```
Minimize: w'Σw  (portfolio variance)
Subject to: w'μ ≥ target_return
            Σwᵢ = 1
            wᵢ ≥ 0
```

Where:
- **w:** Vector of weights
- **Σ:** Covariance matrix
- **μ:** Expected returns vector

### Geometric Portfolio Weights (Our Approach)

We use curvature and impact scores to build weights:

1. **Impact Score:** How much does removing this asset disrupt the network?
   - High impact → systemically important → be careful (lower weight)
   
2. **Curvature Score:** Is this asset a cluster member or a bridge?
   - Negative curvature → bridge/diversifier → good (higher weight)

3. **Volatility Score:** How much does this asset fluctuate?
   - Lower volatility → more stable → good (higher weight)

**Combined Score:**
```
score = 0.4 × (1 - impact_normalized) 
      + 0.4 × (1 - curvature_normalized)
      + 0.2 × (1 - volatility_normalized)
```

Then we convert scores to weights using softmax:
```
weight_i = exp(score_i) / Σexp(score_j)
```

---

## 5. Risk Metrics

### Value at Risk (VaR)

**Intuition:** "The worst loss I expect on 95% (or 99%) of days"

**Example:** "95% VaR of $1M" means:
- On 95% of days, I won't lose more than $1M
- On 5% of days, I might lose more

**Calculation (Historical):**
1. Simulate 10,000 possible portfolio returns
2. Sort them from worst to best
3. The 500th worst (5th percentile) is the 95% VaR

### Conditional VaR (Expected Shortfall)

**Intuition:** "When things go bad, HOW bad?"

VaR tells you the threshold; CVaR tells you the average loss BEYOND that threshold.

**Example:**
- 95% VaR = $1M
- 95% CVaR = $1.5M means: on the worst 5% of days, the AVERAGE loss is $1.5M

### Sharpe Ratio

**Intuition:** Return per unit of risk

```
Sharpe = (Portfolio Return - Risk-free Rate) / Portfolio Volatility
```

**Interpretation:**
- Sharpe > 1: Good
- Sharpe > 2: Very good
- Sharpe > 3: Exceptional (or suspicious)

### Maximum Drawdown

**Intuition:** The worst peak-to-trough decline

**Example:** If your portfolio went from $100 → $120 → $80 → $90
- Peak was $120
- Trough was $80
- Drawdown = ($120 - $80) / $120 = 33%

---

## 6. Stochastic Processes

### Random Walks

**Intuition:** A particle that takes random steps

Stock prices are modeled as random walks:
```
S(t+1) = S(t) × (1 + random_return)
```

### Brownian Motion

**Intuition:** Continuous random movement (like pollen in water)

**Formula:**
```
dS = μ·S·dt + σ·S·dW
```

Where:
- **S:** Stock price
- **μ:** Drift (average direction)
- **σ:** Volatility (randomness)
- **dW:** Random "shock" (Wiener process)

This is the foundation of options pricing.

### Monte Carlo Simulation

**Intuition:** Run thousands of random experiments to estimate probabilities

**Process:**
1. Generate random future scenarios (e.g., 100,000 paths)
2. Calculate portfolio value in each scenario
3. Analyze the distribution of outcomes

---

## 7. Partial Differential Equations

### What is a PDE?

**Intuition:** An equation involving rates of change in multiple variables

**Example:** How does option price change when:
- Stock price changes?
- Time passes?
- Volatility changes?

### The Black-Scholes PDE

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

Where:
- **V:** Option value
- **S:** Stock price
- **t:** Time
- **σ:** Volatility
- **r:** Interest rate

**Solving this tells us the fair price of an option.**

### Finite Difference Methods

**Intuition:** Approximate continuous changes with discrete steps

Instead of calculus, we use:
```
∂V/∂S ≈ (V(S+ΔS) - V(S-ΔS)) / (2ΔS)
```

This turns the PDE into a system of linear equations we can solve on a computer.

---

## Quick Reference: Key Formulas

| Concept | Formula | Use Case |
|---------|---------|----------|
| Mantegna Distance | d = √(2(1-ρ)) | Correlation → Distance |
| Ollivier-Ricci | κ = 1 - W₁/d | Edge curvature |
| Sharpe Ratio | (R - Rf) / σ | Risk-adjusted return |
| VaR | Percentile of losses | Maximum expected loss |
| Black-Scholes | PDE above | Option pricing |

---

## Further Reading

1. **Mantegna & Stanley** — "An Introduction to Econophysics" (beginner-friendly)
2. **Sandhu et al.** — "Ricci curvature: An economic indicator" (our main reference)
3. **Hull** — "Options, Futures, and Other Derivatives" (finance bible)
4. **Ollivier** — "Ricci curvature of Markov chains" (original curvature paper)
