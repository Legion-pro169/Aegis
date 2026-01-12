# AEGIS — Differential Geometry Market Visualization

## Overview
AEGIS (Adaptive Economic Geometry & Intelligence System) — Differential Geometry Visualization Module is a research-grade market structure analysis engine.
It applies Riemannian geometry, network theory, and Ollivier–Ricci curvature to financial markets to understand systemic risk, redundancy, and diversification beyond traditional correlation-based analysis.

This module does NOT predict returns or promise performance.
It focuses on structural diagnostics — answering how the market is shaped, not where prices will go.

---

## Core Idea (Intuition)
Markets are not flat.
Assets live on a curved manifold defined by correlations, dependencies, and structural links.

- Positive curvature → tightly clustered, fragile regimes
- Negative curvature → bridges, diversification, stress absorbers
- Flat regions → neutral redundancy

---

## What This Module Does
1. Builds a correlation network from returns
2. Computes geodesic distances on the market graph
3. Estimates Ollivier–Ricci curvature
4. Aggregates scalar and sectional curvature
5. Computes asset impact scores
6. Generates geometry-aware portfolio weights
7. Visualizes the market as a 3D manifold

---

## What This Module Does NOT Do
- Predict returns
- Guarantee diversification
- Optimize performance

It provides structural intelligence to inform downstream models.

---

## Installation
```bash
pip install -e .
pip install yfinance plotly
```

---

## Usage

### Synthetic Demo
```bash
python run_demo.py
```

### Real Market Experiment
```bash
python run_real_market.py
```

### 3D Visualization
```bash
python run_visualisations.py
```

This generates:
```
market_geometry_3d.html
```

---

## Interpretation Guide
- Height (Z-axis): curvature / fragility
- Color: systemic impact
- Distance: correlation redundancy

---

## Limitations
- Correlations are non-stationary
- Short windows amplify noise
- Structural diagnostics are not forecasts

---

## License
MIT License
