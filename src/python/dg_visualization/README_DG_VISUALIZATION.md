# AEGIS — Differential Geometry Market Visualization Module

## Overview
This directory contains the **Differential Geometry (DG) Visualization module** of the AEGIS
(Advanced Enterprise Geometric Intelligence System) platform.

The DG module is a **research-grade market structure analysis engine** that applies
**network geometry, geodesic distances, and Ollivier–Ricci curvature** to financial markets
to study **systemic risk, redundancy, and diversification** beyond classical correlation analysis.

⚠️ **Important**  
This module is **not a trading strategy**, **not an alpha engine**, and **does not predict returns**.
It provides **structural diagnostics**, not forecasts.

---

## Conceptual Motivation
Financial markets are not flat Euclidean spaces.
Assets lie on a **curved manifold** induced by correlations and dependencies.

Differential geometry allows us to:
- Detect fragile clustering
- Identify systemic connectors
- Understand diversification structurally rather than statistically

---

## Core Geometric Intuition

| Geometric Property | Financial Interpretation |
|-------------------|-------------------------|
| Positive curvature | Tight clustering, fragility |
| Negative curvature | Bridges, diversification |
| High impact score  | Systemically important asset |
| Flat geometry      | Redundancy / neutrality |

---

## What This Module Implements

### 1. Correlation Network Construction
- Converts asset returns into a weighted graph
- Uses k-nearest-neighbor sparsification
- Preserves local market topology

### 2. Geodesic Distance Computation
- Maps correlations to distances
- Computes shortest paths on the market graph
- Approximates the intrinsic market manifold

### 3. Ollivier–Ricci Curvature
- Measures robustness vs fragility locally
- Detects:
  - hubs
  - bridges
  - clustered regimes

### 4. Scalar & Sectional Curvature
- Aggregates edge-level curvature to assets
- Quantifies systemic concentration

### 5. Asset Impact Analysis
- Measures structural disruption when an asset is removed
- Separates:
  - critical assets
  - diversifiers
  - redundant assets

### 6. Geometry-Aware Portfolio Weights
- Penalizes clustered assets
- Rewards structurally diversifying assets
- Produces **risk-aware allocations**
- **Does NOT optimize performance**

### 7. 3D Market Manifold Visualization
- Embeds assets in 3D
- Height encodes curvature
- Color encodes systemic impact
- Makes abstract geometry interpretable

---

## What This Module Does NOT Do

- ❌ Predict future returns
- ❌ Guarantee diversification
- ❌ Optimize Sharpe ratio or alpha
- ❌ Replace portfolio optimization

Instead, it provides **structural intelligence** that can inform downstream models.

---

## Running the DG Module

> Run all commands from the project root (folder containing `setup.py`).

### Demo (Synthetic Data)
Deterministic demo to verify correctness of the geometry pipeline.

```bash
pip install -e .
python run_demo.py
