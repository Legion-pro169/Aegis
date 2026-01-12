"""
Differential Geometry Visualization Module
==========================================

This module provides tools for analyzing portfolio structure using
differential geometry concepts, particularly Ricci curvature.

Key Components:
    - GeometricAnalyzer: Main analysis class
    - ManifoldVisualizer: 3D visualization of market topology
    - curvature: Ollivier-Ricci and sectional curvature computations
    - network: Asset network construction from correlations

Example:
    >>> from aegis.dg_visualization import GeometricAnalyzer
    >>> analyzer = GeometricAnalyzer()
    >>> results = analyzer.run_analysis(returns_df)
    >>> analyzer.plot_manifold_3d(results, output="market.html")

Mathematical Background:
    See docs/mathematics.md for detailed explanations of:
    - Ollivier-Ricci curvature
    - Wasserstein distance
    - Geodesic calculations
    - Portfolio weight construction
"""

from .geometric_analyzer import GeometricAnalyzer
from .manifold_visualization import ManifoldVisualizer
from .config import DGConfig

__all__ = [
    "GeometricAnalyzer",
    "ManifoldVisualizer", 
    "DGConfig"
]
