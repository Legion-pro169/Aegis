"""
Configuration for Differential Geometry Module
==============================================

This file contains all configurable parameters for the DG analysis.
Modify these values to tune the analysis for your specific use case.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DGConfig:
    """
    Configuration parameters for Differential Geometry analysis.
    
    Attributes:
        K: Number of nearest neighbors for network construction.
           Higher K = denser network = smoother curvature estimates.
           Typical range: 3-10. Default: 5.
           
        ALPHA: Laziness parameter for Ollivier-Ricci curvature.
               Probability of staying at current node vs moving to neighbor.
               Range: 0-1. Default: 0.5.
               
        NETWORK_METHOD: How to build the asset network.
               'knn': K-nearest neighbors (default, most common)
               'threshold': Connect all pairs below distance threshold
               'mst_plus': Minimum spanning tree + high-correlation edges
               
        MIN_WEIGHT: Minimum portfolio weight per asset (prevents 0 allocation).
                    Default: 0.01 (1%)
                    
        MAX_WEIGHT: Maximum portfolio weight per asset (prevents concentration).
                    Default: 0.25 (25%)
                    
        REGIME_WINDOW: Rolling window size (days) for regime detection.
                       Default: 22 (approximately 1 trading month)
                       
        STRESS_PERCENTILE: Curvature percentile threshold for stress signal.
                           Default: 90 (signal when curvature > 90th percentile)
    """
    
    # Network construction
    K: int = 5
    ALPHA: float = 0.5
    NETWORK_METHOD: Literal['knn', 'threshold', 'mst_plus'] = 'knn'
    
    # Portfolio constraints  
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.25
    
    # Regime detection
    REGIME_WINDOW: int = 22
    STRESS_PERCENTILE: float = 90
    
    # Visualization
    MANIFOLD_RESOLUTION: int = 50  # Grid resolution for 3D surface
    COLOR_SCHEME: str = 'RdYlGn_r'  # Red=high risk, Green=low risk
    
    # Output
    SAVE_PREFIX: str = "geometric_portfolio"
    
    def validate(self):
        """Validate configuration parameters."""
        assert 1 <= self.K <= 20, "K must be between 1 and 20"
        assert 0 < self.ALPHA < 1, "ALPHA must be between 0 and 1"
        assert self.NETWORK_METHOD in ['knn', 'threshold', 'mst_plus']
        assert 0 <= self.MIN_WEIGHT < self.MAX_WEIGHT <= 1
        assert 0 < self.STRESS_PERCENTILE <= 100
        return True


# Default configuration instance
default_config = DGConfig()
