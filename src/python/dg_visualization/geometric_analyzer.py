"""
Geometric Analyzer - Main Analysis Class
========================================

This is the main entry point for differential geometry portfolio analysis.
It orchestrates all the sub-modules and provides a clean API.

Usage:
------
>>> from aegis.dg_visualization import GeometricAnalyzer
>>> 
>>> # Initialize analyzer
>>> analyzer = GeometricAnalyzer()
>>> 
>>> # Run analysis on returns data
>>> results = analyzer.run_analysis(returns_df)
>>> 
>>> # Generate 3D visualization
>>> analyzer.plot_manifold_3d(results, output="market_topology.html")
>>> 
>>> # Get portfolio weights
>>> weights = results['weights']
"""

import numpy as np
import pandas as pd
import time

from .config import DGConfig, default_config
from .network import (
    correlation_to_distance,
    build_asset_network,
    get_network_statistics
)
from .curvature import (
    compute_geodesic_distances,
    compute_ollivier_ricci_curvature,
    compute_sectional_curvature,
    compute_scalar_curvature,
    compute_curvature_distribution
)
from .portfolio_weights import (
    compute_impact_scores,
    geometric_portfolio_weights,
    compute_portfolio_metrics
)


class GeometricAnalyzer:
    """
    Main class for Differential Geometry Portfolio Analysis.
    
    This class provides a high-level interface to:
    1. Build asset networks from correlation data
    2. Compute curvature measures (Ollivier-Ricci, sectional)
    3. Calculate impact scores for systemic risk
    4. Generate portfolio weights
    5. Visualize the market as a geometric manifold
    
    Attributes
    ----------
    config : DGConfig
        Configuration parameters for the analysis.
        
    results : dict
        Stored results from the last analysis run.
        
    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> returns = pd.DataFrame(
    ...     np.random.randn(252, 10) * 0.02,
    ...     columns=[f'Asset_{i}' for i in range(10)]
    ... )
    >>> 
    >>> # Run analysis
    >>> analyzer = GeometricAnalyzer()
    >>> results = analyzer.run_analysis(returns)
    >>> 
    >>> # Print summary
    >>> print(results['summary'])
    """
    
    def __init__(self, config=None):
        """
        Initialize the GeometricAnalyzer.
        
        Parameters
        ----------
        config : DGConfig, optional
            Configuration object. If None, uses default configuration.
        """
        self.config = config if config is not None else default_config
        self.results = None
        
    def run_analysis(self, data_df, asset_names=None, is_returns=False):
        """
        Run complete differential geometry analysis.
        
        This is the main method that performs all analysis steps:
        1. Data preparation (convert prices to returns if needed)
        2. Build correlation network
        3. Compute geodesic distances
        4. Calculate Ollivier-Ricci curvature
        5. Calculate sectional curvature
        6. Compute impact scores
        7. Generate portfolio weights
        
        Parameters
        ----------
        data_df : pd.DataFrame
            Either price DataFrame or returns DataFrame.
            Columns = assets, rows = time periods.
            
        asset_names : list, optional
            Names of assets. If None, uses column names.
            
        is_returns : bool
            True if data_df contains returns, False for prices.
            If False, log returns will be computed.
            
        Returns
        -------
        results : dict
            Dictionary containing all analysis outputs:
            - 'summary': DataFrame with per-asset metrics
            - 'correlation_matrix': Asset correlations
            - 'distance_matrix': Mantegna distances
            - 'adjacency': Network adjacency matrix
            - 'geodesic_matrix': Shortest path distances
            - 'orc_matrix': Edge curvatures
            - 'weights': Dictionary of {asset: weight}
            - 'config': Configuration used
            - 'network_stats': Network statistics
            - 'curvature_stats': Curvature distribution
        """
        print("=" * 70)
        print("DIFFERENTIAL GEOMETRY PORTFOLIO ANALYSIS")
        print("=" * 70)
        
        results = {}
        start_time = time.time()
        
        # ============================================================
        # STEP 1: Data Preparation
        # ============================================================
        print("\n1) Preparing data...")
        
        if is_returns:
            returns_df = data_df.copy()
        else:
            # Convert prices to log returns
            # Log return = ln(P_t / P_{t-1})
            returns_df = np.log(data_df / data_df.shift(1)).dropna()
        
        if asset_names is None:
            asset_names = returns_df.columns.tolist()
        
        N = len(asset_names)
        T = len(returns_df)
        print(f"   {N} assets, {T} observations")
        
        # Validate K parameter (can't have more neighbors than assets)
        if self.config.K >= N:
            old_k = self.config.K
            self.config.K = max(1, N - 2)
            print(f"   ⚠ Adjusted K from {old_k} to {self.config.K} (must be < N)")
        
        # ============================================================
        # STEP 2: Build Correlation Network
        # ============================================================
        print("\n2) Building correlation network...")
        
        distance_matrix, corr_matrix = correlation_to_distance(returns_df)
        
        adjacency = build_asset_network(
            distance_matrix,
            method=self.config.NETWORK_METHOD,
            k=self.config.K
        )
        
        network_stats = get_network_statistics(adjacency)
        print(f"   Network: {network_stats['n_edges']} edges, "
              f"avg degree: {network_stats['avg_degree']:.1f}")
        
        # ============================================================
        # STEP 3: Compute Geodesic Distances
        # ============================================================
        print("\n3) Computing geodesic distances...")
        
        geodesic_matrix = compute_geodesic_distances(adjacency)
        
        # Check connectivity
        n_disconnected = np.isinf(geodesic_matrix).sum()
        if n_disconnected > 0:
            print(f"   ⚠ {n_disconnected} disconnected pairs detected")
        
        # ============================================================
        # STEP 4: Ollivier-Ricci Curvature
        # ============================================================
        print("\n4) Computing Ollivier-Ricci curvature...")
        
        orc_matrix, orc_per_node = compute_ollivier_ricci_curvature(
            adjacency, 
            geodesic_matrix, 
            alpha=self.config.ALPHA
        )
        
        curvature_stats = compute_curvature_distribution(orc_matrix)
        scalar_curv = compute_scalar_curvature(orc_per_node)
        print(f"   Scalar curvature: {scalar_curv:.4f}")
        print(f"   Positive edges: {curvature_stats['pct_positive']:.1%}, "
              f"Negative edges: {curvature_stats['pct_negative']:.1%}")
        
        # ============================================================
        # STEP 5: Sectional Curvature
        # ============================================================
        print("\n5) Computing sectional curvature...")
        
        sectional_curvature = compute_sectional_curvature(
            geodesic_matrix, 
            adjacency
        )
        
        # ============================================================
        # STEP 6: Impact Scores
        # ============================================================
        print("\n6) Computing asset impact scores...")
        
        # Transpose: assets as rows, time as columns
        data_for_impact = returns_df.values.T
        
        impact_scores, connectivity_loss = compute_impact_scores(
            data_for_impact,
            adjacency,
            geodesic_matrix,
            k=self.config.K
        )
        
        # ============================================================
        # STEP 7: Portfolio Weights
        # ============================================================
        print("\n7) Computing geometric portfolio weights...")
        
        weights = geometric_portfolio_weights(
            returns_df,
            orc_per_node,
            impact_scores,
            min_weight=self.config.MIN_WEIGHT,
            max_weight=self.config.MAX_WEIGHT
        )
        
        portfolio_metrics = compute_portfolio_metrics(weights, returns_df)
        print(f"   Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
        print(f"   Effective assets: {portfolio_metrics['n_effective']:.1f}")
        
        # ============================================================
        # STEP 8: Compile Results
        # ============================================================
        print("\n8) Compiling results...")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'asset': asset_names,
            'ricci_curvature': orc_per_node,
            'sectional_curvature': sectional_curvature,
            'impact_score': impact_scores,
            'connectivity_loss': connectivity_loss,
            'geometric_weight': weights,
            'volatility': returns_df.std().values,
            'mean_return': returns_df.mean().values * 252  # Annualized
        })
        
        # Sort by impact score (most critical first)
        summary_df = summary_df.sort_values(
            'impact_score', 
            ascending=False
        ).reset_index(drop=True)
        
        # Store all results
        results['summary'] = summary_df
        results['returns_df'] = returns_df
        results['correlation_matrix'] = pd.DataFrame(
            corr_matrix, 
            index=asset_names, 
            columns=asset_names
        )
        results['distance_matrix'] = distance_matrix
        results['adjacency'] = adjacency
        results['geodesic_matrix'] = geodesic_matrix
        results['orc_matrix'] = orc_matrix
        results['orc_per_node'] = orc_per_node
        results['sectional_curvature'] = sectional_curvature
        results['impact_scores'] = impact_scores
        results['weights'] = dict(zip(asset_names, weights))
        results['portfolio_metrics'] = portfolio_metrics
        results['network_stats'] = network_stats
        results['curvature_stats'] = curvature_stats
        results['config'] = self.config
        results['asset_names'] = asset_names
        
        # Store for later use
        self.results = results
        
        elapsed = time.time() - start_time
        print(f"\n✓ Analysis complete in {elapsed:.2f} seconds")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print analysis summary."""
        summary = results['summary']
        
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        most_critical = summary.iloc[0]
        most_diversifying = summary.iloc[-1]
        
        print(f"""
MOST CRITICAL ASSET: {most_critical['asset']}
  Impact Score: {most_critical['impact_score']:.4f}
  Ricci Curvature: {most_critical['ricci_curvature']:.4f}
  Recommended Weight: {most_critical['geometric_weight']:.2%}
  → Removing this asset significantly disrupts network structure

MOST DIVERSIFYING ASSET: {most_diversifying['asset']}
  Impact Score: {most_diversifying['impact_score']:.4f}
  Ricci Curvature: {most_diversifying['ricci_curvature']:.4f}
  Recommended Weight: {most_diversifying['geometric_weight']:.2%}
  → This asset provides diversification with minimal systemic risk

PORTFOLIO METRICS:
  Expected Return: {results['portfolio_metrics']['expected_return']:.2%}
  Volatility: {results['portfolio_metrics']['volatility']:.2%}
  Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']:.3f}
  Effective Assets: {results['portfolio_metrics']['n_effective']:.1f} / {len(summary)}
""")
    
    def get_weights(self):
        """Get portfolio weights from last analysis."""
        if self.results is None:
            raise ValueError("No analysis has been run. Call run_analysis() first.")
        return self.results['weights']
    
    def get_summary(self):
        """Get summary DataFrame from last analysis."""
        if self.results is None:
            raise ValueError("No analysis has been run. Call run_analysis() first.")
        return self.results['summary']
    
    def save_results(self, prefix=None):
        """
        Save analysis results to CSV files.
        
        Parameters
        ----------
        prefix : str, optional
            Filename prefix. If None, uses config.SAVE_PREFIX.
        """
        if self.results is None:
            raise ValueError("No analysis has been run. Call run_analysis() first.")
        
        prefix = prefix or self.config.SAVE_PREFIX
        
        # Save summary
        self.results['summary'].to_csv(f"{prefix}_summary.csv", index=False)
        print(f"✓ Saved: {prefix}_summary.csv")
        
        # Save correlation matrix
        self.results['correlation_matrix'].to_csv(f"{prefix}_correlations.csv")
        print(f"✓ Saved: {prefix}_correlations.csv")
        
        # Save weights
        weights_df = pd.DataFrame(
            list(self.results['weights'].items()),
            columns=['asset', 'weight']
        )
        weights_df.to_csv(f"{prefix}_weights.csv", index=False)
        print(f"✓ Saved: {prefix}_weights.csv")


def create_sample_data(n_assets=10, n_days=500, seed=42):
    """
    Create synthetic returns data for testing.
    
    Parameters
    ----------
    n_assets : int
        Number of assets.
        
    n_days : int
        Number of trading days.
        
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    returns_df : pd.DataFrame
        Synthetic returns data with realistic correlation structure.
    """
    np.random.seed(seed)
    
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Create factor-based returns (realistic correlation structure)
    # Market factor affects all assets
    market_factor = np.random.randn(n_days) * 0.01
    
    returns = []
    for i in range(n_assets):
        # Random factor loading (beta)
        beta = 0.3 + 0.6 * np.random.rand()
        
        # Idiosyncratic return
        idio = np.random.randn(n_days) * 0.015
        
        # Total return = beta * market + idiosyncratic
        ret = beta * market_factor + idio
        returns.append(ret)
    
    returns_df = pd.DataFrame(
        np.array(returns).T,
        columns=asset_names,
        index=pd.date_range('2020-01-01', periods=n_days, freq='D')
    )
    
    return returns_df


# Convenience function for quick analysis
def quick_analysis(data_df, is_returns=False, **kwargs):
    """
    Run quick analysis with default settings.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Price or returns data.
        
    is_returns : bool
        True if data is returns.
        
    **kwargs : dict
        Override default config values.
        
    Returns
    -------
    results : dict
        Analysis results.
    """
    config = DGConfig(**kwargs) if kwargs else None
    analyzer = GeometricAnalyzer(config)
    return analyzer.run_analysis(data_df, is_returns=is_returns)
