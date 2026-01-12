"""
Portfolio Weight Construction Module
====================================

This module converts geometric insights into portfolio allocations.

Key Concepts:
-------------

1. IMPACT SCORE:
   How much does removing an asset disrupt the network?
   
   High impact = systemically important = risky to hold too much
   Low impact = redundant/diversifier = safer to hold
   
2. CURVATURE-BASED WEIGHTING:
   - Negative curvature assets: Bridges between clusters → good diversifiers
   - Positive curvature assets: Cluster members → redundant, correlated
   
3. COMBINED SCORING:
   We combine multiple signals:
   - Impact score (lower = better)
   - Curvature (more negative = better)
   - Volatility (lower = better)
   
   Then convert scores to weights using softmax normalization.

Why This Works:
---------------
Traditional mean-variance optimization treats all correlations equally.
Our geometric approach identifies:
- Which assets are truly different (bridges)
- Which assets are secretly similar (clusters)
- Which assets are systemically dangerous (high impact)

This leads to portfolios that are more robust during crises.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_impact_scores(returns_data, adjacency, geodesic_baseline, k=5):
    """
    Compute how much removing each asset affects network structure.
    
    Impact Score measures the systemic importance of an asset.
    We remove each asset one at a time and measure how much the
    network structure changes.
    
    Parameters
    ----------
    returns_data : np.ndarray
        Returns matrix, shape (N_assets, N_timepoints).
        Note: Assets are ROWS here (transposed from usual DataFrame).
        
    adjacency : np.ndarray
        Current network adjacency matrix.
        
    geodesic_baseline : np.ndarray
        Current geodesic distance matrix.
        
    k : int
        K for rebuilding KNN network without removed asset.
        
    Returns
    -------
    impact_scores : np.ndarray
        Impact score for each asset.
        Higher = more systemically important.
        
    connectivity_loss : np.ndarray
        Fraction of network connectivity lost when asset removed.
        
    Interpretation
    --------------
    High impact score: This asset is a critical hub or bridge.
                      Removing it significantly disrupts connections.
                      → Reduce allocation to manage systemic risk.
                      
    Low impact score: This asset is redundant.
                     Network structure survives without it.
                     → Safe to hold, provides diversification.
    """
    N = returns_data.shape[0]  # Number of assets
    impact_scores = np.zeros(N)
    connectivity_loss = np.zeros(N)
    
    # Baseline: how many finite (connected) paths exist?
    baseline_connected = np.isfinite(geodesic_baseline).sum()
    
    for i in range(N):
        # Create index list without asset i
        remaining_idx = [j for j in range(N) if j != i]
        N_sub = len(remaining_idx)
        
        # Skip if too few assets remain
        if N_sub <= k:
            impact_scores[i] = np.nan
            connectivity_loss[i] = 1.0
            continue
        
        # Extract data without asset i
        data_sub = returns_data[remaining_idx]
        
        # Rebuild network without asset i
        # Step 1: Compute pairwise distances
        dist_sub = squareform(pdist(data_sub, metric='euclidean'))
        
        # Step 2: Build KNN adjacency
        adj_sub = np.zeros((N_sub, N_sub))
        
        # n_neighbors includes self, so add 1
        k_neighbors = min(k + 1, N_sub)
        nn = NearestNeighbors(n_neighbors=k_neighbors).fit(data_sub)
        knn_indices = nn.kneighbors(data_sub, return_distance=False)
        
        for r in range(N_sub):
            for c in knn_indices[r]:
                if c != r:  # Skip self-connections
                    adj_sub[r, c] = dist_sub[r, c]
                    adj_sub[c, r] = dist_sub[c, r]
        
        # Step 3: Compute geodesics in reduced network
        G_sub = dijkstra(csr_matrix(adj_sub), directed=False, unweighted=False)
        
        # Extract comparable submatrix from baseline
        G_baseline_sub = geodesic_baseline[np.ix_(remaining_idx, remaining_idx)]
        
        # Compare: how much did distances increase?
        mask_finite = np.isfinite(G_sub) & np.isfinite(G_baseline_sub)
        
        if mask_finite.sum() > 0:
            # Average increase in shortest paths
            diff = G_sub[mask_finite] - G_baseline_sub[mask_finite]
            impact_scores[i] = np.mean(diff)
            
            # Connectivity loss
            sub_connected = np.isfinite(G_sub).sum()
            baseline_sub_connected = np.isfinite(G_baseline_sub).sum()
            
            if baseline_sub_connected > 0:
                connectivity_loss[i] = 1.0 - sub_connected / baseline_sub_connected
        else:
            # Complete disconnection
            impact_scores[i] = np.inf
            connectivity_loss[i] = 1.0
    
    return impact_scores, connectivity_loss


def geometric_portfolio_weights(
    returns_df,
    orc_per_node,
    impact_scores,
    min_weight=0.01,
    max_weight=0.25,
    strategy='balanced'
):
    """
    Construct portfolio weights using geometric insights.
    
    This function combines curvature, impact, and volatility information
    to create portfolio weights that are:
    - Diversified (favors bridge assets)
    - Risk-aware (reduces exposure to systemic hubs)
    - Volatility-adjusted
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame.
        
    orc_per_node : np.ndarray
        Ollivier-Ricci curvature per asset.
        Negative = bridge, Positive = cluster member.
        
    impact_scores : np.ndarray
        Systemic impact scores per asset.
        Higher = more systemically important.
        
    min_weight : float
        Minimum weight per asset (default 1%).
        Ensures no asset is completely excluded.
        
    max_weight : float
        Maximum weight per asset (default 25%).
        Prevents excessive concentration.
        
    strategy : str
        Weight construction strategy:
        - 'risk_reduction': Heavily penalize high-impact assets
        - 'diversification': Heavily favor bridge assets
        - 'balanced': Equal weight to both considerations
        
    Returns
    -------
    weights : np.ndarray
        Portfolio weights summing to 1.0.
        
    How It Works
    ------------
    1. RISK SCORE: Lower impact = better
       risk_score = 1 - normalized(impact)
       
    2. DIVERSIFICATION SCORE: More negative curvature = better
       diversification_score = 1 - normalized(curvature)
       
    3. VOLATILITY SCORE: Lower volatility = better
       volatility_score = 1 - normalized(volatility)
       
    4. COMBINED SCORE:
       combined = w1*risk + w2*diversification + w3*volatility
       
    5. CONVERT TO WEIGHTS via softmax:
       weight_i = exp(score_i) / sum(exp(scores))
       
    6. APPLY CONSTRAINTS:
       Clip to [min_weight, max_weight] and renormalize
    """
    N = len(orc_per_node)
    
    # Clean up NaN and Inf values
    orc_clean = np.nan_to_num(orc_per_node, nan=0.0, posinf=1.0, neginf=-1.0)
    
    impact_median = np.nanmedian(impact_scores)
    impact_max = np.nanmax(impact_scores[np.isfinite(impact_scores)])
    impact_clean = np.nan_to_num(
        impact_scores, 
        nan=impact_median,
        posinf=impact_max
    )
    
    # Normalize to [0, 1] range
    orc_range = np.ptp(orc_clean) + 1e-10  # ptp = max - min
    orc_normalized = (orc_clean - orc_clean.min()) / orc_range
    
    impact_range = np.ptp(impact_clean) + 1e-10
    impact_normalized = (impact_clean - impact_clean.min()) / impact_range
    
    # Compute individual score components
    # Higher score = more desirable for portfolio
    
    # Risk score: LOW impact is GOOD → invert
    risk_score = 1.0 - impact_normalized
    
    # Diversification score: LOW (negative) curvature is GOOD → invert
    diversification_score = 1.0 - orc_normalized
    
    # Volatility score: LOW volatility is GOOD → invert
    volatility = returns_df.std().values
    vol_normalized = volatility / (volatility.max() + 1e-10)
    volatility_score = 1.0 - vol_normalized
    
    # Combine based on strategy
    if strategy == 'risk_reduction':
        combined_score = (
            0.6 * risk_score + 
            0.2 * diversification_score + 
            0.2 * volatility_score
        )
    elif strategy == 'diversification':
        combined_score = (
            0.2 * risk_score + 
            0.6 * diversification_score + 
            0.2 * volatility_score
        )
    else:  # balanced
        combined_score = (
            0.4 * risk_score + 
            0.4 * diversification_score + 
            0.2 * volatility_score
        )
    
    # Convert scores to weights using softmax
    exp_scores = np.exp(combined_score - combined_score.max())
    weights = exp_scores / exp_scores.sum()
    
    # Apply min/max constraints
    weights = np.clip(weights, min_weight, max_weight)
    
    # Renormalize to sum to 1.0
    weights = weights / weights.sum()
    
    return weights


def compute_portfolio_metrics(weights, returns_df, risk_free_rate=0.0):
    """
    Compute standard portfolio performance metrics.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
        
    returns_df : pd.DataFrame
        Asset returns.
        
    risk_free_rate : float
        Annual risk-free rate (default 0).
        
    Returns
    -------
    metrics : dict
        Portfolio metrics:
        - expected_return: Annualized expected return
        - volatility: Annualized volatility
        - sharpe_ratio: Risk-adjusted return
        - max_weight: Largest allocation
        - min_weight: Smallest allocation
        - n_effective: Effective number of assets
    """
    # Portfolio returns
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # Annualize (assuming daily returns, 252 trading days)
    expected_return = portfolio_returns.mean() * 252
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Effective number of assets (Herfindahl index inverse)
    n_effective = 1.0 / (weights ** 2).sum()
    
    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_weight': weights.max(),
        'min_weight': weights.min(),
        'n_effective': n_effective
    }


def compare_strategies(returns_df, orc_per_node, impact_scores, 
                       min_weight=0.01, max_weight=0.25):
    """
    Compare different weighting strategies.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns.
        
    orc_per_node : np.ndarray
        Curvature per asset.
        
    impact_scores : np.ndarray
        Impact scores per asset.
        
    min_weight, max_weight : float
        Weight constraints.
        
    Returns
    -------
    comparison : dict
        Dictionary with metrics for each strategy:
        - 'balanced': Equal weight to risk and diversification
        - 'risk_reduction': Focus on avoiding systemic risk
        - 'diversification': Focus on bridge assets
        - 'equal_weight': Simple 1/N benchmark
    """
    strategies = ['balanced', 'risk_reduction', 'diversification']
    comparison = {}
    
    for strategy in strategies:
        weights = geometric_portfolio_weights(
            returns_df, orc_per_node, impact_scores,
            min_weight=min_weight, max_weight=max_weight,
            strategy=strategy
        )
        metrics = compute_portfolio_metrics(weights, returns_df)
        comparison[strategy] = {
            'weights': weights,
            'metrics': metrics
        }
    
    # Add equal-weight benchmark
    N = len(orc_per_node)
    equal_weights = np.ones(N) / N
    comparison['equal_weight'] = {
        'weights': equal_weights,
        'metrics': compute_portfolio_metrics(equal_weights, returns_df)
    }
    
    return comparison
