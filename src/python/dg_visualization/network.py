"""
Network Construction Module
===========================

This module builds graph networks from asset return correlations.

Key Concepts:
-------------

1. CORRELATION TO DISTANCE:
   We can't use correlation directly for geometry because it's not a 
   proper "metric" (distance measure). We use the Mantegna transformation:
   
       d(i,j) = √(2 × (1 - ρ(i,j)))
   
   This converts correlation ρ ∈ [-1, 1] to distance d ∈ [0, 2]:
   - ρ = +1 (identical) → d = 0 (no distance)
   - ρ = 0 (uncorrelated) → d ≈ 1.41
   - ρ = -1 (perfect hedge) → d = 2 (maximum distance)

2. NETWORK CONSTRUCTION METHODS:
   - KNN: Connect each asset to its K nearest neighbors
   - Threshold: Connect all pairs with distance < threshold
   - MST+: Minimum spanning tree plus high-correlation edges

3. WHY NETWORKS?
   Once we have a network, we can apply graph algorithms:
   - Shortest paths (geodesics)
   - Centrality measures
   - Curvature calculations
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def correlation_to_distance(returns_df):
    """
    Convert correlation matrix to proper metric distance.
    
    This is the Mantegna distance (1999), which satisfies the triangle
    inequality and can be used for proper geometric analysis.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns. Each column is an asset,
        each row is a time period (day).
        
    Returns
    -------
    distance_matrix : np.ndarray
        NxN matrix where entry [i,j] is the distance between assets i and j.
        Symmetric, with zeros on diagonal.
        
    correlation_matrix : np.ndarray
        NxN correlation matrix (useful to keep for analysis).
        
    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Simulate 3 assets over 100 days
    >>> returns = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    >>> dist, corr = correlation_to_distance(returns)
    >>> print(f"Distance A-B: {dist[0,1]:.2f}")
    
    Notes
    -----
    The formula d = √(2(1-ρ)) ensures:
    1. d ≥ 0 (distances are non-negative)
    2. d(i,i) = 0 (distance to self is zero)
    3. d(i,j) = d(j,i) (symmetric)
    4. d(i,j) ≤ d(i,k) + d(k,j) (triangle inequality)
    """
    # Compute correlation matrix
    # corr() computes Pearson correlation between each pair of columns
    corr_matrix = returns_df.corr().values
    
    # Clip to valid range [-1, 1] (numerical precision can cause slight overflows)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    # Mantegna distance transformation
    # Step by step:
    # 1. (1 - corr): transforms [+1, -1] to [0, 2]
    # 2. 2 * (...): scales to [0, 4]  
    # 3. sqrt(...): transforms to [0, 2]
    distance_matrix = np.sqrt(2.0 * (1.0 - corr_matrix))
    
    # Ensure diagonal is exactly zero (self-distance = 0)
    np.fill_diagonal(distance_matrix, 0.0)
    
    return distance_matrix, corr_matrix


def build_knn_network(distance_matrix, k=5):
    """
    Build network by connecting each node to its K nearest neighbors.
    
    This is the most common method for financial networks. Each asset
    is connected to the K most similar (closest) assets.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        NxN distance matrix from correlation_to_distance().
        
    k : int
        Number of nearest neighbors. Typical values: 3-10.
        Larger k = denser network = more edges.
        
    Returns
    -------
    adjacency : np.ndarray
        NxN weighted adjacency matrix.
        adjacency[i,j] = distance if edge exists, 0 otherwise.
        
    Notes
    -----
    The resulting network is typically NOT symmetric in terms of
    neighbor relationships (A's nearest might not include B even if
    B's nearest includes A), but we make the adjacency symmetric
    by including an edge if EITHER direction qualifies.
    """
    N = distance_matrix.shape[0]
    adjacency = np.zeros((N, N))
    
    # Ensure k doesn't exceed N-1 (can't have more neighbors than other nodes)
    k_actual = min(k, N - 1)
    
    for i in range(N):
        # Get distances from node i to all other nodes
        distances = distance_matrix[i].copy()
        
        # Exclude self-connection by setting self-distance to infinity
        distances[i] = np.inf
        
        # Find indices of k smallest distances (k nearest neighbors)
        # argsort returns indices that would sort the array
        # [:k_actual] takes the first k (smallest) indices
        nearest_indices = np.argsort(distances)[:k_actual]
        
        # Create edges to these neighbors
        for j in nearest_indices:
            # Edge weight = distance (lower = stronger connection)
            adjacency[i, j] = distance_matrix[i, j]
            adjacency[j, i] = distance_matrix[j, i]  # Make symmetric
            
    return adjacency


def build_threshold_network(distance_matrix, threshold=None):
    """
    Build network by connecting all pairs below a distance threshold.
    
    This creates edges between all "sufficiently similar" assets.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        NxN distance matrix.
        
    threshold : float, optional
        Maximum distance to create an edge.
        If None, uses median distance as threshold.
        
    Returns
    -------
    adjacency : np.ndarray
        NxN weighted adjacency matrix.
    """
    if threshold is None:
        # Use median of non-zero distances
        nonzero_distances = distance_matrix[distance_matrix > 0]
        threshold = np.median(nonzero_distances) if len(nonzero_distances) > 0 else 1.0
    
    # Create edge where distance < threshold (but not zero = self)
    mask = (distance_matrix > 0) & (distance_matrix < threshold)
    adjacency = np.where(mask, distance_matrix, 0.0)
    
    return adjacency


def build_mst_plus_network(distance_matrix, high_corr_threshold=0.7):
    """
    Build network using Minimum Spanning Tree plus high-correlation edges.
    
    The MST ensures all nodes are connected with minimum total distance.
    We then add extra edges for very high correlations.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        NxN distance matrix.
        
    high_corr_threshold : float
        Correlation threshold for adding extra edges.
        Correlation > this value → add edge.
        Default: 0.7 (equivalent to distance < 0.77)
        
    Returns
    -------
    adjacency : np.ndarray
        NxN weighted adjacency matrix.
        
    Notes
    -----
    MST (Minimum Spanning Tree) is a tree that:
    1. Connects all nodes
    2. Has no cycles
    3. Has minimum total edge weight
    
    This is useful because it captures the "backbone" structure
    of the market while keeping the network sparse.
    """
    # Compute MST using scipy
    # csr_matrix creates a sparse matrix (memory efficient)
    sparse_dist = csr_matrix(distance_matrix)
    mst = minimum_spanning_tree(sparse_dist)
    
    # Convert to dense array and make symmetric
    adjacency = mst.toarray()
    adjacency = adjacency + adjacency.T  # MST is directed; make undirected
    
    # Add high-correlation edges
    # Correlation of 0.7 → distance = sqrt(2 * 0.3) ≈ 0.77
    dist_threshold = np.sqrt(2.0 * (1.0 - high_corr_threshold))
    
    high_corr_mask = (distance_matrix > 0) & (distance_matrix < dist_threshold)
    high_corr_edges = np.where(high_corr_mask, distance_matrix, 0.0)
    
    # Combine: take maximum of MST and high-corr edges
    # (maximum because 0 means no edge)
    adjacency = np.maximum(adjacency, high_corr_edges)
    
    return adjacency


def build_asset_network(distance_matrix, method='knn', k=5, threshold=None):
    """
    Build weighted adjacency matrix from distance matrix.
    
    This is the main entry point for network construction.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        NxN distance matrix from correlation_to_distance().
        
    method : str
        Network construction method:
        - 'knn': K-nearest neighbors (default)
        - 'threshold': Distance threshold
        - 'mst_plus': Minimum spanning tree + high correlation edges
        
    k : int
        Number of neighbors for KNN method. Default: 5.
        
    threshold : float, optional
        Distance threshold for threshold method.
        
    Returns
    -------
    adjacency : np.ndarray
        NxN weighted adjacency matrix.
        Entry [i,j] = distance if edge exists, 0 otherwise.
        
    Example
    -------
    >>> dist_matrix = correlation_to_distance(returns_df)[0]
    >>> adj = build_asset_network(dist_matrix, method='knn', k=5)
    >>> n_edges = (adj > 0).sum() // 2  # Divide by 2 because symmetric
    >>> print(f"Network has {n_edges} edges")
    """
    if method == 'knn':
        return build_knn_network(distance_matrix, k=k)
    elif method == 'threshold':
        return build_threshold_network(distance_matrix, threshold=threshold)
    elif method == 'mst_plus':
        return build_mst_plus_network(distance_matrix)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn', 'threshold', or 'mst_plus'")


def get_network_statistics(adjacency):
    """
    Compute basic statistics about the network.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix.
        
    Returns
    -------
    stats : dict
        Dictionary with:
        - n_nodes: Number of nodes (assets)
        - n_edges: Number of edges
        - avg_degree: Average number of connections per node
        - density: Fraction of possible edges that exist
        - avg_weight: Average edge weight (distance)
    """
    N = adjacency.shape[0]
    
    # Count edges (adjacency is symmetric, so divide by 2)
    n_edges = (adjacency > 0).sum() // 2
    
    # Degree = number of edges per node
    degrees = (adjacency > 0).sum(axis=1)
    avg_degree = degrees.mean()
    
    # Density = actual edges / possible edges
    max_edges = N * (N - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0
    
    # Average edge weight
    weights = adjacency[adjacency > 0]
    avg_weight = weights.mean() if len(weights) > 0 else 0
    
    return {
        'n_nodes': N,
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'density': density,
        'avg_weight': avg_weight
    }
