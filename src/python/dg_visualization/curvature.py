"""
Curvature Computation Module
============================

This module computes various curvature measures on asset networks.

Key Concepts:
-------------

1. WHAT IS CURVATURE?
   Curvature measures how "bent" a space is.
   
   - POSITIVE curvature: Like a sphere - things cluster together
   - ZERO curvature: Like a flat table - parallel lines stay parallel
   - NEGATIVE curvature: Like a saddle - things spread apart
   
   In finance:
   - Positive curvature asset: Part of a tight cluster (redundant)
   - Negative curvature asset: Bridge between clusters (diversifier)

2. OLLIVIER-RICCI CURVATURE:
   Measures curvature of edges (connections between assets).
   
   Formula: κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
   
   Where:
   - W₁ is the Wasserstein distance (how different are the neighborhoods?)
   - d(x,y) is the distance between x and y
   
   Interpretation:
   - κ > 0: Neighbors of x and y are similar (clustered region)
   - κ < 0: Neighbors of x and y are different (bridge/bottleneck)

3. SECTIONAL CURVATURE:
   Measures how geodesics (shortest paths) spread apart.
   Computed using triangle defects.

References:
-----------
- Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces"
- Sandhu et al. (2016). "Ricci curvature: An economic indicator for market fragility"
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import warnings

# Try to import optimal transport library
try:
    import ot  # POT: Python Optimal Transport
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    warnings.warn(
        "POT (Python Optimal Transport) not installed. "
        "Install with: pip install POT. "
        "Using approximate Wasserstein computation instead."
    )


def compute_geodesic_distances(adjacency):
    """
    Compute shortest path distances between all pairs of nodes.
    
    A geodesic is the shortest path between two points. In a network,
    this is found using Dijkstra's algorithm.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix from network construction.
        
    Returns
    -------
    geodesic_matrix : np.ndarray
        NxN matrix where [i,j] is the shortest path distance from i to j.
        If no path exists, value is np.inf.
        
    Notes
    -----
    Dijkstra's algorithm works by:
    1. Start at source node
    2. Visit nearest unvisited neighbor
    3. Update distances to all neighbors
    4. Repeat until all nodes visited
    
    Complexity: O(N² log N) for dense graphs
    """
    # Convert to sparse matrix for efficiency
    sparse_adj = csr_matrix(adjacency)
    
    # Run Dijkstra from each node
    # directed=False because our network is undirected
    # unweighted=False because we want to use edge weights (distances)
    geodesic_matrix = dijkstra(
        sparse_adj, 
        directed=False, 
        unweighted=False
    )
    
    return geodesic_matrix


def _build_probability_measure(adjacency, node_index, alpha=0.5):
    """
    Build probability measure (distribution) centered at a node.
    
    This is used for Ollivier-Ricci curvature computation.
    The measure represents "where probability mass is concentrated"
    around a node.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix.
        
    node_index : int
        Index of the node to build measure for.
        
    alpha : float
        Laziness parameter (0 to 1).
        - alpha = 0: All mass on neighbors
        - alpha = 1: All mass on self
        - alpha = 0.5 (default): Half on self, half spread on neighbors
        
    Returns
    -------
    measure : np.ndarray
        Probability distribution over all nodes.
        measure[i] = probability mass at node i.
        Sum of measure = 1.0.
    """
    N = adjacency.shape[0]
    measure = np.zeros(N)
    
    # Find neighbors (nodes with positive edge weight)
    neighbors = np.where(adjacency[node_index] > 0)[0]
    n_neighbors = len(neighbors)
    
    if n_neighbors == 0:
        # Isolated node: all mass on self
        measure[node_index] = 1.0
    else:
        # Lazy random walk distribution:
        # - alpha probability of staying at current node
        # - (1-alpha) probability split equally among neighbors
        measure[node_index] = alpha
        measure[neighbors] = (1.0 - alpha) / n_neighbors
    
    return measure


def _compute_wasserstein_distance(measure1, measure2, cost_matrix):
    """
    Compute Wasserstein-1 distance between two probability measures.
    
    Also known as Earth Mover's Distance (EMD).
    
    Intuition: If measure1 and measure2 are piles of sand,
    how much "work" (mass × distance) is needed to reshape
    pile 1 into pile 2?
    
    Parameters
    ----------
    measure1, measure2 : np.ndarray
        Probability distributions (must sum to 1).
        
    cost_matrix : np.ndarray
        cost_matrix[i,j] = cost of moving mass from i to j.
        In our case, this is the geodesic distance matrix.
        
    Returns
    -------
    W1 : float
        Wasserstein-1 distance.
    """
    if OT_AVAILABLE:
        # Use proper optimal transport library
        # ot.emd2 solves the Earth Mover's Distance problem
        W1 = ot.emd2(measure1, measure2, cost_matrix)
    else:
        # Fallback: greedy approximation
        W1 = _approximate_wasserstein(measure1, measure2, cost_matrix)
    
    return W1


def _approximate_wasserstein(mu1, mu2, cost_matrix):
    """
    Approximate Wasserstein distance using greedy transport.
    
    This is less accurate than proper optimal transport but works
    when POT library is not available.
    
    Algorithm:
    1. Find nodes with positive mass in both distributions
    2. Greedily move mass from mu1 to mu2
    3. Sum up (mass moved × distance)
    """
    # Find supports (where mass exists)
    support1 = np.where(mu1 > 0)[0]
    support2 = np.where(mu2 > 0)[0]
    
    if len(support1) == 0 or len(support2) == 0:
        return 0.0
    
    # Greedy matching
    total_cost = 0.0
    remaining1 = mu1.copy()
    remaining2 = mu2.copy()
    
    for i in support1:
        for j in support2:
            if remaining1[i] > 0 and remaining2[j] > 0:
                # Move as much mass as possible
                transport_amount = min(remaining1[i], remaining2[j])
                total_cost += transport_amount * cost_matrix[i, j]
                remaining1[i] -= transport_amount
                remaining2[j] -= transport_amount
    
    return total_cost


def compute_ollivier_ricci_curvature(adjacency, geodesic_matrix, alpha=0.5):
    """
    Compute Ollivier-Ricci curvature for all edges in the network.
    
    This is the main curvature measure used in financial network analysis.
    
    Mathematical Definition:
    ------------------------
    For an edge between nodes x and y:
    
        κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
    
    Where:
    - μₓ = probability measure centered at x
    - μᵧ = probability measure centered at y  
    - W₁ = Wasserstein-1 distance
    - d(x,y) = geodesic distance from x to y
    
    Interpretation:
    ---------------
    - κ > 0 (positive curvature):
      Neighbors of x and y overlap significantly.
      The assets are in a tightly clustered region.
      → High systemic risk if this cluster fails.
      
    - κ ≈ 0 (flat):
      Neighbors are somewhat different but not completely.
      
    - κ < 0 (negative curvature):
      Neighbors of x and y are very different.
      This edge is a "bridge" between different regions.
      → Good for diversification.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix.
        
    geodesic_matrix : np.ndarray
        Shortest path distances between all pairs.
        
    alpha : float
        Laziness parameter for probability measures.
        
    Returns
    -------
    orc_matrix : np.ndarray
        NxN matrix of edge curvatures.
        orc_matrix[i,j] = curvature of edge (i,j).
        NaN for non-edges.
        
    orc_per_node : np.ndarray
        Average curvature for each node.
        This summarizes "how clustered" each asset is.
        
    Example
    -------
    >>> orc_matrix, orc_per_node = compute_ollivier_ricci_curvature(adj, geodesic, alpha=0.5)
    >>> # Find the most "bridge-like" asset
    >>> bridge_asset = np.argmin(orc_per_node)
    >>> print(f"Asset {bridge_asset} has curvature {orc_per_node[bridge_asset]:.3f}")
    """
    N = adjacency.shape[0]
    orc_matrix = np.full((N, N), np.nan)  # NaN for non-edges
    
    edges_computed = 0
    
    for i in range(N):
        # Get neighbors of node i
        neighbors_i = np.where(adjacency[i] > 0)[0]
        
        if len(neighbors_i) == 0:
            continue  # Skip isolated nodes
        
        # Build probability measure at node i
        mu_i = _build_probability_measure(adjacency, i, alpha)
        
        for j in neighbors_i:
            # Check if already computed (matrix is symmetric)
            if not np.isnan(orc_matrix[j, i]):
                orc_matrix[i, j] = orc_matrix[j, i]
                continue
            
            # Get neighbors of node j
            neighbors_j = np.where(adjacency[j] > 0)[0]
            
            if len(neighbors_j) == 0:
                continue
            
            # Build probability measure at node j
            mu_j = _build_probability_measure(adjacency, j, alpha)
            
            # Compute Wasserstein distance between measures
            W1 = _compute_wasserstein_distance(mu_i, mu_j, geodesic_matrix)
            
            # Get edge distance
            d_ij = geodesic_matrix[i, j]
            
            # Compute curvature: κ = 1 - W₁/d
            if d_ij > 1e-10:  # Avoid division by zero
                curvature = 1.0 - W1 / d_ij
                orc_matrix[i, j] = curvature
                orc_matrix[j, i] = curvature  # Symmetric
                edges_computed += 1
    
    # Compute per-node average curvature
    # nanmean ignores NaN values (non-edges)
    orc_per_node = np.nanmean(orc_matrix, axis=1)
    
    # Handle isolated nodes (all NaN row)
    orc_per_node = np.nan_to_num(orc_per_node, nan=0.0)
    
    print(f"   Computed ORC for {edges_computed} edges")
    
    return orc_matrix, orc_per_node


def compute_sectional_curvature(geodesic_matrix, adjacency):
    """
    Estimate sectional curvature using geodesic triangle defects.
    
    Sectional curvature measures how geodesics spread apart or converge.
    We estimate this by looking at triangles in the network.
    
    For a triangle with vertices (i, j, k):
    - Direct path: i → k with distance d(i,k)
    - Indirect path: i → j → k with distance d(i,j) + d(j,k)
    - Defect = how much longer is indirect vs direct
    
    In positively curved space (like a sphere), the indirect path
    is relatively longer → positive defect → positive curvature.
    
    In negatively curved space (like a saddle), paths spread out
    → smaller defect or negative.
    
    Parameters
    ----------
    geodesic_matrix : np.ndarray
        Shortest path distances.
        
    adjacency : np.ndarray
        Adjacency matrix (to determine which triangles to consider).
        
    Returns
    -------
    sectional_per_node : np.ndarray
        Average sectional curvature for each node.
        
    Notes
    -----
    Formula for triangle defect:
    
        defect = (d(i,j) + d(j,k) - d(i,k)) / (d(i,j) + d(j,k))
        
    This is normalized by the indirect path length.
    """
    N = geodesic_matrix.shape[0]
    sectional_per_node = np.zeros(N)
    
    for i in range(N):
        curvatures = []
        
        # Consider triangles involving node i and its neighbors
        neighbors_i = np.where(adjacency[i] > 0)[0]
        
        for j in neighbors_i:
            neighbors_j = np.where(adjacency[j] > 0)[0]
            
            for k in neighbors_j:
                # Avoid duplicates and self-loops
                if k <= j or k == i:
                    continue
                
                # Get triangle side lengths
                d_ij = geodesic_matrix[i, j]
                d_jk = geodesic_matrix[j, k]
                d_ik = geodesic_matrix[i, k]
                
                # Skip if any path is infinite (disconnected)
                if np.isinf(d_ij) or np.isinf(d_jk) or np.isinf(d_ik):
                    continue
                
                # Compute triangle defect
                indirect_path = d_ij + d_jk
                
                if indirect_path > 1e-10:
                    defect = (indirect_path - d_ik) / indirect_path
                    curvatures.append(defect)
        
        # Average curvature from all triangles involving this node
        if curvatures:
            sectional_per_node[i] = np.mean(curvatures)
    
    return sectional_per_node


def compute_scalar_curvature(orc_per_node):
    """
    Compute scalar curvature as global summary statistic.
    
    Scalar curvature is a single number summarizing the overall
    curvature of the network.
    
    Parameters
    ----------
    orc_per_node : np.ndarray
        Per-node Ollivier-Ricci curvature.
        
    Returns
    -------
    scalar_curvature : float
        Average curvature across all nodes.
        
    Notes
    -----
    In Riemannian geometry, scalar curvature is the trace of the
    Ricci curvature tensor. For networks, we use the mean.
    
    Interpretation:
    - High scalar curvature: Market is tightly clustered (high systemic risk)
    - Low/negative scalar curvature: Market is dispersed (more diversified)
    """
    return np.mean(orc_per_node)


def compute_curvature_distribution(orc_matrix):
    """
    Analyze the distribution of edge curvatures.
    
    Parameters
    ----------
    orc_matrix : np.ndarray
        Edge curvature matrix from compute_ollivier_ricci_curvature.
        
    Returns
    -------
    stats : dict
        Distribution statistics:
        - mean: Average curvature
        - std: Standard deviation
        - min, max: Range
        - pct_positive: Fraction of positive curvature edges
        - pct_negative: Fraction of negative curvature edges
    """
    # Extract non-NaN values (actual edges)
    edge_curvatures = orc_matrix[~np.isnan(orc_matrix)]
    
    if len(edge_curvatures) == 0:
        return {
            'mean': 0, 'std': 0, 'min': 0, 'max': 0,
            'pct_positive': 0, 'pct_negative': 0
        }
    
    return {
        'mean': np.mean(edge_curvatures),
        'std': np.std(edge_curvatures),
        'min': np.min(edge_curvatures),
        'max': np.max(edge_curvatures),
        'pct_positive': np.mean(edge_curvatures > 0),
        'pct_negative': np.mean(edge_curvatures < 0)
    }
