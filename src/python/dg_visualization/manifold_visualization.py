"""
3D Manifold Visualization Module
================================

This module creates stunning interactive 3D visualizations of market topology
using differential geometry concepts.

Key Visualizations:
-------------------

1. MARKET MANIFOLD SURFACE:
   - Assets positioned on a 2D surface embedded in 3D
   - Height (Z-axis) represents curvature or risk
   - Colors indicate different metrics (curvature, impact, weight)
   - Interactive: rotate, zoom, hover for details

2. NETWORK TOPOLOGY:
   - 3D network graph with assets as nodes
   - Edges show connections (correlation strength)
   - Node size = portfolio weight
   - Node color = curvature

3. CURVATURE LANDSCAPE:
   - Smooth interpolated surface showing market "terrain"
   - Valleys = negative curvature (bridge assets)
   - Hills = positive curvature (clustered assets)
   - Like a topographic map of systemic risk

Why 3D Visualization?
---------------------
- Humans understand 3D space intuitively
- Shows complex relationships at a glance
- Interactive exploration reveals insights
- Very impressive for presentations/demos

Technology:
-----------
- Plotly for interactive HTML output
- Works in any modern browser
- No installation needed to view
- Can be embedded in dashboards
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

# Plotly for interactive 3D visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ManifoldVisualizer:
    """
    Create 3D visualizations of market topology.
    
    This class transforms abstract geometric quantities into
    intuitive visual representations.
    
    Example
    -------
    >>> from aegis.dg_visualization import GeometricAnalyzer, ManifoldVisualizer
    >>> 
    >>> # Run analysis
    >>> analyzer = GeometricAnalyzer()
    >>> results = analyzer.run_analysis(returns_df)
    >>> 
    >>> # Create visualizer
    >>> viz = ManifoldVisualizer(results)
    >>> 
    >>> # Generate 3D manifold
    >>> fig = viz.plot_market_manifold()
    >>> fig.write_html("market_topology.html")
    """
    
    def __init__(self, results):
        """
        Initialize visualizer with analysis results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from GeometricAnalyzer.run_analysis()
        """
        self.results = results
        self.summary = results['summary']
        self.asset_names = results['asset_names']
        self.n_assets = len(self.asset_names)
        
        # Extract key metrics
        self.curvature = results['orc_per_node']
        self.impact = results['impact_scores']
        self.weights = np.array(list(results['weights'].values()))
        self.distance_matrix = results['distance_matrix']
        self.adjacency = results['adjacency']
        
    def _compute_2d_embedding(self, method='mds'):
        """
        Embed assets into 2D space based on distances.
        
        Parameters
        ----------
        method : str
            Embedding method:
            - 'mds': Multidimensional Scaling (preserves distances)
            - 'tsne': t-SNE (preserves local structure)
            - 'pca': PCA on correlation matrix
            
        Returns
        -------
        coords : np.ndarray
            (N, 2) array of 2D coordinates
        """
        if method == 'mds':
            # MDS: Find 2D positions that preserve original distances
            # This is the most geometrically faithful embedding
            mds = MDS(
                n_components=2,
                dissimilarity='precomputed',
                random_state=42,
                normalized_stress='auto'
            )
            coords = mds.fit_transform(self.distance_matrix)
            
        elif method == 'tsne':
            # t-SNE: Emphasizes local structure and clusters
            tsne = TSNE(
                n_components=2,
                metric='precomputed',
                random_state=42,
                perplexity=min(30, self.n_assets - 1)
            )
            coords = tsne.fit_transform(self.distance_matrix)
            
        elif method == 'pca':
            # PCA on correlation matrix
            corr = self.results['correlation_matrix'].values
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(corr)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to [-1, 1] range for nice visualization
        coords = coords - coords.mean(axis=0)
        max_range = np.abs(coords).max()
        coords = coords / (max_range + 1e-10)
        
        return coords
    
    def plot_market_manifold(
        self,
        height_metric='curvature',
        color_metric='impact',
        embedding_method='mds',
        show_surface=True,
        show_network=True,
        resolution=50,
        title="Market Topology Manifold"
    ):
        """
        Create 3D manifold visualization of market structure.
        
        This is the main visualization showing assets on a curved surface
        that represents market topology.
        
        Parameters
        ----------
        height_metric : str
            What determines Z-axis height:
            - 'curvature': Ricci curvature (default)
            - 'impact': Systemic impact score
            - 'volatility': Asset volatility
            - 'weight': Portfolio weight
            
        color_metric : str
            What determines point/surface color:
            - 'impact': Systemic impact (default)
            - 'curvature': Ricci curvature
            - 'weight': Portfolio weight
            - 'sector': Asset grouping (if available)
            
        embedding_method : str
            How to position assets in 2D: 'mds', 'tsne', or 'pca'
            
        show_surface : bool
            Whether to show interpolated surface
            
        show_network : bool
            Whether to show network edges
            
        resolution : int
            Grid resolution for surface interpolation
            
        title : str
            Plot title
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive 3D figure
        """
        # Get 2D embedding
        coords_2d = self._compute_2d_embedding(method=embedding_method)
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        
        # Get height values
        z = self._get_metric_values(height_metric)
        
        # Get color values
        colors = self._get_metric_values(color_metric)
        
        # Normalize height for better visualization
        z_normalized = (z - z.min()) / (z.max() - z.min() + 1e-10)
        z_display = z_normalized * 2 - 1  # Scale to [-1, 1]
        
        # Create figure
        fig = go.Figure()
        
        # Add interpolated surface
        if show_surface:
            fig = self._add_surface(fig, x, y, z_display, colors, resolution)
        
        # Add network edges
        if show_network:
            fig = self._add_network_edges(fig, x, y, z_display)
        
        # Add asset points (spheres)
        fig = self._add_asset_points(fig, x, y, z_display, colors, color_metric)
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Height: {height_metric} | Color: {color_metric}</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title="Embedding Dimension 1",
                yaxis_title="Embedding Dimension 2",
                zaxis_title=height_metric.capitalize(),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            ),
            width=1000,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _get_metric_values(self, metric):
        """Get values for a given metric."""
        if metric == 'curvature':
            return self.curvature
        elif metric == 'impact':
            # Handle NaN/Inf in impact scores
            values = np.nan_to_num(self.impact, nan=0, posinf=0, neginf=0)
            return values
        elif metric == 'volatility':
            return self.summary['volatility'].values
        elif metric == 'weight':
            return self.weights
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _add_surface(self, fig, x, y, z, colors, resolution):
        """Add interpolated surface to figure."""
        # Create grid for interpolation
        xi = np.linspace(x.min() - 0.2, x.max() + 0.2, resolution)
        yi = np.linspace(y.min() - 0.2, y.max() + 0.2, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate Z values onto grid
        # Use cubic interpolation for smooth surface
        zi_grid = griddata(
            (x, y), z,
            (xi_grid, yi_grid),
            method='cubic',
            fill_value=np.nan
        )
        
        # Also interpolate colors
        colors_normalized = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
        ci_grid = griddata(
            (x, y), colors_normalized,
            (xi_grid, yi_grid),
            method='cubic',
            fill_value=np.nan
        )
        
        # Add surface
        fig.add_trace(go.Surface(
            x=xi_grid,
            y=yi_grid,
            z=zi_grid,
            surfacecolor=ci_grid,
            colorscale='RdYlGn_r',  # Red=high risk, Green=low risk
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title="Risk Level",
                x=1.02,
                len=0.5
            ),
            name='Market Surface',
            hoverinfo='skip'
        ))
        
        return fig
    
    def _add_network_edges(self, fig, x, y, z):
        """Add network edges to figure."""
        edge_x = []
        edge_y = []
        edge_z = []
        
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                if self.adjacency[i, j] > 0:
                    # Add line segment from i to j
                    edge_x.extend([x[i], x[j], None])
                    edge_y.extend([y[i], y[j], None])
                    edge_z.extend([z[i], z[j], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(100, 100, 100, 0.3)',
                width=2
            ),
            name='Correlations',
            hoverinfo='skip'
        ))
        
        return fig
    
    def _add_asset_points(self, fig, x, y, z, colors, color_metric):
        """Add asset points as spheres."""
        # Normalize colors for colorscale
        colors_norm = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
        
        # Size based on portfolio weight
        sizes = 10 + 40 * (self.weights / self.weights.max())
        
        # Create hover text
        hover_text = []
        for i, name in enumerate(self.asset_names):
            text = (
                f"<b>{name}</b><br>"
                f"Curvature: {self.curvature[i]:.4f}<br>"
                f"Impact: {self.impact[i]:.4f}<br>"
                f"Weight: {self.weights[i]:.2%}<br>"
                f"Volatility: {self.summary['volatility'].iloc[i]:.4f}"
            )
            hover_text.append(text)
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors_norm,
                colorscale='RdYlGn_r',
                opacity=0.9,
                line=dict(color='white', width=1)
            ),
            text=self.asset_names,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            name='Assets'
        ))
        
        return fig
    
    def plot_curvature_landscape(self, resolution=100, title="Curvature Landscape"):
        """
        Create a topographic-style visualization of market curvature.
        
        This shows the market as a "landscape" where:
        - Mountains = positive curvature (clustered assets)
        - Valleys = negative curvature (bridge assets)
        - Color intensity = curvature magnitude
        
        Parameters
        ----------
        resolution : int
            Grid resolution
            
        title : str
            Plot title
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive 3D figure
        """
        # Get embedding
        coords = self._compute_2d_embedding(method='mds')
        x, y = coords[:, 0], coords[:, 1]
        z = self.curvature
        
        # Create fine grid
        xi = np.linspace(x.min() - 0.3, x.max() + 0.3, resolution)
        yi = np.linspace(y.min() - 0.3, y.max() + 0.3, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate curvature
        zi_grid = griddata(
            (x, y), z,
            (xi_grid, yi_grid),
            method='cubic',
            fill_value=0
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=xi_grid,
            y=yi_grid,
            z=zi_grid,
            colorscale='RdBu_r',  # Red=positive, Blue=negative
            opacity=0.9,
            showscale=True,
            colorbar=dict(
                title="Curvature",
                titleside="right"
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True
                )
            ),
            name='Curvature Surface'
        ))
        
        # Add asset markers
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z + 0.05,  # Slightly above surface
            mode='markers+text',
            marker=dict(
                size=8,
                color='black',
                symbol='diamond'
            ),
            text=self.asset_names,
            textposition='top center',
            name='Assets'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Red=Clustered | Blue=Bridge Assets</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title="Market Dimension 1",
                yaxis_title="Market Dimension 2",
                zaxis_title="Ricci Curvature",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.0))
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_network_3d(self, title="Asset Network Topology"):
        """
        Create 3D network visualization.
        
        Shows assets as nodes connected by edges.
        - Node size = portfolio weight
        - Node height = curvature
        - Node color = impact score
        - Edge opacity = correlation strength
        
        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        # Get embedding
        coords = self._compute_2d_embedding(method='mds')
        x, y = coords[:, 0], coords[:, 1]
        z = self.curvature
        
        fig = go.Figure()
        
        # Add edges with varying opacity based on correlation
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                if self.adjacency[i, j] > 0:
                    # Edge strength (inverse of distance)
                    strength = 1 - self.distance_matrix[i, j] / 2
                    
                    fig.add_trace(go.Scatter3d(
                        x=[x[i], x[j]],
                        y=[y[i], y[j]],
                        z=[z[i], z[j]],
                        mode='lines',
                        line=dict(
                            color=f'rgba(50, 50, 50, {strength * 0.5})',
                            width=3
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add nodes
        impact_norm = (self.impact - self.impact.min()) / (self.impact.max() - self.impact.min() + 1e-10)
        sizes = 15 + 35 * (self.weights / self.weights.max())
        
        hover_text = [
            f"<b>{name}</b><br>Weight: {w:.2%}<br>Curvature: {c:.3f}<br>Impact: {imp:.3f}"
            for name, w, c, imp in zip(self.asset_names, self.weights, self.curvature, self.impact)
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=impact_norm,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Impact"),
                line=dict(color='white', width=2)
            ),
            text=self.asset_names,
            textposition='top center',
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            name='Assets'
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18)),
            scene=dict(
                xaxis_title="Network Dimension 1",
                yaxis_title="Network Dimension 2",
                zaxis_title="Curvature",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_dashboard(self, output_path=None):
        """
        Create comprehensive dashboard with multiple views.
        
        Combines:
        - 3D manifold view
        - Curvature bar chart
        - Weight allocation
        - Risk metrics
        
        Parameters
        ----------
        output_path : str, optional
            If provided, saves HTML to this path
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            subplot_titles=(
                "Market Topology (3D Manifold)",
                "Asset Curvature",
                "Portfolio Allocation"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Get embedding for 3D plot
        coords = self._compute_2d_embedding(method='mds')
        x, y = coords[:, 0], coords[:, 1]
        z = self.curvature
        
        # 1. Add 3D scatter (simplified version)
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(
                    size=10 + 30 * self.weights / self.weights.max(),
                    color=self.curvature,
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=self.asset_names,
                textposition='top center',
                name='Assets'
            ),
            row=1, col=1
        )
        
        # Sort for bar charts
        sorted_idx = np.argsort(self.curvature)[::-1]
        sorted_names = [self.asset_names[i] for i in sorted_idx]
        sorted_curvature = self.curvature[sorted_idx]
        sorted_weights = self.weights[sorted_idx]
        
        # 2. Curvature bar chart
        colors = ['#d73027' if c > 0 else '#1a9850' for c in sorted_curvature]
        fig.add_trace(
            go.Bar(
                x=sorted_names,
                y=sorted_curvature,
                marker_color=colors,
                name='Curvature'
            ),
            row=2, col=1
        )
        
        # 3. Weight bar chart
        fig.add_trace(
            go.Bar(
                x=sorted_names,
                y=sorted_weights * 100,
                marker_color='steelblue',
                name='Weight (%)'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>AEGIS Differential Geometry Portfolio Analysis</b>",
                x=0.5,
                font=dict(size=22)
            ),
            height=900,
            width=1200,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, row=2)
        fig.update_yaxes(title_text="Curvature", row=2, col=1)
        fig.update_yaxes(title_text="Weight (%)", row=2, col=2)
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Dashboard saved to: {output_path}")
        
        return fig
    
    def save_all_visualizations(self, output_dir="."):
        """
        Save all visualizations to HTML files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save files
        """
        import os
        
        # Create directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save each visualization
        visualizations = [
            ("market_manifold.html", self.plot_market_manifold),
            ("curvature_landscape.html", self.plot_curvature_landscape),
            ("network_topology.html", self.plot_network_3d),
            ("analysis_dashboard.html", self.plot_dashboard)
        ]
        
        for filename, plot_func in visualizations:
            filepath = os.path.join(output_dir, filename)
            fig = plot_func()
            fig.write_html(filepath)
            print(f"✓ Saved: {filepath}")
        
        print(f"\n✓ All visualizations saved to: {output_dir}/")


def create_demo_visualization():
    """
    Create a demo visualization with sample data.
    
    This is useful for testing and demonstrations.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    from .geometric_analyzer import GeometricAnalyzer, create_sample_data
    
    # Generate sample data
    returns_df = create_sample_data(n_assets=15, n_days=500)
    
    # Run analysis
    analyzer = GeometricAnalyzer()
    results = analyzer.run_analysis(returns_df, is_returns=True)
    
    # Create visualization
    viz = ManifoldVisualizer(results)
    fig = viz.plot_market_manifold(
        height_metric='curvature',
        color_metric='impact',
        title="Demo: Market Topology Manifold"
    )
    
    return fig, results
