#!/usr/bin/env python3
"""
AEGIS - Differential Geometry Portfolio Analysis Demo
=====================================================

This script demonstrates the differential geometry visualization module.
It can run with either:
1. Synthetic data (default) - for testing and demos
2. Real market data - using Yahoo Finance API

Usage:
------
    # Run with synthetic data
    python main.py
    
    # Run with real data (requires yfinance)
    python main.py --real
    
    # Specify output directory
    python main.py --output ./visualizations

Output:
-------
    - market_manifold.html: Interactive 3D manifold visualization
    - curvature_landscape.html: Topographic curvature map
    - network_topology.html: 3D network graph
    - analysis_dashboard.html: Comprehensive dashboard
    - analysis_summary.csv: Per-asset metrics

Author: [Your Name]
Date: 2026
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


def create_synthetic_portfolio(n_assets=15, n_days=500, seed=42):
    """
    Create synthetic portfolio with realistic characteristics.
    
    Creates assets with:
    - Sector clustering (tech, finance, energy, etc.)
    - Market factor exposure
    - Idiosyncratic risk
    """
    np.random.seed(seed)
    
    # Define sectors
    sectors = {
        'Tech': ['TECH_1', 'TECH_2', 'TECH_3', 'TECH_4'],
        'Finance': ['FIN_1', 'FIN_2', 'FIN_3'],
        'Energy': ['ENRG_1', 'ENRG_2', 'ENRG_3'],
        'Healthcare': ['HLTH_1', 'HLTH_2'],
        'Consumer': ['CONS_1', 'CONS_2', 'CONS_3']
    }
    
    asset_names = []
    sector_labels = []
    for sector, assets in sectors.items():
        asset_names.extend(assets)
        sector_labels.extend([sector] * len(assets))
    
    n_assets = len(asset_names)
    
    # Generate factors
    market_factor = np.random.randn(n_days) * 0.01  # Market
    
    # Sector factors
    sector_factors = {}
    for sector in sectors.keys():
        sector_factors[sector] = np.random.randn(n_days) * 0.008
    
    # Generate returns
    returns = []
    for i, (asset, sector) in enumerate(zip(asset_names, sector_labels)):
        # Market exposure
        beta_market = 0.5 + 0.5 * np.random.rand()
        
        # Sector exposure
        beta_sector = 0.3 + 0.4 * np.random.rand()
        
        # Idiosyncratic
        idio = np.random.randn(n_days) * 0.012
        
        # Combine
        ret = (
            beta_market * market_factor +
            beta_sector * sector_factors[sector] +
            idio
        )
        returns.append(ret)
    
    returns_df = pd.DataFrame(
        np.array(returns).T,
        columns=asset_names,
        index=pd.date_range('2023-01-01', periods=n_days, freq='B')
    )
    
    return returns_df, sector_labels


def fetch_real_data(tickers=None, period='2y'):
    """
    Fetch real market data using yfinance.
    
    Parameters
    ----------
    tickers : list, optional
        List of ticker symbols. If None, uses default portfolio.
        
    period : str
        Data period (e.g., '1y', '2y', '5y')
        
    Returns
    -------
    returns_df : pd.DataFrame
        Daily returns
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Install with: pip install yfinance")
        print("Falling back to synthetic data...")
        return None
    
    if tickers is None:
        # Default diversified portfolio
        tickers = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'NVDA',
            # Finance
            'JPM', 'BAC', 'GS',
            # Energy
            'XOM', 'CVX',
            # Healthcare
            'JNJ', 'UNH',
            # Consumer
            'WMT', 'PG', 'KO',
            # Other
            'SPY'  # S&P 500 ETF for reference
        ]
    
    print(f"Fetching data for {len(tickers)} tickers...")
    
    # Download data
    data = yf.download(tickers, period=period, progress=False)['Adj Close']
    
    # Remove any tickers with missing data
    data = data.dropna(axis=1, how='any')
    
    # Compute log returns
    returns_df = np.log(data / data.shift(1)).dropna()
    
    print(f"✓ Loaded {len(returns_df.columns)} assets, {len(returns_df)} days")
    
    return returns_df


def run_demo(use_real_data=False, output_dir='./output'):
    """
    Run the complete demo analysis.
    
    Parameters
    ----------
    use_real_data : bool
        If True, fetches real market data
        
    output_dir : str
        Directory for output files
    """
    print("=" * 70)
    print("        AEGIS - Differential Geometry Portfolio Analysis")
    print("=" * 70)
    print()
    
    # Import our modules
    from src.python.dg_visualization import GeometricAnalyzer, ManifoldVisualizer
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    if use_real_data:
        print("MODE: Real Market Data")
        returns_df = fetch_real_data()
        if returns_df is None:
            returns_df, _ = create_synthetic_portfolio()
    else:
        print("MODE: Synthetic Data (use --real for market data)")
        returns_df, sector_labels = create_synthetic_portfolio()
    
    print(f"\nPortfolio: {len(returns_df.columns)} assets, {len(returns_df)} observations")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print()
    
    # Run analysis
    analyzer = GeometricAnalyzer()
    results = analyzer.run_analysis(returns_df, is_returns=True)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    viz = ManifoldVisualizer(results)
    
    # 1. Market Manifold (main visualization)
    print("\n1) Creating Market Manifold visualization...")
    fig1 = viz.plot_market_manifold(
        height_metric='curvature',
        color_metric='impact',
        title="Market Topology Manifold"
    )
    fig1.write_html(os.path.join(output_dir, 'market_manifold.html'))
    print(f"   ✓ Saved: {output_dir}/market_manifold.html")
    
    # 2. Curvature Landscape
    print("\n2) Creating Curvature Landscape...")
    fig2 = viz.plot_curvature_landscape(
        title="Ricci Curvature Landscape"
    )
    fig2.write_html(os.path.join(output_dir, 'curvature_landscape.html'))
    print(f"   ✓ Saved: {output_dir}/curvature_landscape.html")
    
    # 3. Network Topology
    print("\n3) Creating Network Topology view...")
    fig3 = viz.plot_network_3d(
        title="Asset Network Topology"
    )
    fig3.write_html(os.path.join(output_dir, 'network_topology.html'))
    print(f"   ✓ Saved: {output_dir}/network_topology.html")
    
    # 4. Dashboard
    print("\n4) Creating Analysis Dashboard...")
    fig4 = viz.plot_dashboard()
    fig4.write_html(os.path.join(output_dir, 'analysis_dashboard.html'))
    print(f"   ✓ Saved: {output_dir}/analysis_dashboard.html")
    
    # Save data
    print("\n5) Saving analysis data...")
    results['summary'].to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
    print(f"   ✓ Saved: {output_dir}/analysis_summary.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"""
Output files saved to: {output_dir}/

Files:
  • market_manifold.html    - Interactive 3D topology (MAIN VISUALIZATION)
  • curvature_landscape.html - Topographic curvature map
  • network_topology.html    - 3D network graph
  • analysis_dashboard.html  - Comprehensive dashboard
  • analysis_summary.csv     - Per-asset metrics

To view visualizations:
  1. Open any .html file in a web browser
  2. Interact: rotate (drag), zoom (scroll), hover for details

Key Insights:
  • RED regions = high risk (positive curvature, clustered assets)
  • GREEN regions = diversifiers (negative curvature, bridge assets)
  • LARGER points = higher portfolio allocation
""")
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='AEGIS Differential Geometry Portfolio Analysis Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with synthetic data
  python main.py --real             # Run with real market data
  python main.py --output ./viz     # Specify output directory
        """
    )
    
    parser.add_argument(
        '--real', 
        action='store_true',
        help='Use real market data (requires yfinance)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='Output directory for visualizations (default: ./output)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_demo(
            use_real_data=args.real,
            output_dir=args.output
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
