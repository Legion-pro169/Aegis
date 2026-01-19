"""
Synthetic Data Generator
========================

Generate realistic synthetic financial data for testing when
real market data is not available.

This is useful when:
- yfinance is not installed
- No internet connection
- Testing/development purposes
- Creating reproducible examples

The synthetic data simulates:
- Factor-based return structure (market, sector factors)
- Realistic correlation patterns
- Sector clustering
- Volatility differences between sectors
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


class SyntheticDataGenerator:
    """
    Generate realistic synthetic stock data.
    
    Example:
        >>> generator = SyntheticDataGenerator(seed=42)
        >>> 
        >>> # Generate simple portfolio
        >>> prices, returns = generator.generate_portfolio(
        ...     n_assets=10,
        ...     n_days=500
        ... )
        >>> 
        >>> # Generate sectored portfolio
        >>> prices, returns = generator.generate_sectored_portfolio()
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Parameters:
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Default sector characteristics
        self.sector_params = {
            'TECH': {
                'names': ['TECH_1', 'TECH_2', 'TECH_3', 'TECH_4'],
                'vol': 0.02,       # Higher volatility
                'beta': 1.2,       # Higher market sensitivity
                'sector_corr': 0.6  # Intra-sector correlation
            },
            'FINANCE': {
                'names': ['FIN_1', 'FIN_2', 'FIN_3'],
                'vol': 0.018,
                'beta': 1.1,
                'sector_corr': 0.5
            },
            'ENERGY': {
                'names': ['ENRG_1', 'ENRG_2', 'ENRG_3'],
                'vol': 0.022,
                'beta': 0.9,
                'sector_corr': 0.55
            },
            'HEALTHCARE': {
                'names': ['HLTH_1', 'HLTH_2'],
                'vol': 0.015,
                'beta': 0.7,
                'sector_corr': 0.4
            },
            'CONSUMER': {
                'names': ['CONS_1', 'CONS_2', 'CONS_3'],
                'vol': 0.014,
                'beta': 0.8,
                'sector_corr': 0.45
            }
        }
    
    def generate_portfolio(
        self,
        n_assets: int = 10,
        n_days: int = 500,
        start_date: str = "2022-01-01",
        annual_return: float = 0.08,
        annual_vol: float = 0.20,
        correlation: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a simple portfolio with uniform characteristics.
        
        Parameters:
            n_assets (int): Number of assets
            n_days (int): Number of trading days
            start_date (str): Start date for index
            annual_return (float): Expected annual return
            annual_vol (float): Expected annual volatility
            correlation (float): Average correlation between assets
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (prices, returns)
        """
        # Create date index (business days only)
        dates = pd.date_range(start_date, periods=n_days, freq='B')
        
        # Asset names
        asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Daily parameters
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Generate correlated returns using Cholesky decomposition
        # Create correlation matrix
        corr_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Cholesky decomposition for correlated samples
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate independent random returns
        independent_returns = np.random.randn(n_days, n_assets)
        
        # Make them correlated
        correlated_returns = independent_returns @ L.T
        
        # Scale to desired volatility and add drift
        returns = daily_return + daily_vol * correlated_returns
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(
            returns,
            index=dates,
            columns=asset_names
        )
        
        # Convert to prices (starting at 100)
        prices_df = 100 * np.exp(returns_df.cumsum())
        
        return prices_df, returns_df
    
    def generate_sectored_portfolio(
        self,
        n_days: int = 500,
        start_date: str = "2022-01-01",
        sectors: List[str] = None,
        market_vol: float = 0.01
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Generate portfolio with sector structure.
        
        This creates more realistic data where:
        - Stocks within same sector are more correlated
        - Different sectors have different volatilities
        - All stocks have some market factor exposure
        
        Parameters:
            n_days (int): Number of trading days
            start_date (str): Start date
            sectors (List[str]): Which sectors to include
                Options: 'TECH', 'FINANCE', 'ENERGY', 'HEALTHCARE', 'CONSUMER'
                Default: All sectors
            market_vol (float): Market factor volatility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[str]]: 
                (prices, returns, sector_labels)
        """
        if sectors is None:
            sectors = list(self.sector_params.keys())
        
        dates = pd.date_range(start_date, periods=n_days, freq='B')
        
        # Generate market factor (affects all stocks)
        market_factor = np.random.randn(n_days) * market_vol
        
        # Generate sector factors
        sector_factors = {}
        for sector in sectors:
            sector_vol = self.sector_params[sector]['vol'] * 0.5
            sector_factors[sector] = np.random.randn(n_days) * sector_vol
        
        # Generate returns for each stock
        all_returns = {}
        sector_labels = []
        
        for sector in sectors:
            params = self.sector_params[sector]
            
            for name in params['names']:
                # Market component
                beta = params['beta'] * (0.8 + 0.4 * np.random.rand())
                market_component = beta * market_factor
                
                # Sector component
                sector_beta = params['sector_corr'] * (0.8 + 0.4 * np.random.rand())
                sector_component = sector_beta * sector_factors[sector]
                
                # Idiosyncratic component
                idio_vol = params['vol'] * (1 - params['sector_corr'])
                idio_component = np.random.randn(n_days) * idio_vol
                
                # Total return
                all_returns[name] = market_component + sector_component + idio_component
                sector_labels.append(sector)
        
        # Create DataFrames
        returns_df = pd.DataFrame(all_returns, index=dates)
        prices_df = 100 * np.exp(returns_df.cumsum())
        
        return prices_df, returns_df, sector_labels
    
    def generate_crisis_scenario(
        self,
        n_assets: int = 10,
        n_days: int = 500,
        crisis_start: int = 300,
        crisis_duration: int = 50,
        crisis_vol_multiplier: float = 3.0,
        crisis_correlation: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate data with a crisis period (useful for testing regime detection).
        
        During the crisis:
        - Volatility increases dramatically
        - Correlations spike (everything moves together)
        - Returns are negative
        
        Parameters:
            n_assets: Number of assets
            n_days: Total days
            crisis_start: Day when crisis begins
            crisis_duration: How long crisis lasts
            crisis_vol_multiplier: Volatility multiplier during crisis
            crisis_correlation: Correlation during crisis
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (prices, returns)
        """
        # Normal period parameters
        normal_vol = 0.015
        normal_corr = 0.3
        
        dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
        asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        all_returns = []
        
        for day in range(n_days):
            if crisis_start <= day < crisis_start + crisis_duration:
                # Crisis period
                vol = normal_vol * crisis_vol_multiplier
                corr = crisis_correlation
                drift = -0.002  # Negative drift during crisis
            else:
                # Normal period
                vol = normal_vol
                corr = normal_corr
                drift = 0.0003  # Small positive drift
            
            # Generate correlated returns for this day
            corr_matrix = np.full((n_assets, n_assets), corr)
            np.fill_diagonal(corr_matrix, 1.0)
            L = np.linalg.cholesky(corr_matrix)
            
            indep = np.random.randn(n_assets)
            correlated = L @ indep
            day_returns = drift + vol * correlated
            all_returns.append(day_returns)
        
        returns_df = pd.DataFrame(
            all_returns,
            index=dates,
            columns=asset_names
        )
        prices_df = 100 * np.exp(returns_df.cumsum())
        
        return prices_df, returns_df


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_sample_data(n_assets: int = 10, n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create simple sample returns data.
    
    Example:
        >>> returns = create_sample_data()
        >>> # Use with DG analysis
    """
    generator = SyntheticDataGenerator(seed=seed)
    _, returns = generator.generate_portfolio(n_assets=n_assets, n_days=n_days)
    return returns


def create_sectored_data(seed: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create sectored sample data.
    
    Returns:
        Tuple of (returns DataFrame, sector labels list)
    """
    generator = SyntheticDataGenerator(seed=seed)
    _, returns, labels = generator.generate_sectored_portfolio()
    return returns, labels


def create_crisis_data(seed: int = 42) -> pd.DataFrame:
    """Create data with a crisis period."""
    generator = SyntheticDataGenerator(seed=seed)
    _, returns = generator.generate_crisis_scenario()
    return returns
