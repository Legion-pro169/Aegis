"""
Market Data Pipeline
====================

This is the main interface for the data pipeline module.
It combines fetching, cleaning, and transforming data into one easy-to-use class.

WHAT THIS MODULE DOES:
----------------------
1. Fetches stock data from Yahoo Finance
2. Cleans and validates the data
3. Calculates returns
4. Provides ready-to-use data for analysis

SIMPLE USAGE:
-------------
    >>> from data_pipeline import DataPipeline
    >>> 
    >>> # Create pipeline
    >>> pipeline = DataPipeline()
    >>> 
    >>> # Get clean returns for a portfolio
    >>> returns = pipeline.get_returns(
    ...     ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
    ...     period="2y"
    ... )
    >>> 
    >>> # Now use in DG analysis
    >>> from dg_visualization import GeometricAnalyzer
    >>> analyzer = GeometricAnalyzer()
    >>> results = analyzer.run_analysis(returns, is_returns=True)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime
import warnings

# Import our submodules
from .yahoo_fetcher import YahooFetcher, YFINANCE_AVAILABLE
from .normalizer import DataNormalizer


class DataPipeline:
    """
    Main data pipeline for fetching and preparing market data.
    
    This class provides a simple, high-level interface to:
    1. Download stock data
    2. Clean and validate it
    3. Calculate returns
    4. Prepare it for analysis
    
    Attributes:
        fetcher (YahooFetcher): Data fetcher
        normalizer (DataNormalizer): Data cleaner
        
    Example:
        >>> pipeline = DataPipeline()
        >>> 
        >>> # Method 1: Get clean returns directly
        >>> returns = pipeline.get_returns(["AAPL", "MSFT"], period="1y")
        >>> 
        >>> # Method 2: Get prices first, then do custom processing
        >>> prices = pipeline.get_prices(["AAPL", "MSFT"], period="1y")
        >>> returns = pipeline.prices_to_returns(prices)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data pipeline.
        
        Parameters:
            verbose (bool): Whether to print progress information
        """
        self.verbose = verbose
        
        # Initialize components
        if YFINANCE_AVAILABLE:
            self.fetcher = YahooFetcher(use_cache=True)
        else:
            self.fetcher = None
            warnings.warn("YahooFetcher not available. Install yfinance.")
        
        self.normalizer = DataNormalizer(verbose=verbose)
        
        if verbose:
            print("=" * 50)
            print("AEGIS Data Pipeline Initialized")
            print("=" * 50)
    
    def get_prices(
        self,
        tickers: Union[str, List[str]],
        period: str = "1y",
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Get clean price data for one or more stocks.
        
        Parameters:
            tickers: Single ticker "AAPL" or list ["AAPL", "MSFT"]
            period: Time period ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
            clean: Whether to clean the data (recommended)
            
        Returns:
            pd.DataFrame: Price data (Adjusted Close)
            
        Example:
            >>> pipeline = DataPipeline()
            >>> prices = pipeline.get_prices(["AAPL", "MSFT", "GOOGL"], period="2y")
        """
        if self.fetcher is None:
            raise RuntimeError("Data fetcher not available. Install yfinance.")
        
        # Handle single ticker
        if isinstance(tickers, str):
            tickers = [tickers]
        
        if self.verbose:
            print(f"\nFetching prices for {len(tickers)} stocks ({period})...")
        
        # Fetch data
        prices = self.fetcher.get_multiple_stocks(
            tickers,
            period=period,
            price_column="Adj Close"
        )
        
        # Clean if requested
        if clean:
            prices = self.normalizer.clean_prices(prices)
        
        return prices
    
    def get_returns(
        self,
        tickers: Union[str, List[str]],
        period: str = "1y",
        return_method: str = "log"
    ) -> pd.DataFrame:
        """
        Get clean returns data for one or more stocks.
        
        This is the most common method - gives you analysis-ready returns.
        
        Parameters:
            tickers: Single ticker or list of tickers
            period: Time period
            return_method: "log" or "simple"
            
        Returns:
            pd.DataFrame: Returns data ready for analysis
            
        Example:
            >>> pipeline = DataPipeline()
            >>> returns = pipeline.get_returns(
            ...     ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
            ...     period="2y"
            ... )
            >>> 
            >>> # Use with DG analysis
            >>> from dg_visualization import GeometricAnalyzer
            >>> analyzer = GeometricAnalyzer()
            >>> results = analyzer.run_analysis(returns, is_returns=True)
        """
        # Get clean prices
        prices = self.get_prices(tickers, period=period, clean=True)
        
        # Calculate returns
        returns = self.normalizer.compute_returns(prices, method=return_method)
        
        if self.verbose:
            print(f"\n✓ Returns calculated ({return_method})")
            print(f"  Shape: {returns.shape}")
            print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        
        return returns
    
    def prices_to_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Convert price DataFrame to returns.
        
        Parameters:
            prices (pd.DataFrame): Price data
            method (str): "log" or "simple"
            
        Returns:
            pd.DataFrame: Returns
        """
        return self.normalizer.compute_returns(prices, method=method)
    
    def get_summary(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for returns.
        
        Parameters:
            returns (pd.DataFrame): Returns data
            
        Returns:
            pd.DataFrame: Statistics per asset
        """
        return self.normalizer.get_summary_stats(returns)
    
    def get_portfolio_data(
        self,
        portfolio_type: str = "diversified",
        period: str = "2y",
        as_returns: bool = True
    ) -> pd.DataFrame:
        """
        Get data for a predefined portfolio.
        
        Parameters:
            portfolio_type (str): Type of portfolio:
                - "diversified": Mix of sectors
                - "tech": Technology stocks
                - "finance": Financial stocks
                - "energy": Energy stocks
                - "healthcare": Healthcare stocks
                - "indices": Major index ETFs
                
            period (str): Time period
            
            as_returns (bool): If True, return returns; else return prices
            
        Returns:
            pd.DataFrame: Portfolio data
            
        Example:
            >>> pipeline = DataPipeline()
            >>> 
            >>> # Get diversified portfolio
            >>> returns = pipeline.get_portfolio_data("diversified", period="2y")
            >>> 
            >>> # Get tech stocks
            >>> tech_returns = pipeline.get_portfolio_data("tech", period="1y")
        """
        from .yahoo_fetcher import (
            DIVERSIFIED_PORTFOLIO, TECH_STOCKS, FINANCE_STOCKS,
            ENERGY_STOCKS, HEALTHCARE_STOCKS, INDEX_ETFS
        )
        
        portfolios = {
            'diversified': DIVERSIFIED_PORTFOLIO,
            'tech': TECH_STOCKS,
            'finance': FINANCE_STOCKS,
            'energy': ENERGY_STOCKS,
            'healthcare': HEALTHCARE_STOCKS,
            'indices': list(INDEX_ETFS.keys())
        }
        
        if portfolio_type not in portfolios:
            raise ValueError(f"Unknown portfolio type: {portfolio_type}. "
                           f"Options: {list(portfolios.keys())}")
        
        tickers = portfolios[portfolio_type]
        
        if self.verbose:
            print(f"\n📊 Loading {portfolio_type.upper()} portfolio: {tickers}")
        
        if as_returns:
            return self.get_returns(tickers, period=period)
        else:
            return self.get_prices(tickers, period=period)
    
    def load_from_csv(
        self,
        filepath: str,
        date_column: str = None,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Parameters:
            filepath (str): Path to CSV file
            date_column (str): Name of date column (if not the index)
            parse_dates (bool): Whether to parse dates
            
        Returns:
            pd.DataFrame: Loaded and cleaned data
            
        Example:
            >>> pipeline = DataPipeline()
            >>> data = pipeline.load_from_csv("my_data.csv", date_column="Date")
        """
        if self.verbose:
            print(f"\nLoading data from: {filepath}")
        
        # Load CSV
        if date_column:
            df = pd.read_csv(filepath, parse_dates=[date_column])
            df = df.set_index(date_column)
        else:
            df = pd.read_csv(filepath, index_col=0, parse_dates=parse_dates)
        
        # Sort by date
        df = df.sort_index()
        
        if self.verbose:
            print(f"  Loaded shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def quality_check(self, data: pd.DataFrame) -> dict:
        """
        Run data quality checks.
        
        Parameters:
            data (pd.DataFrame): Data to check
            
        Returns:
            dict: Quality report
        """
        return self.normalizer.check_data_quality(data)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_portfolio(portfolio_type: str = "diversified", period: str = "2y") -> pd.DataFrame:
    """
    Quick function to get a predefined portfolio's returns.
    
    Example:
        >>> returns = quick_portfolio("tech", period="1y")
    """
    pipeline = DataPipeline(verbose=False)
    return pipeline.get_portfolio_data(portfolio_type, period=period, as_returns=True)


def quick_stocks(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Quick function to get returns for specified stocks.
    
    Example:
        >>> returns = quick_stocks(["AAPL", "MSFT", "GOOGL"], period="2y")
    """
    pipeline = DataPipeline(verbose=False)
    return pipeline.get_returns(tickers, period=period)


# ============================================================
# INTEGRATION WITH DG VISUALIZATION
# ============================================================

def get_data_for_dg_analysis(
    tickers: List[str] = None,
    portfolio_type: str = "diversified",
    period: str = "2y"
) -> pd.DataFrame:
    """
    Get data specifically formatted for DG (Differential Geometry) analysis.
    
    Parameters:
        tickers (List[str]): Custom list of tickers. If None, uses portfolio_type.
        portfolio_type (str): Predefined portfolio if tickers not specified
        period (str): Time period
        
    Returns:
        pd.DataFrame: Returns data ready for GeometricAnalyzer
        
    Example:
        >>> from data_pipeline import get_data_for_dg_analysis
        >>> from dg_visualization import GeometricAnalyzer
        >>> 
        >>> # Get data
        >>> returns = get_data_for_dg_analysis(period="2y")
        >>> 
        >>> # Run analysis
        >>> analyzer = GeometricAnalyzer()
        >>> results = analyzer.run_analysis(returns, is_returns=True)
    """
    pipeline = DataPipeline(verbose=True)
    
    if tickers:
        return pipeline.get_returns(tickers, period=period)
    else:
        return pipeline.get_portfolio_data(portfolio_type, period=period, as_returns=True)
