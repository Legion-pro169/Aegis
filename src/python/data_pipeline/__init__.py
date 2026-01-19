"""
Market Data Pipeline Module
===========================

This module handles data ingestion from various sources.

Quick Start:
------------
    >>> from data_pipeline import DataPipeline
    >>> 
    >>> # Create pipeline
    >>> pipeline = DataPipeline()
    >>> 
    >>> # Get returns for analysis
    >>> returns = pipeline.get_returns(["AAPL", "MSFT", "GOOGL"], period="2y")
    >>> 
    >>> # Or use a predefined portfolio
    >>> returns = pipeline.get_portfolio_data("diversified", period="2y")

Available Classes:
------------------
    - DataPipeline: Main interface (recommended)
    - YahooFetcher: Direct Yahoo Finance access
    - DataNormalizer: Data cleaning utilities

Convenience Functions:
----------------------
    - quick_stocks(tickers, period): Quick stock data
    - quick_portfolio(type, period): Predefined portfolios
    - get_data_for_dg_analysis(): Ready for DG analysis

Predefined Portfolios:
----------------------
    - "diversified": Mix of all sectors
    - "tech": Technology stocks
    - "finance": Financial stocks
    - "energy": Energy stocks
    - "healthcare": Healthcare stocks
    - "indices": Major index ETFs
"""

# Main pipeline (recommended entry point)
from .pipeline import (
    DataPipeline,
    quick_portfolio,
    quick_stocks,
    get_data_for_dg_analysis
)

# Yahoo Finance fetcher
from .yahoo_fetcher import (
    YahooFetcher,
    quick_download,
    YFINANCE_AVAILABLE,
    # Predefined stock lists
    TECH_STOCKS,
    FINANCE_STOCKS,
    ENERGY_STOCKS,
    HEALTHCARE_STOCKS,
    CONSUMER_STOCKS,
    DIVERSIFIED_PORTFOLIO,
    INDEX_ETFS
)

# Data normalizer
from .normalizer import (
    DataNormalizer,
    quick_clean,
    prices_to_returns
)

# Synthetic data generator (for testing without internet)
from .synthetic import (
    SyntheticDataGenerator,
    create_sample_data,
    create_sectored_data,
    create_crisis_data
)

__all__ = [
    # Main classes
    "DataPipeline",
    "YahooFetcher",
    "DataNormalizer",
    "SyntheticDataGenerator",
    # Convenience functions
    "quick_portfolio",
    "quick_stocks",
    "quick_download",
    "quick_clean",
    "prices_to_returns",
    "get_data_for_dg_analysis",
    # Synthetic data
    "create_sample_data",
    "create_sectored_data",
    "create_crisis_data",
    # Constants
    "YFINANCE_AVAILABLE",
    "TECH_STOCKS",
    "FINANCE_STOCKS", 
    "ENERGY_STOCKS",
    "HEALTHCARE_STOCKS",
    "CONSUMER_STOCKS",
    "DIVERSIFIED_PORTFOLIO",
    "INDEX_ETFS"
]
