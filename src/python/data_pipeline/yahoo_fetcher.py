"""
Yahoo Finance Data Fetcher
==========================

This module fetches stock data from Yahoo Finance using the yfinance library.

WHAT YOU'LL LEARN:
------------------
1. How to use external libraries (yfinance)
2. How to handle errors gracefully (try/except)
3. How to work with dates
4. How to validate data
5. How to write clean, documented functions

PREREQUISITES:
--------------
Install yfinance: pip install yfinance

HOW IT WORKS:
-------------
1. You give us a ticker symbol (like "AAPL" for Apple)
2. We ask Yahoo Finance for the price history
3. Yahoo sends back the data
4. We clean it up and return a nice DataFrame

Example:
    >>> from data_pipeline import YahooFetcher
    >>> fetcher = YahooFetcher()
    >>> data = fetcher.get_stock_data("AAPL", period="1y")
    >>> print(data.head())
"""

# ============================================================
# IMPORTS
# ============================================================

# Standard library imports (built into Python)
from datetime import datetime, timedelta
from typing import List, Optional, Union
import warnings

# Third-party imports (need to pip install)
import pandas as pd
import numpy as np

# Try to import yfinance
# We use try/except because the user might not have it installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn(
        "yfinance not installed. Install with: pip install yfinance"
    )


# ============================================================
# YAHOO FETCHER CLASS
# ============================================================

class YahooFetcher:
    """
    Fetches stock data from Yahoo Finance.
    
    This class provides methods to download historical stock prices,
    company information, and other financial data.
    
    Attributes:
        cache (dict): Stores previously fetched data to avoid repeat downloads
        
    Example:
        >>> fetcher = YahooFetcher()
        >>> 
        >>> # Get 1 year of Apple data
        >>> apple = fetcher.get_stock_data("AAPL", period="1y")
        >>> 
        >>> # Get multiple stocks
        >>> portfolio = fetcher.get_multiple_stocks(
        ...     ["AAPL", "GOOGL", "MSFT"],
        ...     period="2y"
        ... )
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the Yahoo Fetcher.
        
        Parameters:
            use_cache (bool): If True, stores data to avoid re-downloading.
                             Default is True (saves time and API calls).
        """
        # Check if yfinance is available
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required but not installed.\n"
                "Install it with: pip install yfinance"
            )
        
        # Initialize cache (a dictionary to store downloaded data)
        # Dictionary = key-value pairs, like a real dictionary
        # Key: "AAPL_1y" → Value: DataFrame of Apple's 1-year data
        self.use_cache = use_cache
        self.cache = {}
        
        print("✓ YahooFetcher initialized")
    
    def get_stock_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download historical data for a single stock.
        
        Parameters:
            ticker (str): Stock symbol (e.g., "AAPL", "GOOGL", "MSFT")
            
            period (str): How far back to get data. Options:
                - "1d", "5d" = 1 or 5 days
                - "1mo", "3mo", "6mo" = 1, 3, or 6 months
                - "1y", "2y", "5y", "10y" = 1, 2, 5, or 10 years
                - "ytd" = Year to date
                - "max" = All available data
                
            interval (str): Time between data points. Options:
                - "1m", "5m", "15m", "30m" = minutes (limited history)
                - "1h" = hourly
                - "1d" = daily (most common)
                - "1wk" = weekly
                - "1mo" = monthly
                
            start_date (str): Specific start date "YYYY-MM-DD" (optional)
            end_date (str): Specific end date "YYYY-MM-DD" (optional)
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - Open, High, Low, Close = prices
                - Adj Close = adjusted close (USE THIS!)
                - Volume = shares traded
                
        Example:
            >>> fetcher = YahooFetcher()
            >>> 
            >>> # Using period
            >>> data = fetcher.get_stock_data("AAPL", period="1y")
            >>> 
            >>> # Using specific dates
            >>> data = fetcher.get_stock_data(
            ...     "AAPL",
            ...     start_date="2023-01-01",
            ...     end_date="2024-01-01"
            ... )
        """
        # Clean the ticker (uppercase, remove spaces)
        ticker = ticker.strip().upper()
        
        # Create cache key
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check if we already have this data cached
        if self.use_cache and cache_key in self.cache:
            print(f"  Using cached data for {ticker}")
            return self.cache[cache_key].copy()
        
        print(f"  Downloading {ticker}...", end=" ")
        
        # TRY/EXCEPT: Handle errors gracefully
        # "try" = attempt this code
        # "except" = if an error occurs, do this instead
        try:
            # Create a Ticker object
            # This represents the stock in yfinance
            stock = yf.Ticker(ticker)
            
            # Download the historical data
            if start_date and end_date:
                # Use specific date range
                data = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
            else:
                # Use period
                data = stock.history(
                    period=period,
                    interval=interval
                )
            
            # Check if we got any data
            if data.empty:
                print("✗ No data returned")
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Clean the data
            data = self._clean_data(data)
            
            # Store in cache
            if self.use_cache:
                self.cache[cache_key] = data.copy()
            
            print(f"✓ {len(data)} rows")
            return data
            
        except Exception as e:
            # Something went wrong
            print(f"✗ Error: {e}")
            raise
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
        price_column: str = "Adj Close"
    ) -> pd.DataFrame:
        """
        Download data for multiple stocks and combine them.
        
        This is useful for portfolio analysis where you need
        multiple stocks in one DataFrame.
        
        Parameters:
            tickers (List[str]): List of stock symbols
                Example: ["AAPL", "GOOGL", "MSFT", "AMZN"]
                
            period (str): Time period (see get_stock_data)
            
            interval (str): Data interval (see get_stock_data)
            
            price_column (str): Which price to use:
                - "Adj Close" = adjusted close (recommended)
                - "Close" = regular close
                - "Open" = opening price
                
        Returns:
            pd.DataFrame: DataFrame where:
                - Each column is a stock
                - Each row is a date
                - Values are the selected price
                
        Example:
            >>> fetcher = YahooFetcher()
            >>> portfolio = fetcher.get_multiple_stocks(
            ...     ["AAPL", "GOOGL", "MSFT", "JPM", "XOM"],
            ...     period="2y"
            ... )
            >>> print(portfolio.head())
                            AAPL    GOOGL    MSFT     JPM     XOM
            Date
            2022-01-03    150.23   145.67   302.45   158.34   61.23
            2022-01-04    149.87   144.89   300.12   157.98   62.01
            ...
        """
        print(f"\nFetching {len(tickers)} stocks...")
        
        # Dictionary to store each stock's data
        all_data = {}
        
        # Track failed downloads
        failed = []
        
        # Loop through each ticker
        for ticker in tickers:
            try:
                # Get data for this stock
                data = self.get_stock_data(
                    ticker,
                    period=period,
                    interval=interval
                )
                
                # Extract just the price column we want
                if price_column in data.columns:
                    all_data[ticker] = data[price_column]
                else:
                    print(f"  Warning: {price_column} not found for {ticker}")
                    failed.append(ticker)
                    
            except Exception as e:
                print(f"  Failed to get {ticker}: {e}")
                failed.append(ticker)
        
        # Combine all stocks into one DataFrame
        # pd.DataFrame(dict) creates a DataFrame from a dictionary
        if all_data:
            combined = pd.DataFrame(all_data)
            
            # Drop rows where ANY stock has missing data
            # This ensures all stocks have the same dates
            combined = combined.dropna()
            
            print(f"\n✓ Successfully fetched {len(combined.columns)} stocks")
            print(f"  Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
            print(f"  Total rows: {len(combined)}")
            
            if failed:
                print(f"  ⚠ Failed tickers: {failed}")
            
            return combined
        else:
            raise ValueError("Could not fetch any stock data")
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Get company information for a stock.
        
        Parameters:
            ticker (str): Stock symbol
            
        Returns:
            dict: Company information including:
                - longName: Full company name
                - sector: Industry sector
                - marketCap: Market capitalization
                - and many more...
                
        Example:
            >>> fetcher = YahooFetcher()
            >>> info = fetcher.get_stock_info("AAPL")
            >>> print(info['longName'])
            'Apple Inc.'
            >>> print(info['sector'])
            'Technology'
        """
        ticker = ticker.strip().upper()
        
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            print(f"Error getting info for {ticker}: {e}")
            return {}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the downloaded data.
        
        This is a "private" method (starts with _) meaning it's
        meant to be used internally, not called directly by users.
        
        Steps:
        1. Remove any duplicate rows
        2. Sort by date
        3. Handle missing values
        4. Ensure correct data types
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Remove duplicate dates (if any)
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date (oldest first)
        df = df.sort_index()
        
        # Forward-fill missing values (use previous day's price)
        # This handles days when market was closed unexpectedly
        df = df.ffill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        return df
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters:
            prices (pd.DataFrame): DataFrame of prices
            
            method (str): How to calculate returns:
                - "simple": (P_t - P_{t-1}) / P_{t-1}
                  Also called "arithmetic" or "percentage" returns
                  
                - "log": ln(P_t / P_{t-1})
                  Also called "logarithmic" or "continuous" returns
                  Better for analysis because they're additive
                  
        Returns:
            pd.DataFrame: Returns (same shape as input, minus one row)
            
        Why Log Returns?
        ----------------
        Log returns have nice properties:
        1. Additive: r_monthly = r_day1 + r_day2 + ... + r_day22
        2. Symmetric: +10% and -10% cancel out properly
        3. No negative prices possible when simulating
        
        Example:
            >>> prices = fetcher.get_multiple_stocks(["AAPL", "MSFT"])
            >>> returns = fetcher.calculate_returns(prices, method="log")
        """
        if method == "log":
            # Log return = ln(price_today / price_yesterday)
            # np.log is natural logarithm (ln)
            returns = np.log(prices / prices.shift(1))
        elif method == "simple":
            # Simple return = (price_today - price_yesterday) / price_yesterday
            # pct_change() calculates this automatically
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")
        
        # Remove the first row (it's NaN because there's no previous price)
        returns = returns.dropna()
        
        return returns
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        print("✓ Cache cleared")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_download(
    tickers: Union[str, List[str]],
    period: str = "1y"
) -> pd.DataFrame:
    """
    Quick function to download stock data.
    
    Parameters:
        tickers: Single ticker "AAPL" or list ["AAPL", "MSFT"]
        period: Time period
        
    Returns:
        DataFrame of adjusted close prices
        
    Example:
        >>> # Single stock
        >>> apple = quick_download("AAPL")
        >>> 
        >>> # Multiple stocks
        >>> portfolio = quick_download(["AAPL", "MSFT", "GOOGL"])
    """
    fetcher = YahooFetcher()
    
    if isinstance(tickers, str):
        # Single ticker
        data = fetcher.get_stock_data(tickers, period=period)
        return data[['Adj Close']].rename(columns={'Adj Close': tickers})
    else:
        # Multiple tickers
        return fetcher.get_multiple_stocks(tickers, period=period)


# ============================================================
# PREDEFINED PORTFOLIOS
# ============================================================

# These are ready-to-use stock lists for different purposes

TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
FINANCE_STOCKS = ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC']
ENERGY_STOCKS = ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
HEALTHCARE_STOCKS = ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV']
CONSUMER_STOCKS = ['WMT', 'PG', 'KO', 'PEP', 'COST']

# Diversified portfolio
DIVERSIFIED_PORTFOLIO = [
    'AAPL', 'MSFT', 'GOOGL',  # Tech
    'JPM', 'BAC',              # Finance
    'XOM', 'CVX',              # Energy
    'JNJ', 'UNH',              # Healthcare
    'WMT', 'PG',               # Consumer
    'SPY'                      # S&P 500 ETF (benchmark)
]

# Major indices as ETFs
INDEX_ETFS = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    'DIA': 'Dow Jones',
    'IWM': 'Russell 2000',
    'VTI': 'Total US Market'
}


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    """
    This code runs when you execute this file directly:
    python yahoo_fetcher.py
    
    It won't run when you import the module.
    """
    print("=" * 60)
    print("YAHOO FETCHER DEMO")
    print("=" * 60)
    
    # Create fetcher
    fetcher = YahooFetcher()
    
    # Download single stock
    print("\n1. Downloading single stock (AAPL)...")
    apple = fetcher.get_stock_data("AAPL", period="1mo")
    print(f"   Columns: {list(apple.columns)}")
    print(f"   Shape: {apple.shape}")
    print(apple.tail())
    
    # Download multiple stocks
    print("\n2. Downloading portfolio...")
    portfolio = fetcher.get_multiple_stocks(
        ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM'],
        period="6mo"
    )
    print(portfolio.tail())
    
    # Calculate returns
    print("\n3. Calculating returns...")
    returns = fetcher.calculate_returns(portfolio, method="log")
    print(returns.tail())
    
    # Summary statistics
    print("\n4. Summary Statistics:")
    print(f"   Annualized Return: {(returns.mean() * 252 * 100).round(2)}%")
    print(f"   Annualized Vol: {(returns.std() * np.sqrt(252) * 100).round(2)}%")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE ✓")
    print("=" * 60)
