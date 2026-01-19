"""
Data Normalizer
===============

This module cleans, validates, and transforms financial data
to prepare it for analysis.

WHAT YOU'LL LEARN:
------------------
1. Data cleaning techniques
2. Handling missing values
3. Detecting outliers
4. Data transformations
5. Quality checks

WHY NORMALIZE DATA?
-------------------
Raw data from APIs can have problems:
- Missing values (market holidays, trading halts)
- Outliers (data errors, flash crashes)
- Different scales (AAPL at $150 vs BRK.A at $500,000)
- Corporate actions (splits, dividends)

Normalization fixes these issues so our analysis is accurate.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import warnings


class DataNormalizer:
    """
    Clean and normalize financial data for analysis.
    
    Methods:
        clean_prices: Handle missing values and outliers in price data
        normalize_scale: Scale data to comparable ranges
        align_dates: Align multiple time series to same dates
        detect_outliers: Find suspicious data points
        compute_returns: Calculate returns with various methods
        
    Example:
        >>> normalizer = DataNormalizer()
        >>> clean_data = normalizer.clean_prices(raw_prices)
        >>> returns = normalizer.compute_returns(clean_data)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the normalizer.
        
        Parameters:
            verbose (bool): If True, print information about data issues found
        """
        self.verbose = verbose
    
    def clean_prices(
        self,
        prices: pd.DataFrame,
        fill_method: str = "ffill",
        max_missing_pct: float = 10.0,
        remove_outliers: bool = True,
        outlier_std: float = 5.0
    ) -> pd.DataFrame:
        """
        Clean price data by handling missing values and outliers.
        
        Parameters:
            prices (pd.DataFrame): Raw price data (columns=assets, rows=dates)
            
            fill_method (str): How to handle missing values:
                - "ffill": Forward fill (use previous day's price)
                - "bfill": Backward fill (use next day's price)
                - "interpolate": Linear interpolation
                - "drop": Remove rows with missing values
                
            max_missing_pct (float): Maximum allowed missing percentage per column.
                Columns with more missing data are dropped.
                Default 10% (10.0)
                
            remove_outliers (bool): Whether to detect and fix outliers
            
            outlier_std (float): Number of standard deviations to consider outlier.
                Default 5.0 (very conservative - only extreme values)
                
        Returns:
            pd.DataFrame: Cleaned price data
            
        Example:
            >>> normalizer = DataNormalizer()
            >>> clean = normalizer.clean_prices(raw_data, fill_method="ffill")
        """
        if self.verbose:
            print("\n--- Cleaning Price Data ---")
            print(f"Input shape: {prices.shape}")
        
        df = prices.copy()
        
        # Step 1: Check for columns with too many missing values
        missing_pct = (df.isna().sum() / len(df)) * 100
        bad_columns = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        if bad_columns and self.verbose:
            print(f"⚠ Dropping columns with >{max_missing_pct}% missing: {bad_columns}")
        
        df = df.drop(columns=bad_columns)
        
        # Step 2: Handle missing values
        original_missing = df.isna().sum().sum()
        
        if fill_method == "ffill":
            # Forward fill: use the previous valid value
            # If Monday is missing, use Friday's price
            df = df.ffill()
            # Also backward fill for any remaining NaN at the start
            df = df.bfill()
            
        elif fill_method == "bfill":
            df = df.bfill()
            df = df.ffill()
            
        elif fill_method == "interpolate":
            # Linear interpolation between known values
            df = df.interpolate(method='linear')
            df = df.ffill().bfill()
            
        elif fill_method == "drop":
            df = df.dropna()
        
        if self.verbose and original_missing > 0:
            print(f"  Filled {original_missing} missing values using {fill_method}")
        
        # Step 3: Handle outliers
        if remove_outliers:
            outliers_fixed = 0
            
            for col in df.columns:
                # Calculate returns to detect outliers
                # (A single-day 50% move is likely an error)
                returns = df[col].pct_change()
                
                # Find outliers using z-score method
                mean_ret = returns.mean()
                std_ret = returns.std()
                
                if std_ret > 0:
                    z_scores = np.abs((returns - mean_ret) / std_ret)
                    outlier_mask = z_scores > outlier_std
                    
                    if outlier_mask.any():
                        # Replace outliers with interpolated values
                        outlier_indices = returns[outlier_mask].index
                        df.loc[outlier_indices, col] = np.nan
                        df[col] = df[col].interpolate()
                        outliers_fixed += outlier_mask.sum()
            
            if self.verbose and outliers_fixed > 0:
                print(f"  Fixed {outliers_fixed} outlier values")
        
        # Step 4: Final cleanup
        df = df.dropna()  # Remove any remaining NaN rows
        
        if self.verbose:
            print(f"Output shape: {df.shape}")
        
        return df
    
    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log",
        period: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters:
            prices (pd.DataFrame): Price data
            
            method (str): Return calculation method:
                - "log": Log returns ln(P_t / P_{t-1})
                  Best for statistical analysis, additive over time
                  
                - "simple": Simple returns (P_t - P_{t-1}) / P_{t-1}
                  Easier to interpret (actual percentage gain)
                  
            period (int): Return period in days
                - 1: Daily returns (default)
                - 5: Weekly returns
                - 21: Monthly returns
                
        Returns:
            pd.DataFrame: Returns data
            
        Example:
            >>> returns = normalizer.compute_returns(prices, method="log")
            >>> monthly_returns = normalizer.compute_returns(prices, period=21)
        """
        if method == "log":
            returns = np.log(prices / prices.shift(period))
        elif method == "simple":
            returns = prices.pct_change(periods=period)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns.dropna()
    
    def normalize_scale(
        self,
        data: pd.DataFrame,
        method: str = "zscore"
    ) -> pd.DataFrame:
        """
        Normalize data to a common scale.
        
        Parameters:
            data (pd.DataFrame): Input data
            
            method (str): Normalization method:
                - "zscore": (x - mean) / std
                  Centers at 0, std = 1
                  
                - "minmax": (x - min) / (max - min)
                  Scales to [0, 1] range
                  
                - "rebase": x / x[0] * 100
                  Rebases all series to start at 100
                  Good for comparing performance
                  
        Returns:
            pd.DataFrame: Normalized data
            
        Example:
            >>> # Compare performance of different stocks
            >>> rebased = normalizer.normalize_scale(prices, method="rebase")
            >>> # All stocks start at 100, making comparison easy
        """
        if method == "zscore":
            return (data - data.mean()) / data.std()
            
        elif method == "minmax":
            return (data - data.min()) / (data.max() - data.min())
            
        elif method == "rebase":
            return data / data.iloc[0] * 100
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def align_dates(
        self,
        dataframes: List[pd.DataFrame],
        how: str = "inner"
    ) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to the same dates.
        
        Parameters:
            dataframes (List[pd.DataFrame]): List of DataFrames with DatetimeIndex
            
            how (str): How to align:
                - "inner": Keep only dates present in ALL DataFrames
                - "outer": Keep all dates, fill missing with NaN
                
        Returns:
            List[pd.DataFrame]: Aligned DataFrames
        """
        if how == "inner":
            # Find dates present in all DataFrames
            common_dates = dataframes[0].index
            for df in dataframes[1:]:
                common_dates = common_dates.intersection(df.index)
            
            return [df.loc[common_dates] for df in dataframes]
            
        elif how == "outer":
            # Find all unique dates
            all_dates = dataframes[0].index
            for df in dataframes[1:]:
                all_dates = all_dates.union(df.index)
            
            return [df.reindex(all_dates) for df in dataframes]
        
        else:
            raise ValueError(f"Unknown method: {how}")
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers in the data.
        
        Parameters:
            data (pd.DataFrame): Input data
            
            method (str): Detection method:
                - "zscore": Based on standard deviations from mean
                - "iqr": Based on interquartile range
                
            threshold (float): 
                - For zscore: number of std deviations (default 3)
                - For iqr: multiplier for IQR (default 1.5)
                
        Returns:
            pd.DataFrame: Boolean DataFrame (True = outlier)
        """
        if method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
            
        elif method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (data < lower) | (data > upper)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_summary_stats(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for returns.
        
        Parameters:
            returns (pd.DataFrame): Returns data (daily)
            
        Returns:
            pd.DataFrame: Summary statistics per asset
        """
        # Annualization factor (252 trading days per year)
        ann_factor = 252
        
        stats = pd.DataFrame({
            'Mean (Daily)': returns.mean(),
            'Std (Daily)': returns.std(),
            'Mean (Annual)': returns.mean() * ann_factor,
            'Std (Annual)': returns.std() * np.sqrt(ann_factor),
            'Sharpe Ratio': (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)),
            'Min': returns.min(),
            'Max': returns.max(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        })
        
        return stats.round(4)
    
    def check_data_quality(self, data: pd.DataFrame) -> dict:
        """
        Comprehensive data quality check.
        
        Parameters:
            data (pd.DataFrame): Data to check
            
        Returns:
            dict: Quality metrics and issues found
        """
        report = {
            'shape': data.shape,
            'date_range': (data.index[0], data.index[-1]),
            'missing_values': data.isna().sum().to_dict(),
            'total_missing': data.isna().sum().sum(),
            'duplicated_rows': data.index.duplicated().sum(),
            'constant_columns': [],
            'issues': []
        }
        
        # Check for constant columns (no variation)
        for col in data.columns:
            if data[col].std() == 0:
                report['constant_columns'].append(col)
                report['issues'].append(f"Column {col} has no variation")
        
        # Check for negative prices (shouldn't happen)
        if (data < 0).any().any():
            report['issues'].append("Negative values detected")
        
        # Check for gaps in dates
        expected_days = pd.date_range(data.index[0], data.index[-1], freq='B')
        missing_days = len(expected_days) - len(data)
        if missing_days > 0:
            report['missing_trading_days'] = missing_days
        
        return report


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_clean(prices: pd.DataFrame) -> pd.DataFrame:
    """Quick cleaning with default settings."""
    normalizer = DataNormalizer(verbose=False)
    return normalizer.clean_prices(prices)


def prices_to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Quick conversion from prices to returns."""
    normalizer = DataNormalizer(verbose=False)
    clean = normalizer.clean_prices(prices)
    return normalizer.compute_returns(clean, method=method)
