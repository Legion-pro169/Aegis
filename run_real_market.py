import numpy as np
import pandas as pd
import yfinance as yf

from dg_visualization import GeometricAnalyzer
from dg_visualization.config import DGConfig

# -----------------

TICKERS = [
    "RELIANCE.NS",
    "GC=F",
    "USDINR=X",
    "CL=F",
    "^NSEI",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS"
]

print("Downloading real market data...")

prices = yf.download(
    TICKERS,
    start="2020-02-01",
    end="2020-06-01",
    progress=False,
    group_by="ticker",
    auto_adjust=True
)

adj_close = pd.DataFrame({
    ticker: prices[ticker]["Close"]
    for ticker in prices.columns.levels[0]
    if "Close" in prices[ticker]
})

# Drop completely empty columns
adj_close = adj_close.dropna(axis=1, how="all")

# Align on common dates (IMPORTANT for heterogeneous assets)
adj_close = adj_close.dropna(how="any")

print(f"Aligned data shape: {adj_close.shape}")

if adj_close.shape[1] < 3:
    raise ValueError("Not enough assets after date alignment")

returns = adj_close.pct_change(fill_method=None).dropna()

print(f"Real data loaded: {returns.shape[1]} assets, {returns.shape[0]} observations")

config = DGConfig(K=3)
analyzer = GeometricAnalyzer(config)

results = analyzer.run_analysis(returns, is_returns=True)

print("\nFINAL PORTFOLIO WEIGHTS (REAL DATA):")
for asset, weight in results["weights"].items():
    print(asset, round(weight, 3))
