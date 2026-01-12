import pandas as pd
import yfinance as yf

from dg_visualization import GeometricAnalyzer
from dg_visualization.manifold_visualization import ManifoldVisualizer
from dg_visualization.config import DGConfig

# -------------------------
# Asset universe (heterogeneous)
# -------------------------
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "USDINR=X",
    "GC=F",
    "CL=F",
    "^NSEI"
]

START_DATE = "2020-02-01"
END_DATE   = "2020-04-30"

print("Downloading market data...")

prices = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    progress=False,
    group_by="ticker",
    auto_adjust=True
)

# -------------------------
# Build aligned price matrix
# -------------------------
adj_close = pd.DataFrame({
    t: prices[t]["Close"]
    for t in prices.columns.levels[0]
    if "Close" in prices[t]
})

adj_close = adj_close.dropna(how="any")

returns = adj_close.pct_change(fill_method=None).dropna()

print(f"Using {returns.shape[1]} assets, {returns.shape[0]} observations")

# -------------------------
# Run AEGIS analysis
# -------------------------
config = DGConfig(K=3)
analyzer = GeometricAnalyzer(config)
results = analyzer.run_analysis(returns, is_returns=True)

# -------------------------
# 3D Manifold Visualization
# -------------------------
viz = ManifoldVisualizer(results)

fig = viz.plot_market_manifold(
    height_metric="curvature",   # z-axis
    color_metric="impact",       # node color
    title="AEGIS — Market Geometry (Stress Regime)"
)

# Save interactive HTML
fig.write_html("market_geometry_3d.html")
print("Saved: market_geometry_3d.html")
