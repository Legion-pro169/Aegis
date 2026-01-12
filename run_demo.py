
# Now imports will work


import numpy as np
import pandas as pd

from dg_visualization import GeometricAnalyzer
from dg_visualization.config import DGConfig

np.random.seed(42)

returns = pd.DataFrame(
    np.random.randn(300, 6) * 0.01,
    columns=[f"Asset_{i}" for i in range(6)]
)

config = DGConfig(K=2)  # 👈 CRITICAL FIX
analyzer = GeometricAnalyzer(config)

results = analyzer.run_analysis(returns, is_returns=True)

print("\nFinal portfolio weights:")
for asset, weight in results["weights"].items():
    print(asset, round(weight, 3))

