# Aegis Architecture Guide

## Overview

Aegis follows a **modular, layered architecture** designed for:

1. **Performance** — C++ for computation-heavy tasks
2. **Flexibility** — Python for rapid prototyping and ML
3. **Enterprise Readiness** — Java for robust, scalable services
4. **Extensibility** — Plugin system for custom models

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Java Dashboard │  │  REST API       │  │  CLI Tools  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Aegis Intelligence (AI Copilot)                        ││
│  │  - Query interpretation                                 ││
│  │  - Result explanation                                   ││
│  │  - Natural language interface                           ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    ANALYTICS LAYER                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ Geometric  │ │    AI      │ │    LLM     │ │ Systemic │ │
│  │ Analysis   │ │ Optimizer  │ │ Analysis   │ │   Risk   │ │
│  │  Engine    │ │            │ │            │ │ Monitor  │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    COMPUTATION LAYER                        │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │   Monte Carlo Engine    │  │    PDE Pricing Engine   │  │
│  │        (C++)            │  │         (C++)           │  │
│  │  - VaR, CVaR, ES        │  │  - Black-Scholes        │  │
│  │  - Parallel simulation  │  │  - Heston               │  │
│  │  - Stress testing       │  │  - Local volatility     │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │  Market    │ │ Historical │ │Alternative │ │  Cache   │ │
│  │   Feeds    │ │    Data    │ │   Data     │ │ (Redis)  │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Secure Gateway: Auth • RBAC • Encryption • Audit      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Module Details

### 1. Monte Carlo Risk Engine (C++)

**Purpose:** High-performance risk calculations

**Capabilities:**
- Value-at-Risk (VaR) at arbitrary confidence levels
- Conditional VaR (Expected Shortfall)
- Stress testing and scenario analysis
- Greeks calculation via bump-and-reprice

**Performance Target:** 1M+ simulations/second on modern hardware

**Key Files:**
```
src/cpp/monte_carlo/
├── CMakeLists.txt
├── include/
│   ├── monte_carlo_engine.hpp
│   ├── random_generator.hpp
│   └── portfolio.hpp
└── src/
    ├── monte_carlo_engine.cpp
    └── main.cpp
```

---

### 2. PDE Pricing Engine (C++)

**Purpose:** Derivatives pricing using partial differential equations

**Models Supported:**
- Black-Scholes (constant volatility)
- Heston (stochastic volatility)
- Local volatility (calibrated surface)

**Numerical Methods:**
- Explicit finite difference
- Implicit finite difference
- Crank-Nicolson (second-order accurate)

**Key Files:**
```
src/cpp/pde_pricing/
├── CMakeLists.txt
├── include/
│   ├── black_scholes.hpp
│   ├── heston.hpp
│   └── finite_difference.hpp
└── src/
    └── ...
```

---

### 3. Differential Geometry Engine (Python)

**Purpose:** Visualize market structure as geometric manifold

**Mathematical Foundation:**
- **Ollivier-Ricci Curvature:** Measures how "clustered" vs "bridging" each asset is
- **Sectional Curvature:** Local geometry via geodesic triangle defects
- **Wasserstein Distance:** Optimal transport for comparing probability distributions

**Key Insight:** During market stress, correlations increase → network tightens → curvature rises. This is a leading indicator of fragility.

**Key Files:**
```
src/python/dg_visualization/
├── __init__.py
├── geometric_analyzer.py      # Core analysis
├── manifold_visualization.py  # 3D plotting
├── curvature.py              # Ricci & sectional curvature
├── network.py                # Graph construction
└── portfolio_weights.py      # Weight optimization
```

---

### 4. AI Portfolio Optimizer (Python)

**Purpose:** Intelligent portfolio allocation

**Techniques:**
- Reinforcement Learning (PPO, SAC algorithms)
- Bayesian Optimization for hyperparameters
- Multi-objective optimization (risk vs return)

**Key Files:**
```
src/python/ai_optimizer/
├── __init__.py
├── rl_agent.py
├── bayesian_optimizer.py
└── constraints.py
```

---

### 5. LLM Financial Intelligence (Python)

**Purpose:** Extract insights from unstructured text

**Capabilities:**
- Sentiment analysis of earnings calls
- Risk indicator extraction from 10-K filings
- News impact scoring
- Anomaly detection (fraud signals)

**Key Files:**
```
src/python/llm_intelligence/
├── __init__.py
├── sentiment_analyzer.py
├── document_parser.py
└── anomaly_detector.py
```

---

### 6. Market Data Pipeline (Python)

**Purpose:** Unified data ingestion and normalization

**Data Sources:**
- Real-time: Yahoo Finance, Alpha Vantage
- Historical: CRSP, Quandl
- Alternative: ESG scores, sentiment indices

**Key Files:**
```
src/python/data_pipeline/
├── __init__.py
├── fetchers/
│   ├── yahoo_fetcher.py
│   └── alpha_vantage_fetcher.py
├── normalizers/
│   └── price_normalizer.py
└── cache.py
```

---

### 7. Quantitative Dashboard (Java)

**Purpose:** Real-time visualization and monitoring

**Features:**
- Live KPI display (Sharpe, Beta, VaR, Drawdown)
- Alert system for threshold breaches
- Historical trend analysis
- Export to PDF/Excel

**Technology:** JavaFX + Spring Boot

**Key Files:**
```
src/java/dashboard/
├── build.gradle
├── src/main/java/com/aegis/dashboard/
│   ├── Application.java
│   ├── controllers/
│   ├── services/
│   └── views/
└── src/main/resources/
```

---

### 8. Secure Gateway (Java)

**Purpose:** Enterprise security and access control

**Features:**
- Role-Based Access Control (RBAC)
- JWT authentication
- Data encryption (AES-256)
- Audit logging
- Multi-tenant isolation

**Key Files:**
```
src/java/gateway/
├── build.gradle
└── src/main/java/com/aegis/gateway/
    ├── security/
    ├── audit/
    └── config/
```

---

## Data Flow

```
[Market Data Sources]
        │
        ▼
┌───────────────────┐
│   Data Pipeline   │ ──→ Normalize, validate, cache
└───────────────────┘
        │
        ├──────────────────┬────────────────────┐
        ▼                  ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Geometric   │   │     AI       │   │    LLM       │
│   Engine     │   │  Optimizer   │   │  Analysis    │
└──────────────┘   └──────────────┘   └──────────────┘
        │                  │                    │
        └──────────┬───────┴────────────────────┘
                   ▼
        ┌─────────────────────┐
        │   Risk Aggregator   │
        └─────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Monte Carlo  │      │  Dashboard   │
│   Engine     │      │              │
└──────────────┘      └──────────────┘
```

---

## Inter-Module Communication

| From | To | Method | Data Format |
|------|----|--------|-------------|
| Data Pipeline | All Python modules | Direct import | Pandas DataFrame |
| Python modules | C++ engines | Python bindings (pybind11) | NumPy arrays |
| C++ engines | Dashboard | REST API | JSON |
| All modules | Gateway | HTTP/gRPC | Protobuf/JSON |

---

## Deployment Options

### Development
```bash
# Run all Python components
python -m aegis.main

# Run C++ engines separately
./build/monte_carlo_engine
./build/pde_engine
```

### Production (Docker)
```bash
docker-compose up -d
```

### Cloud (Kubernetes)
```bash
kubectl apply -f k8s/
```

---

## Performance Considerations

1. **C++ Engines:** Use OpenMP for CPU parallelism, optional CUDA for GPU
2. **Python Analytics:** Vectorized NumPy operations, avoid loops
3. **Data Caching:** Redis for frequently accessed data
4. **Dashboard:** WebSocket for real-time updates

---

## Security Model

1. **Authentication:** OAuth 2.0 / JWT tokens
2. **Authorization:** RBAC with fine-grained permissions
3. **Encryption:** TLS 1.3 in transit, AES-256 at rest
4. **Audit:** Complete request logging with tamper-proof storage

---

## Extending Aegis

### Adding a New Python Module

1. Create directory under `src/python/`
2. Implement with `__init__.py` exposing public API
3. Add tests under `tests/`
4. Document in `docs/`

### Adding a Custom Model (Plugin)

```python
from aegis.plugins import BasePlugin

class MyCustomModel(BasePlugin):
    def analyze(self, data):
        # Your logic here
        return results
```

Register in `config/plugins.yaml`:
```yaml
plugins:
  - name: my_custom_model
    class: my_module.MyCustomModel
    enabled: true
```
