# Risk-Neutral-Density-Estimation-from-Options
Extracts SPY risk-neutral probability distributions from option prices using the Breeden–Litzenberger formula

## Overview

This project constructs **risk-neutral probability distributions for SPY** using real-time option chain data and the **Breeden–Litzenberger formula**. It pulls option quotes via the Alpaca API, preprocesses the option surface to handle market microstructure effects, and numerically recovers the implied density across strikes and maturities.

## Key Features

- **Live SPY option chain ingestion** using Alpaca (calls & puts)
- **Risk-neutral PDF extraction** via second-strike derivatives (Breeden–Litzenberger)
- **Robust preprocessing pipeline**, including:
  - Short-maturity filtering (avoids \( T \to 0 \) instability)
  - Mild bid–ask spread filtering (reduces curvature noise)
  - Interpolation onto a **uniform log-moneyness grid** for numerical stability
- **Forward-consistent discounting**, with:
  - Treasury yield interpolation (1M / 3M)
  - Safe fallback to a constant risk-free rate for reproducibility
- Produces a **clean, surface-ready DataFrame** for downstream analysis, visualization, or modeling
