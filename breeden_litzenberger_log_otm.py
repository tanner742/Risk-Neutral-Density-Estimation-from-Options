
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def compute_rn_density_log_otm(
    df_exp: pd.DataFrame,
    *,
    trim_lo: float = 0.01,
    trim_hi: float = 0.99,
    n_grid: int = 401,
    smooth_window: int = 11,
    clip_negative: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build risk-neutral PDF/CDF for ONE expiration using:
      - Fix #2: OTM-only + put-call parity (uses puts for K<F, calls for K>=F)
      - Fix #1: uniform log-moneyness grid k=ln(K/F), chain-rule density

    Input df_exp must be a single-expiration slice from get_spy_surface_inputs(), containing:
      strike, call_bid, call_ask, put_bid, put_ask, spot, T_years, r_annual

    Returns:
      out_df with columns: strike, q_density, cdf
      meta dict with: exp, spot, F, D_fwd, T, r, K_atm, total_mass_raw, area_trim
    """

    g = df_exp.copy()

    # numeric coercion
    for col in ["strike", "call_bid", "call_ask", "put_bid", "put_ask", "spot", "T_years", "r_annual"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    g = g.dropna(subset=["strike", "spot", "T_years", "r_annual"])
    g = g.sort_values("strike").reset_index(drop=True)

    if g.empty or len(g) < 20:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    exp = g["expiration"].iloc[0] if "expiration" in g.columns else None
    spot = float(g["spot"].iloc[0])
    T = float(g["T_years"].iloc[0])
    r = float(g["r_annual"].iloc[0])

    # quote validity masks
    call_ok = (
        g["call_bid"].notna() & g["call_ask"].notna() &
        (g["call_bid"] > 0) & (g["call_ask"] > 0) &
        (g["call_ask"] >= g["call_bid"])
    )
    put_ok = (
        g["put_bid"].notna() & g["put_ask"].notna() &
        (g["put_bid"] > 0) & (g["put_ask"] > 0) &
        (g["put_ask"] >= g["put_bid"])
    )

    g = g[call_ok | put_ok].copy()
    if g.empty or len(g) < 20:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    g["call_mid"] = 0.5 * (g["call_bid"] + g["call_ask"])
    g["put_mid"] = 0.5 * (g["put_bid"] + g["put_ask"])

    # mild spread filter, prevents second derivative from being dominated by microstructure noise 
    g["call_rel_spr"] = (g["call_ask"] - g["call_bid"]) / g["call_mid"]
    g["put_rel_spr"] = (g["put_ask"] - g["put_bid"]) / g["put_mid"]
    g.loc[~call_ok, "call_rel_spr"] = np.nan
    g.loc[~put_ok, "put_rel_spr"] = np.nan
    g = g[
        (g["call_rel_spr"].isna() | (g["call_rel_spr"] < 0.35)) &
        (g["put_rel_spr"].isna() | (g["put_rel_spr"] < 0.35))
    ].copy()

    if g.empty or len(g) < 20:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    #Estimate discounted forward D_fwd = S e^{-qT} from near-ATM strike with both quotes
    both = g.dropna(subset=["call_mid", "put_mid"]).copy()
    if both.empty:
        # cannot do parity reliably without both sides somewhere
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    i_atm = (both["strike"] - spot).abs().idxmin()
    K_atm = float(both.loc[i_atm, "strike"])
    C_atm = float(both.loc[i_atm, "call_mid"])
    P_atm = float(both.loc[i_atm, "put_mid"])

    D_fwd = C_atm - P_atm + K_atm * np.exp(-r * T)
    if not np.isfinite(D_fwd) or D_fwd <= 0:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    F = D_fwd * np.exp(r * T)

    #Build synthetic call curve using OTM-only selection + parity
    K = g["strike"].to_numpy(float)
    call_mid = g["call_mid"].to_numpy(float)
    put_mid = g["put_mid"].to_numpy(float)
    discK = K * np.exp(-r * T)

    C_synth = np.full_like(K, np.nan, dtype=float)
    mask_call = (K >= F) & np.isfinite(call_mid)
    C_synth[mask_call] = call_mid[mask_call]
    mask_put = (K < F) & np.isfinite(put_mid)
    C_synth[mask_put] = put_mid[mask_put] + D_fwd - discK[mask_put]

    keep = np.isfinite(C_synth)
    K = K[keep]
    C_synth = C_synth[keep]
    if len(K) < 25:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    # de-dup strikes
    tmp = pd.DataFrame({"K": K, "C": C_synth}).groupby("K", as_index=False)["C"].mean().sort_values("K")
    K = tmp["K"].to_numpy(float)
    C_synth = tmp["C"].to_numpy(float)

    # uniform log-moneyness grid, uniform spacing in log-moneyness makes the second derivative numerically stable, and economically meaningful
    k_obs = np.log(K / F)

    # trim observed k-range to reduce boundary derivative blow-ups
    k_lo = np.quantile(k_obs, 0.02)
    k_hi = np.quantile(k_obs, 0.98)
    if not np.isfinite(k_lo) or not np.isfinite(k_hi) or k_hi <= k_lo:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    k_grid = np.linspace(k_lo, k_hi, int(n_grid))
    K_grid = F * np.exp(k_grid)

    # interpolate call prices to k-grid
    C_grid = np.interp(k_grid, k_obs, C_synth)

    # smooth in k-space (moving average)
    w = int(smooth_window)
    if w >= 3 and w % 2 == 1 and len(C_grid) > w:
        kernel = np.ones(w) / w
        C_pad = np.pad(C_grid, (w // 2, w // 2), mode="edge")
        C_grid = np.convolve(C_pad, kernel, mode="valid")

    # derivatives wrt k
    dC_dk = np.gradient(C_grid, k_grid)
    d2C_dk2 = np.gradient(dC_dk, k_grid)

    # chain rule: d2C/dK2 = (d2C/dk2 - dC/dk) / K^2
    d2C_dK2 = (d2C_dk2 - dC_dk) / (K_grid ** 2)

    q = np.exp(r * T) * d2C_dK2
    if clip_negative:
        q = np.where(np.isfinite(q), np.maximum(q, 0.0), np.nan)

    mask = np.isfinite(q)
    Kp = K_grid[mask]
    qp = q[mask]

    if len(Kp) < 30:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    # CDF for trimming
    dKp = np.diff(Kp, prepend=Kp[0])
    cdf_raw = np.cumsum(qp * dKp)
    total_mass = cdf_raw[-1] if len(cdf_raw) else 0.0
    if total_mass <= 0 or not np.isfinite(total_mass):
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    cdf_unit = cdf_raw / total_mass
    keep2 = (cdf_unit >= trim_lo) & (cdf_unit <= trim_hi)
    Kp = Kp[keep2]
    qp = qp[keep2]
    if len(Kp) < 30:
        return pd.DataFrame(columns=["strike", "q_density", "cdf"]), {}

    # normalize on trimmed region
    area_trim = np.trapezoid(qp, Kp)
    if area_trim > 0 and np.isfinite(area_trim):
        qn = qp / area_trim
    else:
        qn = qp

    # CDF on trimmed region (0..1)
    dKp2 = np.diff(Kp, prepend=Kp[0])
    cdf = np.cumsum(qn * dKp2)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    cdf = np.clip(cdf, 0.0, 1.0)

    out = pd.DataFrame({
        "strike": Kp,
        "q_density": qn,
        "cdf": cdf,
    })

    meta = {
        "expiration": exp,
        "spot": spot,
        "T_years": T,
        "r_annual": r,
        "K_atm": K_atm,
        "D_fwd": D_fwd,
        "forward_F": F,
        "total_mass_raw": float(total_mass),
        "area_trim": float(area_trim) if np.isfinite(area_trim) else np.nan,
        "trim_lo": trim_lo,
        "trim_hi": trim_hi,
    }

    return out, meta
