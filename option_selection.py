from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest

from spot_price import get_spy_spot


DEFAULT_RF = 0.045  # 4.5% annualized fallback risk-free rate, in case there are problems reaching FRED website (there was)


# Parse OCC-style option symbol
# Example: SPY260128P00845000
def _parse_alpaca_option_symbol(sym: str) -> Optional[Tuple[str, date, str, float]]:
    i = 0
    while i < len(sym) and not sym[i].isdigit():
        i += 1
    if i == 0 or i + 15 > len(sym):
        return None

    underlying = sym[:i]
    yymmdd = sym[i:i + 6]
    right = sym[i + 6:i + 7]
    strike_str = sym[i + 7:i + 15]

    if right not in ("C", "P"):
        return None

    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    exp = date(2000 + yy, mm, dd)

    strike = int(strike_str) / 1000.0
    return underlying, exp, right, strike


# Risk-free rate (Treasury proxy via FRED) with fallback
def _fred_latest_percent(series_id: str, timeout: int = 10) -> float:
    """
    Returns the latest non-missing value for the requested FRED series as a PERCENT
    (e.g., 5.12 for 5.12%).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    lines = r.text.strip().splitlines()
    for line in reversed(lines[1:]):
        _, val = line.split(",")
        if val and val != ".":
            return float(val)

    raise RuntimeError(f"No valid FRED data for {series_id}")


def _risk_free_rate_decimal_for_exp(exp: date, today: date, timeout: int = 10) -> float:
    """
    Returns annualized risk-free rate as a DECIMAL (e.g., 0.045 for 4.5%).
    Tries FRED 1M/3M yields; falls back to DEFAULT_RF if unreachable/slow/etc.
    """
    days = (exp - today).days
    if days <= 0:
        return 0.0

    try:
        y1m = _fred_latest_percent("DGS1MO", timeout) / 100.0
        y3m = _fred_latest_percent("DGS3MO", timeout) / 100.0

        if days <= 30:
            return y1m
        if days >= 90:
            return y3m

        w = (days - 30.0) / 60.0
        return (1.0 - w) * y1m + w * y3m

    except (requests.RequestException, RuntimeError, ValueError) as e:
        # Network/timeouts, HTTP errors, parsing errors, missing series values, etc.
        print(f"[WARN] FRED unavailable ({e}). Using DEFAULT_RF={DEFAULT_RF:.3f}")
        return DEFAULT_RF


#Retreives data
def get_spy_surface_inputs(
    opt_client: OptionHistoricalDataClient,
    stock_client: StockHistoricalDataClient,
    *,
    days_forward: int = 45,
    strikes_per_exp: int = 30,
    strike_band: float = 50.0,
    fred_timeout: int = 10,
) -> pd.DataFrame:

    #Need spot price to center contracts around
    spot = float(get_spy_spot(stock_client))

    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=days_forward)

    
    chain = opt_client.get_option_chain(
        OptionChainRequest(
            underlying_symbol="SPY",
            expiration_date_gte=today,
            expiration_date_lte=cutoff,
            strike_price_gte=str(spot - strike_band),
            strike_price_lte=str(spot + strike_band),
        )
    )

    # exp -> strike -> {C,P}
    by_exp: Dict[date, Dict[float, Dict[str, str]]] = {}

    for sym in chain.keys():
        parsed = _parse_alpaca_option_symbol(sym)
        if not parsed:
            continue

        underlying, exp, right, strike = parsed
        if underlying != "SPY":
            continue

        by_exp.setdefault(exp, {}).setdefault(strike, {})[right] = sym

    # Select nearest strikes
    rows: List[dict] = []
    symbols: List[str] = []

    for exp in sorted(by_exp.keys()):
        strike_map = by_exp[exp]
        strikes = sorted(strike_map.keys(), key=lambda k: abs(k - spot))
        nearest = strikes[:strikes_per_exp]

        r_annual = _risk_free_rate_decimal_for_exp(exp, today, fred_timeout)
        T_years = (exp - today).days / 365.0

        for k in sorted(nearest):
            c = strike_map[k].get("C")
            p = strike_map[k].get("P")

            if c:
                symbols.append(c)
            if p:
                symbols.append(p)

            rows.append({
                "expiration": exp,
                "T_years": T_years,
                "strike": k,
                "call_symbol": c,
                "put_symbol": p,
                "spot": spot,
                "r_annual": r_annual,
            })

    if not rows:
        return pd.DataFrame()

    # SNAPSHOT FIX: batch ≤ 100, Alpaca free API limits you to 100 per querry
    symbols = list(dict.fromkeys(symbols))  # dedupe

    def _chunks(lst, n=100):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    snaps = {}
    for batch in _chunks(symbols):
        snaps.update(
            opt_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=batch)
            )
        )

    def _bid_ask(sym: Optional[str]):
        if not sym:
            return None, None
        s = snaps.get(sym)
        if not s or not s.latest_quote:
            return None, None
        q = s.latest_quote
        return (
            float(q.bid_price) if q.bid_price is not None else None,
            float(q.ask_price) if q.ask_price is not None else None,
        )

    # Attach quotes
    for row in rows:
        cb, ca = _bid_ask(row["call_symbol"])
        pb, pa = _bid_ask(row["put_symbol"])
        row["call_bid"] = cb
        row["call_ask"] = ca
        row["put_bid"] = pb
        row["put_ask"] = pa

    df = pd.DataFrame(rows)
    return df.sort_values(["expiration", "strike"]).reset_index(drop=True)
