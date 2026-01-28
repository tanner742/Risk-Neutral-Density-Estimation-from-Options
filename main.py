
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient

from option_selection import get_spy_surface_inputs
from breeden_litzenberger_log_otm import compute_rn_density_log_otm


def main():
    load_dotenv()
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")

    opt_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
    stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    #Pull option chain data 
    df = get_spy_surface_inputs(
        opt_client,
        stock_client,
        days_forward=45,
        strikes_per_exp=180,
        strike_band=400.0,
    )

    print("\nSurface inputs preview:")
    print(df.head(10))
    print("\nrows:", len(df))

    #Filter out short term matruites, Breeden-Litzenberger fails to be accurate round matuiry near 0
    #As maturity shrinks the second derivative go tomards an infinite spike
    min_T_years = 5 / 365
    df_valid = df[df["T_years"] > min_T_years].copy()
    if df_valid.empty:
        print("\nNo maturities with T_years > 5/365.")
        return

    #Choose one date to create pdf and cdf for
    expirations = sorted(df_valid["expiration"].unique())
    exp = expirations[0]
    df_exp = df_valid[df_valid["expiration"] == exp].copy()

    rn_df, meta = compute_rn_density_log_otm(
        df_exp,
        trim_lo=0.01,
        trim_hi=0.99,
        n_grid=401,
        smooth_window=11,
        clip_negative=True,
    )

    if rn_df.empty:
        print(f"\nNo RN density returned for exp={exp}.")
        return

    print("\nMeta:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    #Plotting PDF results
    spot = float(meta["spot"])
    F = float(meta["forward_F"])

    plt.figure(figsize=(10, 5))
    plt.plot(rn_df["strike"], rn_df["q_density"], linewidth=2,
             label=f"RN PDF (log+OTM, exp={meta['expiration']})")
    plt.axvline(spot, linestyle="--", label="Spot")
    plt.axvline(F, linestyle=":", label="Forward (est)")
    plt.xlabel("Strike (K)")
    plt.ylabel("Risk-neutral density q(K) (trimmed+normalized)")
    plt.title("SPY Risk-Neutral Density via Breeden–Litzenberger (OTM + log-moneyness)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    pdf_path = f"rn_pdf_{meta['expiration']}_log_otm.png"
    plt.savefig(pdf_path, dpi=150)
    print(f"\nSaved PDF plot: {pdf_path}")
    plt.close()

    #Plotting CDF results
    plt.figure(figsize=(10, 5))
    plt.plot(rn_df["strike"], rn_df["cdf"], linewidth=2,
             label=f"RN CDF (log+OTM, exp={meta['expiration']})")
    plt.axvline(spot, linestyle="--", label="Spot")
    plt.axvline(F, linestyle=":", label="Forward (est)")
    plt.xlabel("Strike (K)")
    plt.ylabel("Risk-neutral CDF  P_Q(S_T ≤ K)")
    plt.title("SPY Risk-Neutral CDF vs Strike (OTM + log-moneyness)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    cdf_path = f"rn_cdf_{meta['expiration']}_log_otm.png"
    plt.savefig(cdf_path, dpi=150)
    print(f"Saved CDF plot: {cdf_path}")
    plt.close()


if __name__ == "__main__":
    main()
