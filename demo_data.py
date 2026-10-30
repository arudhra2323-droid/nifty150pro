"""DEMO DATA ONLY. Deterministic synthetic prices for testing layouts and calculations.

These are NOT real market prices and must never be presented as recommendations.
The symbol list is a demo sample, not the official Nifty 150 constituent list.
"""
import zlib

import numpy as np
import pandas as pd

DEMO_UNIVERSE = [
    ("RELIANCE", "Reliance Industries", "Energy"),
    ("HDFCBANK", "HDFC Bank", "Financials"),
    ("ICICIBANK", "ICICI Bank", "Financials"),
    ("SBIN", "State Bank of India", "Financials"),
    ("AXISBANK", "Axis Bank", "Financials"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Financials"),
    ("BAJFINANCE", "Bajaj Finance", "Financials"),
    ("INFY", "Infosys", "IT"),
    ("TCS", "Tata Consultancy Services", "IT"),
    ("HCLTECH", "HCL Technologies", "IT"),
    ("WIPRO", "Wipro", "IT"),
    ("BHARTIARTL", "Bharti Airtel", "Telecom"),
    ("ITC", "ITC", "FMCG"),
    ("HINDUNILVR", "Hindustan Unilever", "FMCG"),
    ("NESTLEIND", "Nestle India", "FMCG"),
    ("ASIANPAINT", "Asian Paints", "Consumer"),
    ("TITAN", "Titan Company", "Consumer"),
    ("MARUTI", "Maruti Suzuki", "Auto"),
    ("TATAMOTORS", "Tata Motors", "Auto"),
    ("M&M", "Mahindra & Mahindra", "Auto"),
    ("LT", "Larsen & Toubro", "Industrials"),
    ("ULTRACEMCO", "UltraTech Cement", "Materials"),
    ("GRASIM", "Grasim Industries", "Materials"),
    ("SUNPHARMA", "Sun Pharma", "Pharma"),
    ("DRREDDY", "Dr. Reddy's Labs", "Pharma"),
    ("CIPLA", "Cipla", "Pharma"),
    ("POWERGRID", "Power Grid Corp", "Utilities"),
    ("NTPC", "NTPC", "Utilities"),
    ("ONGC", "ONGC", "Energy"),
    ("COALINDIA", "Coal India", "Energy"),
]


def load_demo_ohlcv(symbol: str, n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    drift = rng.normal(0.0004, 0.0006)
    rets = drift + 0.016 * rng.standard_normal(n)
    close = 1000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + 0.004 * rng.standard_normal(n))
    hi = np.maximum(open_, close) * (1 + np.abs(0.007 * rng.standard_normal(n)))
    lo = np.minimum(open_, close) * (1 - np.abs(0.007 * rng.standard_normal(n)))
    base = rng.uniform(1e6, 5e6)
    vol = base * (1 + 0.3 * rng.random(n)) * (1 + 1.5 * np.abs(rets) / 0.016)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx
    )
