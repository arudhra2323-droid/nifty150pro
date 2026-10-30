"""Provider adapter layer. Calculation code never imports a vendor directly.

Set MARKET_DATA_PROVIDER in .env:
  demo  -> synthetic, clearly labeled DEMO data (default)
  live  -> not yet implemented; raises a clear error instead of silently falling back.
"""
import os

from .demo_data import DEMO_UNIVERSE, load_demo_ohlcv


def provider_name() -> str:
    return os.getenv("MARKET_DATA_PROVIDER", "demo").lower()


def load() -> tuple[list[dict], dict, str]:
    """Return (universe_rows, {symbol: ohlcv_df}, source_label)."""
    p = provider_name()
    if p == "demo":
        universe = [{"symbol": s, "name": n, "sector": sec} for s, n, sec in DEMO_UNIVERSE]
        frames = {u["symbol"]: load_demo_ohlcv(u["symbol"]) for u in universe}
        return universe, frames, "demo"
    raise NotImplementedError(
        f"Market data provider '{p}' is not implemented yet. "
        "Write an adapter in backend/market_data.py or set MARKET_DATA_PROVIDER=demo."
    )
