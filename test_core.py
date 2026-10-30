import numpy as np
import pandas as pd
import pytest

from backend import analysis, indicators as ind
from backend.demo_data import load_demo_ohlcv


def test_rsi_bounds_and_direction():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).iloc[-1] > 90
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down).iloc[-1] < 10


def test_atr_positive_and_ema_tracks_constant():
    df = load_demo_ohlcv("TEST", n=100)
    assert (ind.atr(df).dropna() > 0).all()
    const = pd.Series([10.0] * 50)
    assert ind.ema(const, 20).iloc[-1] == pytest.approx(10.0)


def test_features_are_causal_no_lookahead():
    """Features at bar k must be identical whether or not future bars exist."""
    df = load_demo_ohlcv("CAUSAL", n=300)
    full = analysis.features(df)
    k = 250
    part = analysis.features(df.iloc[: k + 1])
    cols = ["ema20", "ema200", "rsi", "macd_hist", "atr", "st_dir", "relvol", "hi20", "lo20", "wk_ema10"]
    for c in cols:
        a, b = full[c].iloc[k], part[c].iloc[k]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9), c


def test_simulate_stop_and_target_logic():
    n = 40
    o = np.full(n, 100.0); h = np.full(n, 101.0); l = np.full(n, 99.0); c = np.full(n, 100.0)
    A = {"open": o, "high": h, "low": l, "close": c, "atr": np.full(n, 2.0)}
    # Stop: entry=100, atr=2 -> stop=97, risk=3%, target=106. Force low below stop on bar 2.
    A["low"] = A["low"].copy(); A["low"][6] = 90.0
    t = analysis._simulate(A, 5)
    assert t["outcome"] == "stop" and t["net"] < 0
    # Target hit: high reaches 107 (>= 106) on bar 7, low never reaches the 97 stop.
    A2 = {k: v.copy() for k, v in A.items()}
    A2["low"] = np.full(n, 99.0)
    A2["high"][7] = 107.0
    t2 = analysis._simulate(A2, 5)
    assert t2["outcome"] == "target" and t2["net"] > 0
    assert t2["gross"] == pytest.approx(0.06, rel=1e-6)


def test_wilson_lower_bound_is_below_point_estimate():
    assert 0 < analysis.wilson_lower(0.7, 30) < 0.7
    assert analysis.wilson_lower(0.7, 0) == 0.0


def test_demo_snapshot_never_qualifies_anything():
    """Demo data must never validate a probability, so nothing qualifies."""
    snap = analysis.get_snapshot(refresh=True)
    assert snap["source"] == "demo"
    assert snap["opportunities"] == []
    for key, v in snap["validation"].items():
        if v["status"] != "not_implemented":
            assert v["status"] == "not_validated_demo"
    dash = analysis.dashboard()
    assert dash["top_opportunities"] == []
    assert "No qualifying opportunities" in dash["no_opportunity_reason"]


def test_fundamental_strategy_not_evaluated():
    snap = analysis.get_snapshot()
    assert snap["validation"]["fundamental_momentum"]["status"] == "not_implemented"


def test_explanation_structure_and_missing_data_not_positive():
    sym = analysis.get_snapshot()["universe"][0]["symbol"]
    e = analysis.explanation(sym)
    assert e["symbol"] == sym
    fund = [x for x in e["evidence"] if x["indicator"] == "Fundamentals"][0]
    assert fund["contribution"] == "Neutral" and fund["warning"]
    assert e["decision_summary"] in {
        "Qualifies — entry conditions confirmed", "Insufficient validation — probability cannot be trusted",
        "Watchlist only — additional evidence required", "Not qualifying — no strategy rules satisfied"}


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.app import app
    c = TestClient(app)
    assert c.get("/api/status").json()["mode"] == "DEMO"
    assert c.get("/api/dashboard").status_code == 200
    assert c.get("/api/universe").status_code == 200
    r = c.get("/api/screener", params={"min_prob": 0.7, "include_excluded": True})
    assert r.status_code == 200 and r.json()["total"] == 0
    assert c.get("/api/stocks/NOTREAL/explanation").status_code == 404
    sym = c.get("/api/universe").json()[0]["symbol"]
    assert c.get(f"/api/stocks/{sym}/explanation").json()["symbol"] == sym
    assert c.get(f"/api/stocks/{sym}/candles", params={"bars": 30}).json()[-1]["close"] > 0
    assert c.get("/").status_code == 200
