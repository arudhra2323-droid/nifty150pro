"""Core analysis pipeline: features -> strategy signals -> pooled backtest -> validation -> ranking."""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import indicators as ind
from . import market_data
from .strategies import REGISTRY

COST_ROUNDTRIP = 0.0025      # 0.25% assumed round-trip cost and slippage (conservative placeholder)
HORIZON = 15                 # max holding period in sessions (upper end of 2-15 day default)
ATR_STOP_MULT = 1.5
R_MULTIPLE = 2.0
MIN_PROB = 0.70              # default qualification threshold
MIN_OOS_TRADES = 30          # minimum out-of-sample trades before a probability is shown as validated
OOS_FRACTION = 0.6           # first 60% in-sample, last 40% out-of-sample (chronological)
WARMUP = 200

FEATURE_COLS = ["open", "high", "low", "close", "volume", "ema20", "ema50", "ema200", "rsi",
                "rsi_prev", "macd_line", "macd_sig", "macd_hist", "macd_hist_prev", "atr",
                "st_dir", "relvol", "hi20", "lo20", "wk_ema10"]

_state = {"snap": None, "built_at": None}


def features(df: pd.DataFrame) -> pd.DataFrame:
    F = df.copy()
    c = F["close"]
    F["ema20"], F["ema50"], F["ema200"] = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    F["rsi"] = ind.rsi(c)
    F["rsi_prev"] = F["rsi"].shift(1)
    line, sig, hist = ind.macd(c)
    F["macd_line"], F["macd_sig"], F["macd_hist"] = line, sig, hist
    F["macd_hist_prev"] = hist.shift(1)
    F["atr"] = ind.atr(F)
    F["st_dir"] = ind.supertrend(F).dir
    F["relvol"] = F["volume"] / F["volume"].rolling(20).mean()
    F["hi20"] = F["high"].shift(1).rolling(20).max()   # prior 20 bars, excludes current bar
    F["lo20"] = F["low"].shift(1).rolling(20).min()
    # Weekly EMA10 shifted one week so a daily bar never sees its own week's close (no look-ahead)
    weekly = c.resample("W-FRI").last().ewm(span=10, adjust=False).mean().shift(1)
    F["wk_ema10"] = weekly.reindex(F.index, method="ffill")
    return F


def _arrays(F: pd.DataFrame) -> dict:
    return {k: F[k].to_numpy(dtype=float) for k in FEATURE_COLS}


def _simulate(A: dict, i: int):
    o, h, l, c = A["open"], A["high"], A["low"], A["close"]
    entry = o[i + 1]
    atr = A["atr"][i]
    if not (np.isfinite(atr) and np.isfinite(entry) and entry > 0):
        return None
    stop = entry - ATR_STOP_MULT * atr
    if stop <= 0:
        return None
    risk = (entry - stop) / entry
    target = entry * (1 + R_MULTIPLE * risk)
    for j in range(i + 1, i + 1 + HORIZON):
        if l[j] <= stop:                       # stop checked first (conservative when both hit)
            return _trade(i + 1, j, entry, min(stop, o[j]), "stop", risk)
        if h[j] >= target:
            return _trade(i + 1, j, entry, max(target, o[j]), "target", risk)
    j = i + HORIZON
    return _trade(i + 1, j, entry, c[j], "time", risk)


def _trade(ei, xi, entry, exit_p, outcome, risk):
    gross = exit_p / entry - 1
    net = gross - COST_ROUNDTRIP
    return {"entry_idx": ei, "exit_idx": xi, "gross": gross, "net": net,
            "win": net > 0, "outcome": outcome, "risk": risk}


def _backtest_symbol(key: str, A: dict, n: int) -> list[dict]:
    fn = REGISTRY[key]["fn"]
    split = int(n * OOS_FRACTION)
    trades, i = [], WARMUP
    while i <= n - HORIZON - 2:
        triggered, _ = fn(A, i)
        if triggered:
            t = _simulate(A, i)
            if t:
                t["oos"] = t["entry_idx"] >= split
                trades.append(t)
                i = t["exit_idx"] + 1          # no overlapping trades per symbol
                continue
        i += 1
    return trades


def _stats(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": None, "expectancy_net": None, "avg_win": None,
                "avg_loss": None, "profit_factor": None, "max_drawdown": None,
                "target_hit_rate": None, "stop_hit_rate": None, "avg_hold_sessions": None}
    nets = np.array([t["net"] for t in trades])
    gains, losses = nets[nets > 0], nets[nets <= 0]
    eq = np.cumsum(nets)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    pf = float(gains.sum() / abs(losses.sum())) if losses.sum() != 0 else None
    return {
        "n": n,
        "win_rate": float(np.mean(nets > 0)),
        "expectancy_net": float(nets.mean()),
        "avg_win": float(gains.mean()) if len(gains) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": pf,
        "max_drawdown": dd,  # on cumulative sum of trade returns, in fraction
        "target_hit_rate": float(np.mean([t["outcome"] == "target" for t in trades])),
        "stop_hit_rate": float(np.mean([t["outcome"] == "stop" for t in trades])),
        "avg_hold_sessions": float(np.mean([t["exit_idx"] - t["entry_idx"] + 1 for t in trades])),
    }


def wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / denom)


def _validate_strategies(arrays: dict, lengths: dict, source: str) -> dict:
    out = {}
    for key, meta in REGISTRY.items():
        if meta["fn"] is None:
            out[key] = {"status": "not_implemented", "note": meta.get("note", ""),
                        "stats_all": _stats([]), "stats_in": _stats([]), "stats_oos": _stats([]),
                        "probability": None}
            continue
        trades = []
        for s, A in arrays.items():
            trades += [dict(t, symbol=s) for t in _backtest_symbol(key, A, lengths[s])]
        oos = [t for t in trades if t["oos"]]
        ins = [t for t in trades if not t["oos"]]
        st_oos = _stats(oos)
        if source != "live":
            status = f"not_validated_{source}"
        elif len(oos) < MIN_OOS_TRADES:
            status = "insufficient_validation"
        else:
            status = "validated_oos"
        p = st_oos["win_rate"]
        out[key] = {
            "status": status,
            "stats_all": _stats(trades), "stats_in": _stats(ins), "stats_oos": st_oos,
            "probability": p,
            "probability_lower95": wilson_lower(p, st_oos["n"]) if p is not None else None,
        }
    return out


def _plan(A: dict, key: str, validation: dict) -> dict:
    i = len(A["close"]) - 1
    ref = A["close"][i]
    atr = A["atr"][i]
    stop = ref - ATR_STOP_MULT * atr
    risk = (ref - stop) / ref
    target = ref * (1 + R_MULTIPLE * risk)
    v = validation[key]
    p = v["probability"]
    if p is not None:
        ev = p * (R_MULTIPLE * risk - COST_ROUNDTRIP) - (1 - p) * (risk + COST_ROUNDTRIP)
    else:
        ev = None
    score = ev / (risk + COST_ROUNDTRIP) if ev is not None else None
    return {"reference_close": ref, "entry_zone": [round(ref * 0.995, 2), round(ref * 1.005, 2)],
            "stop": round(stop, 2), "target": round(target, 2),
            "stop_loss_pct": -risk * 100, "gross_target_pct": R_MULTIPLE * risk * 100,
            "rr_ratio": R_MULTIPLE, "probability": p,
            "expected_net_return_pct": None if ev is None else ev * 100,
            "risk_adjusted_score": score, "hold_sessions_avg": v["stats_oos"]["avg_hold_sessions"]}


def _build_rows(universe, frames_feat, arrays, lengths, validation, source, asof):
    rows = {}
    for u in universe:
        s = u["symbol"]
        A = arrays[s]
        F = frames_feat[s]
        i = lengths[s] - 1
        g = lambda k: float(A[k][i])
        triggered, conds_all = {}, {}
        for key, meta in REGISTRY.items():
            if meta["fn"] is None:
                continue
            t, conds = meta["fn"](A, i)
            conds_all[key] = conds
            if t:
                triggered[key] = _plan(A, key, validation)
        candidates = []
        for key, plan in triggered.items():
            v = validation[key]
            reasons = []
            if not v["status"].startswith("validated"):
                reasons.append(f"Probability {v['status'].replace('_', ' ')}")
            elif v["probability"] < MIN_PROB:
                reasons.append(f"Probability {v['probability']:.0%} below {MIN_PROB:.0%} threshold")
            if plan["expected_net_return_pct"] is None or plan["expected_net_return_pct"] <= 0:
                reasons.append("Expected net return not positive")
            candidates.append({"strategy": key, "plan": plan, "exclusion_reasons": reasons,
                               "qualifies": not reasons})
        candidates.sort(key=lambda c: (c["qualifies"], c["plan"]["risk_adjusted_score"] or -9),
                        reverse=True)
        best = candidates[0] if candidates else None
        price = g("close")
        prev = float(A["close"][i - 1])
        rows[s] = {
            "symbol": s, "name": u["name"], "sector": u["sector"],
            "price": price, "change_pct": (price / prev - 1) * 100,
            "last_candle": str(F.index[i].date()), "as_of": asof, "source": source,
            "trend": ("bullish" if g("ema20") > g("ema50") and price > g("ema50")
                      else "bearish" if g("ema20") < g("ema50") and price < g("ema50")
                      else "neutral"),
            "indicators": {k: g(k) for k in ["ema20", "ema50", "ema200", "rsi", "macd_line",
                                             "macd_sig", "macd_hist", "st_dir", "relvol", "atr",
                                             "hi20", "lo20", "wk_ema10", "close"]},
            "conditions": {k: v for k, v in conds_all.items()},
            "triggered": list(triggered.keys()),
            "candidates": candidates,
            "best": best,
            "qualifies": bool(best and best["qualifies"]),
        }
    return rows


def build_snapshot() -> dict:
    universe, frames, source = market_data.load()
    asof = datetime.now(timezone.utc).isoformat(timespec="seconds")
    feats = {s: features(df) for s, df in frames.items()}
    arrays = {s: _arrays(F) for s, F in feats.items()}
    lengths = {s: len(F) for s, F in feats.items()}
    validation = _validate_strategies(arrays, lengths, source)
    rows = _build_rows(universe, feats, arrays, lengths, validation, source, asof)
    opps = [r for r in rows.values() if r["qualifies"]]
    opps.sort(key=lambda r: r["best"]["plan"]["risk_adjusted_score"], reverse=True)
    for rank, r in enumerate(opps, 1):
        r["rank"] = rank
    return {"rows": rows, "validation": validation, "source": source, "as_of": asof,
            "opportunities": [r["symbol"] for r in opps], "universe": universe}


def get_snapshot(refresh: bool = False) -> dict:
    if refresh or _state["snap"] is None:
        t0 = time.time()
        _state["snap"] = build_snapshot()
        _state["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _state["build_seconds"] = round(time.time() - t0, 2)
    return _state["snap"]


def status() -> dict:
    snap = get_snapshot()
    return {"mode": "DEMO" if snap["source"] == "demo" else "LIVE", "source": snap["source"],
            "symbols_monitored": len(snap["rows"]), "last_successful_update": _state["built_at"],
            "build_seconds": _state.get("build_seconds"),
            "min_probability_default": MIN_PROB, "horizon_sessions": HORIZON,
            "cost_roundtrip_pct": COST_ROUNDTRIP * 100}


def dashboard() -> dict:
    snap = get_snapshot()
    rows = list(snap["rows"].values())
    counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for r in rows:
        counts[r["trend"]] += 1
    top = [snap["rows"][s] for s in snap["opportunities"][:5]]
    candidates = [r for r in rows if r["triggered"] and not r["qualifies"]]
    candidates.sort(key=lambda r: r["best"]["plan"]["risk_adjusted_score"] or -9, reverse=True)
    by_chg = sorted(rows, key=lambda r: r["change_pct"], reverse=True)
    return {
        "mode": "DEMO" if snap["source"] == "demo" else "LIVE",
        "source": snap["source"], "as_of": snap["as_of"],
        "total_monitored": len(rows), "trend_counts": counts,
        "qualifying_count": len(snap["opportunities"]),
        "top_opportunities": top,
        "no_opportunity_reason": None if top else (
            "No qualifying opportunities currently meet your requirements. "
            "Each strategy's probability must be validated out-of-sample, be at least "
            f"{MIN_PROB:.0%}, and have a positive expected net return. "
            + ("This session uses DEMO data, so no probability can be validated."
               if snap["source"] == "demo" else "")),
        "candidates_unvalidated": candidates[:10],
        "top_gainers": by_chg[:5], "top_losers": by_chg[::-1][:5],
        "validation_summary": {k: {"status": v["status"], "probability": v["probability"],
                                   "oos_trades": v["stats_oos"]["n"]}
                               for k, v in snap["validation"].items()},
        "news": {"status": "not_configured", "message": "No licensed news provider connected."},
    }


def screener(filters: dict) -> dict:
    snap = get_snapshot()
    rows = list(snap["rows"].values())
    min_prob = filters.get("min_prob") or 0.0
    min_ret = filters.get("min_return")
    sector = filters.get("sector")
    included, excluded = [], []
    for r in rows:
        best = r["best"]
        prob = snap["validation"][best["strategy"]]["probability"] if best else None
        ret = best["plan"]["expected_net_return_pct"] if best else None
        item = {"symbol": r["symbol"], "name": r["name"], "sector": r["sector"],
                "price": r["price"], "change_pct": r["change_pct"], "trend": r["trend"],
                "strategy": best["strategy"] if best else None, "probability": prob,
                "expected_net_return_pct": ret,
                "rr_ratio": best["plan"]["rr_ratio"] if best else None,
                "risk_adjusted_score": best["plan"]["risk_adjusted_score"] if best else None,
                "qualifies": r["qualifies"],
                "exclusion_reasons": best["exclusion_reasons"] if best else
                ["No strategy rules triggered on the latest bar"]}
        if sector and r["sector"].lower() != sector.lower():
            continue
        if filters.get("require_signal") and not r["triggered"]:
            continue
        if min_prob and (prob is None or prob < min_prob):
            excluded.append(item)
            continue
        if min_ret is not None and (ret is None or ret < min_ret):
            excluded.append(item)
            continue
        included.append(item)
    key = filters.get("sort") or "risk_adjusted_score"
    included.sort(key=lambda x: (x.get(key) is None, -(x.get(key) or 0)))
    size = int(filters.get("page_size") or 10)
    page = int(filters.get("page") or 1)
    total = len(included)
    start = (page - 1) * size
    return {"total": total, "page": page, "page_size": size,
            "results": included[start:start + size],
            "excluded": excluded if filters.get("include_excluded") else None,
            "excluded_count": len(excluded)}


def universe() -> list[dict]:
    snap = get_snapshot()
    return [
        {"symbol": r["symbol"], "name": r["name"], "sector": r["sector"], "price": r["price"],
         "change_pct": r["change_pct"], "trend": r["trend"], "strategies": r["triggered"],
         "qualifies": r["qualifies"], "last_candle": r["last_candle"], "source": r["source"],
         "market_cap_cr": None, "market_cap_note": "Requires a market-cap data provider",
         "fundamental_score": None, "fundamental_note": "Requires a fundamentals provider"}
        for r in snap["rows"].values()
    ]


def stock(symbol: str) -> dict:
    snap = get_snapshot()
    r = snap["rows"].get(symbol.upper())
    if r is None:
        raise KeyError(symbol)
    return r


def candles(symbol: str, bars: int = 120) -> list[dict]:
    frame = _frame(symbol)
    tail = frame.tail(bars)
    return [{"date": str(ix.date()), "open": float(o), "high": float(h), "low": float(l),
             "close": float(c), "volume": float(v)}
            for ix, o, h, l, c, v in zip(tail.index, tail.open, tail.high, tail.low,
                                          tail.close, tail.volume)]


_frames_cache = {}


def _frame(symbol: str) -> pd.DataFrame:
    if not _frames_cache:
        _, frames, _ = market_data.load()
        _frames_cache.update(frames)
    if symbol.upper() not in _frames_cache:
        raise KeyError(symbol)
    return _frames_cache[symbol.upper()]


def explanation(symbol: str) -> dict:
    snap = get_snapshot()
    r = stock(symbol)
    src = r["source"]
    ind_ = r["indicators"]
    close = ind_["close"]
    asof = r["as_of"]
    base = {"timeframe": "1D", "source": src, "last_candle": r["last_candle"], "as_of": asof}

    def contrib(ok, neutral=False):
        return "Neutral" if neutral else ("Positive" if ok else "Negative")

    e20, e50, e200 = ind_["ema20"], ind_["ema50"], ind_["ema200"]
    rsi_v = ind_["rsi"]
    relvol = ind_["relvol"]
    evidence = [
        {**base, "indicator": "EMA 20 / 50 / 200", "calculation": "EMA(20), EMA(50), EMA(200) on daily closes",
         "reading": f"{e20:.2f} / {e50:.2f} / {e200:.2f}",
         "interpretation": "Trend alignment: EMA20 > EMA50 > EMA200 is bullish",
         "contribution": contrib(e20 > e50 > e200, neutral=not (e20 > e50 > e200) and e20 > e50)},
        {**base, "indicator": "RSI", "calculation": "RSI(14), Wilder smoothing",
         "reading": f"{rsi_v:.1f}", "interpretation": "Above 50 = bullish momentum; below 30 = oversold",
         "contribution": contrib(rsi_v > 50, neutral=30 <= rsi_v <= 50)},
        {**base, "indicator": "MACD", "calculation": "MACD(12,26,9)",
         "reading": f"Line {ind_['macd_line']:.3f}, Signal {ind_['macd_sig']:.3f}, Hist {ind_['macd_hist']:.3f}",
         "interpretation": "Line above signal = bullish momentum",
         "contribution": contrib(ind_["macd_line"] > ind_["macd_sig"])},
        {**base, "indicator": "Supertrend", "calculation": "Supertrend(10, 3.0)",
         "reading": "Bullish" if ind_["st_dir"] == 1 else "Bearish",
         "interpretation": "Trend direction from ATR bands", "contribution": contrib(ind_["st_dir"] == 1)},
        {**base, "indicator": "Volume", "calculation": "Volume / 20-day average volume",
         "reading": f"{relvol:.2f}x", "interpretation": "Participation: >=1.5x confirms, <1x weak",
         "contribution": contrib(relvol >= 1.5, neutral=1 <= relvol < 1.5)},
        {**base, "indicator": "Support / Resistance", "calculation": "Prior 20-bar high and low",
         "reading": f"Resistance {ind_['hi20']:.2f}, Support {ind_['lo20']:.2f}",
         "interpretation": "Close above resistance = breakout; close near support = pullback zone",
         "contribution": contrib(close > ind_["hi20"], neutral=close <= ind_["hi20"])},
        {"indicator": "Fundamentals", "reading": "Not available", "interpretation":
         "No fundamentals provider connected. Missing data is not treated as positive.",
         "contribution": "Neutral", "source": "none", "as_of": asof, "timeframe": "n/a",
         "calculation": "n/a", "last_candle": r["last_candle"], "warning": True},
        {"indicator": "News / Events", "reading": "Not available", "interpretation":
         "No licensed news or corporate-event provider connected.",
         "contribution": "Neutral", "source": "none", "as_of": asof, "timeframe": "n/a",
         "calculation": "n/a", "last_candle": r["last_candle"], "warning": True},
    ]

    strategy_match = []
    for key, meta in REGISTRY.items():
        if meta["fn"] is None:
            strategy_match.append({"key": key, "name": meta["name"], "version": meta["version"],
                                   "triggered": False, "note": meta["note"]})
            continue
        conds = r["conditions"].get(key, [])
        best_c = next((c for c in r["candidates"] if c["strategy"] == key), None)
        v = snap["validation"][key]
        strategy_match.append({
            "key": key, "name": meta["name"], "version": meta["version"],
            "triggered": key in r["triggered"],
            "conditions_satisfied": [c["label"] + (f" ({c['detail']})" if c["detail"] else "")
                                     for c in conds if c["passed"]],
            "conditions_not_satisfied": [c["label"] + (f" ({c['detail']})" if c["detail"] else "")
                                         for c in conds if not c["passed"]],
            "entry_trigger": meta["entry"], "stop_logic": meta["stop"],
            "target_logic": meta["target"], "invalidation": meta["invalidation"],
            "backtest_period": {"from_index": 0, "bars_per_symbol": snap_len(symbol),
                                "oos_fraction": 1 - OOS_FRACTION},
            "historical_trades_all": v["stats_all"]["n"],
            "historical_win_rate_all": v["stats_all"]["win_rate"],
            "out_of_sample": v["stats_oos"], "probability": v["probability"],
            "probability_status": v["status"],
            "qualifies": bool(best_c and best_c["qualifies"]),
            "exclusion_reasons": best_c["exclusion_reasons"] if best_c else [],
        })

    best = r["best"]
    triggered_list = [s for s in strategy_match if s["triggered"]]
    if best and best["qualifies"]:
        state = "Qualifies — entry conditions confirmed"
    elif triggered_list:
        state = "Insufficient validation — probability cannot be trusted"
    else:
        partial = max((sum(c["passed"] for c in conds) for conds in r["conditions"].values()),
                      default=0)
        state = ("Watchlist only — additional evidence required" if partial >= 3
                 else "Not qualifying — no strategy rules satisfied")

    not_met = [c["label"] for conds in r["conditions"].values() for c in conds if not c["passed"]]
    what_could_go_wrong = []
    if relvol < 1.5:
        what_could_go_wrong.append(f"Volume is {relvol:.2f}x average; breakouts need >=1.5x confirmation.")
    if close > ind_["ema20"] * 1.05:
        what_could_go_wrong.append("Price is extended more than 5% above EMA20; mean reversion risk.")
    if ind_["rsi"] > 70:
        what_could_go_wrong.append(f"RSI {ind_['rsi']:.1f} is overbought.")
    what_could_go_wrong.append(f"Stop-loss breach invalidates the setup (about {abs(best['plan']['stop_loss_pct']):.2f}% below reference close)." if best else "No active trade plan.")
    what_could_go_wrong.append("Unmodelled event risk: earnings, results, or sector shocks are not covered by the price model.")
    if src == "demo":
        what_could_go_wrong.append("DEMO data: prices are synthetic and no result here reflects a real market.")

    reasons_not_to_trade = []
    if not r["qualifies"]:
        if best:
            reasons_not_to_trade.extend(best["exclusion_reasons"])
        else:
            reasons_not_to_trade.append("No strategy rules were satisfied on the latest bar.")
    reasons_not_to_trade.append("Fundamental and news data were not evaluated (no provider connected).")

    selection = []
    if triggered_list:
        selection.append(f"Triggered on the latest bar ({r['last_candle']}) by: "
                         + ", ".join(s["name"] for s in triggered_list) + ".")
        selection.append("Every required rule of the triggered strategy was satisfied (see Strategy Match).")
    else:
        selection.append("No strategy rules were fully satisfied on the latest bar.")
    if best:
        selection.append("Probability status: "
                         + snap["validation"][best["strategy"]]["status"].replace("_", " ") + ".")

    return {
        "symbol": r["symbol"], "name": r["name"], "sector": r["sector"], "mode": "DEMO" if src == "demo" else "LIVE",
        "generated_at": asof, "price": r["price"], "change_pct": r["change_pct"],
        "decision_summary": state,
        "why_selected": selection,
        "evidence": evidence,
        "strategy_match": strategy_match,
        "probability": {
            "strategy": best["strategy"] if best else None,
            "value": snap["validation"][best["strategy"]]["probability"] if best else None,
            "meaning": "Share of out-of-sample historical trades of this strategy that finished with a positive net return within the holding horizon. This is a strategy-level estimate, not a per-stock forecast.",
            "horizon_sessions": HORIZON, "costs_included": True,
            "sample_size_oos": snap["validation"][best["strategy"]]["stats_oos"]["n"] if best else 0,
            "validation_status": snap["validation"][best["strategy"]]["status"] if best else "no_signal",
            "wilson_lower_95": snap["validation"][best["strategy"]].get("probability_lower95") if best else None,
            "note": "Historical win rate is not a guarantee for this trade.",
        },
        "trade_plan": best["plan"] if best else None,
        "rank": r.get("rank"),
        "rank_method": "Ranked among qualifying opportunities by risk-adjusted score = expected net return / (stop distance + round-trip cost).",
        "rank_components": best["plan"] if best else None,
        "what_could_go_wrong": what_could_go_wrong,
        "unmet_conditions": not_met,
        "reasons_not_to_trade": reasons_not_to_trade,
        "ai_narrative": None,
        "ai_narrative_note": "AI narrative not configured. Structured evidence above is the authoritative content.",
    }


def snap_len(symbol: str) -> int:
    return len(_frame(symbol))
