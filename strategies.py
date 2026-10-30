"""Five candidate swing-trading strategies. Each rule set is a versioned, explicit list of conditions.

Each strategy function takes a dict of numpy feature arrays and a bar index i (data up to and
including bar i only) and returns (triggered: bool, conditions: list[dict]).
"""
import numpy as np

VERSION = "1.0.0"


def _c(label, passed, detail=""):
    return {"label": label, "passed": bool(passed), "detail": detail}


def trend_pullback(A, i):
    g = lambda k: A[k][i]
    conds = [
        _c("EMA20 > EMA50 > EMA200", g("ema20") > g("ema50") > g("ema200"),
           f"EMA20 {g('ema20'):.2f}, EMA50 {g('ema50'):.2f}, EMA200 {g('ema200'):.2f}"),
        _c("Pullback touched EMA20 zone and close reclaimed EMA20",
           g("low") <= g("ema20") * 1.01 and g("close") > g("ema20"),
           f"Low {g('low'):.2f} vs EMA20 {g('ema20'):.2f}"),
        _c("RSI between 40 and 60 and rising", 40 <= g("rsi") <= 60 and g("rsi") > g("rsi_prev"),
           f"RSI {g('rsi'):.1f} (prev {g('rsi_prev'):.1f})"),
        _c("MACD histogram rising", g("macd_hist") > g("macd_hist_prev"),
           f"Hist {g('macd_hist'):.3f} (prev {g('macd_hist_prev'):.3f})"),
        _c("Bullish candle (close > open)", g("close") > g("open")),
    ]
    return all(c["passed"] for c in conds), conds


def breakout_volume(A, i):
    g = lambda k: A[k][i]
    conds = [
        _c("Close above prior 20-bar high", g("close") > g("hi20"),
           f"Close {g('close'):.2f} vs 20-bar high {g('hi20'):.2f}"),
        _c("Relative volume >= 1.5x 20-day average", g("relvol") >= 1.5,
           f"Relative volume {g('relvol'):.2f}x"),
        _c("EMA20 > EMA50", g("ema20") > g("ema50")),
        _c("RSI between 50 and 75", 50 <= g("rsi") <= 75, f"RSI {g('rsi'):.1f}"),
        _c("Not extended (close < 8% above EMA20)", g("close") < g("ema20") * 1.08),
    ]
    return all(c["passed"] for c in conds), conds


def supertrend_continuation(A, i):
    g = lambda k: A[k][i]
    conds = [
        _c("Supertrend bullish", g("st_dir") == 1),
        _c("MACD line above signal line", g("macd_line") > g("macd_sig")),
        _c("EMA20 > EMA50", g("ema20") > g("ema50")),
        _c("Weekly trend: close above 10-week EMA", g("close") > g("wk_ema10"),
           f"Close {g('close'):.2f} vs weekly EMA10 {g('wk_ema10'):.2f}"),
        _c("Not chasing (close < 6% above EMA20)", g("close") < g("ema20") * 1.06),
    ]
    return all(c["passed"] for c in conds), conds


def rsi_support_recovery(A, i):
    g = lambda k: A[k][i]
    conds = [
        _c("Broader trend valid: EMA50 > EMA200", g("ema50") > g("ema200")),
        _c("RSI recovered above 35 from <= 35 on prior bar",
           g("rsi_prev") <= 35 < g("rsi"), f"RSI {g('rsi_prev'):.1f} -> {g('rsi'):.1f}"),
        _c("Low within 2% of prior 20-bar low (support)", g("low") <= g("lo20") * 1.02,
           f"Low {g('low'):.2f} vs support {g('lo20'):.2f}"),
        _c("Bullish candle (close > open)", g("close") > g("open")),
        _c("Support holding: close above prior 20-bar low", g("close") > g("lo20")),
    ]
    return all(c["passed"] for c in conds), conds


REGISTRY = {
    "trend_pullback": {
        "name": "Trend Pullback Confirmation", "version": VERSION, "fn": trend_pullback,
        "entry": "Next session open after a bullish close that reclaims EMA20 following a pullback.",
        "stop": "Entry minus 1.5 x ATR(14).",
        "target": "Entry plus 2R, where R = (entry - stop) / entry.",
        "invalidation": "Close below the stop, or EMA20 falling below EMA50.",
    },
    "breakout_volume": {
        "name": "Breakout with Volume Confirmation", "version": VERSION, "fn": breakout_volume,
        "entry": "Next session open after a close above the prior 20-bar high on >=1.5x volume.",
        "stop": "Entry minus 1.5 x ATR(14).",
        "target": "Entry plus 2R.",
        "invalidation": "Close back below the breakout level (false breakout) or stop hit.",
    },
    "supertrend_continuation": {
        "name": "Supertrend + Multi-Timeframe Continuation", "version": VERSION,
        "fn": supertrend_continuation,
        "entry": "Next session open when daily Supertrend is bullish and weekly trend agrees.",
        "stop": "Entry minus 1.5 x ATR(14).",
        "target": "Entry plus 2R.",
        "invalidation": "Supertrend flips bearish or close below the stop.",
    },
    "rsi_support_recovery": {
        "name": "RSI Recovery Near Support", "version": VERSION, "fn": rsi_support_recovery,
        "entry": "Next session open after RSI crosses above 35 near 20-bar support with a bullish candle.",
        "stop": "Entry minus 1.5 x ATR(14). Simplified support: prior 20-bar low.",
        "target": "Entry plus 2R.",
        "invalidation": "Support fails (close below prior 20-bar low) or EMA50 falls below EMA200.",
    },
    "fundamental_momentum": {
        "name": "Fundamental Quality + Relative Strength", "version": VERSION, "fn": None,
        "entry": "Not evaluated.", "stop": "Not evaluated.", "target": "Not evaluated.",
        "invalidation": "Not evaluated.",
        "note": "Requires a connected fundamentals provider. Missing data is never treated as positive.",
    },
}
