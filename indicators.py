"""Reusable technical indicator functions. All calculations are causal (use only past/current bars)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No down-moves with some up-moves: RSI is 100 by definition (avoid 0/0 -> NaN)
    return out.where(~((ad == 0) & (au > 0)), 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Standard Supertrend. direction: +1 bullish, -1 bearish."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    ub, lb = hl2 + mult * a, hl2 - mult * a
    close = df["close"].to_numpy()
    m = len(df)
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(m)
    for i in range(1, m):
        if not (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]):
            fub[i] = fub[i - 1]
        if not (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]):
            flb[i] = flb[i - 1]
        if direction[i - 1] == 1 and close[i] < flb[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and close[i] > fub[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction == 1, flb, fub)
    return pd.DataFrame({"st": line, "dir": direction}, index=df.index)
