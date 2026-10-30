from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analysis

app = FastAPI(title="Nifty150 Pro", version="0.1.0")


@app.exception_handler(NotImplementedError)
async def _not_impl(_, exc):
    return JSONResponse(status_code=503, content={"error": "provider_not_configured",
                                                   "message": str(exc)})


@app.get("/api/status")
def status():
    return analysis.status()


@app.post("/api/refresh")
def refresh():
    analysis.get_snapshot(refresh=True)
    return analysis.status()


@app.get("/api/dashboard")
def dashboard():
    return analysis.dashboard()


@app.get("/api/screener")
def screener(
    sector: str | None = None,
    min_prob: float | None = Query(None, ge=0, le=1),
    min_return: float | None = None,
    require_signal: bool = False,
    sort: str = Query("risk_adjusted_score", pattern="^(probability|expected_net_return_pct|rr_ratio|risk_adjusted_score|change_pct|price)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    include_excluded: bool = False,
):
    return analysis.screener({"sector": sector, "min_prob": min_prob, "min_return": min_return,
                              "require_signal": require_signal, "sort": sort, "page": page,
                              "page_size": page_size, "include_excluded": include_excluded})


@app.get("/api/universe")
def universe():
    return analysis.universe()


@app.get("/api/strategies")
def strategies():
    from .strategies import REGISTRY
    snap = analysis.get_snapshot()
    out = []
    for key, meta in REGISTRY.items():
        v = snap["validation"][key]
        out.append({"key": key, "name": meta["name"], "version": meta["version"],
                    "entry": meta["entry"], "stop": meta["stop"], "target": meta["target"],
                    "invalidation": meta["invalidation"], "status": v["status"],
                    "probability": v["probability"], "probability_lower95": v.get("probability_lower95"),
                    "stats_all": v["stats_all"], "stats_in_sample": v["stats_in"],
                    "stats_out_of_sample": v["stats_oos"], "note": meta.get("note")})
    return {"source": snap["source"], "as_of": snap["as_of"], "strategies": out}


@app.get("/api/stocks/{symbol}")
def stock(symbol: str):
    try:
        return analysis.stock(symbol)
    except KeyError:
        raise HTTPException(404, f"Symbol {symbol} is not in the monitored universe")


@app.get("/api/stocks/{symbol}/explanation")
def explanation(symbol: str):
    try:
        return analysis.explanation(symbol)
    except KeyError:
        raise HTTPException(404, f"Symbol {symbol} is not in the monitored universe")


@app.get("/api/stocks/{symbol}/candles")
def candles(symbol: str, bars: int = Query(120, ge=20, le=500)):
    try:
        return analysis.candles(symbol, bars)
    except KeyError:
        raise HTTPException(404, f"Symbol {symbol} is not in the monitored universe")


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
