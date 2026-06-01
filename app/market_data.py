"""Live market data helpers (yfinance)."""

from __future__ import annotations

from typing import Any

import yfinance as yf
from ml_core import configure_logging

logger = configure_logging("market-data")


def fetch_symbol_snapshot(symbol: str) -> dict[str, Any]:
    """Return recent price/volume stats for a ticker."""
    sym = symbol.upper().strip()
    ticker = yf.Ticker(sym)
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        raise ValueError(f"No market data returned for {sym}")

    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else last
    change_pct = float((last["Close"] - prev["Close"]) / prev["Close"] * 100)

    info: dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception as exc:  # pragma: no cover - yfinance metadata can fail
        logger.warning("ticker.info failed for %s: %s", sym, exc)

    return {
        "symbol": sym,
        "last_close": round(float(last["Close"]), 4),
        "change_percent": round(change_pct, 2),
        "volume": int(last["Volume"]),
        "sector": info.get("sector") or "unknown",
        "short_name": info.get("shortName") or sym,
    }


def format_research_content(snapshot: dict[str, Any]) -> str:
    """Human-readable research blurb from live snapshot."""
    return (
        f"Market Research for {snapshot['symbol']} ({snapshot['short_name']}):\n\n"
        f"Recent Performance:\n"
        f"- Last close: ${snapshot['last_close']}\n"
        f"- 1-day change: {snapshot['change_percent']}%\n"
        f"- Session volume: {snapshot['volume']:,}\n"
        f"- Sector: {snapshot['sector']}\n\n"
        f"Recommendation:\n"
        f"- Monitor for entry opportunities\n"
        f"- Watch support/resistance around recent close\n"
        f"- Track earnings and sector news"
    )


def detect_volume_anomalies(symbols: list[str]) -> list[str]:
    """Flag simple volume spikes vs 5-day average."""
    anomalies: list[str] = []
    for sym in symbols:
        try:
            hist = yf.Ticker(sym.upper()).history(period="10d", interval="1d")
            if len(hist) < 3:
                continue
            avg_vol = float(hist["Volume"].iloc[:-1].mean())
            last_vol = float(hist["Volume"].iloc[-1])
            if avg_vol > 0 and last_vol > avg_vol * 1.5:
                anomalies.append(
                    f"Volume spike on {sym.upper()}: {last_vol:,.0f} vs 5d avg {avg_vol:,.0f}"
                )
        except Exception as exc:
            logger.warning("anomaly check failed for %s: %s", sym, exc)
    return anomalies or ["No significant volume anomalies in watched symbols"]
