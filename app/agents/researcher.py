"""ResearchAgent — fetches synthetic market data and analyses trends.

Price data is generated via a seeded random walk with drift so results are
deterministic (given the same symbol) and realistic in shape.  No external
finance libraries are required.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import ClassVar, Literal

from ml_core import configure_logging

logger = configure_logging("researcher")

# ---------------------------------------------------------------------------
# Base prices and drift parameters — one per supported symbol
# ---------------------------------------------------------------------------

_SYMBOL_PARAMS: dict[str, dict] = {
    "AAPL": {"base": 175.0, "drift": 0.0005, "sigma": 0.015},
    "GOOGL": {"base": 140.0, "drift": 0.0004, "sigma": 0.016},
    "MSFT": {"base": 380.0, "drift": 0.0006, "sigma": 0.013},
    "AMZN": {"base": 185.0, "drift": 0.0005, "sigma": 0.018},
    "TSLA": {"base": 220.0, "drift": 0.0003, "sigma": 0.035},
    "BTC": {"base": 65000.0, "drift": 0.0008, "sigma": 0.040},
    "ETH": {"base": 3500.0, "drift": 0.0007, "sigma": 0.038},
    "SPY": {"base": 520.0, "drift": 0.0003, "sigma": 0.009},
    "QQQ": {"base": 440.0, "drift": 0.0004, "sigma": 0.012},
    "GLD": {"base": 220.0, "drift": 0.0001, "sigma": 0.008},
}

SUPPORTED_SYMBOLS = set(_SYMBOL_PARAMS.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PriceBar:
    """Single OHLCV bar."""

    date: str  # ISO format YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MarketDataSeries:
    """Historical price series for one symbol."""

    symbol: str
    bars: list[PriceBar] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def closes(self) -> list[float]:
        """Return list of close prices ordered oldest→newest."""
        return [b.close for b in self.bars]

    @property
    def volumes(self) -> list[int]:
        return [b.volume for b in self.bars]

    @property
    def latest_close(self) -> float:
        return self.bars[-1].close if self.bars else 0.0

    @property
    def latest_date(self) -> str:
        return self.bars[-1].date if self.bars else ""


@dataclass
class TrendAnalysis:
    """Trend metrics for a symbol."""

    symbol: str
    direction: Literal["up", "down", "sideways"]
    momentum: float  # rate of change over the period, -1..1
    volatility: float  # annualised std-dev
    period_return: float  # total return over requested period
    avg_volume: float


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Maintains a synthetic market data store and exposes analysis helpers."""

    # Class-level cache so repeated calls are fast within a process
    _cache: ClassVar[dict[str, MarketDataSeries]] = {}

    def fetch_data(self, symbol: str, days: int = 30) -> MarketDataSeries:
        """Generate (or retrieve) synthetic historical data for *symbol*.

        Uses a seeded geometric random walk so output is deterministic per
        (symbol, days) combination.
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}:{days}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if symbol not in _SYMBOL_PARAMS:
            raise ValueError(
                f"Unsupported symbol '{symbol}'. " f"Supported: {sorted(SUPPORTED_SYMBOLS)}"
            )

        params = _SYMBOL_PARAMS[symbol]
        base = params["base"]
        drift = params["drift"]
        sigma = params["sigma"]

        rng = random.Random(hash(symbol) % (2**31))

        bars: list[PriceBar] = []
        price = base
        today = date(2026, 5, 2)  # fixed reference date for reproducibility

        for i in range(days, 0, -1):
            bar_date = today - timedelta(days=i)
            # Skip weekends for equities (but keep all days for crypto)
            if symbol not in ("BTC", "ETH") and bar_date.weekday() >= 5:
                continue

            # Geometric Brownian Motion step
            z = rng.gauss(0, 1)
            ret = drift + sigma * z
            open_price = price
            close_price = open_price * math.exp(ret)
            intra_range = abs(close_price - open_price) * rng.uniform(1.2, 2.5)
            high = max(open_price, close_price) + intra_range * 0.5
            low = min(open_price, close_price) - intra_range * 0.5
            volume = int(rng.randint(500_000, 5_000_000) * (1 + abs(z)))

            bars.append(
                PriceBar(
                    date=bar_date.isoformat(),
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close_price, 4),
                    volume=volume,
                )
            )
            price = close_price

        series = MarketDataSeries(symbol=symbol, bars=bars)
        self._cache[cache_key] = series
        logger.info(f"[ResearchAgent] fetched {len(bars)} bars for {symbol}")
        return series

    def analyze_trend(self, symbol: str, days: int = 30) -> TrendAnalysis:
        """Return trend direction, momentum, and volatility for *symbol*."""
        series = self.fetch_data(symbol, days=days)
        closes = series.closes

        if len(closes) < 2:
            return TrendAnalysis(
                symbol=symbol,
                direction="sideways",
                momentum=0.0,
                volatility=0.0,
                period_return=0.0,
                avg_volume=0.0,
            )

        # Period return
        period_return = (closes[-1] - closes[0]) / closes[0]

        # Log returns for volatility
        log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        n = len(log_rets)
        mean_ret = sum(log_rets) / n
        variance = sum((r - mean_ret) ** 2 for r in log_rets) / max(n - 1, 1)
        daily_std = math.sqrt(variance)
        annualised_vol = daily_std * math.sqrt(252)

        # Momentum: normalised rate of change clamped to [-1, 1]
        momentum = max(-1.0, min(1.0, period_return * 5))

        # Direction
        if period_return > 0.01:
            direction: Literal["up", "down", "sideways"] = "up"
        elif period_return < -0.01:
            direction = "down"
        else:
            direction = "sideways"

        avg_volume = sum(series.volumes) / max(len(series.volumes), 1)

        return TrendAnalysis(
            symbol=symbol,
            direction=direction,
            momentum=round(momentum, 4),
            volatility=round(annualised_vol, 4),
            period_return=round(period_return, 6),
            avg_volume=round(avg_volume, 0),
        )
