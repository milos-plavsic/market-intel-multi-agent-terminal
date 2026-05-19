"""AnalystAgent — real technical indicator implementations.

All indicators are computed from pure Python math; no external finance
libraries are required.

Indicators implemented
----------------------
- SMA  : Simple Moving Average (20 and 50 day crossover)
- RSI  : Relative Strength Index (14-period)
- Bollinger Bands : 20-period, 2-sigma
- MACD : 12/26 EMA difference with 9-period signal line
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ml_core import configure_logging

from app.agents.researcher import MarketDataSeries

logger = configure_logging("analyst")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A single technical indicator signal."""

    name: str
    value: float
    direction: Literal["buy", "sell", "hold"]
    strength: float  # 0.0 (weak) to 1.0 (strong)
    description: str = ""


# ---------------------------------------------------------------------------
# Low-level indicator maths
# ---------------------------------------------------------------------------


def _sma(prices: list[float], period: int) -> list[float]:
    """Compute Simple Moving Average for *period* days.

    Returns a list the same length as *prices*.  Positions < period-1 are NaN.
    """
    result: list[float] = []
    for i, _ in enumerate(prices):
        if i < period - 1:
            result.append(float("nan"))
        else:
            window = prices[i - period + 1 : i + 1]
            result.append(sum(window) / period)
    return result


def _ema(prices: list[float], period: int) -> list[float]:
    """Compute Exponential Moving Average.

    Uses the standard multiplier k = 2 / (period + 1).
    Returns list the same length as *prices*; first value equals prices[0].
    """
    if not prices:
        return []
    k = 2.0 / (period + 1)
    ema_vals: list[float] = [prices[0]]
    for p in prices[1:]:
        ema_vals.append(p * k + ema_vals[-1] * (1 - k))
    return ema_vals


def _rsi(prices: list[float], period: int = 14) -> float:
    """Return the most recent RSI value (0-100).

    Wilder smoothing (EMA-style with alpha = 1/period).
    """
    if len(prices) < period + 1:
        return 50.0  # not enough data — neutral

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(c, 0.0) for c in changes]
    losses = [abs(min(c, 0.0)) for c in changes]

    # Initial averages over first *period* bars
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth remaining bars (Wilder)
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Return (upper_band, middle_band, lower_band) for the last bar.

    Requires at least *period* price points.
    """
    if len(prices) < period:
        last = prices[-1] if prices else 0.0
        return last, last, last

    window = prices[-period:]
    middle = sum(window) / period
    variance = sum((p - middle) ** 2 for p in window) / period
    std = math.sqrt(variance)
    return middle + num_std * std, middle, middle - num_std * std


def _macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """Return (macd_line, signal_line, histogram) for the last bar."""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0

    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]

    return macd_line[-1], signal_line[-1], histogram[-1]


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------


class AnalystAgent:
    """Computes technical signals from a MarketDataSeries."""

    def compute_signals(self, data: MarketDataSeries) -> list[Signal]:
        """Return a list of signals for *data*."""
        closes = data.closes
        symbol = data.symbol

        if len(closes) < 2:
            logger.warning(f"[AnalystAgent] insufficient data for {symbol}")
            return []

        signals: list[Signal] = []

        # ---- SMA crossover -----------------------------------------------
        sma20 = _sma(closes, min(20, len(closes)))
        sma50 = _sma(closes, min(50, len(closes)))

        # Find last non-NaN values
        last_sma20 = next((v for v in reversed(sma20) if not math.isnan(v)), None)
        last_sma50 = next((v for v in reversed(sma50) if not math.isnan(v)), None)

        if last_sma20 is not None and last_sma50 is not None:
            diff_pct = (last_sma20 - last_sma50) / last_sma50
            if diff_pct > 0.01:
                sma_dir: Literal["buy", "sell", "hold"] = "buy"
                sma_strength = min(abs(diff_pct) * 20, 1.0)
            elif diff_pct < -0.01:
                sma_dir = "sell"
                sma_strength = min(abs(diff_pct) * 20, 1.0)
            else:
                sma_dir = "hold"
                sma_strength = 0.3

            signals.append(
                Signal(
                    name="SMA_crossover",
                    value=round(last_sma20, 4),
                    direction=sma_dir,
                    strength=round(sma_strength, 4),
                    description=(
                        f"SMA20={last_sma20:.2f} vs SMA50={last_sma50:.2f} " f"({diff_pct:+.2%})"
                    ),
                )
            )

        # ---- RSI -------------------------------------------------------------
        rsi_val = _rsi(closes, period=min(14, len(closes) - 1))
        if rsi_val < 30:
            rsi_dir: Literal["buy", "sell", "hold"] = "buy"
            rsi_strength = round((30 - rsi_val) / 30, 4)
        elif rsi_val > 70:
            rsi_dir = "sell"
            rsi_strength = round((rsi_val - 70) / 30, 4)
        else:
            rsi_dir = "hold"
            rsi_strength = round(abs(rsi_val - 50) / 50 * 0.5, 4)

        signals.append(
            Signal(
                name="RSI",
                value=round(rsi_val, 4),
                direction=rsi_dir,
                strength=rsi_strength,
                description=f"RSI(14)={rsi_val:.1f}",
            )
        )

        # ---- Bollinger Bands ------------------------------------------------
        upper, mid, lower = _bollinger(closes, period=min(20, len(closes)))
        last_close = closes[-1]

        if last_close > upper:
            bb_dir: Literal["buy", "sell", "hold"] = "sell"
            bb_strength = round(min((last_close - upper) / (upper - mid + 1e-9), 1.0), 4)
        elif last_close < lower:
            bb_dir = "buy"
            bb_strength = round(min((lower - last_close) / (mid - lower + 1e-9), 1.0), 4)
        else:
            bb_dir = "hold"
            band_width = upper - lower if upper != lower else 1.0
            position = (last_close - lower) / band_width  # 0=at lower, 1=at upper
            bb_strength = round(abs(position - 0.5) * 0.8, 4)

        signals.append(
            Signal(
                name="Bollinger_Bands",
                value=round(last_close, 4),
                direction=bb_dir,
                strength=bb_strength,
                description=(
                    f"Price={last_close:.2f}, BB upper={upper:.2f}, "
                    f"mid={mid:.2f}, lower={lower:.2f}"
                ),
            )
        )

        # ---- MACD -----------------------------------------------------------
        macd_line, signal_line, histogram = _macd(closes)

        if macd_line > signal_line and histogram > 0:
            macd_dir: Literal["buy", "sell", "hold"] = "buy"
            macd_strength = round(min(abs(histogram) / (abs(macd_line) + 1e-9), 1.0), 4)
        elif macd_line < signal_line and histogram < 0:
            macd_dir = "sell"
            macd_strength = round(min(abs(histogram) / (abs(macd_line) + 1e-9), 1.0), 4)
        else:
            macd_dir = "hold"
            macd_strength = 0.2

        signals.append(
            Signal(
                name="MACD",
                value=round(macd_line, 6),
                direction=macd_dir,
                strength=macd_strength,
                description=(
                    f"MACD={macd_line:.4f}, Signal={signal_line:.4f}, " f"Hist={histogram:.4f}"
                ),
            )
        )

        logger.info(f"[AnalystAgent] computed {len(signals)} signals for {symbol}")
        return signals
