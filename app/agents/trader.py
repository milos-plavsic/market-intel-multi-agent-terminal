"""TraderAgent — aggregates technical signals into a trade recommendation.

Signals are weighted by risk tolerance and combined into an overall score
that drives the final action (buy / sell / hold), confidence, and position
sizing guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_core import configure_logging

from app.agents.analyst import Signal

logger = configure_logging("trader")

# ---------------------------------------------------------------------------
# Risk tolerance profiles
# Risk tolerance maps to signal weight multipliers and position constraints.
# ---------------------------------------------------------------------------

_RISK_PROFILES: dict[str, dict] = {
    "conservative": {
        "signal_weights": {
            "SMA_crossover": 0.50,
            "RSI": 0.30,
            "Bollinger_Bands": 0.15,
            "MACD": 0.05,
        },
        "max_position_pct": 0.05,  # 5 % of portfolio max
        "buy_threshold": 0.65,
        "sell_threshold": 0.65,
    },
    "moderate": {
        "signal_weights": {
            "SMA_crossover": 0.35,
            "RSI": 0.25,
            "Bollinger_Bands": 0.20,
            "MACD": 0.20,
        },
        "max_position_pct": 0.10,
        "buy_threshold": 0.55,
        "sell_threshold": 0.55,
    },
    "aggressive": {
        "signal_weights": {
            "SMA_crossover": 0.20,
            "RSI": 0.20,
            "Bollinger_Bands": 0.25,
            "MACD": 0.35,
        },
        "max_position_pct": 0.20,
        "buy_threshold": 0.45,
        "sell_threshold": 0.45,
    },
}

# Default profile for unrecognised risk tolerances
_DEFAULT_PROFILE = "moderate"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class TradeRecommendation:
    """Aggregated trade recommendation."""

    symbol: str
    action: Literal["buy", "sell", "hold"]
    confidence: float  # 0.0 - 1.0
    position_size_pct: float  # recommended portfolio allocation (0 - 1)
    stop_loss: float | None  # price level (None if hold)
    take_profit: float | None  # price level (None if hold)
    rationale: str
    signal_summary: list[dict]


# ---------------------------------------------------------------------------
# TraderAgent
# ---------------------------------------------------------------------------


class TraderAgent:
    """Aggregates technical signals into a trade recommendation."""

    def generate_recommendation(
        self,
        signals: list[Signal],
        current_price: float,
        symbol: str = "UNKNOWN",
        risk_tolerance: str = "moderate",
    ) -> TradeRecommendation:
        """Generate a trade recommendation from *signals*.

        Parameters
        ----------
        signals:
            List of Signal objects from AnalystAgent.
        current_price:
            Latest close price used for stop-loss / take-profit calculation.
        symbol:
            Ticker symbol (for labelling).
        risk_tolerance:
            One of "conservative", "moderate", "aggressive".
        """
        profile = _RISK_PROFILES.get(risk_tolerance, _RISK_PROFILES[_DEFAULT_PROFILE])
        weights = profile["signal_weights"]
        max_pos = profile["max_position_pct"]
        buy_thresh = profile["buy_threshold"]
        sell_thresh = profile["sell_threshold"]

        if not signals:
            return TradeRecommendation(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                position_size_pct=0.0,
                stop_loss=None,
                take_profit=None,
                rationale="No signals available.",
                signal_summary=[],
            )

        # Compute weighted buy / sell scores separately
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        signal_summary = []
        for sig in signals:
            w = weights.get(sig.name, 0.10)  # fallback weight for unknown signals
            weighted_strength = w * sig.strength
            if sig.direction == "buy":
                buy_score += weighted_strength
            elif sig.direction == "sell":
                sell_score += weighted_strength
            total_weight += w
            signal_summary.append(
                {
                    "name": sig.name,
                    "direction": sig.direction,
                    "value": sig.value,
                    "strength": sig.strength,
                    "weight": w,
                }
            )

        # Normalise
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        net_score = buy_score - sell_score  # positive → buy, negative → sell

        if net_score > 0 and buy_score >= buy_thresh:
            action: Literal["buy", "sell", "hold"] = "buy"
            confidence = round(min(buy_score, 1.0), 4)
            position_size_pct = round(max_pos * confidence, 4)
            stop_loss = round(current_price * 0.95, 4)  # 5 % trailing stop
            take_profit = round(current_price * 1.10, 4)  # 10 % target
            rationale = (
                f"Buy signal: weighted buy score={buy_score:.2f} > "
                f"threshold={buy_thresh}. "
                f"Strongest signals: "
                + ", ".join(
                    s["name"]
                    for s in sorted(signal_summary, key=lambda x: -x["strength"])
                    if s["direction"] == "buy"
                )[:3]
                + "."
            )
        elif net_score < 0 and sell_score >= sell_thresh:
            action = "sell"
            confidence = round(min(sell_score, 1.0), 4)
            position_size_pct = round(max_pos * confidence, 4)
            stop_loss = round(current_price * 1.05, 4)  # 5 % stop-loss above for shorts
            take_profit = round(current_price * 0.90, 4)
            rationale = (
                f"Sell signal: weighted sell score={sell_score:.2f} > " f"threshold={sell_thresh}. "
            )
        else:
            action = "hold"
            confidence = round(max(buy_score, sell_score), 4)
            position_size_pct = 0.0
            stop_loss = None
            take_profit = None
            rationale = (
                f"Hold: net score={net_score:.2f} does not meet thresholds "
                f"(buy={buy_thresh}, sell={sell_thresh})."
            )

        logger.info(
            f"[TraderAgent] {symbol} action={action}, confidence={confidence:.2f}, "
            f"risk={risk_tolerance}"
        )

        return TradeRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size_pct=position_size_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale,
            signal_summary=signal_summary,
        )
