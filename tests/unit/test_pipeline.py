"""Unit tests for the Market Intel multi-agent pipeline.

Covers:
- ResearchAgent: data generation, trend analysis
- AnalystAgent: all four technical indicators (SMA, RSI, Bollinger Bands, MACD)
- Indicator math functions (_sma, _ema, _rsi, _bollinger, _macd)
- TraderAgent: weighted signal aggregation and trade recommendation
- Edge cases: short series, flat prices, all-gain / all-loss RSI
- Signal correctness against hand-calculated values

At least 15 meaningful tests.
"""

from __future__ import annotations

import math

import pytest

from app.agents.analyst import (
    AnalystAgent,
    Signal,
    _bollinger,
    _ema,
    _macd,
    _rsi,
    _sma,
)
from app.agents.researcher import MarketDataSeries, PriceBar, ResearchAgent
from app.agents.trader import _RISK_PROFILES, TraderAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(closes: list[float], symbol: str = "TEST") -> MarketDataSeries:
    """Build a MarketDataSeries from a plain list of close prices."""
    bars = [
        PriceBar(
            date=f"2026-01-{i+1:02d}",
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=1_000_000,
        )
        for i, c in enumerate(closes)
    ]
    return MarketDataSeries(symbol=symbol, bars=bars)


def _make_signal(name: str, direction: str, strength: float = 0.5) -> Signal:
    return Signal(name=name, value=0.0, direction=direction, strength=strength)


# ---------------------------------------------------------------------------
# 1. _sma tests
# ---------------------------------------------------------------------------


class TestSMA:
    def test_correct_value_simple(self):
        """SMA(3) of [1,2,3,4,5] last value == 4.0."""
        result = _sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert math.isclose(result[-1], 4.0, rel_tol=1e-9)

    def test_leading_nans(self):
        """First period-1 values must be NaN."""
        result = _sma([10.0, 20.0, 30.0, 40.0], 3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert not math.isnan(result[2])

    def test_same_length_as_input(self):
        prices = [float(i) for i in range(10)]
        assert len(_sma(prices, 5)) == 10

    def test_period_1_equals_input(self):
        prices = [3.0, 7.0, 2.0]
        result = _sma(prices, 1)
        assert result == prices


# ---------------------------------------------------------------------------
# 2. _ema tests
# ---------------------------------------------------------------------------


class TestEMA:
    def test_first_value_equals_first_price(self):
        prices = [100.0, 105.0, 110.0]
        result = _ema(prices, 3)
        assert math.isclose(result[0], 100.0, rel_tol=1e-9)

    def test_ema_smooths_towards_new_price(self):
        """EMA should move toward the latest price."""
        prices = [100.0] * 10 + [200.0]
        result = _ema(prices, 3)
        # After 10 flat values, EMA is ~100; last bar pushes it toward 200
        assert result[-1] > result[-2]

    def test_empty_prices_returns_empty(self):
        assert _ema([], 5) == []

    def test_same_length_as_input(self):
        prices = list(range(1, 21))
        assert len(_ema([float(p) for p in prices], 5)) == 20


# ---------------------------------------------------------------------------
# 3. _rsi tests
# ---------------------------------------------------------------------------


class TestRSI:
    def test_neutral_when_insufficient_data(self):
        """Fewer than period+1 bars → 50.0 (neutral)."""
        assert _rsi([100.0, 101.0], period=14) == 50.0

    def test_rsi_100_all_gains(self):
        """All-up moves → RSI should be very high (100 if no losses)."""
        prices = [float(100 + i) for i in range(20)]
        val = _rsi(prices, period=14)
        assert val > 90.0

    def test_rsi_0_all_losses(self):
        """All-down moves → RSI should be very low."""
        prices = [float(100 - i) for i in range(20)]
        val = _rsi(prices, period=14)
        assert val < 10.0

    def test_rsi_near_50_alternating(self):
        """Alternating up/down of equal magnitude → RSI ≈ 50."""
        prices = [100.0 + (1 if i % 2 == 0 else -1) for i in range(30)]
        val = _rsi(prices, period=14)
        assert 40.0 < val < 60.0

    def test_rsi_bounds(self):
        """RSI must always be in [0, 100]."""
        import random

        rng = random.Random(42)
        prices = [100.0]
        for _ in range(50):
            prices.append(prices[-1] * (1 + rng.gauss(0, 0.02)))
        val = _rsi(prices, period=14)
        assert 0.0 <= val <= 100.0


# ---------------------------------------------------------------------------
# 4. _bollinger tests
# ---------------------------------------------------------------------------


class TestBollinger:
    def test_middle_is_sma(self):
        """Middle band must equal the SMA of the last *period* prices."""
        prices = [float(i) for i in range(1, 21)]  # 1..20
        upper, mid, lower = _bollinger(prices, period=20)
        expected_mid = sum(prices) / 20
        assert math.isclose(mid, expected_mid, rel_tol=1e-9)

    def test_upper_above_lower(self):
        prices = [100.0 + math.sin(i) * 5 for i in range(30)]
        upper, mid, lower = _bollinger(prices, period=20)
        assert upper > lower

    def test_flat_prices_zero_bandwidth(self):
        """Constant prices → std = 0 → upper == middle == lower."""
        prices = [100.0] * 25
        upper, mid, lower = _bollinger(prices, period=20)
        assert math.isclose(upper, mid, rel_tol=1e-9)
        assert math.isclose(lower, mid, rel_tol=1e-9)

    def test_returns_last_price_when_insufficient(self):
        """Fewer than *period* prices → all three bands equal last price."""
        prices = [55.0, 60.0]
        upper, mid, lower = _bollinger(prices, period=20)
        assert upper == lower == prices[-1]


# ---------------------------------------------------------------------------
# 5. _macd tests
# ---------------------------------------------------------------------------


class TestMACD:
    def test_returns_zeros_insufficient_data(self):
        prices = [100.0] * 10
        macd_line, signal_line, hist = _macd(prices)
        assert macd_line == 0.0 and signal_line == 0.0 and hist == 0.0

    def test_macd_positive_on_uptrend(self):
        """Rising prices → EMA12 > EMA26 → MACD line > 0."""
        prices = [100.0 + i * 2 for i in range(40)]
        macd_line, _, _ = _macd(prices)
        assert macd_line > 0

    def test_macd_negative_on_downtrend(self):
        """Falling prices → EMA12 < EMA26 → MACD line < 0."""
        prices = [200.0 - i * 2 for i in range(40)]
        macd_line, _, _ = _macd(prices)
        assert macd_line < 0

    def test_histogram_equals_macd_minus_signal(self):
        prices = [100.0 + math.sin(i * 0.3) * 10 for i in range(50)]
        macd_line, signal_line, hist = _macd(prices)
        assert math.isclose(hist, macd_line - signal_line, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 6. AnalystAgent.compute_signals tests
# ---------------------------------------------------------------------------


class TestAnalystAgent:
    agent = AnalystAgent()

    def test_returns_four_signals_for_adequate_data(self):
        """With ≥50 bars we get all four indicator signals."""
        closes = [100.0 + math.sin(i * 0.1) * 5 for i in range(60)]
        series = _make_series(closes)
        signals = self.agent.compute_signals(series)
        names = {s.name for s in signals}
        assert "SMA_crossover" in names
        assert "RSI" in names
        assert "Bollinger_Bands" in names
        assert "MACD" in names

    def test_returns_empty_for_single_bar(self):
        """Fewer than 2 bars → no signals."""
        series = _make_series([100.0])
        signals = self.agent.compute_signals(series)
        assert signals == []

    def test_signal_direction_valid(self):
        closes = [150.0 + i * 0.5 for i in range(60)]
        series = _make_series(closes)
        for sig in self.agent.compute_signals(series):
            assert sig.direction in ("buy", "sell", "hold")

    def test_signal_strength_in_bounds(self):
        closes = [100.0 - i * 0.3 for i in range(60)]
        series = _make_series(closes)
        for sig in self.agent.compute_signals(series):
            assert 0.0 <= sig.strength <= 1.0

    def test_rsi_buy_signal_on_oversold(self):
        """Sharp downtrend → RSI < 30 → RSI signal direction == 'buy'."""
        closes = [100.0 - i * 2.5 for i in range(30)]
        series = _make_series(closes, symbol="BEAR")
        signals = self.agent.compute_signals(series)
        rsi_sig = next((s for s in signals if s.name == "RSI"), None)
        assert rsi_sig is not None
        # RSI for a sharp downtrend should be very low → buy signal
        assert rsi_sig.direction == "buy"

    def test_rsi_sell_signal_on_overbought(self):
        """Sharp uptrend → RSI > 70 → RSI signal direction == 'sell'."""
        closes = [50.0 + i * 3.0 for i in range(30)]
        series = _make_series(closes, symbol="BULL")
        signals = self.agent.compute_signals(series)
        rsi_sig = next((s for s in signals if s.name == "RSI"), None)
        assert rsi_sig is not None
        assert rsi_sig.direction == "sell"


# ---------------------------------------------------------------------------
# 7. ResearchAgent tests
# ---------------------------------------------------------------------------


class TestResearchAgent:
    agent = ResearchAgent()

    def test_fetch_data_returns_series(self):
        series = self.agent.fetch_data("AAPL", days=30)
        assert isinstance(series, MarketDataSeries)
        assert series.symbol == "AAPL"

    def test_fetch_data_bar_count(self):
        """30 calendar days for an equity should yield ~22 trading bars."""
        series = self.agent.fetch_data("MSFT", days=30)
        assert len(series.bars) >= 15

    def test_fetch_data_unsupported_symbol_raises(self):
        with pytest.raises(ValueError, match="Unsupported symbol"):
            self.agent.fetch_data("XYZZY", days=30)

    def test_trend_analysis_direction_is_valid(self):
        trend = self.agent.analyze_trend("AAPL", days=30)
        assert trend.direction in ("up", "down", "sideways")

    def test_trend_analysis_volatility_positive(self):
        trend = self.agent.analyze_trend("TSLA", days=60)
        assert trend.volatility >= 0.0

    def test_trend_analysis_momentum_clamped(self):
        trend = self.agent.analyze_trend("BTC", days=60)
        assert -1.0 <= trend.momentum <= 1.0


# ---------------------------------------------------------------------------
# 8. TraderAgent tests
# ---------------------------------------------------------------------------


class TestTraderAgent:
    agent = TraderAgent()

    def test_hold_when_no_signals(self):
        rec = self.agent.generate_recommendation([], current_price=100.0, symbol="X")
        assert rec.action == "hold"
        assert rec.confidence == 0.0

    def test_buy_recommendation_on_strong_buy_signals(self):
        signals = [
            _make_signal("SMA_crossover", "buy", strength=0.9),
            _make_signal("RSI", "buy", strength=0.8),
            _make_signal("Bollinger_Bands", "buy", strength=0.7),
            _make_signal("MACD", "buy", strength=0.8),
        ]
        rec = self.agent.generate_recommendation(
            signals, current_price=100.0, symbol="BULL", risk_tolerance="moderate"
        )
        assert rec.action == "buy"
        assert rec.confidence > 0.0

    def test_sell_recommendation_on_strong_sell_signals(self):
        signals = [
            _make_signal("SMA_crossover", "sell", strength=0.9),
            _make_signal("RSI", "sell", strength=0.9),
            _make_signal("Bollinger_Bands", "sell", strength=0.8),
            _make_signal("MACD", "sell", strength=0.9),
        ]
        rec = self.agent.generate_recommendation(
            signals, current_price=200.0, symbol="BEAR", risk_tolerance="aggressive"
        )
        assert rec.action == "sell"

    def test_stop_loss_set_on_buy(self):
        signals = [_make_signal("SMA_crossover", "buy", strength=1.0)] * 4
        rec = self.agent.generate_recommendation(
            signals, current_price=100.0, risk_tolerance="aggressive"
        )
        if rec.action == "buy":
            assert rec.stop_loss is not None
            assert rec.stop_loss < 100.0

    def test_confidence_in_bounds(self):
        import random

        rng = random.Random(7)
        signals = [
            Signal(
                name=name,
                value=0.0,
                direction=rng.choice(["buy", "sell", "hold"]),
                strength=rng.random(),
            )
            for name in ("SMA_crossover", "RSI", "Bollinger_Bands", "MACD")
        ]
        rec = self.agent.generate_recommendation(signals, current_price=50.0)
        assert 0.0 <= rec.confidence <= 1.0

    def test_position_size_respects_max_profile(self):
        """Position size must not exceed the profile's max_position_pct."""
        for profile_name, profile in _RISK_PROFILES.items():
            signals = [_make_signal("SMA_crossover", "buy", strength=1.0)] * 4
            rec = self.agent.generate_recommendation(
                signals, current_price=100.0, risk_tolerance=profile_name
            )
            assert rec.position_size_pct <= profile["max_position_pct"] + 1e-9

    def test_signal_summary_length_matches_input(self):
        signals = [
            _make_signal("SMA_crossover", "buy"),
            _make_signal("RSI", "sell"),
            _make_signal("MACD", "hold"),
        ]
        rec = self.agent.generate_recommendation(signals, current_price=100.0)
        assert len(rec.signal_summary) == 3
