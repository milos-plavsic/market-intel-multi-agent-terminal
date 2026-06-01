"""Market intelligence multi-agent system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ml_core import configure_logging
from ml_core.exceptions import ApplicationError

from app.market_data import (
    detect_volume_anomalies,
    fetch_symbol_snapshot,
    format_research_content,
)

logger = configure_logging("market-intel")


class AgentRole(str, Enum):
    """Agent roles in the system."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    TRADER = "trader"
    MONITOR = "monitor"


@dataclass
class MarketData:
    """Market data point."""

    symbol: str
    price: float
    change_percent: float
    volume: int
    timestamp: str


@dataclass
class MarketInsight:
    """Market insight from agent."""

    agent: str
    role: AgentRole
    title: str
    content: str
    confidence: float
    timestamp: str


@dataclass
class TradeSignal:
    """Trade signal from system."""

    symbol: str
    action: str  # buy, sell, hold
    strength: float  # 0.0 to 1.0
    target_price: float | None = None
    stop_loss: float | None = None
    rationale: str = ""


class ResearchAgent:
    """Research agent for market analysis."""

    def __init__(self, name: str = "ResearchBot"):
        """Initialize a new ResearchAgent instance."""
        self.name = name
        self.role = AgentRole.RESEARCHER

    async def analyze_market(self, symbols: list[str]) -> list[MarketInsight]:
        """Analyze market for given symbols."""
        logger.info(f"{self.name} analyzing {len(symbols)} symbols")

        insights = []
        for symbol in symbols:
            snapshot = fetch_symbol_snapshot(symbol)
            change = abs(snapshot["change_percent"])
            confidence = min(0.92, 0.65 + change / 100)
            insight = MarketInsight(
                agent=self.name,
                role=self.role,
                title=f"Research Report: {snapshot['symbol']}",
                content=format_research_content(snapshot),
                confidence=round(confidence, 3),
                timestamp=datetime.utcnow().isoformat(),
            )
            insights.append(insight)

        logger.info(f"Generated {len(insights)} insights")
        return insights


class AnalysisAgent:
    """Analysis agent for deep dives."""

    def __init__(self, name: str = "AnalysisBot"):
        """Initialize a new AnalysisAgent instance."""
        self.name = name
        self.role = AgentRole.ANALYST

    async def analyze_sector(self, sector: str) -> MarketInsight:
        """Analyze sector trends."""
        logger.info(f"{self.name} analyzing sector: {sector}")

        insight = MarketInsight(
            agent=self.name,
            role=self.role,
            title=f"Sector Analysis: {sector}",
            content=f"""
Sector Analysis Report: {sector}

Market Dynamics:
- Growth trajectory: Positive
- Competitive landscape: Consolidating
- Regulation: Favorable
- Investment flows: Increasing

Key Drivers:
1. Demand fundamentals improving
2. Supply constraints easing
3. Technology adoption accelerating
4. Capital allocation favorable

Outlook:
- Bullish momentum likely to continue
- Risk/reward ratio attractive
- Volatility expected near earnings
""",
            confidence=0.82,
            timestamp=datetime.utcnow().isoformat(),
        )

        return insight


class TradeAgent:
    """Trading agent for signal generation."""

    def __init__(self, name: str = "TradeBot"):
        """Initialize a new TradeAgent instance."""
        self.name = name
        self.role = AgentRole.TRADER

    async def generate_signals(
        self,
        insights: list[MarketInsight],
    ) -> list[TradeSignal]:
        """Generate trade signals from insights."""
        logger.info(f"{self.name} generating signals from {len(insights)} insights")

        signals = []

        # Aggregate insights to generate signals
        for insight in insights:
            if insight.confidence > 0.75:
                # Extract symbol (simplified)
                symbols = [word for word in insight.title.split() if len(word) <= 5]
                symbol = symbols[-1] if symbols else "UNKNOWN"

                signal = TradeSignal(
                    symbol=symbol,
                    action="buy" if insight.confidence > 0.80 else "hold",
                    strength=insight.confidence,
                    target_price=None,
                    stop_loss=None,
                    rationale=f"Based on {insight.agent} research",
                )
                signals.append(signal)

        logger.info(f"Generated {len(signals)} trade signals")
        return signals


class MarketMonitor:
    """Monitor for market anomalies."""

    def __init__(self, name: str = "MonitorBot"):
        """Initialize a new MarketMonitor instance."""
        self.name = name
        self.role = AgentRole.MONITOR

    async def check_anomalies(self, symbols: list[str] | None = None) -> list[str]:
        """Check for market anomalies using recent volume data."""
        logger.info(f"{self.name} checking for anomalies")
        watched = symbols or ["SPY", "QQQ"]
        anomalies = detect_volume_anomalies(watched)
        logger.info(f"Found {len(anomalies)} potential anomalies")
        return anomalies


class MultiAgentSystem:
    """Orchestrate multiple trading agents."""

    def __init__(self):
        """Initialize a new MultiAgentSystem instance."""
        self.researcher = ResearchAgent("MarketResearcher")
        self.analyst = AnalysisAgent("MarketAnalyst")
        self.trader = TradeAgent("TradeSignalGenerator")
        self.monitor = MarketMonitor("MarketMonitor")
        logger.info("Initialized multi-agent system")

    async def run_analysis(self, symbols: list[str], sector: str) -> dict:
        """Run full market analysis workflow."""
        logger.info(f"Running analysis for {len(symbols)} symbols in {sector}")

        try:
            # Step 1: Research
            research_insights = await self.researcher.analyze_market(symbols)

            # Step 2: Analysis
            sector_insight = await self.analyst.analyze_sector(sector)

            # Step 3: Trading signals
            all_insights = [*research_insights, sector_insight]
            trade_signals = await self.trader.generate_signals(all_insights)

            # Step 4: Monitoring
            anomalies = await self.monitor.check_anomalies(symbols)

            logger.info("Analysis complete")

            return {
                "research_insights": [
                    {
                        "agent": i.agent,
                        "title": i.title,
                        "confidence": i.confidence,
                        "timestamp": i.timestamp,
                    }
                    for i in research_insights
                ],
                "sector_insight": {
                    "title": sector_insight.title,
                    "confidence": sector_insight.confidence,
                },
                "trade_signals": [
                    {
                        "symbol": s.symbol,
                        "action": s.action,
                        "strength": s.strength,
                        "rationale": s.rationale,
                    }
                    for s in trade_signals
                ],
                "anomalies": anomalies,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise ApplicationError(f"Market analysis failed: {e}") from e


def build_brief(ticker: str, sector: str = "technology") -> dict:
    """Build a synchronous market brief for a single ticker.

    Wraps :class:`MultiAgentSystem` so it's reachable from a request handler
    that doesn't want to manage an event loop. For higher throughput call
    ``MultiAgentSystem().run_analysis(...)`` directly from an async handler.
    """
    import asyncio

    if not ticker or not isinstance(ticker, str):
        raise ApplicationError("ticker must be a non-empty string")

    system = MultiAgentSystem()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context; spin up a worker loop.
            new_loop = asyncio.new_event_loop()
            try:
                result = new_loop.run_until_complete(system.run_analysis([ticker], sector))
            finally:
                new_loop.close()
            return result
    except RuntimeError:
        pass
    return asyncio.run(system.run_analysis([ticker], sector))


async def main():
    """Main entry point."""
    logger.info("Starting Market Intelligence System")

    system = MultiAgentSystem()

    symbols = ["AAPL", "MSFT", "GOOGL"]
    sector = "Technology"

    result = await system.run_analysis(symbols, sector)

    logger.info("\n" + "=" * 70)
    logger.info("MARKET INTELLIGENCE REPORT")
    logger.info("=" * 70)
    logger.info(f"\nSymbols Analyzed: {', '.join(symbols)}")
    logger.info(f"Sector: {sector}")
    logger.info(f"\nResearch Insights: {len(result['research_insights'])}")
    logger.info(f"Trade Signals: {len(result['trade_signals'])}")
    logger.info(f"Anomalies Detected: {len(result['anomalies'])}")
    logger.info("\nAnomalies:")
    for anomaly in result["anomalies"]:
        logger.info(f"  - {anomaly}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
