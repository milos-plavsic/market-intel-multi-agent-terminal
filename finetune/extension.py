"""Fine-tuning for multi-agent finance research (disclaimer-aware tone, scenario conditioning)."""

from ml_core import configure_logging

logger = configure_logging(__name__)


def market_training_guide() -> dict:
    """Return notes for market-narrative model fine-tuning."""
    return {
        "compliance_first": [
            "SFT on analyst-style briefs with mandatory risk and disclaimer sections.",
            "DPO to prefer cautious language when volatility features spike.",
        ],
        "multi_agent": [
            "Per-agent LoRA adapters (macro vs sentiment) sharing one base model.",
        ],
        "non_llm": "Backtest-driven weight tuning for scenario simulator knobs (grid search / Bayesian).",
    }


def main() -> None:
    """Main."""
    import json

    logger.info(json.dumps(market_training_guide(), indent=2))


if __name__ == "__main__":
    main()
