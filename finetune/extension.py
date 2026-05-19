"""Fine-tuning for multi-agent finance research (disclaimer-aware tone, scenario conditioning)."""

from ml_core import configure_logging

logger = configure_logging(__name__)


def describe_market_llm_finetune_playbook() -> dict:
    """Execute the describe market llm finetune playbook routine."""
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
    """Execute the main routine."""
    import json

    logger.info(json.dumps(describe_market_llm_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
