"""Fine-tuning for multi-agent finance research (disclaimer-aware tone, scenario conditioning)."""


def describe_market_llm_finetune_playbook() -> dict:
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
    import json

    print(json.dumps(describe_market_llm_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
