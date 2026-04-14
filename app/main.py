import os


def build_brief(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "macro_agent": "neutral",
        "sentiment_agent": "slightly bullish",
        "risk_manager": "medium risk",
    }


def main() -> None:
    ticker = os.getenv("DEMO_TICKER", "NVDA")
    result = build_brief(ticker)
    print("Market Intel Multi-Agent Terminal")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
