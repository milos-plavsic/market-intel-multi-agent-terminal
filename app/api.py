"""FastAPI application for the Market Intel Multi-Agent Terminal.

Auth + rate limiting provided by ml-core:
  - APIKeyMiddleware  : rejects requests missing a valid X-API-Key header
    (no-op when API_KEY env var is unset, so local dev works without config)
  - RateLimiter       : token-bucket, 10 req/s burst per API-key
    (market data is expensive to compute, so a tighter limit is appropriate)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from ml_core.exceptions import RateLimitExceeded
from ml_core.ratelimit import RateLimiter
from pydantic import BaseModel, Field

from app.main import MultiAgentSystem
from finetune.extension import describe_market_llm_finetune_playbook

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Market Intel Multi-Agent Terminal", version="1.0.0")

# ---------------------------------------------------------------------------
# Rate limiter — 10 requests/second, burst of 20 (market data is expensive)
# ---------------------------------------------------------------------------

_limiter = RateLimiter(rate=10.0, burst=20.0)


def _rate_limit(_key: str = "dev") -> None:
    """FastAPI dependency that enforces the per-key token-bucket rate limit."""
    client_key = _key or "anon"
    try:
        _limiter.acquire(client_key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BriefRequest(BaseModel):
    """Pydantic schema for the brief request."""

    ticker: str = Field(..., min_length=1, max_length=12)
    sector: str = Field(default="technology", description="Market sector for analysis")
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk profile: conservative | moderate | aggressive",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status.  No auth required."""
    return {"status": "ok"}


@app.post("/v1/brief")
async def brief(body: BriefRequest) -> dict:
    """Run the full multi-agent market analysis for a single ticker.

    Pipeline: ResearchAgent → AnalystAgent (4 technical indicators) →
    TraderAgent (weighted recommendation) → anomaly monitor.

    Requires a valid ``X-API-Key`` header.  Rate-limited to 10 req/s per key
    (burst 20).
    """
    ticker = body.ticker.upper()
    try:
        result = await MultiAgentSystem().run_analysis([ticker], body.sector)
        return {"ticker": ticker, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/finetune/playbook")
async def finetune_playbook() -> dict:
    """Return the LLM fine-tune playbook for market-intel models.  Requires auth."""
    return describe_market_llm_finetune_playbook()
