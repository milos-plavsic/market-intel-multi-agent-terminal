from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import build_brief
from finetune.extension import describe_market_llm_finetune_playbook

app = FastAPI(title="Market Intel Multi-Agent Terminal", version="0.1.0")


class BriefRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/brief")
def brief(body: BriefRequest) -> dict:
    return build_brief(body.ticker.upper())


@app.get("/v1/finetune/playbook")
def finetune_playbook() -> dict:
    return describe_market_llm_finetune_playbook()
