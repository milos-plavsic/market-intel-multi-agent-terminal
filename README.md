# 06 - Market Intelligence Multi-Agent Terminal

[![CI](https://github.com/milos-plavsic/market-intel-multi-agent-terminal/actions/workflows/ci.yml/badge.svg)](https://github.com/milos-plavsic/market-intel-multi-agent-terminal/actions/workflows/ci.yml)
[![Python3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A financial research terminal powered by specialized agents (macro, sentiment, technical, risk) that collaborate to generate scenario-aware market briefs.

## Quickstart

```bash
make install
make run
make api
make test
```

Docker API: `make docker-api`.

## API

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`
- Brief: `POST /v1/brief` with JSON body `{"ticker":"..."}`

## Architecture

```mermaid
flowchart LR
  D[Data ingest] --> M[Macro]
  D --> S[Sentiment]
  D --> T[Technical]
  M --> Y[Synthesis]
  S --> Y
  T --> Y
  Y --> R[Risk manager]
```

## Core Capabilities

- News and market data ingestion.
- Sentiment extraction and event clustering.
- Scenario simulation (`rate hike`, `recession`, `risk-on`).
- Portfolio recommendation drafts with risk commentary.
- Backtesting snapshots against selected benchmarks.

## Architecture (Graph)

`data_ingest -> macro_agent + sentiment_agent + technical_agent -> synthesis_agent -> risk_manager -> scenario_simulator -> report_generator`
