# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.factors import build_llm_factor_summary, build_quant_factor_context


def _bars(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    base = pd.Series(range(rows), dtype=float)
    close = 100 + base * 0.5
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000000 + base * 10000,
            "amount": 100000000 + base * 1000000,
        }
    )


def test_build_quant_factor_context_includes_window_and_capital_flow() -> None:
    bars = _bars()
    factor_summary = build_llm_factor_summary(bars, code="600519")
    context = build_quant_factor_context(
        bars,
        code="600519",
        source="storage.get_data_range",
        factor_summary=factor_summary,
        fundamental_context={
            "capital_flow": {
                "status": "available",
                "data": {
                    "stock_flow": {
                        "main_net_inflow": 1200000,
                        "inflow_5d": 5000000,
                        "inflow_10d": -1000000,
                    },
                    "sector_rankings": {
                        "top": [{"name": "白酒"}],
                        "bottom": [{"name": "地产"}],
                    },
                },
            },
            "valuation": {
                "status": "available",
                "data": {"pe_ratio": 28.5, "pb_ratio": 6.2, "total_mv": 2000000000000},
            },
            "belong_boards": [{"name": "白酒"}, {"name": "消费"}],
            "earnings": {
                "status": "partial",
                "data": {
                    "financial_report": {
                        "report_date": "2026-03-31",
                        "revenue": 1000000000,
                        "net_profit_parent": 300000000,
                    }
                },
            },
        },
        event_context={
            "status": "available",
            "as_of_date": "2026-05-24",
            "items": [
                {
                    "type": "risk_event",
                    "title": "贵州茅台未来12天存在限售股解禁",
                    "publish_time": "2026-06-05",
                    "source": "akshare.stock_restricted_release_detail_em",
                    "summary": "解禁日期=2026-06-05；占解禁前流通市值比例=6.2%",
                    "tags": ["risk_event", "unlock", "risk"],
                    "risk_level": "high",
                    "is_confirmed": True,
                }
            ],
            "event_digest": {"event_bias": "negative", "negative_risks": ["解禁"]},
        },
    )

    assert context["status"] == "available"
    assert context["window_policy"]["decision_windows"] == [1, 3, 5, 10, 20]
    assert context["technical"]["window_returns_pct"]["5d"] is not None
    assert context["capital_flow"]["stock_flow"]["bias"] == "inflow"
    assert context["capital_flow"]["sector_rankings"]["top"] == ["白酒"]
    assert context["valuation"]["pe_ratio"] == 28.5
    assert "白酒" in context["industry"]["boards"]
    assert context["fundamentals"]["financial_report"]["lag_days"] is not None
    assert context["margin"]["status"] == "not_supported"
    assert context["event_unlock"]["status"] == "available"
    assert context["event_unlock"]["nearest_date"] == "2026-06-05"


def test_build_quant_factor_context_marks_missing_flow_without_ohlc_fallback() -> None:
    context = build_quant_factor_context(
        pd.DataFrame(),
        code="AAPL",
        fundamental_context={
            "valuation": {
                "status": "available",
                "data": {"pe_ratio": 31.2, "pb_ratio": 9.4},
            }
        },
    )

    assert context["status"] == "available"
    assert context["technical"]["missing_reason"] == "daily_bars_missing"
    assert context["capital_flow"]["missing_reason"] == "capital_flow_missing"
    assert context["event_unlock"]["missing_reason"] == "event_context_missing"
    assert "capital_flow_available_only_when_A_share_source_supports_it" in context["limitations"]


def test_build_quant_factor_context_falls_back_to_realtime_valuation_snapshot() -> None:
    context = build_quant_factor_context(
        _bars(),
        code="600183",
        factor_summary=build_llm_factor_summary(_bars(), code="600183"),
        fundamental_context={
            "valuation": {
                "status": "not_supported",
                "data": {
                    "pe_ratio": None,
                    "pb_ratio": None,
                    "total_mv": None,
                    "circ_mv": None,
                },
            }
        },
        realtime_quote=SimpleNamespace(
            pe_ratio=101.9,
            pb_ratio=25.04,
            total_mv=400319000000.0,
            circ_mv=394614000000.0,
        ),
    )

    assert context["valuation"]["status"] == "available"
    assert context["valuation"]["pe_ratio"] == 101.9
    assert context["valuation"]["pb_ratio"] == 25.04
    assert context["valuation"]["total_mv"] == 400319000000.0
