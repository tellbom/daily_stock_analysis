# -*- coding: utf-8 -*-

from datetime import date, timedelta

import pandas as pd

from src.factors import build_llm_factor_summary


def _bars(count: int = 65) -> pd.DataFrame:
    start = date(2026, 1, 1)
    rows = []
    for i in range(count):
        close = 10 + i * 0.1
        rows.append(
            {
                "date": start + timedelta(days=i),
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1_000_000 + i * 10_000,
                "amount": (1_000_000 + i * 10_000) * close,
                "data_source": "unit-test",
            }
        )
    return pd.DataFrame(rows)


def test_build_llm_factor_summary_from_daily_bars() -> None:
    summary = build_llm_factor_summary(_bars(), code="600519")

    assert summary["status"] == "available"
    assert summary["source"] == "unit-test"
    assert summary["trend"]["ma_alignment"] == "bullish"
    assert summary["trend"]["ma5_distance_pct"] > 0
    assert summary["momentum"]["macd_state"] in {"bullish", "recovering"}
    assert "rsi_6" in summary["momentum"]
    assert "kdj_k" in summary["momentum"]
    assert "adx_14" in summary["momentum"]
    assert "cci_14" in summary["momentum"]
    assert "willr_14" in summary["momentum"]
    assert "stoch_k" in summary["momentum"]
    assert "atr_14_pct" in summary["volatility"]
    assert "obv_z_20d" in summary["volatility"]
    assert "volume_ratio_20d" in summary["volume"]
    assert "technical" in summary
    assert "kdj" in summary["technical"]
    assert "breadth" in summary["technical"]
    assert summary["warmup"]["is_full_warmup"] is True
    assert summary["short_term"]["limit_ratio_pct"] == 10.0


def test_build_llm_factor_summary_marks_short_term_risks() -> None:
    bars = _bars(65)
    last = bars.index[-1]
    prev_close = float(bars.loc[last - 1, "close"])
    bars.loc[last, "open"] = prev_close * 1.02
    bars.loc[last, "high"] = prev_close * 1.08
    bars.loc[last, "low"] = prev_close * 1.01
    bars.loc[last, "close"] = prev_close * 1.005
    bars.loc[last, "volume"] = bars["volume"].tail(20).mean() * 4
    bars.loc[last, "amount"] = bars.loc[last, "volume"] * bars.loc[last, "close"]

    summary = build_llm_factor_summary(bars, code="600584")

    flags = summary["risk"]["flags"]
    assert flags["high_volume"] is True
    assert flags["high_volume_stall"] is True
    assert "high_volume_stall" in summary["risk"]["active"]
    assert "volume_ratio_20d" in summary["risk"]["drivers"]


def test_build_llm_factor_summary_marks_warmup_partial() -> None:
    summary = build_llm_factor_summary(_bars(8), code="300750")

    assert summary["status"] == "partial"
    assert summary["warnings"] == ["factor_warmup_partial"]
    assert summary["short_term"]["limit_ratio_pct"] == 20.0


def test_build_llm_factor_summary_missing_for_invalid_bars() -> None:
    summary = build_llm_factor_summary(pd.DataFrame([{"date": "2026-01-01"}]))

    assert summary["status"] == "missing"
    assert summary["missing_reason"] == "insufficient_daily_bars"
