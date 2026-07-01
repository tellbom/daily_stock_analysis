# -*- coding: utf-8 -*-
"""Windowed quant factor context for LLM analysis.

This module borrows the factor-family shape from the sibling stock-analysis
project, but it only uses data already available in DSA. It emits a compact
single-stock/small-batch context instead of training features or cross-project
runtime dependencies.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.factors.llm_factor_summary import (
    _compact_number,
    _format_date,
    _normalize_bars,
    _overlay_realtime_quote,
    _period_return,
    _range_position,
    _safe_float,
    _strip_empty,
)


DECISION_WINDOWS = (1, 3, 5, 10, 20)
COMPUTE_WINDOW_BARS = 120


def build_quant_factor_context(
    bars: Optional[pd.DataFrame],
    *,
    code: str = "",
    source: Optional[str] = None,
    factor_summary: Optional[Dict[str, Any]] = None,
    fundamental_context: Optional[Dict[str, Any]] = None,
    realtime_quote: Optional[Any] = None,
    event_context: Optional[Dict[str, Any]] = None,
    decision_windows: Iterable[int] = DECISION_WINDOWS,
) -> Dict[str, Any]:
    """Build an LLM-ready windowed factor context from local DSA artifacts."""
    df = _normalize_bars(bars) if isinstance(bars, pd.DataFrame) else pd.DataFrame()
    if not df.empty:
        df = _overlay_realtime_quote(df.tail(COMPUTE_WINDOW_BARS).copy(), realtime_quote)

    technical = _build_technical_windows(df, factor_summary, decision_windows)
    capital_flow = _build_capital_flow(fundamental_context)
    valuation = _build_valuation(fundamental_context, realtime_quote)
    industry = _build_industry(fundamental_context)
    fundamentals = _build_fundamentals(fundamental_context, technical.get("as_of"))
    events = _build_event_context(event_context)
    margin = _unsupported("margin", "margin_source_not_integrated")
    lockup_event = _build_unlock_event(event_context)

    warnings = _collect_warnings(
        technical,
        capital_flow,
        valuation,
        industry,
        fundamentals,
        events,
        margin,
        lockup_event,
    )
    status = _overall_status(technical, capital_flow, valuation, industry, fundamentals, events)

    return _strip_empty(
        {
            "status": status,
            "code": code,
            "source": source or technical.get("source") or "local_context",
            "as_of": technical.get("as_of") or _as_of_from_summary(factor_summary),
            "window_policy": {
                "mode": "single_stock_llm_context",
                "compute_window_bars": COMPUTE_WINDOW_BARS,
                "decision_windows": list(decision_windows),
                "primary_horizon": "next_trading_day",
                "background_windows": [60, 120],
            },
            "technical": technical,
            "capital_flow": capital_flow,
            "valuation": valuation,
            "industry": industry,
            "fundamentals": fundamentals,
            "margin": margin,
            "event_unlock": lockup_event,
            "events": events,
            "limitations": _limitations(capital_flow, industry, margin, lockup_event),
            "warnings": warnings,
        }
    )


def _build_technical_windows(
    df: pd.DataFrame,
    factor_summary: Optional[Dict[str, Any]],
    decision_windows: Iterable[int],
) -> Dict[str, Any]:
    summary = factor_summary if isinstance(factor_summary, dict) else {}
    if df.empty and not summary:
        return _missing("technical", "daily_bars_missing")

    as_of = _format_date(df.iloc[-1].get("date")) if not df.empty else _as_of_from_summary(summary)
    returns = summary.get("returns") if isinstance(summary.get("returns"), dict) else {}
    volume_summary = summary.get("volume") if isinstance(summary.get("volume"), dict) else {}
    trend = summary.get("trend") if isinstance(summary.get("trend"), dict) else {}
    momentum = summary.get("momentum") if isinstance(summary.get("momentum"), dict) else {}
    short_term = summary.get("short_term") if isinstance(summary.get("short_term"), dict) else {}
    risk = summary.get("risk") if isinstance(summary.get("risk"), dict) else {}
    warmup = summary.get("warmup") if isinstance(summary.get("warmup"), dict) else {}

    window_returns: Dict[str, Any] = {}
    volume_windows: Dict[str, Any] = {}
    if not df.empty:
        latest_volume = _safe_float(df.iloc[-1].get("volume"))
        for window in decision_windows:
            key = f"{int(window)}d"
            window_returns[key] = _period_return(df, int(window))
            volume_windows[key] = _volume_ratio(df, int(window), latest_volume)
    else:
        for key, summary_key in (("1d", "return_1d_pct"), ("5d", "return_5d_pct"), ("20d", "return_20d_pct")):
            window_returns[key] = returns.get(summary_key)

    range_20d = short_term.get("range_position_20d")
    if range_20d is None and not df.empty:
        range_20d = _range_position(df, 20)

    status = _factor_status(summary.get("status"))
    if not summary and not df.empty:
        status = "partial"

    return _strip_empty(
        {
            "status": status,
            "source": summary.get("source") or "daily_bars",
            "as_of": as_of,
            "bar_count": summary.get("bar_count") or (int(len(df)) if not df.empty else None),
            "window_returns_pct": window_returns,
            "volume_ratios": volume_windows,
            "trend": {
                "ma_alignment": trend.get("ma_alignment"),
                "ma5_distance_pct": trend.get("ma5_distance_pct"),
                "ma20_distance_pct": trend.get("ma20_distance_pct"),
            },
            "momentum": {
                "macd_state": momentum.get("macd_state"),
                "rsi_6": momentum.get("rsi_6"),
                "rsi_12": momentum.get("rsi_12"),
                "kdj_j": momentum.get("kdj_j"),
                "adx_14": momentum.get("adx_14"),
                "roc_10_pct": momentum.get("roc_10_pct"),
            },
            "short_term": {
                "range_position_20d": range_20d,
                "volume_ratio_5d": volume_summary.get("volume_ratio_5d"),
                "volume_ratio_20d": volume_summary.get("volume_ratio_20d"),
                "volume_pct_rank_20d": volume_summary.get("volume_pct_rank_20d"),
                "active_risk_flags": risk.get("active") if isinstance(risk.get("active"), list) else None,
            },
            "warmup": {
                "is_full_warmup": warmup.get("is_full_warmup"),
                "minimum_available_bars": warmup.get("minimum_available_bars"),
                "full_warmup_bars": warmup.get("full_warmup_bars"),
            },
        }
    )


def _build_capital_flow(fundamental_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    block = _fundamental_block(fundamental_context, "capital_flow")
    data = _block_data(block)
    stock_flow = data.get("stock_flow") if isinstance(data.get("stock_flow"), dict) else {}
    sector_rankings = (
        data.get("sector_rankings")
        if isinstance(data.get("sector_rankings"), dict)
        else {}
    )
    main = _safe_float(stock_flow.get("main_net_inflow"))
    flow_5d = _safe_float(stock_flow.get("inflow_5d"))
    flow_10d = _safe_float(stock_flow.get("inflow_10d"))
    available = any(value is not None for value in (main, flow_5d, flow_10d)) or bool(
        sector_rankings.get("top") or sector_rankings.get("bottom")
    )
    if not available:
        return _with_block_status(block, "capital_flow", "capital_flow_missing")

    return _strip_empty(
        {
            "status": _block_status(block, "available"),
            "source": _source_from_block(block, "fundamental_context.capital_flow"),
            "stock_flow": {
                "main_net_inflow": _compact_number(main),
                "inflow_5d": _compact_number(flow_5d),
                "inflow_10d": _compact_number(flow_10d),
                "bias": _flow_bias(main, flow_5d, flow_10d),
                "reversal": _flow_reversal(main, flow_5d, flow_10d),
            },
            "sector_rankings": {
                "top": _sector_names(sector_rankings.get("top")),
                "bottom": _sector_names(sector_rankings.get("bottom")),
            },
            "usage": "price_position_filter",
        }
    )


def _build_valuation(
    fundamental_context: Optional[Dict[str, Any]],
    realtime_quote: Optional[Any],
) -> Dict[str, Any]:
    block = _fundamental_block(fundamental_context, "valuation")
    data = _block_data(block)
    if realtime_quote is not None:
        data = {
            **data,
            "pe_ratio": _first_non_empty(data.get("pe_ratio"), getattr(realtime_quote, "pe_ratio", None)),
            "pb_ratio": _first_non_empty(data.get("pb_ratio"), getattr(realtime_quote, "pb_ratio", None)),
            "total_mv": _first_non_empty(data.get("total_mv"), getattr(realtime_quote, "total_mv", None)),
            "circ_mv": _first_non_empty(data.get("circ_mv"), getattr(realtime_quote, "circ_mv", None)),
        }
    if not any(data.get(key) is not None for key in ("pe_ratio", "pb_ratio", "total_mv", "circ_mv")):
        return _with_block_status(block, "valuation", "valuation_missing")
    status = _block_status(block, "available")
    if status in {"missing", "not_supported"}:
        status = "available"

    return _strip_empty(
        {
            "status": status,
            "source": _source_from_block(block, "fundamental_context.valuation"),
            "pe_ratio": _compact_number(data.get("pe_ratio")),
            "pb_ratio": _compact_number(data.get("pb_ratio")),
            "total_mv": _compact_number(data.get("total_mv")),
            "circ_mv": _compact_number(data.get("circ_mv")),
            "note": "raw_snapshot_only_no_cross_section_rank",
        }
    )


def _build_industry(fundamental_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(fundamental_context, dict):
        return _missing("industry", "fundamental_context_missing")
    belong_boards = fundamental_context.get("belong_boards")
    boards = _block_data(_fundamental_block(fundamental_context, "boards"))
    concept_boards = _block_data(_fundamental_block(fundamental_context, "concept_boards"))
    names = []
    if isinstance(belong_boards, list):
        names.extend(_board_names(belong_boards))
    names.extend(_board_names(boards.get("boards") if isinstance(boards, dict) else None))
    names.extend(_board_names(concept_boards.get("boards") if isinstance(concept_boards, dict) else None))
    names = _dedupe(names)
    if not names:
        return {
            "status": "not_supported",
            "factor_family": "industry",
            "missing_reason": "industry_rank_source_not_integrated",
        }
    return {
        "status": "partial",
        "factor_family": "industry",
        "boards": names[:8],
        "note": "board_membership_only_no_industry_rank",
    }


def _build_fundamentals(
    fundamental_context: Optional[Dict[str, Any]],
    as_of: Optional[str],
) -> Dict[str, Any]:
    earnings = _block_data(_fundamental_block(fundamental_context, "earnings"))
    growth = _block_data(_fundamental_block(fundamental_context, "growth"))
    financial_report = (
        earnings.get("financial_report")
        if isinstance(earnings.get("financial_report"), dict)
        else {}
    )
    dividend = earnings.get("dividend") if isinstance(earnings.get("dividend"), dict) else {}
    if not any((financial_report, dividend, growth)):
        return _with_block_status(
            _fundamental_block(fundamental_context, "earnings"),
            "fundamentals",
            "fundamental_metrics_missing",
        )

    report_date = _first_non_empty(
        financial_report.get("report_date"),
        financial_report.get("period_end"),
        growth.get("report_date") if isinstance(growth, dict) else None,
    )
    return _strip_empty(
        {
            "status": "partial",
            "source": "fundamental_context",
            "financial_report": {
                "report_date": report_date,
                "lag_days": _lag_days(as_of, report_date),
                "revenue": financial_report.get("revenue"),
                "net_profit_parent": financial_report.get("net_profit_parent"),
                "operating_cash_flow": financial_report.get("operating_cash_flow"),
                "roe": financial_report.get("roe"),
            },
            "dividend": {
                "ttm_dividend_yield_pct": dividend.get("ttm_dividend_yield_pct"),
                "ttm_cash_dividend_per_share": dividend.get("ttm_cash_dividend_per_share"),
            },
            "growth": growth if isinstance(growth, dict) else None,
        }
    )


def _build_event_context(event_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(event_context, dict):
        return _missing("events", "event_context_missing")
    digest = event_context.get("event_digest") if isinstance(event_context.get("event_digest"), dict) else {}
    items = event_context.get("items") if isinstance(event_context.get("items"), list) else []
    return _strip_empty(
        {
            "status": _factor_status(event_context.get("status")),
            "as_of": event_context.get("as_of_date"),
            "event_bias": digest.get("event_bias"),
            "positive_catalysts": digest.get("positive_catalysts", [])[:3],
            "negative_risks": digest.get("negative_risks", [])[:3],
            "important_event_count": len(items),
        }
    )


def _build_unlock_event(event_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(event_context, dict):
        return _missing("event_unlock", "event_context_missing")
    items = event_context.get("items") if isinstance(event_context.get("items"), list) else []
    unlock_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        if item.get("type") == "risk_event" and "unlock" in {str(tag) for tag in tags}:
            unlock_items.append(item)

    warnings = [
        str(warning)
        for warning in event_context.get("warnings", [])
        if "unlock_event" in str(warning)
    ] if isinstance(event_context.get("warnings"), list) else []
    if unlock_items:
        nearest = sorted(unlock_items, key=lambda item: str(item.get("publish_time") or ""))[0]
        return _strip_empty(
            {
                "status": "available",
                "source": nearest.get("source") or "event_context",
                "nearest_date": nearest.get("publish_time"),
                "risk_level": nearest.get("risk_level"),
                "title": nearest.get("title"),
                "summary": nearest.get("summary"),
                "event_count": len(unlock_items),
                "usage": "risk_event_filter",
            }
        )
    if warnings:
        return {
            "status": "failed",
            "factor_family": "event_unlock",
            "missing_reason": "unlock_event_parse_failed",
            "warnings": warnings,
        }
    return {
        "status": "missing",
        "factor_family": "event_unlock",
        "missing_reason": "no_upcoming_unlock_event_found",
    }


def _volume_ratio(df: pd.DataFrame, window: int, latest_volume: Optional[float]) -> Optional[float]:
    if latest_volume is None or latest_volume <= 0 or len(df) < window:
        return None
    mean_volume = _safe_float(df.tail(window)["volume"].mean()) if "volume" in df.columns else None
    if mean_volume is None or mean_volume == 0:
        return None
    return _compact_number(latest_volume / mean_volume)


def _fundamental_block(context: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    block = context.get(name)
    return block if isinstance(block, dict) else {}


def _block_data(block: Dict[str, Any]) -> Dict[str, Any]:
    data = block.get("data") if isinstance(block, dict) else None
    return data if isinstance(data, dict) else {}


def _block_status(block: Dict[str, Any], default: str) -> str:
    status = str(block.get("status") or default).strip().lower() if isinstance(block, dict) else default
    if status in {"ok", "success"}:
        return "available"
    if status in {"failed", "fetch_failed"}:
        return "failed"
    if status in {"missing", "not_supported", "partial", "available"}:
        return status
    return default


def _factor_status(status: Any) -> str:
    text = str(status or "").strip().lower()
    if text in {"available", "partial", "missing", "not_supported", "failed"}:
        return text
    return "partial"


def _source_from_block(block: Dict[str, Any], default: str) -> str:
    chain = block.get("source_chain") if isinstance(block, dict) else None
    if isinstance(chain, list) and chain:
        providers = []
        for item in chain:
            if isinstance(item, dict) and item.get("provider"):
                providers.append(str(item.get("provider")))
            elif isinstance(item, str):
                providers.append(item)
        if providers:
            return ",".join(providers[:3])
    return default


def _with_block_status(block: Dict[str, Any], family: str, reason: str) -> Dict[str, Any]:
    status = _block_status(block, "missing")
    if status in {"available", "partial"}:
        status = "missing"
    return {
        "status": status,
        "factor_family": family,
        "missing_reason": reason,
    }


def _missing(family: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "missing",
        "factor_family": family,
        "missing_reason": reason,
    }


def _unsupported(family: str, reason: str) -> Dict[str, Any]:
    return {
        "status": "not_supported",
        "factor_family": family,
        "missing_reason": reason,
    }


def _flow_bias(main: Optional[float], flow_5d: Optional[float], flow_10d: Optional[float]) -> str:
    positives = sum(1 for value in (main, flow_5d, flow_10d) if value is not None and value > 0)
    negatives = sum(1 for value in (main, flow_5d, flow_10d) if value is not None and value < 0)
    if positives >= 2:
        return "inflow"
    if negatives >= 2:
        return "outflow"
    if main is not None and main > 0:
        return "short_inflow"
    if main is not None and main < 0:
        return "short_outflow"
    return "neutral"


def _flow_reversal(main: Optional[float], flow_5d: Optional[float], flow_10d: Optional[float]) -> Optional[str]:
    if main is None:
        return None
    longer = flow_5d if flow_5d is not None else flow_10d
    if longer is None:
        return None
    if main > 0 and longer < 0:
        return "short_term_turning_in"
    if main < 0 and longer > 0:
        return "short_term_turning_out"
    return "consistent"


def _sector_names(items: Any) -> list[str]:
    names = []
    if isinstance(items, list):
        for item in items[:5]:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("sector") or "").strip()
            else:
                name = str(item or "").strip()
            if name:
                names.append(name)
    return names


def _board_names(items: Any) -> list[str]:
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("board_name") or item.get("板块名称")
        else:
            name = item
        text = str(name or "").strip()
        if text:
            names.append(text)
    return names


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_non_empty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _as_of_from_summary(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    return summary.get("as_of") if isinstance(summary, dict) else None


def _lag_days(as_of: Optional[str], report_date: Any) -> Optional[int]:
    if not as_of or not report_date:
        return None
    try:
        as_of_date = pd.to_datetime(as_of).date()
        report = pd.to_datetime(report_date).date()
    except Exception:
        return None
    if not isinstance(as_of_date, date) or not isinstance(report, date):
        return None
    return (as_of_date - report).days


def _overall_status(*families: Dict[str, Any]) -> str:
    statuses = [family.get("status") for family in families if isinstance(family, dict)]
    if any(status == "available" for status in statuses):
        return "available"
    if any(status == "partial" for status in statuses):
        return "partial"
    if any(status == "not_supported" for status in statuses):
        return "partial"
    return "missing"


def _collect_warnings(*families: Dict[str, Any]) -> list[str]:
    warnings = []
    for family in families:
        if not isinstance(family, dict):
            continue
        status = family.get("status")
        reason = family.get("missing_reason")
        name = family.get("factor_family")
        if status in {"missing", "failed", "not_supported"} and reason:
            warnings.append(f"{name or 'factor'}:{reason}")
    return warnings


def _limitations(
    capital_flow: Dict[str, Any],
    industry: Dict[str, Any],
    margin: Dict[str, Any],
    lockup_event: Dict[str, Any],
) -> list[str]:
    items = []
    if capital_flow.get("status") not in {"available", "partial"}:
        items.append("capital_flow_available_only_when_A_share_source_supports_it")
    if industry.get("status") != "available":
        items.append("industry_cross_section_rank_not_computed_in_single_stock_context")
    if margin.get("status") == "not_supported":
        items.append("margin_balance_factor_not_integrated")
    if lockup_event.get("status") == "failed":
        items.append("unlock_event_parse_failed")
    return items
