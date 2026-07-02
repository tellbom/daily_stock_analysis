# -*- coding: utf-8 -*-
"""Best-effort US technology market summary for A-share context."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


_INDEX_SYMBOLS: Tuple[Tuple[str, str, str], ...] = (
    ("nasdaq", "^IXIC", "纳斯达克综合指数"),
    ("sp500_technology", "XLK", "标普科技板块ETF(XLK代理)"),
    ("philadelphia_semiconductor_index", "^SOX", "费城半导体指数"),
)

_GROUPS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "mega_cap_tech": (
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet"),
        ("META", "Meta"),
        ("AMZN", "Amazon"),
        ("TSLA", "Tesla"),
    ),
    "ai_semiconductor_chain": (
        ("NVDA", "NVIDIA"),
        ("AMD", "AMD"),
        ("AVGO", "Broadcom"),
        ("MU", "Micron"),
        ("QCOM", "Qualcomm"),
        ("INTC", "Intel"),
    ),
    "ev_chain": (
        ("TSLA", "Tesla"),
        ("RIVN", "Rivian"),
        ("LCID", "Lucid"),
    ),
}


def build_us_tech_summary() -> Dict[str, Any]:
    """Return a fail-open US tech summary for market review and stock prompts."""

    symbols = _unique_symbols(
        [symbol for _, symbol, _ in _INDEX_SYMBOLS]
        + [symbol for group in _GROUPS.values() for symbol, _ in group]
    )
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        return _missing_summary(f"import_yfinance:{type(exc).__name__}")

    try:
        frame = yf.download(
            tickers=" ".join(symbols),
            period="7d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=8,
        )
    except Exception as exc:
        logger.warning("US tech summary yfinance fetch failed: %s", exc)
        return _missing_summary(f"fetch_yfinance:{type(exc).__name__}")

    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        quote = _extract_quote(frame, symbol)
        if quote is not None:
            quotes[symbol] = quote

    if not quotes:
        return _missing_summary("no_yfinance_rows")

    payload: Dict[str, Any] = {
        "status": "available" if len(quotes) >= max(3, len(symbols) // 2) else "partial",
        "as_of": _latest_as_of(quotes),
        "source": "yfinance",
        "limitations": [],
    }

    for key, symbol, name in _INDEX_SYMBOLS:
        payload[key] = _quote_block(symbol=symbol, name=name, quote=quotes.get(symbol))

    for key, group in _GROUPS.items():
        payload[key] = _group_block(group, quotes)

    missing_symbols = [symbol for symbol in symbols if symbol not in quotes]
    if missing_symbols:
        payload["limitations"].append(f"missing_symbols:{','.join(missing_symbols[:8])}")

    payload["summary"] = _build_summary(payload)
    payload["possible_impact_on_a_share"] = _build_a_share_impact(payload)
    return payload


def _missing_summary(reason: str) -> Dict[str, Any]:
    return {
        "status": "missing",
        "as_of": None,
        "source": "yfinance",
        "missing_reason": reason,
        "nasdaq": {"status": "missing", "symbol": "^IXIC"},
        "sp500_technology": {"status": "missing", "symbol": "XLK"},
        "philadelphia_semiconductor_index": {"status": "missing", "symbol": "^SOX"},
        "mega_cap_tech": {"status": "missing", "items": []},
        "ai_semiconductor_chain": {"status": "missing", "items": []},
        "ev_chain": {"status": "missing", "items": []},
        "summary": "美股科技链数据暂不可用。",
        "possible_impact_on_a_share": "外盘科技映射缺失，A股科技/半导体/新能源链条仅按本地数据判断。",
        "limitations": [reason],
    }


def _unique_symbols(symbols: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for symbol in symbols:
        if symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)
    return unique


def _extract_quote(frame: Any, symbol: str) -> Optional[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    try:
        if hasattr(frame, "columns") and getattr(frame.columns, "nlevels", 1) > 1:
            if symbol not in frame.columns.get_level_values(0):
                return None
            data = frame[symbol]
        else:
            data = frame
        if data is None or getattr(data, "empty", True):
            return None
        close = data["Close"] if "Close" in data else None
        if close is None:
            return None
        close = close.dropna()
        if len(close) < 2:
            return None
        latest = close.iloc[-1]
        previous = close.iloc[-2]
        if pd is not None and (pd.isna(latest) or pd.isna(previous)):
            return None
        latest_float = float(latest)
        previous_float = float(previous)
        if previous_float == 0:
            return None
        as_of = close.index[-1]
        if hasattr(as_of, "date"):
            as_of_text = as_of.date().isoformat()
        else:
            as_of_text = str(as_of)[:10]
        change_pct = (latest_float / previous_float - 1) * 100
        return {
            "status": "available",
            "symbol": symbol,
            "last_close": round(latest_float, 4),
            "change_pct": round(change_pct, 4),
            "as_of": as_of_text,
        }
    except Exception:
        return None


def _quote_block(*, symbol: str, name: str, quote: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not quote:
        return {"status": "missing", "symbol": symbol, "name": name}
    block = dict(quote)
    block["name"] = name
    return block


def _group_block(group: Iterable[Tuple[str, str]], quotes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for symbol, name in group:
        quote = quotes.get(symbol)
        if not quote:
            continue
        item = dict(quote)
        item["name"] = name
        items.append(item)
    if not items:
        return {"status": "missing", "items": []}
    avg_change = sum(float(item.get("change_pct") or 0) for item in items) / len(items)
    leaders = sorted(items, key=lambda item: float(item.get("change_pct") or 0), reverse=True)[:3]
    laggards = sorted(items, key=lambda item: float(item.get("change_pct") or 0))[:3]
    return {
        "status": "available",
        "avg_change_pct": round(avg_change, 4),
        "leaders": [{"symbol": item["symbol"], "change_pct": item["change_pct"]} for item in leaders],
        "laggards": [{"symbol": item["symbol"], "change_pct": item["change_pct"]} for item in laggards],
        "items": items,
    }


def _latest_as_of(quotes: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    dates = [str(quote.get("as_of") or "") for quote in quotes.values() if quote.get("as_of")]
    return max(dates) if dates else datetime.utcnow().date().isoformat()


def _build_summary(payload: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key, label in (
        ("nasdaq", "纳指"),
        ("philadelphia_semiconductor_index", "费半"),
    ):
        block = payload.get(key)
        if isinstance(block, Mapping) and block.get("status") == "available":
            parts.append(f"{label}{float(block.get('change_pct') or 0):+.2f}%")
    ai_chain = payload.get("ai_semiconductor_chain")
    if isinstance(ai_chain, Mapping) and ai_chain.get("status") == "available":
        parts.append(f"AI/半导体链均值{float(ai_chain.get('avg_change_pct') or 0):+.2f}%")
    return "，".join(parts) + "。" if parts else "美股科技链数据可用但信号不完整。"


def _build_a_share_impact(payload: Mapping[str, Any]) -> str:
    semi = payload.get("philadelphia_semiconductor_index")
    ai_chain = payload.get("ai_semiconductor_chain")
    values = []
    for block in (semi, ai_chain):
        if isinstance(block, Mapping) and block.get("status") == "available":
            values.append(float(block.get("change_pct") or block.get("avg_change_pct") or 0))
    if not values:
        return "外盘科技信号不完整，A股科技/半导体/AI链条以本地量价和资金流为主。"
    avg = sum(values) / len(values)
    if avg >= 1.0:
        return "外盘科技风险偏好偏强，可能对A股半导体、AI算力和消费电子形成情绪映射，但仍需A股资金流确认。"
    if avg <= -1.0:
        return "外盘科技链偏弱，可能压制A股半导体、AI算力和消费电子的短线风险偏好，追高需更谨慎。"
    return "外盘科技链影响中性，A股相关板块更应关注本地资金流、板块排名和个股事件。"
