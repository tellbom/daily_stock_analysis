# -*- coding: utf-8 -*-
"""Lightweight stock event context for LLM analysis prompts.

This module intentionally lives inside the current project.  It borrows only
endpoint ideas from external research code and does not import or reference any
other local project at runtime.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

from data_provider.base import normalize_stock_code


logger = logging.getLogger(__name__)

EVENT_TYPES = {"news", "announcement", "research", "industry_event", "risk_event"}
_EVENT_PRIORITY = {
    "risk_event": 0,
    "announcement": 0,
    "research": 1,
    "industry_event": 2,
    "news": 3,
}
_POSITIVE_KEYWORDS = (
    "增长",
    "预增",
    "扭亏",
    "中标",
    "合同",
    "合作",
    "回购",
    "增持",
    "分红",
    "突破",
    "上调",
    "买入",
    "推荐",
)
_NEGATIVE_KEYWORDS = (
    "下降",
    "预减",
    "亏损",
    "处罚",
    "监管",
    "问询",
    "减持",
    "诉讼",
    "风险",
    "终止",
    "下调",
    "卖出",
)


class EventFetcher(Protocol):
    def fetch(
        self,
        stock_code: str,
        stock_name: str = "",
        *,
        as_of_date: Optional[Any] = None,
        fundamental_context: Optional[Dict[str, Any]] = None,
        lookback_days: int = 7,
        max_items: int = 40,
    ) -> Dict[str, Any]:
        ...


class AkshareEventFetcher:
    """Fetch a compact event set from AKShare, with endpoint-level fail-open."""

    def fetch(
        self,
        stock_code: str,
        stock_name: str = "",
        *,
        as_of_date: Optional[Any] = None,
        fundamental_context: Optional[Dict[str, Any]] = None,
        lookback_days: int = 7,
        max_items: int = 40,
    ) -> Dict[str, Any]:
        target_date = _coerce_date(as_of_date) or date.today()
        code = _plain_stock_code(stock_code)
        warnings: List[str] = []
        source_chain: List[Dict[str, Any]] = []
        items: List[Dict[str, Any]] = []

        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            return {
                "items": _industry_events_from_fundamentals(
                    fundamental_context,
                    target_date=target_date,
                ),
                "warnings": [f"akshare_import_failed:{type(exc).__name__}"],
                "source_chain": [
                    {"provider": "akshare", "result": "failed", "reason": type(exc).__name__}
                ],
            }

        endpoint_calls: Sequence[Tuple[str, Any, Dict[str, Any]]] = (
            ("announcement", getattr(ak, "stock_individual_notice_report", None), {}),
            ("news", getattr(ak, "stock_news_em", None), {"symbol": code}),
            ("research", getattr(ak, "stock_research_report_em", None), {"symbol": code}),
        )
        for event_type, func, kwargs in endpoint_calls:
            if not callable(func):
                warnings.append(f"{event_type}_endpoint_missing")
                source_chain.append({"provider": f"akshare.{event_type}", "result": "missing"})
                continue
            try:
                if event_type == "announcement":
                    endpoint_items = self._fetch_notices(func, code, stock_name, target_date, lookback_days)
                else:
                    df = func(**kwargs)
                    endpoint_items = _normalise_dataframe_events(
                        df,
                        event_type=event_type,
                        stock_code=code,
                        stock_name=stock_name,
                        source=f"akshare.{getattr(func, '__name__', event_type)}",
                    )
                items.extend(endpoint_items)
                source_chain.append(
                    {
                        "provider": f"akshare.{getattr(func, '__name__', event_type)}",
                        "result": "ok",
                        "count": len(endpoint_items),
                    }
                )
            except Exception as exc:
                logger.debug("%s event fetch failed for %s: %s", event_type, code, exc, exc_info=True)
                warnings.append(f"{event_type}_fetch_failed:{type(exc).__name__}")
                source_chain.append(
                    {
                        "provider": f"akshare.{getattr(func, '__name__', event_type)}",
                        "result": "failed",
                        "reason": type(exc).__name__,
                    }
                )

        for endpoint_name, func in (
            ("stock_yjyg_em", getattr(ak, "stock_yjyg_em", None)),
            ("stock_yjkb_em", getattr(ak, "stock_yjkb_em", None)),
        ):
            if not callable(func):
                continue
            try:
                endpoint_items = self._fetch_earnings_events(
                    func,
                    endpoint_name=endpoint_name,
                    stock_code=code,
                    stock_name=stock_name,
                    target_date=target_date,
                )
                items.extend(endpoint_items)
                source_chain.append(
                    {"provider": f"akshare.{endpoint_name}", "result": "ok", "count": len(endpoint_items)}
                )
            except Exception as exc:
                logger.debug("%s fetch failed for %s: %s", endpoint_name, code, exc, exc_info=True)
                warnings.append(f"{endpoint_name}_fetch_failed:{type(exc).__name__}")
                source_chain.append(
                    {
                        "provider": f"akshare.{endpoint_name}",
                        "result": "failed",
                        "reason": type(exc).__name__,
                    }
                )

        unlock_func = getattr(ak, "stock_restricted_release_detail_em", None)
        if callable(unlock_func):
            try:
                endpoint_items = self._fetch_unlock_events(
                    unlock_func,
                    stock_code=code,
                    stock_name=stock_name,
                    target_date=target_date,
                )
                items.extend(endpoint_items)
                source_chain.append(
                    {
                        "provider": "akshare.stock_restricted_release_detail_em",
                        "result": "ok",
                        "count": len(endpoint_items),
                    }
                )
            except Exception as exc:
                logger.debug("unlock event fetch failed for %s: %s", code, exc, exc_info=True)
                warnings.append(f"unlock_event_fetch_failed:{type(exc).__name__}")
                source_chain.append(
                    {
                        "provider": "akshare.stock_restricted_release_detail_em",
                        "result": "failed",
                        "reason": type(exc).__name__,
                    }
                )
        else:
            warnings.append("unlock_event_endpoint_missing")
            source_chain.append(
                {
                    "provider": "akshare.stock_restricted_release_detail_em",
                    "result": "missing",
                }
            )

        items.extend(_industry_events_from_fundamentals(fundamental_context, target_date=target_date))
        items.sort(key=lambda item: 0 if item.get("type") == "risk_event" else 1)
        return {
            "items": items[: max(1, int(max_items))],
            "warnings": warnings,
            "source_chain": source_chain,
        }

    def _fetch_notices(
        self,
        func: Any,
        stock_code: str,
        stock_name: str,
        target_date: date,
        lookback_days: int,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        days = max(1, min(int(lookback_days or 1), 7))
        begin_date = (target_date - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        end_date = target_date.strftime("%Y-%m-%d")
        df = func(
            security=stock_code,
            symbol="全部",
            begin_date=begin_date,
            end_date=end_date,
        )
        items.extend(
            _normalise_dataframe_events(
                _filter_by_stock(df, stock_code, stock_name),
                event_type="announcement",
                stock_code=stock_code,
                stock_name=stock_name,
                source="akshare.stock_individual_notice_report",
            )
        )
        return items

    def _fetch_unlock_events(
        self,
        func: Any,
        *,
        stock_code: str,
        stock_name: str,
        target_date: date,
        lookahead_days: int = 180,
    ) -> List[Dict[str, Any]]:
        start_date = target_date.strftime("%Y%m%d")
        end_date = (target_date + timedelta(days=max(1, int(lookahead_days)))).strftime("%Y%m%d")
        df = func(start_date=start_date, end_date=end_date)
        filtered = _filter_by_stock(df, stock_code, stock_name)
        if filtered is None or not hasattr(filtered, "empty") or filtered.empty:
            return []

        items: List[Dict[str, Any]] = []
        for record in filtered.to_dict(orient="records"):
            if not isinstance(record, dict):
                continue
            unlock_date = _pick_value(record, "解禁时间", "解禁日期", "date")
            parsed_unlock_date = _coerce_date(unlock_date)
            if parsed_unlock_date is None:
                continue
            days_to_unlock = (parsed_unlock_date - target_date).days
            if days_to_unlock < 0:
                continue
            unlock_type = _pick_text(record, "限售股类型", "类型") or "限售股解禁"
            actual_qty = _safe_float(_pick_value(record, "实际解禁数量", "解禁数量"))
            market_value = _safe_float(_pick_value(record, "实际解禁市值", "解禁市值"))
            ratio_raw = _safe_float(_pick_value(record, "占解禁前流通市值比例", "占流通市值比例", "占总股本比例"))
            ratio_pct = _normalise_ratio_pct(ratio_raw)
            risk_level = _unlock_risk_level(ratio_pct, market_value, days_to_unlock)
            display_name = stock_name or _pick_text(record, "股票简称", "名称") or stock_code
            title = f"{display_name}未来{days_to_unlock}天存在限售股解禁"
            summary_parts = [
                f"解禁日期={parsed_unlock_date.isoformat()}",
                f"限售股类型={unlock_type}",
                f"实际解禁数量={_compact_number(actual_qty)}",
                f"实际解禁市值={_compact_number(market_value)}",
                f"占解禁前流通市值比例={_compact_number(ratio_pct)}%",
                f"风险级别={risk_level}",
            ]
            items.append(
                {
                    "type": "risk_event",
                    "title": title,
                    "publish_time": parsed_unlock_date.isoformat(),
                    "source": "akshare.stock_restricted_release_detail_em",
                    "source_id": _event_source_id(
                        "akshare.stock_restricted_release_detail_em",
                        record,
                        title,
                    ),
                    "summary": "；".join(part for part in summary_parts if "None" not in part),
                    "tags": ["risk_event", "unlock", "risk"],
                    "risk_level": risk_level,
                    "is_confirmed": True,
                }
            )
        return items

    def _fetch_earnings_events(
        self,
        func: Any,
        *,
        endpoint_name: str,
        stock_code: str,
        stock_name: str,
        target_date: date,
    ) -> List[Dict[str, Any]]:
        periods = _recent_report_periods(target_date)
        items: List[Dict[str, Any]] = []
        for period in periods:
            df = func(date=period)
            filtered = _filter_by_stock(df, stock_code, stock_name)
            items.extend(
                _normalise_dataframe_events(
                    filtered,
                    event_type="announcement",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    source=f"akshare.{endpoint_name}",
                    forced_tags=["earnings"],
                )
            )
        return items


def build_event_context(
    stock_code: str,
    stock_name: str = "",
    *,
    as_of_date: Optional[Any] = None,
    fundamental_context: Optional[Dict[str, Any]] = None,
    fetcher: Optional[EventFetcher] = None,
    lookback_days: int = 7,
    max_items: int = 20,
) -> Dict[str, Any]:
    """Return normalized news/event context for prompt injection."""

    target_date = _coerce_date(as_of_date) or date.today()
    warnings: List[str] = []
    source_chain: List[Dict[str, Any]] = []
    raw_items: List[Dict[str, Any]] = []

    provider = fetcher or AkshareEventFetcher()
    try:
        fetched = provider.fetch(
            stock_code,
            stock_name,
            as_of_date=target_date,
            fundamental_context=fundamental_context,
            lookback_days=lookback_days,
            max_items=max(max_items * 2, 40),
        )
    except Exception as exc:
        logger.debug("event context fetcher failed for %s: %s", stock_code, exc, exc_info=True)
        fetched = {
            "items": [],
            "warnings": [f"event_fetcher_failed:{type(exc).__name__}"],
            "source_chain": [{"provider": type(provider).__name__, "result": "failed"}],
        }

    if isinstance(fetched, dict):
        raw_items = [item for item in fetched.get("items", []) if isinstance(item, dict)]
        warnings.extend(_list_text(fetched.get("warnings")))
        source_chain.extend(item for item in fetched.get("source_chain", []) if isinstance(item, dict))
    elif isinstance(fetched, list):
        raw_items = [item for item in fetched if isinstance(item, dict)]

    normalised, filter_warnings = _standardise_events(
        raw_items,
        stock_code=stock_code,
        stock_name=stock_name,
        as_of_date=target_date,
        lookback_days=lookback_days,
    )
    warnings.extend(filter_warnings)
    events = _dedupe_events(normalised)[: max(1, int(max_items or 20))]
    counts = Counter(str(item.get("type")) for item in events)
    digest = build_event_digest(events)

    failed_sources = [item for item in source_chain if item.get("result") == "failed"]
    if events:
        status = "partial" if failed_sources or warnings else "available"
    elif failed_sources or warnings:
        status = "failed"
    else:
        status = "missing"

    return {
        "status": status,
        "as_of_date": target_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "items": events,
        "event_digest": digest,
        "counts_by_type": dict(counts),
        "warnings": _unique_text(warnings),
        "source_chain": source_chain,
    }


def build_event_digest(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {
            "status": "no_recent_events_found",
            "positive_catalysts": [],
            "negative_risks": [],
            "uncertainties": ["no_recent_events_found"],
            "event_bias": "no_recent_events_found",
            "important_events": [],
        }

    positive: List[str] = []
    negative: List[str] = []
    uncertainties: List[str] = []
    important: List[Dict[str, Any]] = []
    score = 0

    for item in events:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        tags = set(_list_text(item.get("tags")))
        is_confirmed = bool(item.get("is_confirmed"))
        risk_level = str(item.get("risk_level") or "medium")
        line = f"{item.get('publish_time', '')} {title}".strip()

        if "risk" in tags or risk_level == "high":
            negative.append(line)
            score -= 2 if is_confirmed else 1
        if "catalyst" in tags:
            positive.append(line)
            score += 2 if is_confirmed else 1
        if not is_confirmed or item.get("type") == "news":
            uncertainties.append(f"{line}（未确认新闻/衍生事件，需公告或权威来源验证）")
        if item.get("type") == "announcement" or risk_level == "high":
            important.append(_digest_event(item))

    if score >= 2:
        bias = "positive"
    elif score <= -2:
        bias = "negative"
    elif positive or negative:
        bias = "mixed"
    else:
        bias = "neutral"

    return {
        "status": "available",
        "positive_catalysts": _unique_text(positive)[:5],
        "negative_risks": _unique_text(negative)[:5],
        "uncertainties": _unique_text(uncertainties)[:5],
        "event_bias": bias,
        "important_events": important[:5],
    }


def _normalise_dataframe_events(
    df: Any,
    *,
    event_type: str,
    stock_code: str,
    stock_name: str,
    source: str,
    forced_tags: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if df is None or not hasattr(df, "empty") or df.empty:
        return []
    records = df.to_dict(orient="records")
    result: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        title = _pick_text(
            record,
            "标题",
            "公告标题",
            "报告名称",
            "研报名称",
            "新闻标题",
            "title",
            "name",
        )
        summary = _summary_from_record(record, title)
        publish_value = _pick_value(
            record,
            "发布时间",
            "发布日期",
            "公告日期",
            "报告日期",
            "日期",
            "time",
            "datetime",
            "publish_time",
        )
        url = _pick_text(record, "链接", "公告链接", "研报链接", "url", "URL", "article_url")
        source_name = _pick_text(record, "文章来源", "媒体", "机构", "评级机构", "source") or source
        tags = _classify_tags(title, summary, event_type, forced_tags=forced_tags)
        risk_level = _risk_level(title, summary, tags)
        result.append(
            {
                "type": event_type if event_type in EVENT_TYPES else "news",
                "title": title,
                "publish_time": publish_value,
                "source": source_name,
                "url": url or None,
                "source_id": _event_source_id(source, record, title),
                "summary": summary,
                "tags": tags,
                "risk_level": risk_level,
                "is_confirmed": event_type in {"announcement", "research"},
            }
        )
    return result


def _standardise_events(
    items: Sequence[Dict[str, Any]],
    *,
    stock_code: str,
    stock_name: str,
    as_of_date: date,
    lookback_days: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    end_dt = datetime.combine(as_of_date, time.max)
    start_dt = end_dt - timedelta(days=max(0, int(lookback_days or 0)))
    result: List[Dict[str, Any]] = []
    for item in items:
        event_type = str(item.get("type") or "news").strip()
        event_type = event_type if event_type in EVENT_TYPES else "news"
        title = _safe_text(item.get("title"))
        publish_dt = _coerce_datetime(item.get("publish_time"))
        if not title:
            warnings.append("event_missing_title")
            continue
        if publish_dt is None:
            warnings.append("event_missing_publish_time")
            continue
        compare_dt = publish_dt.replace(tzinfo=None) if publish_dt.tzinfo else publish_dt
        is_forward_risk_event = event_type == "risk_event"
        if compare_dt > end_dt and not is_forward_risk_event:
            warnings.append("event_future_filtered")
            continue
        if event_type not in {"announcement", "risk_event"} and compare_dt < start_dt:
            warnings.append("event_out_of_window_filtered")
            continue

        summary = _safe_text(item.get("summary")) or title
        tags = _classify_tags(title, summary, event_type, forced_tags=_list_text(item.get("tags")))
        source = _safe_text(item.get("source")) or "unknown"
        url = _safe_text(item.get("url"))
        source_id = _safe_text(item.get("source_id")) or _event_source_id(source, item, title)
        result.append(
            {
                "type": event_type,
                "title": title,
                "publish_time": publish_dt.date().isoformat(),
                "source": source,
                "url": url or None,
                "source_id": source_id,
                "summary": summary,
                "tags": tags,
                "risk_level": _safe_text(item.get("risk_level")) or _risk_level(title, summary, tags),
                "is_confirmed": bool(item.get("is_confirmed")) if event_type != "announcement" else True,
                "stock_code": normalize_stock_code(stock_code),
                "stock_name": stock_name or None,
            }
        )
    result.sort(
        key=lambda item: (
            item.get("publish_time") or "",
            -_EVENT_PRIORITY.get(str(item.get("type")), 9),
        ),
        reverse=True,
    )
    return result, warnings


def _dedupe_events(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = _normalise_title_key(str(item.get("title") or ""))
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = dict(item)
            continue
        current_rank = _EVENT_PRIORITY.get(str(current.get("type")), 9)
        item_rank = _EVENT_PRIORITY.get(str(item.get("type")), 9)
        if item_rank < current_rank:
            best[key] = dict(item)
    deduped = list(best.values())
    deduped.sort(
        key=lambda item: (
            item.get("publish_time") or "",
            -_EVENT_PRIORITY.get(str(item.get("type")), 9),
        ),
        reverse=True,
    )
    return deduped


def _industry_events_from_fundamentals(
    fundamental_context: Optional[Dict[str, Any]],
    *,
    target_date: date,
) -> List[Dict[str, Any]]:
    if not isinstance(fundamental_context, dict):
        return []
    data = fundamental_context.get("data") if isinstance(fundamental_context.get("data"), dict) else fundamental_context
    candidates: List[str] = []
    for path in (
        ("concept_boards", "data", "top"),
        ("concept_boards", "data", "bottom"),
        ("boards", "data", "top"),
        ("boards", "data", "bottom"),
        ("capital_flow", "data", "sector_rankings", "top"),
        ("capital_flow", "data", "sector_rankings", "bottom"),
    ):
        rows = _nested(data, *path)
        if isinstance(rows, list):
            for row in rows[:3]:
                name = _safe_text(row.get("name") if isinstance(row, dict) else row)
                change = row.get("change_pct") if isinstance(row, dict) else None
                if name:
                    suffix = f"({change}%)" if change not in (None, "") else ""
                    candidates.append(f"{name}{suffix}")
    candidates = _unique_text(candidates)
    if not candidates:
        return []
    title = "相关行业/板块事件摘要"
    summary = "相关行业/板块近期表现：" + "、".join(candidates[:8])
    return [
        {
            "type": "industry_event",
            "title": title,
            "publish_time": target_date.isoformat(),
            "source": "fundamental_context.board_rankings",
            "source_id": _hash_text(summary),
            "summary": summary,
            "tags": ["industry_event", "sector"],
            "risk_level": "medium",
            "is_confirmed": False,
        }
    ]


def _filter_by_stock(df: Any, stock_code: str, stock_name: str = "") -> Any:
    if df is None or not hasattr(df, "empty") or df.empty:
        return df
    code = _plain_stock_code(stock_code)
    columns = [str(col) for col in getattr(df, "columns", [])]
    code_columns = [col for col in columns if col in {"代码", "股票代码", "证券代码", "code", "symbol"}]
    name_columns = [col for col in columns if col in {"名称", "股票简称", "股票名称", "证券简称", "name"}]
    mask = pd.Series([False] * len(df), index=df.index)
    for col in code_columns:
        mask = mask | df[col].astype(str).str.replace(r"\D", "", regex=True).str.endswith(code)
    if stock_name:
        for col in name_columns:
            mask = mask | df[col].astype(str).str.contains(re.escape(stock_name), na=False)
    return df[mask] if mask.any() else df.iloc[0:0]


def _classify_tags(
    title: str,
    summary: str,
    event_type: str,
    *,
    forced_tags: Optional[Sequence[str]] = None,
) -> List[str]:
    text = f"{title} {summary}"
    tags = [event_type]
    tags.extend(str(tag) for tag in forced_tags or [] if str(tag).strip())
    if any(keyword in text for keyword in _POSITIVE_KEYWORDS):
        tags.append("catalyst")
    if any(keyword in text for keyword in _NEGATIVE_KEYWORDS):
        tags.append("risk")
    if "业绩" in text or "财报" in text or "利润" in text:
        tags.append("earnings")
    if "评级" in text or "买入" in text or "增持" in text or "下调" in text:
        tags.append("rating")
    if "行业" in text or "板块" in text:
        tags.append("sector")
    return _unique_text(tags)


def _risk_level(title: str, summary: str, tags: Sequence[str]) -> str:
    text = f"{title} {summary}"
    if "risk" in tags and any(keyword in text for keyword in ("处罚", "监管", "诉讼", "亏损", "下调")):
        return "high"
    if "risk" in tags:
        return "medium"
    return "low" if "catalyst" in tags else "medium"


def _summary_from_record(record: Dict[str, Any], title: str) -> str:
    parts: List[str] = []
    for key in (
        "摘要",
        "内容",
        "新闻内容",
        "业绩变动",
        "业绩变动原因",
        "预测指标",
        "评级",
        "评级变动",
        "机构",
    ):
        text = _safe_text(record.get(key))
        if text and text not in parts:
            parts.append(f"{key}: {text}")
    return "；".join(parts[:4]) or title


def _recent_report_periods(target_date: date) -> List[str]:
    periods: List[str] = []
    for year in (target_date.year, target_date.year - 1):
        for suffix in ("0331", "0630", "0930", "1231"):
            period = f"{year}{suffix}"
            if period <= target_date.strftime("%Y%m%d"):
                periods.append(period)
    return periods[-4:]


def _coerce_date(value: Optional[Any]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _coerce_datetime(value)
    return parsed.date() if parsed else None


def _coerce_datetime(value: Optional[Any]) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if hasattr(parsed, "to_pydatetime"):
        return parsed.to_pydatetime()
    return None


def _pick_value(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _pick_text(record: Dict[str, Any], *keys: str) -> str:
    value = _pick_value(record, *keys)
    return _safe_text(value)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "None", "nan"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _compact_number(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _normalise_ratio_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if abs(value) <= 1:
        return value * 100
    return value


def _unlock_risk_level(
    ratio_pct: Optional[float],
    market_value: Optional[float],
    days_to_unlock: int,
) -> str:
    if (
        (ratio_pct is not None and ratio_pct >= 5)
        or (market_value is not None and market_value >= 1_000_000_000)
    ):
        return "high"
    if (
        (ratio_pct is not None and ratio_pct >= 1)
        or (market_value is not None and market_value >= 200_000_000)
        or days_to_unlock <= 14
    ):
        return "medium"
    return "low"


def _list_text(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _unique_text(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _plain_stock_code(stock_code: str) -> str:
    normalized = normalize_stock_code(str(stock_code or "").strip())
    digits = re.sub(r"\D", "", normalized)
    return digits[-6:] if len(digits) >= 6 else normalized


def _normalise_title_key(title: str) -> str:
    return re.sub(r"[\s:：,，。.!！?？/（）()【】\[\]\\-]+", "", title).lower()


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _event_source_id(source: str, record: Dict[str, Any], title: str) -> str:
    raw = "|".join(
        _safe_text(value)
        for value in (
            source,
            record.get("id"),
            record.get("代码"),
            record.get("股票代码"),
            record.get("公告代码"),
            title,
        )
    )
    return _hash_text(raw or title or source)


def _digest_event(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": item.get("type"),
        "title": item.get("title"),
        "publish_time": item.get("publish_time"),
        "source": item.get("source"),
        "risk_level": item.get("risk_level"),
        "is_confirmed": item.get("is_confirmed"),
        "tags": item.get("tags"),
    }


def _nested(value: Dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
