# -*- coding: utf-8 -*-
"""Build lightweight market factor summaries from already-fetched OHLCV bars.

The helpers in this module intentionally avoid network calls and external
project imports.  They convert the daily bars already available in the DSA
pipeline into compact, dimensionless signals that can be safely injected into
LLM context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def build_llm_factor_summary(
    bars: pd.DataFrame,
    *,
    code: str = "",
    source: Optional[str] = None,
    realtime_quote: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return an LLM-ready factor summary from OHLCV bars."""
    df = _normalize_bars(bars)
    if df.empty or len(df) < 2:
        return {
            "status": "missing",
            "source": source or "daily_bars",
            "missing_reason": "insufficient_daily_bars",
        }

    df = _overlay_realtime_quote(df, realtime_quote)
    df = _add_indicators(df)
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    close = _safe_float(latest.get("close"))
    prev_close = _safe_float(previous.get("close"))
    as_of = _format_date(latest.get("date"))

    summary = {
        "status": "available" if len(df) >= 20 else "partial",
        "source": source or _latest_text(df.get("data_source")) or "daily_bars",
        "as_of": as_of,
        "bar_count": int(len(df)),
        "price": _compact_number(close),
        "returns": {
            "return_1d_pct": _pct_change(close, prev_close),
            "return_5d_pct": _period_return(df, 5),
            "return_20d_pct": _period_return(df, 20),
        },
        "trend": _trend_summary(latest),
        "momentum": _momentum_summary(df, latest),
        "volatility": _volatility_summary(df, latest),
        "volume": _volume_summary(df, latest),
        "short_term": _short_term_summary(df, code, latest, previous),
    }
    if len(df) < 20:
        summary["warnings"] = ["factor_warmup_partial"]
    return _strip_empty(summary)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "涨跌幅": "pct_chg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    for col in ("open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def _overlay_realtime_quote(df: pd.DataFrame, quote: Optional[Any]) -> pd.DataFrame:
    if quote is None or df.empty:
        return df
    price = _safe_float(getattr(quote, "price", None))
    if price is None or price <= 0:
        return df
    work = df.copy()
    last_idx = work.index[-1]
    work.at[last_idx, "close"] = price
    for attr, col in (
        ("open_price", "open"),
        ("high", "high"),
        ("low", "low"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("change_pct", "pct_chg"),
        ("turnover_rate", "turnover_rate"),
    ):
        value = _safe_float(getattr(quote, attr, None))
        if value is not None:
            work.at[last_idx, col] = value
    return work


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    close = work["close"]
    high = work["high"]
    low = work["low"]
    for window in (5, 10, 20, 60):
        work[f"ma{window}"] = close.rolling(window, min_periods=window).mean()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    work["macd_dif"] = ema_fast - ema_slow
    work["macd_dea"] = work["macd_dif"].ewm(span=9, adjust=False).mean()
    work["macd_bar"] = (work["macd_dif"] - work["macd_dea"]) * 2
    work["macd_bar_delta"] = work["macd_bar"].diff()

    for period in (6, 12, 24):
        work[f"rsi_{period}"] = _rsi(close, period)

    boll_mid = close.rolling(20, min_periods=20).mean()
    boll_std = close.rolling(20, min_periods=20).std(ddof=0)
    boll_upper = boll_mid + 2 * boll_std
    boll_lower = boll_mid - 2 * boll_std
    width = boll_upper - boll_lower
    work["boll_percent_b"] = (close - boll_lower) / width.replace(0, np.nan)
    work["boll_width_pct"] = width / boll_mid.replace(0, np.nan) * 100

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    work["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    work["atr_14_pct"] = work["atr_14"] / close.replace(0, np.nan) * 100

    if "volume" in work.columns:
        vol = work["volume"]
        work["volume_ma5"] = vol.rolling(5, min_periods=5).mean()
        work["volume_ma20"] = vol.rolling(20, min_periods=20).mean()
        work["volume_ratio_20d"] = vol / work["volume_ma20"].replace(0, np.nan)
        work["volume_pct_rank_20d"] = vol.rolling(20, min_periods=5).rank(pct=True)
    return work


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100).where(avg_loss > 0, 100)


def _trend_summary(latest: pd.Series) -> Dict[str, Any]:
    close = _safe_float(latest.get("close"))
    distances = {
        f"ma{window}_distance_pct": _distance_pct(close, _safe_float(latest.get(f"ma{window}")))
        for window in (5, 10, 20, 60)
    }
    ma5 = _safe_float(latest.get("ma5"))
    ma10 = _safe_float(latest.get("ma10"))
    ma20 = _safe_float(latest.get("ma20"))
    if None not in (ma5, ma10, ma20) and ma5 > ma10 > ma20:
        alignment = "bullish"
    elif None not in (ma5, ma10, ma20) and ma5 < ma10 < ma20:
        alignment = "bearish"
    elif None not in (ma5, ma10, ma20):
        alignment = "mixed"
    else:
        alignment = "insufficient"
    return _strip_empty({"ma_alignment": alignment, **distances})


def _momentum_summary(df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
    dif = _safe_float(latest.get("macd_dif"))
    dea = _safe_float(latest.get("macd_dea"))
    bar = _safe_float(latest.get("macd_bar"))
    prev_bar = _safe_float(df.iloc[-2].get("macd_bar")) if len(df) >= 2 else None
    if None not in (dif, dea) and dif > dea and dif > 0:
        macd_state = "bullish"
    elif None not in (dif, dea) and dif < dea and dif < 0:
        macd_state = "bearish"
    elif None not in (dif, dea) and dif > dea:
        macd_state = "recovering"
    elif None not in (dif, dea):
        macd_state = "weakening"
    else:
        macd_state = "insufficient"

    return _strip_empty(
        {
            "macd_state": macd_state,
            "macd_dif_pct": _price_ratio(dif, latest.get("close")),
            "macd_dea_pct": _price_ratio(dea, latest.get("close")),
            "macd_bar_pct": _price_ratio(bar, latest.get("close")),
            "macd_bar_delta_pct": _price_ratio(
                bar - prev_bar if None not in (bar, prev_bar) else None,
                latest.get("close"),
            ),
            "rsi_6": _compact_number(latest.get("rsi_6")),
            "rsi_12": _compact_number(latest.get("rsi_12")),
            "rsi_24": _compact_number(latest.get("rsi_24")),
            "roc_10_pct": _period_return(df, 10),
        }
    )


def _volatility_summary(df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
    return _strip_empty(
        {
            "atr_14_pct": _compact_number(latest.get("atr_14_pct")),
            "boll_percent_b": _compact_number(latest.get("boll_percent_b")),
            "boll_width_pct": _compact_number(latest.get("boll_width_pct")),
            "realized_vol_20d_pct": _compact_number(
                df["close"].pct_change().rolling(20, min_periods=10).std(ddof=0).iloc[-1] * 100
            ),
        }
    )


def _volume_summary(df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
    return _strip_empty(
        {
            "volume_ratio_20d": _compact_number(latest.get("volume_ratio_20d")),
            "volume_pct_rank_20d": _compact_number(latest.get("volume_pct_rank_20d")),
            "amount": _compact_number(latest.get("amount")),
            "turnover_rate_pct": _compact_number(
                _first_present(latest, "turnover_rate", "turnover", "换手率")
            ),
        }
    )


def _short_term_summary(
    df: pd.DataFrame,
    code: str,
    latest: pd.Series,
    previous: pd.Series,
) -> Dict[str, Any]:
    close = _safe_float(latest.get("close"))
    prev_close = _safe_float(previous.get("close"))
    limit_ratio = _a_share_limit_ratio(code)
    limit_up = prev_close * (1 + limit_ratio) if prev_close and limit_ratio else None
    limit_down = prev_close * (1 - limit_ratio) if prev_close and limit_ratio else None
    return _strip_empty(
        {
            "range_position_20d": _range_position(df, 20),
            "days_above_ma5": _days_above_ma(df, "ma5", 10),
            "limit_ratio_pct": _compact_number(limit_ratio * 100 if limit_ratio else None),
            "distance_to_limit_up_pct": _distance_pct(limit_up, close),
            "distance_to_limit_down_pct": _distance_pct(close, limit_down),
        }
    )


def _period_return(df: pd.DataFrame, periods: int) -> Optional[float]:
    if len(df) <= periods:
        return None
    latest = _safe_float(df.iloc[-1].get("close"))
    base = _safe_float(df.iloc[-periods - 1].get("close"))
    return _pct_change(latest, base)


def _range_position(df: pd.DataFrame, window: int) -> Optional[float]:
    if len(df) < 2:
        return None
    tail = df.tail(window)
    low = _safe_float(tail["low"].min())
    high = _safe_float(tail["high"].max())
    close = _safe_float(df.iloc[-1].get("close"))
    if None in (low, high, close) or high == low:
        return None
    return _compact_number((close - low) / (high - low))


def _days_above_ma(df: pd.DataFrame, ma_col: str, lookback: int) -> Optional[int]:
    if ma_col not in df.columns or len(df) < 2:
        return None
    tail = df.tail(lookback)
    valid = tail[ma_col].notna()
    if not valid.any():
        return None
    return int((tail.loc[valid, "close"] > tail.loc[valid, ma_col]).sum())


def _a_share_limit_ratio(code: str) -> Optional[float]:
    c = str(code or "").upper().replace("SH", "").replace("SZ", "").replace("BJ", "")
    c = c.split(".")[0]
    if len(c) != 6 or not c.isdigit():
        return None
    if c.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if c.startswith(("688", "300")):
        return 0.20
    return 0.10


def _pct_change(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or base is None or base == 0:
        return None
    return _compact_number((value - base) / base * 100)


def _distance_pct(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or base is None or base == 0:
        return None
    return _compact_number((value - base) / base * 100)


def _price_ratio(value: Optional[float], close: Any) -> Optional[float]:
    close_value = _safe_float(close)
    if value is None or close_value is None or close_value == 0:
        return None
    return _compact_number(value / close_value * 100)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _compact_number(value: Any, digits: int = 4) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _first_present(row: pd.Series, *keys: str) -> Optional[Any]:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _latest_text(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    for value in reversed(series.dropna().tolist()):
        text = str(value).strip()
        if text:
            return text
    return None


def _format_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return str(value)
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _strip_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            for cleaned in [_strip_empty(item)]
            if cleaned not in (None, {}, [])
        }
    if isinstance(value, list):
        return [item for item in (_strip_empty(v) for v in value) if item not in (None, {}, [])]
    return value
