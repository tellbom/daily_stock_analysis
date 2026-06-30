# -*- coding: utf-8 -*-
"""Build LLM-ready short-horizon factor summaries from existing OHLCV bars.

This module internalises the stock-analysis short-term factor ideas without
importing that sibling project at runtime and without using its LightGBM path.
It works only on bars that DSA has already fetched/stored, then emits compact,
dimensionless context for the LLM prompt.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


_MIN_AVAILABLE_BARS = 20
_WARMUP_BARS = 60


def build_llm_factor_summary(
    bars: pd.DataFrame,
    *,
    code: str = "",
    source: Optional[str] = None,
    realtime_quote: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return an LLM-ready factor summary from qfq-compatible OHLCV bars."""
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
    bar_count = int(len(df))

    summary = {
        "status": "available" if bar_count >= _MIN_AVAILABLE_BARS else "partial",
        "source": source or _latest_text(df.get("data_source")) or "daily_bars",
        "as_of": as_of,
        "bar_count": bar_count,
        "price": _compact_number(close),
        "returns": {
            "return_1d_pct": _pct_change(close, prev_close),
            "return_5d_pct": _period_return(df, 5),
            "return_20d_pct": _period_return(df, 20),
        },
        "trend": _trend_summary(latest),
        "momentum": _momentum_summary(df, latest),
        "volatility": _volatility_summary(df, latest),
        "volume": _volume_summary(latest),
        "short_term": _short_term_summary(df, code, latest, previous),
        "technical": _technical_summary(latest),
        "risk": _risk_summary(df, code, latest, previous),
        "warmup": _warmup_summary(bar_count),
    }
    warnings = _warnings_for_bar_count(bar_count)
    if warnings:
        summary["warnings"] = warnings
    return _strip_empty(summary)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    rename = {
        "\u65e5\u671f": "date",
        "\u5f00\u76d8": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6536\u76d8": "close",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u6362\u624b\u7387": "turnover_rate",
        "\u6da8\u8dcc\u5e45": "pct_chg",
        "turnover": "turnover_rate",
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

    # Core stock-analysis indicators: min_periods=1, with warm-up exposed
    # separately so downstream consumers know when values are weak.
    for window in (5, 10, 20, 60):
        work[f"ma{window}"] = close.rolling(window, min_periods=1).mean()

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    work["macd_dif"] = ema_fast - ema_slow
    work["macd_dea"] = work["macd_dif"].ewm(span=9, adjust=False).mean()
    work["macd_bar"] = (work["macd_dif"] - work["macd_dea"]) * 2
    work["macd_bar_delta"] = work["macd_bar"].diff()

    for period in (6, 12, 24):
        work[f"rsi_{period}"] = _rsi(close, period)

    _add_kdj(work)
    _add_boll(work)
    true_range = _true_range(high, low, close)
    work["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    work["atr_14_pct"] = work["atr_14"] / close.replace(0, np.nan) * 100
    _add_breadth_indicators(work, true_range)
    _add_volume_indicators(work)
    return work


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _add_kdj(work: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> None:
    low_min = work["low"].rolling(window=n, min_periods=1).min()
    high_max = work["high"].rolling(window=n, min_periods=1).max()
    rsv = (work["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    work["kdj_k"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    work["kdj_d"] = work["kdj_k"].ewm(com=m2 - 1, adjust=False).mean()
    work["kdj_j"] = 3 * work["kdj_k"] - 2 * work["kdj_d"]


def _add_boll(work: pd.DataFrame, period: int = 20, std_multiplier: int = 2) -> None:
    close = work["close"]
    mid = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std()
    upper = mid + std_multiplier * std
    lower = mid - std_multiplier * std
    width = upper - lower
    work["boll_upper"] = upper
    work["boll_mid"] = mid
    work["boll_lower"] = lower
    work["boll_percent_b"] = (close - lower) / width.replace(0, np.nan)
    work["boll_width_pct"] = width / mid.replace(0, np.nan) * 100


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _add_breadth_indicators(work: pd.DataFrame, true_range: pd.Series) -> None:
    high = work["high"]
    low = work["low"]
    close = work["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_sum = true_range.rolling(14, min_periods=14).sum().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / atr_sum
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / atr_sum
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    work["adx_14"] = dx.rolling(14, min_periods=14).mean()

    typical_price = (high + low + close) / 3
    tp_ma = typical_price.rolling(14, min_periods=14).mean()
    mean_dev = typical_price.rolling(14, min_periods=14).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    work["cci_14"] = (typical_price - tp_ma) / (0.015 * mean_dev.replace(0, np.nan))

    work["roc_10_pct"] = close.pct_change(10) * 100
    highest_14 = high.rolling(14, min_periods=14).max()
    lowest_14 = low.rolling(14, min_periods=14).min()
    range_14 = (highest_14 - lowest_14).replace(0, np.nan)
    work["willr_14"] = -100 * (highest_14 - close) / range_14
    stoch_k = 100 * (close - lowest_14) / range_14
    work["stoch_k"] = stoch_k
    work["stoch_d"] = stoch_k.rolling(3, min_periods=3).mean()

    volume = work.get("volume")
    if volume is not None:
        direction = np.sign(close.diff()).fillna(0)
        work["obv"] = (direction * volume.fillna(0)).cumsum()
        obv_diff = work["obv"].diff()
        obv_mean = obv_diff.rolling(20, min_periods=5).mean()
        obv_std = obv_diff.rolling(20, min_periods=5).std(ddof=1).replace(0, np.nan)
        work["obv_z_20d"] = (obv_diff - obv_mean) / obv_std


def _add_volume_indicators(work: pd.DataFrame) -> None:
    if "volume" not in work.columns:
        return
    vol = work["volume"]
    work["volume_ma5"] = vol.rolling(5, min_periods=5).mean()
    work["volume_ma20"] = vol.rolling(20, min_periods=20).mean()
    work["volume_ratio_5d"] = vol / work["volume_ma5"].replace(0, np.nan)
    work["volume_ratio_20d"] = vol / work["volume_ma20"].replace(0, np.nan)
    work["volume_pct_rank_20d"] = vol.rolling(20, min_periods=5).rank(pct=True)


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
            "roc_10_pct": _compact_number(latest.get("roc_10_pct")),
            "kdj_k": _compact_number(latest.get("kdj_k")),
            "kdj_d": _compact_number(latest.get("kdj_d")),
            "kdj_j": _compact_number(latest.get("kdj_j")),
            "stoch_k": _compact_number(latest.get("stoch_k")),
            "stoch_d": _compact_number(latest.get("stoch_d")),
            "willr_14": _compact_number(latest.get("willr_14")),
            "cci_14": _compact_number(latest.get("cci_14")),
            "adx_14": _compact_number(latest.get("adx_14")),
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
            "obv_z_20d": _compact_number(latest.get("obv_z_20d")),
        }
    )


def _volume_summary(latest: pd.Series) -> Dict[str, Any]:
    return _strip_empty(
        {
            "volume_ratio_5d": _compact_number(latest.get("volume_ratio_5d")),
            "volume_ratio_20d": _compact_number(latest.get("volume_ratio_20d")),
            "volume_pct_rank_20d": _compact_number(latest.get("volume_pct_rank_20d")),
            "amount": _compact_number(latest.get("amount")),
            "turnover_rate_pct": _compact_number(latest.get("turnover_rate")),
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


def _technical_summary(latest: pd.Series) -> Dict[str, Any]:
    return _strip_empty(
        {
            "ma": {
                "ma5": _compact_number(latest.get("ma5")),
                "ma10": _compact_number(latest.get("ma10")),
                "ma20": _compact_number(latest.get("ma20")),
                "ma60": _compact_number(latest.get("ma60")),
            },
            "macd": {
                "dif": _compact_number(latest.get("macd_dif")),
                "dea": _compact_number(latest.get("macd_dea")),
                "bar": _compact_number(latest.get("macd_bar")),
            },
            "rsi": {
                "rsi6": _compact_number(latest.get("rsi_6")),
                "rsi12": _compact_number(latest.get("rsi_12")),
                "rsi24": _compact_number(latest.get("rsi_24")),
            },
            "boll": {
                "upper": _compact_number(latest.get("boll_upper")),
                "mid": _compact_number(latest.get("boll_mid")),
                "lower": _compact_number(latest.get("boll_lower")),
                "percent_b": _compact_number(latest.get("boll_percent_b")),
            },
            "kdj": {
                "k": _compact_number(latest.get("kdj_k")),
                "d": _compact_number(latest.get("kdj_d")),
                "j": _compact_number(latest.get("kdj_j")),
            },
            "breadth": {
                "atr_14_pct": _compact_number(latest.get("atr_14_pct")),
                "adx_14": _compact_number(latest.get("adx_14")),
                "cci_14": _compact_number(latest.get("cci_14")),
                "roc_10_pct": _compact_number(latest.get("roc_10_pct")),
                "willr_14": _compact_number(latest.get("willr_14")),
                "stoch_k": _compact_number(latest.get("stoch_k")),
                "stoch_d": _compact_number(latest.get("stoch_d")),
                "obv_z_20d": _compact_number(latest.get("obv_z_20d")),
            },
        }
    )


def _risk_summary(
    df: pd.DataFrame,
    code: str,
    latest: pd.Series,
    previous: pd.Series,
) -> Dict[str, Any]:
    close = _safe_float(latest.get("close"))
    open_price = _safe_float(latest.get("open"))
    high = _safe_float(latest.get("high"))
    low = _safe_float(latest.get("low"))
    prev_close = _safe_float(previous.get("close"))
    volume_ratio_20d = _safe_float(latest.get("volume_ratio_20d"))
    volume_pct_rank = _safe_float(latest.get("volume_pct_rank_20d"))
    rsi_6 = _safe_float(latest.get("rsi_6"))
    rsi_12 = _safe_float(latest.get("rsi_12"))
    ma5_distance = _distance_pct(close, _safe_float(latest.get("ma5")))
    ma20_distance = _distance_pct(close, _safe_float(latest.get("ma20")))
    range_position = _range_position(df, 20)
    pct_change = _pct_change(close, prev_close)
    intraday_ret = _pct_change(close, open_price)
    upper_shadow_pct = None
    day_range = None
    if None not in (high, low, close) and close:
        day_range = high - low
        upper_shadow_pct = (high - close) / close * 100

    limit_ratio = _a_share_limit_ratio(code)
    limit_up = prev_close * (1 + limit_ratio) if prev_close and limit_ratio else None
    limit_up_distance = _distance_pct(limit_up, close)
    high_volume = (
        (volume_ratio_20d is not None and volume_ratio_20d >= 1.8)
        or (volume_pct_rank is not None and volume_pct_rank >= 0.9)
    )

    flags = {
        "high_volume": bool(high_volume),
        "high_volume_stall": bool(
            high_volume
            and pct_change is not None
            and pct_change < 1.0
            and upper_shadow_pct is not None
            and upper_shadow_pct >= 1.5
        ),
        "breakdown_below_ma20": _is_breakdown_below_ma20(latest, previous),
        "short_term_overheated": bool(
            (rsi_6 is not None and rsi_6 >= 80)
            or (rsi_12 is not None and rsi_12 >= 70)
            or (ma5_distance is not None and ma5_distance >= 5)
            or (range_position is not None and range_position >= 0.9)
        ),
        "close_near_limit_up": bool(limit_up_distance is not None and limit_up_distance <= 1.0),
        "large_intraday_reversal": bool(
            day_range is not None
            and day_range > 0
            and upper_shadow_pct is not None
            and upper_shadow_pct >= 2.0
            and intraday_ret is not None
            and intraday_ret <= 0
        ),
    }
    active = [name for name, enabled in flags.items() if enabled]
    return _strip_empty(
        {
            "flags": flags,
            "active": active,
            "drivers": {
                "pct_change": pct_change,
                "intraday_return_pct": intraday_ret,
                "ma5_distance_pct": ma5_distance,
                "ma20_distance_pct": ma20_distance,
                "range_position_20d": range_position,
                "volume_ratio_20d": _compact_number(volume_ratio_20d),
                "volume_pct_rank_20d": _compact_number(volume_pct_rank),
                "rsi_6": _compact_number(rsi_6),
                "rsi_12": _compact_number(rsi_12),
                "upper_shadow_pct": _compact_number(upper_shadow_pct),
                "distance_to_limit_up_pct": limit_up_distance,
            },
        }
    )


def _is_breakdown_below_ma20(latest: pd.Series, previous: pd.Series) -> bool:
    close = _safe_float(latest.get("close"))
    ma20 = _safe_float(latest.get("ma20"))
    prev_close = _safe_float(previous.get("close"))
    prev_ma20 = _safe_float(previous.get("ma20"))
    return bool(
        None not in (close, ma20, prev_close, prev_ma20)
        and close < ma20
        and prev_close >= prev_ma20
    )


def _warmup_summary(bar_count: int) -> Dict[str, Any]:
    return {
        "bar_count": bar_count,
        "minimum_available_bars": _MIN_AVAILABLE_BARS,
        "full_warmup_bars": _WARMUP_BARS,
        "is_full_warmup": bar_count >= _WARMUP_BARS,
    }


def _warnings_for_bar_count(bar_count: int) -> list[str]:
    warnings = []
    if bar_count < _MIN_AVAILABLE_BARS:
        warnings.append("factor_warmup_partial")
    elif bar_count < _WARMUP_BARS:
        warnings.append("factor_warmup_not_full")
    return warnings


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
