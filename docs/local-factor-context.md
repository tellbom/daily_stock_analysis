# Local Factor Context

This document records the LLM factor-context boundary for the local Python
service and normal stock analysis path.

## Goal

DSA internalises the short-horizon data and factor conventions from the sibling
`stock-analysis` project for LLM context only. It does not import that project at
runtime, does not require `quant_platform`, and does not use the LightGBM,
training, backtest or Parquet-lake workflow.

## Data Input

- A-share daily OHLCV uses the stock-analysis-style AKShare order:
  `ak.stock_zh_a_daily(..., adjust="qfq")` first, then
  `ak.stock_zh_a_hist(..., adjust="qfq")` fallback.
- Bars are normalised to DSA standard columns: `date`, `open`, `high`, `low`,
  `close`, `volume`, `amount`, `turnover_rate`, `pct_chg`.
- Normal analysis prefetches 120 trading-day daily bars so T+5 context has
  enough warm-up for MA60, ADX, OBV, BOLL, ATR and volume percentile signals.
- `factor_summary` only uses bars already fetched or stored by the DSA pipeline;
  it does not trigger a separate external project data run.
- `factor_summary` reads stored daily bars across a 180-natural-day window,
  which is intended to cover roughly 120 trading days when the local database
  has enough history.

## Factor Families

- Core indicators: MA5/10/20/60, MACD, RSI6/12/24, BOLL and KDJ.
- Breadth indicators: ATR14, OBV 20-day z-score, ADX14, CCI14, ROC10, WillR14
  and Stoch K/D.
- Short-horizon context: 1/5/20-day returns, realised 20-day volatility,
  20-day range position, 5/20-day volume ratios, 20-day volume percentile,
  amount, turnover and A-share limit-up/limit-down distance.
- Risk flags: high volume, high-volume stall, MA20 breakdown, short-term
  overheating, close-near-limit-up and large intraday reversal.

## Warm-Up And Missing Data

Core MA/MACD/RSI/BOLL/KDJ calculations follow the stock-analysis style of
computing from the first available row. The summary still exposes warm-up state:

- `factor_warmup_partial`: fewer than 20 bars, only weak reference.
- `factor_warmup_not_full`: at least 20 but fewer than 60 bars, long-window
  indicators should be down-weighted.
- `warmup.is_full_warmup=true`: at least 60 bars.

Unavailable values are omitted from nested blocks rather than fabricated. The
LLM prompt describes the factor table as auxiliary evidence and instructs the
model not to replace news, fundamentals, capital flow or risk-event evidence
with factor-only signals.
