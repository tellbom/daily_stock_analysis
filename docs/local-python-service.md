# 本地 Python 服务部署说明

本文记录当前仓库的本地 Python Web/API 服务启动方式，适合交给其他 agent 在另一台机器复现部署。

## 前置条件

- Python 3.10+
- macOS / Linux / WSL / Git Bash
- 能访问 Python 包源和必要的行情接口

## 配置文件

启动脚本读取仓库根目录 `.env`。当前本地部署至少需要：

```bash
ALPHASIFT_ENABLED=true
STOCK_LIST=603629,002409,600183

LLM_CHANNELS=deepseek
LLM_DEEPSEEK_PROTOCOL=deepseek
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_DEEPSEEK_API_KEY=<deepseek_api_key>
LLM_DEEPSEEK_MODELS=deepseek-v4-pro
LITELLM_MODEL=deepseek/deepseek-v4-pro
OPENAI_API_KEY=<deepseek_api_key>
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro
```

`.env` 被 `.gitignore` 忽略，不会随代码提交。迁移到另一台机器时，需要单独提供该文件或同等环境变量。

## 启动

```bash
./scripts/start-local-service.sh
```

脚本行为：

- 读取 `.env`
- 如果缺少 `.venv`，自动创建虚拟环境
- 执行 `pip install -r requirements.txt`
- 执行 `python main.py --serve-only`

启动成功后访问：

- WebUI: `http://127.0.0.1:8000`
- API 文档: `http://127.0.0.1:8000/docs`
- 健康检查: `http://127.0.0.1:8000/api/health`

## 停止

前台运行时按 `Ctrl+C`。后台运行时先查进程再停止：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

## 数据源说明

当前普通 A 股日线默认先走 `StockAnalysisAkshareFetcher`，该源使用已迁入当前项目内部的抓取顺序：

1. `ak.stock_zh_a_daily`，新浪日线，前复权
2. `ak.stock_zh_a_hist`，东方财富日线，前复权

未配置 `TUSHARE_TOKEN` 时，`TushareFetcher` 不会实例化。即使配置了 token，Tushare 也只作为后备候选，排在 `StockAnalysisAkshareFetcher` 之后。

后续 fallback 仍保留当前项目已有链路：`EfinanceFetcher`、`TencentFetcher`、`AkshareFetcher`、`PytdxFetcher`、`BaostockFetcher`、`YfinanceFetcher` 等。

## LLM 因子输入

普通分析会基于已写入本项目数据库的日线数据生成 `factor_summary`，并在 Prompt 中以结构化表格注入给 LLM。日线预取窗口为 120 个交易日，`factor_summary` 从数据库读取 180 个自然日窗口，尽量覆盖约 120 个交易日用于 MA60、ADX、OBV、BOLL、ATR 和量能分位等短线指标 warm-up。当前覆盖：

- 均线偏离与均线排列
- MACD、RSI、布林、ATR
- 1/5/20 日收益、20 日区间位置
- 量能比例、量能分位、成交额、换手率
- A 股涨跌停比例与距离

该能力不读取外部项目源码目录，不依赖 `stock-analysis` / `stock_analysis` / `quant_platform` 运行时包，也不会迁移训练、回测、标签、LightGBM、MLflow、DVC 或 Parquet lake 体系。
