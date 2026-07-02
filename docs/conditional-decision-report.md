# 条件型短线决策报告

个股 LLM 报告在保留旧版 `operation_advice`、`decision_type`、`dashboard` 字段的基础上，新增条件型短线决策扩展，用于把“单一结论”改造成明日可执行的多情景计划。

## 新增结构

- `conditional_decision`：解析器会把 LLM 顶层条件字段镜像到 `dashboard.conditional_decision`，便于报告、通知和历史 raw_result 复用。
- `trigger_strategy`：必须包含 `bearish_continue`、`neutral_observe`、`bullish_repair`，分别表示走弱延续、中性观察和修复转强时的处理条件。
- `position_advice`：按 `empty_position`、`light_position`、`heavy_position` 给出空仓、轻仓、重仓建议。
- `data_completeness`：记录已使用数据、缺失数据和缺失项对置信度的影响。
- `external_market_reference`：当大盘上下文提供美股科技链摘要时，科技、电子、半导体、AI、算力、新能源车相关 A 股需要说明外盘映射；不相关标记为 `not_relevant`。

## 通知展示

企业微信通知将买入、风控、目标拆为多行：

- 买入参考 / 潜在买点条件
- 风控参考
- 反弹/目标参考

当建议为观望、减仓、卖出或回避时，不强制输出立即买入价，买入位置展示为“潜在买点条件”。风险提示最多展示 3 条，超过部分提示查看完整报告。

## 大盘上下文

A 股大盘复盘会 best-effort 生成 `us_tech_summary`。数据源不可用时返回 `status=missing` 和原因，不阻断大盘复盘或个股分析。单股 Prompt 只消费每日大盘上下文中的安全摘要，不在个股链路重新抓取美股数据。
