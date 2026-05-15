# yfinance + 备用 OHLCV 双源校验

采用 yfinance 作为主力数据源 + Stooq（首选）/ Alpaca / Polygon 三级备用 OHLCV 源的交叉校验架构。

**理由**：yfinance 作为免费数据源，过去 3 年内发生过 4 次 breaking change 和 2 次封禁事件。单点依赖意味着任何一次上游变更都可能导致日频扫描流水线完全停摆，破坏工程转正所需的 60 天连续稳定性。

**校验机制**：每日对 Top 100 + Top 20 标的拉取备用源 OHLCV，字段级对比（Open/High/Low/Close 任一差异 > 0.5% → 写入 data_source_diff 表）。当日告警 ≥ 5 次触发飞书推送。连续 3 天 yfinance 失败率 > 30% 则自动全量切换备用源。

**权衡**：增加了 MVP 阶段的实现复杂度（额外的数据拉取逻辑 + SQLite 差异记录表 + 三层降级路径），但避免了 yfinance 失效导致的 Pipeline 停摆——后者对工程转正达标的威胁远大于前者。

**Considered Options**: 单靠 yfinance（放弃 — 单点风险不可接受）、引入付费数据源如 Polygon Starter（放弃 — MVP 阶段成本控制优先，FMP Free tier 已有每月 250 req 预算）。
