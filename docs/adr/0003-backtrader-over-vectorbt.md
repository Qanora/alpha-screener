# 选择 backtrader 而非 vectorbt 作为回测引擎

回测引擎选用 backtrader（事件驱动），而非 vectorbt（全量向量化）。

**硬约束**：部署目标为 4 vCPU / 8 GB RAM 单机。在 2,000 只美股 × 2 年日线数据的规模下，vectorbt 的全量向量化内存峰值 3-5 GB，叠加 daily_scan 的 4 GB 峰值后会超过 8 GB 上限。backtrader 事件驱动模式在同等数据规模下内存 < 1 GB。

**权衡**：backtrader 单次全量回测耗时更长（约 4 小时 vs vectorbt 约 30 分钟），但通过增量回测优先 + 月度全量安排在夜间的方式缓解。

**升级路径**：若后续升级到 8c16g（触发条件见 PRD 10.3），可恢复 vectorbt 全量向量化。

**Considered Options**: vectorbt（放弃 — 4c8g 内存超限）、zipline（放弃 — 维护活跃度低）。
