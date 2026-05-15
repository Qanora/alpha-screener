# MVP 阶段冻结全部因子权重

MVP（V1.0 第一阶段）全部 13 因子权重冻结不变（因子版本 v1.0.0），仅启用 CUSUM 快速层监控做保护性告警。慢速层权重调整延后到工程转正后（V1.0 第二阶段），由环境变量 `EVOLUTION_WEIGHT_ADJUST_ENABLED` 控制。

**理由**：MVP 最宝贵的产出是 Paper Trading 期间积累的真实 IC 数据。如果 MVP 阶段就让进化系统调整权重，后续无法区分"因子 alpha 本身的变化"和"调参导致的过拟合"，策略转正所需的 walk-forward 验证将失去干净的基线。

**Consequences**: 因子生命周期状态机（proposed/probation/degraded/retired/rejected）的代码实现延迟到 V1.0 第二阶段，但枚举和表结构在 MVP 阶段预定义。
