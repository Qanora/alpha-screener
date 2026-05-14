# AlphaScreener

面向美股的 AI-Native 量化策略实验/验证平台。每日扫描标的池，通过多因子粗筛 + LLM 风险审计精筛，在 T+7 窗口内识别候选爆发标的。

## Language

### 核心概念

**扫描 (Scan)**:
每日执行的选股流水线，从标的池到最终信号输出的全过程。扫描、选股、信号三者可互换使用。
_Avoid_: 无

**爆发 (Breakout)**:
T+1 开盘至 T+7 收盘期间，股价涨幅 ≥ 10%。二分类预测的目标变量。
_Avoid_: 大涨

**信号 (Signal)**:
一次扫描产出的单个选股推荐，包含 ticker + Coarse_Score + Refined_Score + BreakoutAssessment。
_Avoid_: 选股结果、推荐

**信号文件 (Signal Parquet)**:
每日流水线输出的 Parquet 分区：`signals_refined.parquet`（含 LLM 轨）、`signals_refined_pure.parquet`（Ablation 纯因子轨）。
_Avoid_: 结果文件

### 筛选流水线

**预过滤 (Pre-filtering)**:
标的池准入过滤。基于指数白名单（SP500 ∪ Russell 1000）、市值（> $300M）、成交额（> $20M）、股价（> $5）、上市时长（≥ 12 个月）。输入 ~6,000 → 输出 ~2,000。
_Avoid_: 无

**硬过滤 (Hard Filtering)**:
因子信号触发的强制筛选（Phase 1）。基于 MOM_5D、ATR_RATIO、RSI、MFI/VOL_ANOMALY 等因子实时值。输入 ~2,000 → 输出 ~100。
_Avoid_: Phase 1 粗筛、硬规则过滤

**行业去重 (Sector/Industry Cap)**:
MVP 阶段轻量规则：GICS Sector 最多 3 只、Industry 最多 2 只，按 Coarse_Score 降序保留。V1.5 起 HDBSCAN 聚类做主去重，本规则作为二次兜底。
_Avoid_: 聚类去重、行业上限

### 评分体系

**Coarse_Score**:
Phase 2 加权评分输出，13 因子 z-score 截断后的加权和 `Σ(w_i × z_capped_i)`。用于粗筛排序和行业去重。
_Avoid_: Breakout_Score, Coarse_Final_Score, Final_Score, 粗筛评分

**Refined_Score**:
Coarse_Score 经 PM 风险审计修正后的最终排序分数：`Coarse_Score × score_correction × risk_filter`（risk_filter = 0 if 硬杀伤 tag ∈ risk_tags else 1）。
_Avoid_: 最终得分

**score_i**:
单因子 UI 展示分，`50 + z_capped_i × (50/3)`，映射到 [0, 100]。仅用于飞书卡片和 Parquet 可读输出，不参与排序。
_Avoid_: 因子分

**score_correction (修正系数)**:
PM 输出的有限修正乘数，区间 [0.90, 1.05]。默认 1.00，据风险下调至 0.95/0.90；仅催化剂一致 + 无风险标签 + 流动性充足三项同时满足时上调至 1.05。
_Avoid_: risk_penalty

**risk_tags**:
PM 输出的结构化风险标签数组。枚举：`no_risk` / `data_conflict` / `liquidity_trap` / `delisting_risk` / `earnings_timing_mismatch` / `catalyst_already_passed`。其中 `data_conflict` 和 `delisting_risk` 为硬杀伤标签（触发后 Refined_Score = 0）。
_Avoid_: 风险标记

### 回测与验证

**回测 (Backtest)**:
以历史 OHLCV 数据模拟策略执行的统称。"历史回测"（标签已知）和"Paper Trading"（按真实时间推进，标签滞后 T+7 回填）两种执行方式。
_Avoid_: 无

**Ablation 双轨**:
每日并行记录的消融实验：**纯因子 (pure)** 以 Coarse_Score 排序，**含 LLM (llm)** 以 Refined_Score 排序。ΔLift@20 ≥ 0.05 才允许 LLM 参与最终排序。MVP 强制执行。
_Avoid_: A/B 轨、消融

**Lift@K**:
`Precision@K / base_rate`，策略选股精度相对市场 base_rate 的提升倍数。**核心验收指标**。
_Avoid_: 提升度

**Precision@K**:
Top K 信号中实际爆发的比例。辅助指标，支撑 Lift@K 计算。
_Avoid_: 准确率@K

**base_rate**:
全市场 T+7 自然爆发率，不依赖任何策略。Lift@K 的分母基准。
_Avoid_: 基准概率

**IC (Information Coefficient)**:
`spearman(rank_by_score, actual_return)`，Coarse_Score 排序与真实收益的秩相关性。> 0.03 视为有预测能力。
_Avoid_: 信息系数

**工程转正 (Tier 1)**:
验证系统稳定运行：≥ 60 天无 L3/L4 熔断、调度成功率 ≥ 95%、NaN < 5%。通过后启用 V1.0 第二阶段，但不允许实盘建议。工程转正 ≠ alpha 已被证明。
_Avoid_: 系统转正

**策略转正 (Tier 2)**:
验证 alpha 真实性：≥ 2 年 walk-forward + ≥ 6 个月 live shadow，Lift@20 CI 下界 ≥ 1.05，ΔLift@20 ≥ 0.05。通过后给出实盘建议。
_Avoid_: alpha 转正

### 因子生命周期

**active**:
正常参与评分的因子。MVP 阶段全部 13 因子均为 active，权重冻结。
_Avoid_: 启用、在线

**probation**:
新因子观察期，权重 2-3%，观察 30 个交易日后决定晋升或拒绝。
_Avoid_: 试用

**degraded**:
因 CUSUM 持续告警或 IC 衰减而冻结的因子，权重冻结 15 个交易日等待恢复或退役。
_Avoid_: 降级

**proposed**:
人工或 LLM 提案的新因子，尚未通过 AST 验证和沙箱回测。
_Avoid_: 候选

**retired**:
正式退役的因子，移入 archived，权重归零。
_Avoid_: 删除

**rejected**:
提案阶段被否决的因子，60 天内不重试。
_Avoid_: 驳回

### 进化与监控

**CUSUM 监控 (快速层)**:
T+8 标签回填后对每个 active 因子计算 CUSUM 统计量，检测预测能力异常偏离。仅监控 + 告警，不调权。MVP 已启用。
_Avoid_: 快速进化

**慢速层 (权重调整)**:
每两周由 LLM Strategy Review Agent 发起，基于 30 天 IC/CUSUM/回测偏差，输出权重微调 / 因子增删提案。必须人工 approve。V1.0 第二阶段启用。
_Avoid_: 双周进化、biweekly_evolution

**动态阈值 (Dynamic Threshold)**:
纯规则驱动的硬过滤阈值自动调节。按每日过滤率自动放宽或收紧，单次 ±10%，冷却期 ≥ 3 日，累计 ≤ 30%。不依赖 LLM，独立于双层之外。
_Avoid_: 阈值自适应

**市场 Regime**:
市场运行环境：`normal` / `low_activity` / `style_rotation` / `crisis`。MVP 阶段预定义枚举，慢速层 V1.0 第二阶段启用诊断逻辑。
_Avoid_: 市场状态、市场环境

**成本熔断 (Cost Circuit Breaker)**:
LLM 调用成本的四级自动保护，统一以滚动 30 日日均 USD 判定：L1 警告（日 ≥ $0.80，批大小缩减）、L2 降级（日 ≥ $1.00，暂停精筛）、L3 节约（30 日均 ≥ $2.67，Top 10）、L4 熔断（30 日均 ≥ $3.17，完全停止）。
_Avoid_: 预算熔断

### LLM 精筛角色

**Analyst 层**:
信息收集层。Market/News/Fundamentals/Breakout 四位分析师分别生成结构化分析报告。Social Analyst 延后到 V1.5。
_Avoid_: 分析层

**Researcher 层**:
多空对抗层。Bull Researcher 和 Bear Researcher 并行，基于 Analyst 报告从对立方论证 7 天爆发/回落可能性。
_Avoid_: 辩论层

**PM 层**:
风险审计层。Portfolio Manager 综合 Bull/Bear 报告，输出 score_correction + risk_tags，对 Coarse_Score 做有限修正。
_Avoid_: 决策层、PM 精筛

## Relationships

- 一次 **扫描** 产出 Top 20 **信号**，每个信号含一个 **Coarse_Score** 和一个 **Refined_Score**
- 一个 **信号** 在 T+7 后回溯验证是否为 **爆发**（正样本）
- 止损踢出的标的算 **失败**，不论后续是否达到 +10%
- **预过滤** → **硬过滤** → **Coarse_Score** 排序 → **行业去重** → **LLM 精筛**（Analyst → Researcher → PM）→ **Refined_Score** 排序 → **信号文件** 输出
- 一个 **Factor** 的生命周期：**proposed** → **probation** → **active** → **degraded** → **retired**（或被 **rejected**）
- **CUSUM 监控** 检测因子异常 → 触发告警；**慢速层** 在 V1.0 第二阶段根据 CUSUM 累计数据调整权重
- **Ablation 双轨** 并行运行：纯因子轨产出 `signals_refined_pure.parquet`，含 LLM 轨产出 `signals_refined.parquet`

## Example dialogue

> **Dev:** "一个标的的 Coarse_Score 很高，但 PM 给它打了 `delisting_risk` 标签，这个标的还会出现在最终 Top 20 里吗？"
> **Domain expert:** "不会。`delisting_risk` 是硬杀伤标签，Refined_Score 直接归零，不会被选入最终信号。"
>
> **Dev:** "那 Ablation 双轨里，纯因子轨会包含它吗？"
> **Domain expert:** "会的。纯因子轨只看 Coarse_Score，不看 risk_tags。只有在 ΔLift@20 ≥ 0.05 时 LLM 修正层的价值才被确认。"
>
> **Dev:** "如果 CUSUM 告警触发了但系统还没到工程转正，会怎样？"
> **Domain expert:** "飞书告警，但权重不动。工程转正后慢速层才被允许调权。"

## Flagged ambiguities

- "信号" 在风险章节偶尔指"交易信号源"这种更高层概念 — 当前允许互换，但策略转正时需明确区分
- `data_conflict` 同时出现在 risk_tags 枚举和独立 bool 字段中 — 已解决：移除 bool 字段，仅保留 risk_tags 中的标签
