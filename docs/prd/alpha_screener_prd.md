# AlphaScreener 产品需求文档（PRD）

| 项目 | 内容 |
|------|------|
| 产品名称 | AlphaScreener |
| 文档版本 | v1.0 |
| 发布日期 | 2026-05-09 |
| 文档状态 | Ready for Development |
| 部署目标 | 4 vCPU / 8 GB RAM / 100 GB SSD 单机 |
| 核心依赖 | [TradingAgents](https://github.com/TauricResearch/TradingAgents)（git+commit 锁定） |

---

## 1. 产品概述

### 1.1 产品定位

AlphaScreener 是一个面向美股（V1.5 起扩展港股）的 **AI-Native 量化策略验证平台**，深度融合多因子量化筛选、LLM 风险审计与自适应进化回路，在 **T+7 时间窗口** 内识别候选爆发标的，并通过严格的 walk-forward 验证持续测量 alpha 真实性。

> **重要定性**：MVP 阶段系统是"量化策略实验/验证平台"，不是已证明的生产级交易信号源。最宝贵的产出是 Paper Trading 期间积累的真实 IC 数据，用于支撑后续 alpha 显著性验证。

### 1.2 预测目标

- **预测窗口**：T+1 开盘至 T+7 收盘
- **目标变量**：`y = 1 if (Close_T+7 / Open_T+1 - 1) ≥ 10% else 0`
- **信号性质**：前瞻性预测（非回顾性确认）

### 1.3 核心价值主张

| 维度 | 传统量化选股 | AlphaScreener |
|------|--------------|---------------|
| 筛选逻辑 | 静态规则 / 固定权重 | 动态因子组合 + LLM 风险审计 |
| 多标的处理 | 逐个跑模型 | 批量扫描 + 结构化辩论排序 |
| 解释性 | 黑盒信号 | 多空辩论生成可读的投资叙事 |
| 进化能力 | 人工调参 | 回测反馈驱动的因子权重自优化 |
| 时效性 | 日频 / 周频 | 日频扫描，聚焦 7 日爆发窗口 |

### 1.4 目标市场

| 市场 | 原始候选池 | 预过滤后规模 | 启用版本 |
|------|------------|--------------|----------|
| 美股（NYSE/NASDAQ） | 6,000+ | 约 2,000 只 | MVP |
| 港股（主板） | 2,000+ | 约 800 只 | V1.5 |

**美股预过滤条件**（4c8g 资源约束加严）：

| 维度 | 阈值 |
|------|------|
| 指数白名单 | SP500 ∪ Russell 1000 |
| 近 20 日平均成交额 | > $20M |
| 总市值 | > $300M |
| 股价 | > $5 |
| 上市时长 | ≥ 12 个月 |
| 状态 | 非停牌 / 非退市 |

### 1.5 版本规划

| 版本 | 周期 | 核心交付 |
|------|------|----------|
| **MVP（V1.0 第一阶段）** | 4 周 | 美股扫描引擎、13 因子粗筛、LLM 风险审计、Paper Trading、Alpha 验收口径 |
| **V1.0 第二阶段** | +4 周 | 慢速层权重进化、完整回测系统、CLI、Walk-forward 月报 |
| **V1.5** | +4 周 | 港股支持、HDBSCAN 聚类、IsolationForest、LLM 自主因子发现 |
| **策略转正** | +4-8 周 | 通过 ≥ 2 年 walk-forward + 6 个月 live shadow 验证 |
| **V2.0** | +8 周 | 实时盘中扫描（需升级 8c16g） |

### 1.6 MVP 范围冻结

| 模块 | IN（必须实现） | NOT-IN-MVP（明确不做） |
|------|----------------|------------------------|
| 标的池 | 美股 ~2,000 只 | 全市场 6,000+、ETF、ADR、OTC |
| 因子体系 | 13 active 因子（PEAD 粗筛降级为哑变量） | 期权 IV、Short Interest、宏观因子 |
| 评分模型 | Phase 1 硬过滤 + Phase 2 加权评分 | 机器学习预测模型 |
| 行业去重 | 轻量规则（Sector cap≤3，Industry cap≤2） | HDBSCAN（V1.5 启用） |
| LLM 精筛 | Top 20 × 3 调用，4 位 Analyst | 多轮深度辩论、多模型投票 |
| 回测 | backtrader 增量回测 | vectorbt 全量向量化 |
| Alpha 验收 | base_rate / Precision@20 / Lift@20 / Bootstrap CI | 风险因子归因分解 |
| 自进化 | 全部因子权重冻结，仅 CUSUM 监控告警 | 慢速层权重调整（延后到 V1.0 第二阶段） |
| 数据源 | yfinance 主 + 备用 OHLCV + FMP Free | Bloomberg / Refinitiv |
| 推送 | 飞书每日卡片 | 微信 / 钉钉 / 邮件 |
| 部署 | systemd + APScheduler | Docker / k8s |

---

## 2. 系统架构

### 2.1 模块全景

```
+---------------+----------------+----------------+----------------+
| 数据接入层 | 因子计算层 | 智能筛选层 | 决策输出层 |
+---------------+----------------+----------------+----------------+
| yfinance | 动量因子 | 粗筛引擎 | 精筛 Top 20 |
| 备用 OHLCV | 波动率因子 | (硬规则过滤) | 风险审计报告 |
| FMP (辅助) | 资金流因子 | 行业去重 | 回测验证标签 |
| | 情绪因子(V1.5) | LLM 精筛 | 飞书推送 |
| | 技术形态因子 | (风险审计层) | 进化诊断日志 |
| | 基本面因子 | | |
+---------------+----------------+----------------+----------------+
^ |
| v
+-------- 回测与进化反馈回路（双层） <-----------+
```

### 2.2 功能清单

| ID | 功能 | 输入 | 输出 |
|----|------|------|------|
| F1 | 广域扫描引擎 | ~2,000 只标的 | 通过硬过滤的候选池 (~100) |
| F2 | 爆发潜力评分 | 候选池因子向量 | Breakout_Score 加权排序 |
| F3 | 行业去重 | Top 30 | Top 20（Sector/Industry cap） |
| F4 | LLM 风险审计精筛 | Top 20 | 评级 + 风险标签 + 修正系数 |
| F5 | 7 天爆发预测 | 精筛结果 | Top 标的 + 催化剂 + 仓位建议 |
| F6 | 回测验证系统 | 历史选股记录 | 胜率、夏普、回撤、IC |
| F7 | 自我进化回路 | 回测 + 实盘对账 | 因子权重 / 因子增删建议 |

---

## 3. 选股引擎设计

### 3.1 因子体系

> **数据窗口原则**：技术因子回看 ≤ 63 个交易日；基本面因子受财报频率约束，允许回溯 2-3 季度。

#### 3.1.1 因子权重表

| 类别 | 因子 | MVP 权重 | V1.5 权重 | 学术依据 |
|------|------|----------|-----------|----------|
| **动量** | MOM_5D | 14% | 12% | Chen et al. (2025) |
| | PTH（63 日高点） | 12% | 10% | George & Hwang (2024) |
| | MOM_SLOPE | 10% | 8% | Intraday Momentum (2025) |
| **波动率** | BB_SQUEEZE | 13% | 12% | Volatility Regime (2025) |
| | ATR_RATIO | 9% | 8% | ATR Indicator (2025) |
| **资金流** | MFI_14 | 9.5% | 8% | ScienceDirect (2025) |
| | CMF_21 | 8.5% | 7% | Chaikin Money Flow (2025) |
| | VOL_ANOMALY | 4.5% | 3% | 微观结构研究 (2025) |
| **情绪**（V1.5） | SOCIAL_SENT | 0% | 8% | Zhang et al. (2025) |
| | NEWS_SENT | 0% | 7% | FinGPT (2024) |
| **技术形态** | RSI_OVERSOLD | 4.5% | 4% | RSI/MACD (2024) |
| | MACD_CROSS | 3.5% | 3% | Sage (2024) |
| | GOLDEN_CROSS | 3.5% | 3% | 技术分析综述 (2025) |
| **基本面** | PEAD | 0%（粗筛）/ 4.34%（精筛） | 4% | ACL (2025) |
| | INSIDER_BUY | 4.50% | 2% | SEC Form 4 研究 (2025) |
| | REV_ACCEL | 3.50% | 1% | JFQA (2025) |

#### 3.1.2 关键因子公式

**MOM_5D**：`(Close_t - Close_{t-5}) / Close_{t-5}`

**MOM_SLOPE**：`LinearRegression_Slope(Daily_Returns[t-10:t])`，正斜率 = 动量加速；负斜率 = 动量减速

**PTH**：`Close_t / max(Close, 过去 63 个交易日)`，硬过滤阈值 > 0.90

**BB_SQUEEZE**：`1 if BB_Width < Percentile(60d, 20) else 0`

**ATR_RATIO**：`ATR(5) / ATR(20)`，压缩信号 < 0.8

**MFI_14**：`100 - 100/(1 + Σ(Positive_MF, 14d) / Σ(Negative_MF, 14d))`

**CMF_21**：`Σ(MF_Volume, 21d) / Σ(Volume, 21d)`

**VOL_ANOMALY**：`1 if Volume_ZScore(50d) > 2.0 and Close > SMA(5) else 0`

**RSI_OVERSOLD**：`1 if RSI < 30 and Close > SMA(20) else 0`

**MACD_CROSS**：`1 if MACD > Signal and Histogram crosses 0 else 0`

**PEAD（精筛阶段）**：`(Actual_EPS_Q - Consensus_EPS_Q) / abs(Consensus_EPS_Q + 0.01)`

> **PEAD 分层策略**：
> - **粗筛阶段**：PEAD 降级为 0/1 哑变量（`PEAD_FLAG = 1 if 近 30 日内有财报发布 else 0`），仅作为 Phase 1 可选加分项，不参与 Phase 2 加权评分
> - **精筛阶段**：对 Top 20 标的调用 FMP 真值计算季度同质 PEAD，权重 4.34%，注入 PM Prompt 作为 context
> - **数据源**：粗筛使用 yfinance `earnings_dates`；精筛使用 FMP `earning_surprises` API

**INSIDER_BUY**：`1 if Σ(Buy_Amount, 60d) / Market_Cap > 0.001 else 0`

**REV_ACCEL**：`Rev_Growth_Q - Rev_Growth_{Q-1}`

#### 3.1.3 因子标准化

```
Step 1: Z-score 标准化 z_i = (f_i - μ_30d) / σ_30d
Step 2: 极值截断 z_capped_i = clip(z_i, -3, +3)
Step 3: 单因子展示分 score_i = 50 + z_capped_i × (50/3) ∈ [0, 100]，仅 UI/Parquet
Step 4: 加权综合分 Final_Score = Σ(w_i × z_capped_i) 用于排序与回测
```

> **重要约束**：`Final_Score` 与 `score_i` 不同量纲。排序、入场、回测均以 `Final_Score` 为准；UI 展示用 `score_i`。

#### 3.1.4 缺失数据与异常处理

- 价格/成交量缺失：剔除该标的当日评分
- 单因子缺失：得分置中性值（50），权重不变
- 缺失因子 z-score 置 0（即对加权综合分无贡献），UI 展示分 score_i 置 50
- 情绪因子（V1.5）缺失：该因子 z-score = 0，不影响当前 MVP 评分
- 基本面因子缺失（非财报期）：PEAD_FLAG = 0（不激活），INSIDER_BUY 缺失时 z-score = 0
- 任意标的因子缺失率 > 30%：标记"数据不足"，不进入推荐池
- Z-score > 3σ：截断；价格日涨跌 > 50% 检查拆股；成交量 > 10× 均量检查大宗

### 3.2 粗筛规则

#### 3.2.1 Phase 1 硬过滤

**必须满足**（全部）：
1. `MOM_5D > 0`
2. `VOL_ANOMALY = 1` 或 `MFI_14 > 40`
3. `ATR_RATIO < 0.8`
4. `RSI ∈ [25, 75]`

**可选加分**（满足越多越优先）：
- `BB_SQUEEZE = 1`、`PTH > 0.90`、`CMF_21 > 0`、`PEAD_FLAG = 1`、`INSIDER_BUY = 1`

#### 3.2.2 动态阈值调节

按每日过滤率自动调节硬条件阈值，单次幅度 ±10%，冷却期 ≥ 3 交易日，累计放宽 ≤ 30%。

| 过滤率 | 状态 | 动作 |
|--------|------|------|
| 80-92% | 正常 | 无调节 |
| 92-95% | 偏紧 | 告警 |
| 95-98% | 过紧 | 自动放宽硬条件 |
| > 98% | 极端 | 全条件放宽 10% + 进化 Agent regime 诊断 |
| < 70% | 过松 | 自动收紧 |

**放宽方向规则**：`X < 阈值` 放宽 = +Δ；`X > 阈值` 放宽 = -Δ；`X ∈ [a,b]` 放宽 = `[a-Δ, b+Δ]`。

**数值示例**（单次放宽 10%）：

| 硬条件 | 原始阈值 | 放宽后阈值 | 方向 |
|--------|---------|-----------|------|
| MOM_5D > X | > 0% | > -0.5% | 下调 |
| ATR_RATIO < X | < 0.8 | < 0.88 | 上调 |
| RSI ∈ [a, b] | [25, 75] | [22.5, 77.5] | 双向扩展 |
| MFI_14 > X 或 VOL_ANOMALY = 1 | MFI > 40 | MFI > 36 | 下调 |

#### 3.2.3 Phase 2 加权评分

```
Breakout_Score = Σ(w_i × z_capped_i) # 加权 z-score 综合分
排序: Breakout_Score 降序 → Top 30 → 行业去重 → Top 20
```

### 3.3 行业去重（轻量规则，MVP 必做）

| 项 | 设计 |
|----|------|
| 数据源 | yfinance `Ticker.info["sector"]` + `industry"]`（GICS 11 Sector / ~25 Industry） |
| Sector 上限 | 同一 GICS Sector 最多保留 **3** 个标的 |
| Industry 上限 | 同一 GICS Industry 最多保留 **2** 个标的 |
| 排序优先 | Breakout_Score 降序 |
| 输出 | Top 30 → Top 20（不足时按原始评分补足） |
| 元数据缓存 | 每月 1 日刷新 `~/.alphascreener/data/universe_meta.parquet` |

**V1.5 扩展**：HDBSCAN 聚类（特征 = 行业 one-hot + 因子 z-score，24-26 维），本节规则继续作为二次保险。

### 3.4 时间语义与 Look-ahead Bias 防护

| 时点 | 含义 |
|------|------|
| T | 选股日（信号生成日，美股交易日） |
| T 收盘 | 美股 T 日收盘（约 UTC 21:00） |
| T+1 开盘 | 回测/实盘买入时点 |
| T+7 收盘 | T 后第 7 个**交易日**（不计美股休市日）的收盘时点；若该日休市则顺延 |

**可见性约束**：

| 因子类别 | 规则 |
|----------|------|
| OHLCV 类 | 仅使用 ≤ T 的数据 |
| 财报类（PEAD/REV_ACCEL） | `earnings_release_date ≤ T - 1` |
| 内部人交易 | SEC Form 4 `filing_date ≤ T` |
| 新闻/社交 | `published_at ≤ T 收盘 (UTC 21:00)` |
| 共识 EPS（精筛） | FMP `estimate_date ≤ T - 1` |

**禁止操作**：使用 future window（如 `mean(Close, T-5:T+5)`）；使用 T 之后已知的事件结果。

---

## 4. LLM 多空辩论精筛（风险审计层定位）

### 4.1 设计原则

LLM（GPT-4o-mini）相对 13 因子量化模型**没有信息优势**，因此 LLM 在本系统中**重定位为风险审计层**，不再作为最终概率预测器：

- **核心输出**：结构化风险标签 + 修正系数 ∈ [0.9, 1.05]，对粗筛 Final_Score 做有限修正
- **价值边界**：识别因子模型无法捕捉的（a）数据冲突（SEC/yfinance 不一致）、（b）催化剂时序错位、（c）流动性陷阱、（d）退市风险
- **强制 Ablation**：Paper Trading 必须并行记录"含 LLM"与"纯因子"两套结果，60 天 ΔLift@20 ≥ 0.05 才允许 LLM 参与最终排序

### 4.2 精筛流水线

```
Top 20 标的 (聚类去重后)
│
├──> Analyst 团队 (4 位: Market / News / Fundamentals / Breakout)
│ ↓
├──> Bull Researcher ∥ Bear Researcher (Stage 1，并行)
│ ↓
└──> Portfolio Manager 风险审计 (Stage 2，注入 Bull/Bear 报告)
↓
BreakoutAssessment 结构化输出
```

#### 4.2.1 Analyst 团队职能定义

在现有 4 位分析师基础上，新增 **Breakout Analyst（爆发分析师）**。

> **MVP 阶段实际启用**：4 位（Social Analyst 跳过）。**V1.5 起启用全部 5 位**（启用 Social Analyst 需 SOCIAL_SENT 数据源就绪）。

| 分析师 | 职责 | 输出 | MVP 启用 |
|--------|------|------|---------|
| Market Analyst | 技术面 + 量价分析 | 技术形态报告 | ✅ |
| News Analyst | 近期新闻事件 | 事件催化剂列表 | ✅ |
| Social Analyst | 社交媒体情绪 | 情绪极值信号 | ❌（V1.5 启用） |
| Fundamentals Analyst | 基本面变化 | 业绩/估值触发点 | ✅ |
| **Breakout Analyst** | **爆发形态专项识别** | **历史相似爆发案例 + 当前形态匹配度** | ✅ |

**Breakout Analyst 历史相似案例检索实现**：

| 项 | 设计 |
|----|------|
| **案例库** | 历史所有正样本（T+7 涨幅 ≥ 10%）的因子向量 + 涨幅元数据，存于 `~/.alphascreener/data/case_library/cases.parquet` |
| **检索方式** | 当前标的因子向量 vs 案例库做余弦相似度，取 Top 5（仅返回相似度 ≥ 0.85 的案例） |
| **加速结构** | `faiss.IndexFlatIP`（向量数 < 50K）或 `faiss.IndexHNSWFlat`（≥ 50K） |
| **更新频率** | 每月 1 日全量重建索引；每日新增正样本增量写入案例库 |
| **输出格式** | `[{"ticker": "AAPL", "date": "2023-05-15", "similarity": 0.91, "actual_pnl": 0.135}, ...]` |
| **MVP 阶段** | 案例库为空时返回 `[]`，PM Output `similar_cases` 字段允许为空数组 |

> **角色关系**：4 位 Analyst 各自生成结构化分析报告 → 报告聚合为 Context → 注入 Bull Researcher 和 Bear Researcher 的 Prompt → Bull/Bear 并行生成多空论点 → PM 综合两方论点进行风险审计，输出 BreakoutAssessment。

### 4.3 输出 Schema (BreakoutAssessment)

```python
class BreakoutAssessment(ResearchPlan):
# 评级（软建议）
final_rating: Literal["Strong Buy", "Buy", "Hold", "Avoid"]
breakout_probability: float # 0.0-1.0，仅可解释性
confidence: int # 0-100

# 风险审计层核心
score_correction: float # 0.9-1.05，进入最终排序
risk_tags: List[str] # enum: no_risk / data_conflict / liquidity_trap /
# delisting_risk / earnings_timing_mismatch /
# catalyst_already_passed
data_conflict_detected: bool
catalyst_consistency: Literal["consistent", "ambiguous", "contradicted"]

# 多空权重
bull_weight: int # bull_weight + bear_weight = 100
bear_weight: int

# 催化与案例
catalyst_events: List[str]
catalyst_timeline: str
similar_cases: List[str]
risk_factors: List[str]

# 交易建议
optimal_entry_window: str
expected_move_pct: float
position_suggestion: float # 0.0-5.0 %
key_reasoning: str # ≤ 100 字
stop_loss_trigger: str
```

### 4.4 最终排序公式

```
Refined_Score = Coarse_Final_Score
× score_correction # ∈ [0.9, 1.05]
× (0 if "delisting_risk" ∈ risk_tags else 1)
× (0 if data_conflict_detected else 1)
```

### 4.5 Prompt 模板设计

本节定义精筛阶段核心 Agent 的完整 Prompt 模板。Prompt 中 `{variable}` 为系统运行时注入的动态变量；输出必须严格符合 JSON Schema，违反约束的字段触发"校验失败"路径（详见 4.5.5）。

> **Prompt 版本号绑定**：当前 Prompt 版本 = PRD 版本（v1.0）。任何字段增删或 Role 重定义必须升级 PRD 版本。

#### 4.5.1 Bull Researcher Prompt

```
## Role
你是一位专注于短期爆发机会的多头研究员。你的任务是为给定标的论证 7 天内上涨的催化剂和技术信号支撑。

## Context (由系统注入)
- 标的: {ticker}
- 当前价格: {price} | 5 日动量: {mom_5d}%
- 因子评分摘要: {factor_scores_summary} (粗筛 Top 5 因子)
- 近期新闻摘要 (3 条): {news_summary}
- 技术形态: {technical_pattern}

## Task
基于以上信息，从以下 4 个维度论证该标的 7 天内爆发的可能性:
1. **催化剂识别**: 是否存在明确的事件催化剂(财报/产品发布/行业利好)?
2. **技术突破信号**: 当前技术形态是否支持短期突破(布林带收窄后扩张、量价配合)?
3. **资金流向**: MFI/CMF 信号是否显示主力资金进场?
4. **动量延续性**: 当前动量是否有持续性(非单日脉冲)?

## Output Format (严格 JSON, 所有 confidence 字段为 0-100 整数)
{
"bull_confidence": 75,
"primary_catalyst": "iPhone 17 出货指引上修",
"supporting_signals": ["BB 带收窄 12 日突破", "MFI 14 上穿 60", "成交量放大 2.3x"],
"price_target_7d": "$185-$195",
"risk_to_bull_case": "FOMC 鹰派可能压制估值"
}
```

**Context 变量与字段约束**：

| 变量 / 字段 | 来源 | 类型 / 约束 |
|-------------|------|-------------|
| `{ticker}` | 粗筛输出 | 股票代码 |
| `{price}` | yfinance 实时价 | float，2 位小数 |
| `{mom_5d}` | MOM_5D 因子 | float，1 位小数（百分比） |
| `{factor_scores_summary}` | 粗筛 Top 5 因子 | string，≤ 200 tokens |
| `{news_summary}` | News Analyst | string，≤ 300 tokens |
| `{technical_pattern}` | Market Analyst | string，≤ 150 tokens |
| `bull_confidence` | 输出 | int，0-100 |
| `primary_catalyst` | 输出 | string，≤ 50 字 |
| `supporting_signals` | 输出 | string[]，长度 ≥ 1，每项 ≤ 30 字 |
| `price_target_7d` | 输出 | string，形如 `"$185-$195"` |
| `risk_to_bull_case` | 输出 | string，≤ 50 字 |

#### 4.5.2 Bear Researcher Prompt

```
## Role
你是一位专注于识别假突破和下行风险的空头研究员。你的任务是为给定标的论证 7 天内回落或假突破的可能性。

## Context (由系统注入)
- 标的: {ticker}
- 当前价格: {price} | 5 日动量: {mom_5d}%
- 因子评分摘要: {factor_scores_summary}
- 近期新闻摘要 (3 条): {news_summary}
- 技术形态: {technical_pattern}
- 历史假突破率: {false_breakout_rate}% (过去 90 天)

## Task
基于以上信息，从以下 4 个维度评估该标的的下行风险:
1. **假突破特征**: 当前走势是否符合历史假突破模式(低量突破、缺口未回补)?
2. **估值压力**: 当前估值是否已过度拉伸(RSI 超买、偏离均线过大)?
3. **利空隐患**: 是否存在被市场忽视的潜在利空(竞争加剧/监管/内部人卖出)?
4. **流动性陷阱**: 当前成交量是否足以支撑持续上涨?

## Output Format (严格 JSON, 所有 confidence/probability 字段为 0-100 整数)
{
"bear_confidence": 60,
"primary_risk": "RSI 76 超买且偏离 MA20 +12%",
"false_breakout_probability": 35,
"warning_signals": ["内部人 60 日累计卖出 $4.2M", "财报临近回调风险"],
"downside_target_7d": "-3% ~ -8%"
}
```

**Context 变量与字段约束**（除以下额外变量外，其余同 4.5.1）：

| 变量 / 字段 | 来源 | 类型 / 约束 |
|-------------|------|-------------|
| `{false_breakout_rate}` | 历史回测统计 | int，0-100 |
| `bear_confidence` | 输出 | int，0-100 |
| `primary_risk` | 输出 | string，≤ 50 字 |
| `false_breakout_probability` | 输出 | int，0-100 |
| `warning_signals` | 输出 | string[]，长度 ≥ 1，每项 ≤ 30 字 |
| `downside_target_7d` | 输出 | string，形如 `"-3% ~ -8%"` |

> **`false_breakout_rate` 计算**：过去 90 个交易日内所有"突破日"（满足 `Close_t > max(Close_{t-20:t-1})`）中，后续 5 日内最低价 ≤ 突破日收盘价 × 0.97 的占比（向下取整）。无突破事件时返回 50（中性值）。

#### 4.5.3 Portfolio Manager Prompt（风险审计层）

```
## Role
你是 AlphaScreener 的风险审计经理。你的核心职责是**审计粗筛因子模型的输出**，识别因子模型无法捕捉的数据冲突、催化剂时序问题、流动性陷阱、退市风险等结构化风险信号；
你**不是**"是否爆发"的最终判官 —— 爆发概率主要由 13 因子量化模型决定，你的输出只能对粗筛分数做有限修正 × [0.9, 1.05]，不允许大幅上调。

## Context (由系统注入)
- 标的: {ticker} | 价格: {price}
- 粗筛评分: {screening_score}/100 (来自 13 因子加权模型, 是基础概率信号)
- Bull Researcher 报告: {bull_report_json}
- Bear Researcher 报告: {bear_report_json}
- PEAD 季度真值 (FMP): {pead_quarterly_value}
- 数据源差异告警: {data_source_diff_flag} (本标的 OHLCV 主备差异 > 0.5% 时为 true)
- 流动性指标: {avg_daily_volume_20d_usd} (近 20 日平均成交额 USD)
- 退市/停牌状态: {delisting_status} ("normal" / "halted" / "delisting_pending")

## Task
以**风险审计**为核心，输出以下 5 类判断:
1. **数据冲突识别**: Bull 和 Bear 报告中的事实声称是否一致? 是否与 {data_source_diff_flag} 一致? 是否存在 SEC/yfinance/FMP 多源矛盾?
2. **催化剂时序一致性**: Bull 报告的催化剂日期是否真实落在 T+1 ~ T+7 窗口内? 是否存在"催化剂已过去"或"催化剂尚未到达"的时序错位?
3. **流动性与退市风险**: 是否识别出流动性陷阱({avg_daily_volume_20d_usd} 偏低但 Bull 仍预期大幅突破)或退市风险?
4. **粗筛分数修正**: 默认 1.00; 中等风险 → 0.95; 严重风险 → 0.90; 仅当催化剂明确一致 + 流动性充足 + 无数据冲突三项同时满足才允许上调至 1.05。
5. **可解释性附加输出**: final_rating / breakout_probability / catalyst_events 等字段保留作为飞书卡片展示和回测分析用，不进入最终排序。

## Output Format (严格 JSON, 字段命名与 4.3 BreakoutAssessment 对齐)
{
"final_rating": "Buy",
"breakout_probability": 0.62,
"confidence": 72,
"score_correction": 1.00,
"risk_tags": ["no_risk"],
"data_conflict_detected": false,
"catalyst_consistency": "consistent",
"bull_weight": 60,
"bear_weight": 40,
"catalyst_events": ["iPhone 17 出货指引", "Q1 财报预披露"],
"catalyst_timeline": "3-5 天内",
"similar_cases": ["AAPL 2024-Q4 突破"],
"risk_factors": ["FOMC 鹰派", "供应链不确定"],
"optimal_entry_window": "T+1 开盘后 30 分钟内",
"expected_move_pct": 6.5,
"position_suggestion": 3.0,
"key_reasoning": "粗筛分 78 + 催化剂一致 + 流动性充足, 无数据冲突, score_correction 维持 1.00",
"stop_loss_trigger": "T+1 买入价回撤 8%"
}
```

**Context 变量与字段约束**（按"风险审计核心字段"vs"可解释性字段"分组）：

| 字段 | 类型 | 约束 | 进入最终排序 |
|------|------|------|--------------|
| **`score_correction`** | float | **0.90 ≤ x ≤ 1.05**，2 位小数 | ✅ |
| **`risk_tags`** | string[] | 1-3 项，enum: `no_risk` / `data_conflict` / `liquidity_trap` / `delisting_risk` / `earnings_timing_mismatch` / `catalyst_already_passed` | ✅ |
| **`data_conflict_detected`** | bool | true / false | ✅（=true 时 Refined_Score = 0） |
| **`catalyst_consistency`** | enum | `consistent` / `ambiguous` / `contradicted` | ❌ 决策辅助 |
| `final_rating` | enum | `Strong Buy` / `Buy` / `Hold` / `Avoid` | ❌ 卡片展示 |
| `breakout_probability` | float | 0.0 - 1.0 | ❌ 可解释性 |
| `confidence` | int | 0 - 100 | ❌ 可解释性 |
| `bull_weight` / `bear_weight` | int | 0 - 100，且 sum = 100 | ❌ 可解释性 |
| `catalyst_events` | string[] | 长度 ≥ 1，每项 ≤ 30 字 | ❌ 卡片展示 |
| `catalyst_timeline` | string | 形如 `"2-3 天内"` | ❌ 卡片展示 |
| `similar_cases` | string[] | 0-5 项，每项 ≤ 50 字 | ❌ 卡片展示 |
| `risk_factors` | string[] | 长度 ≥ 1 | ❌ 卡片展示 |
| `optimal_entry_window` | string | ≤ 30 字 | ❌ 卡片展示 |
| `expected_move_pct` | float | -50.0 ~ +50.0，1 位小数 | ❌ 可解释性 |
| `position_suggestion` | float | 0.0 - 5.0（单位 %） | ❌ 仓位建议 |
| `key_reasoning` | string | ≤ 100 字，必须包含 score_correction 理由 | ❌ 审计追踪 |
| `stop_loss_trigger` | string | ≤ 50 字 | ❌ 风控指令 |

**Context 输入变量**：

| 变量 | 来源 | 格式 |
|------|------|------|
| `{ticker}` / `{price}` | 粗筛输出 / yfinance | 股票代码 / float 2 位小数 |
| `{screening_score}` | 粗筛 Breakout_Score | int 0-100 |
| `{bull_report_json}` | Bull Researcher 输出 | string，≤ 600 tokens |
| `{bear_report_json}` | Bear Researcher 输出 | string，≤ 600 tokens |
| `{pead_quarterly_value}` | FMP 精筛阶段重算 | float（季度同质 PEAD） |
| `{data_source_diff_flag}` | 备用 OHLCV 校验 | bool |
| `{avg_daily_volume_20d_usd}` | 近 20 日平均成交额 | float（USD） |
| `{delisting_status}` | 退市/停牌状态 | enum: `normal` / `halted` / `delisting_pending` |

#### 4.5.4 评级 ↔ 概率 ↔ score_correction 映射规则

**评级 ↔ 概率（软约束，建议但不强制）**：

| final_rating | 建议 breakout_probability |
|--------------|---------------------------|
| Strong Buy | ≥ 0.70 |
| Buy | 0.50 ~ 0.70 |
| Hold | 0.30 ~ 0.50 |
| Avoid | < 0.30 |

> 允许 LLM 输出"Hold + 0.62"等不严格匹配的组合（保留校准信息）；仅当 final_rating 与 breakout_probability **反向**（如 Strong Buy + 0.20）才视为校验失败。

**score_correction 硬约束**：
- 必须 `0.90 ≤ score_correction ≤ 1.05`，超出范围视为校验失败
- 默认 1.00；下调路径：1.00 → 0.95（中等风险）→ 0.90（严重风险，需 risk_tags 至少 2 项非 `no_risk`）
- 上调至 1.05 必须三项同时满足：`catalyst_consistency = "consistent"` + `risk_tags = ["no_risk"]` + `data_conflict_detected = false`

#### 4.5.5 输出校验失败处理

| 失败类型 | 处置 |
|----------|------|
| JSON 解析失败 | 重试 1 次（同 Prompt + 显式追加"请严格输出 JSON"），仍失败 → score_correction = 1.00 + risk_tags = `["no_risk"]` 兜底 |
| score_correction 越界 | clamp 到 [0.90, 1.05] + 日志告警 |
| risk_tags 含非 enum 值 | 过滤非法值；若过滤后为空 → `["no_risk"]` |
| final_rating 与 probability 反向 | 保留两值 + 日志告警（不阻断排序） |
| 字段缺失 | 缺失字段使用类型默认值（数值 0 / 字符串 "" / 数组 []）+ 日志告警 |

#### 4.5.6 Prompt 修改审批

| 修改类型 | 审批流程 |
|----------|----------|
| Context 字段内容选择（如新增/移除某分析师摘要） | 进化 Agent 自动执行 + 变更日志 |
| Task 描述措辞优化 | 人工 review + 灰度 A/B（10% 流量观察 14 天） |
| Output JSON 字段增删 / Role 重定义 | PRD 版本升级（MINOR）+ 产品评审 |

### 4.6 性能与成本

#### 4.6.1 执行模型（分批串行 + 单标的两阶段 DAG）

| 参数 | 数值 |
|------|------|
| 每日精筛标的数 | Top 20 |
| 批大小 | 3 标的 / 批 |
| 单标的 DAG | (Bull ∥ Bear) → PM |
| 总批次数 | 7 批（3+3+3+3+3+3+2） |
| 批内 Stage 1 并发 | 6（3 标的 × Bull/Bear） |
| 批内 Stage 2 并发 | 3（3 标的 × PM） |
| 单批耗时 | ~6-8 秒 |
| 全量耗时 | ~49 秒（最坏 70 秒） |
| 进程内存峰值 | < 200 MB |

#### 4.6.2 Token 与成本预算

| Agent | Input Budget | Output Budget |
|-------|--------------|---------------|
| Bull Researcher | 1500 | 600 |
| Bear Researcher | 1500 | 600 |
| Portfolio Manager | 2500 | 1000 |
| Breakout Analyst | 2000 | 800 |

```
日成本: 20 标的 × 3 调用 × (2500 input × $0.15/1M + 1000 output × $0.6/1M) ≈ $0.06/日
月成本: ~$1.8 / $100 月预算
```

#### 4.6.3 成本熔断

| 级别 | 触发条件 | 动作 |
|------|----------|------|
| L1 警告 | 日成本 ≥ $0.80 | 批大小 3 → 2 |
| L2 降级 | 日成本 ≥ $1.0 | 暂停精筛，仅输出粗筛 |
| L3 节约 | 月成本 ≥ $80 | 精筛缩减至 Top 10 |
| L4 熔断 | 月成本 ≥ $95 | 完全停止 LLM 调用 |

#### 4.6.4 Rate Limit 处理

| 机制 | 实现 | 参数 |
|------|------|------|
| 客户端限速 | `asyncio.Semaphore` + 令牌桶 | 5 RPS |
| 指数退避重试 | tenacity | 初始 1s，最大 60s，5 次 |
| 429 处理 | 解析 `Retry-After` header | 优先级高于本地退避 |
| 超时控制 | 单次 60s | 超时计入失败 |
| 降级路径 | 重试用尽 → 跳过 LLM，仅粗筛 | 记录到日志 |

### 4.7 LLM Ablation 测试（强制）

每日并行记录两套排序：

| 轨道 | 排序公式 | 写入 |
|------|----------|------|
| **A. 纯因子** | `Refined_Score_pure = Coarse_Final_Score` | `signals_refined_pure.parquet` |
| **B. 含 LLM 修正** | `Coarse_Final_Score × score_correction × risk_filter` | `signals_refined.parquet` |

**LLM 增量门槛**（60 天滚动）：

| ΔLift@20 | 处置 |
|----------|------|
| ≥ 0.05 | 通过：LLM 修正层启用 |
| [0, 0.05) | 边缘：LLM 仅作可解释性输出，不参与排序 |
| < 0 | 失败：禁用 LLM 修正层，触发 PRD review |

---

## 5. 回测系统

### 5.1 回测引擎选型

| 引擎 | 内存占用（2,000 股 × 2 年） | 选用 |
|------|------------------------------|------|
| vectorbt | 3-5 GB（全量向量化） | ❌ 超 8GB 限制 |
| **backtrader** | **< 1 GB**（事件驱动） | ✅ |
| zipline | 1-2 GB | ❌ 维护活跃度低 |

### 5.2 回测规则

| 参数 | 设定 |
|------|------|
| 持仓周期 | 7 个交易日 |
| 买入规则 | T+1 开盘市价，`entry_price = Open_{T+1}` |
| 卖出规则 | ① T+7 收盘市价；② 盘中跌破 `entry_price × 0.92`；③ 停牌/退市 |
| 止损执行细节 | ② 触发后以触发当日收盘价成交（backtrader daily bar 模式下用 Low ≤ entry×0.92 判定触发，以 Close 近似成交价）；如 T+7 前未触发则按 T+7 收盘正常退出 |
| 资金管理 | 每只标的等额分配 |
| 最大持仓 | 20 只 |
| 摩擦成本 | 手续费 0.1% + 滑点 0.2% |
| 绩效基准 | SPY（美股）/ 2800.HK（港股，V1.5+） |

### 5.3 绩效指标

胜率、平均收益、盈亏比、年化收益、夏普比率、最大回撤、因子贡献分析。

### 5.4 Paper Trading 转正机制（双层）

#### Tier 1 - 工程转正（≥ 60 个交易日，验证系统稳定性）

| 条件 | 阈值 |
|------|------|
| 积累交易日 | ≥ 60 天 |
| 系统稳定性 | 60 天内无 L3/L4 成本熔断 |
| 数据完整性 | factor_nan_count 日均 < 全标的 5% |
| 调度准时 | daily_scan 成功率 ≥ 95% |
| 备用数据源校验 | 主备 OHLCV 差异 > 0.5% 告警 ≤ 5 次/月 |

> **工程转正 ≠ alpha 已证明**。完成后系统进入持续运行，可启用 V1.0 第二阶段进化反馈，但不允许实盘建议。

#### Tier 2 - 策略转正（实盘前必须达成）

| 条件 | 阈值 |
|------|------|
| Walk-forward 历史回测 | ≥ 2 年；Lift@20 bootstrap 95% CI 下界 > 1.10 |
| Live shadow 时长 | ≥ 6 个月（约 126 个交易日） |
| Lift@20（live） | 滚动 60 天均值 > 1.10 且 CI 下界 > 1.05 |
| Precision@20（live） | 滚动 60 天均值 > base_rate × 1.10 |
| LLM ablation | ΔLift@20 ≥ 0.05（如启用 LLM 层） |
| 因子 IC 持续性 | 滚动 60 天 IC 与历史 IC 衰减 < 50% |

### 5.5 Alpha 验收口径

| 指标 | 公式 | 用途 |
|------|------|------|
| **base_rate** | 全市场 T+7 涨幅 ≥ 10% 标的占比 | 基准概率 |
| **Precision@K** | Top K 中实际命中比率（K = 10, 20） | 选股精度 |
| **Lift@K** | `Precision@K / base_rate` | 核心指标 |
| **Recall@K** | Top K 命中数 / 全市场命中数 | 覆盖率 |
| **IC** | `spearman(rank_by_score, actual_return)` | 排序一致性 |
| **Block-bootstrap 95% CI 下界** | block size = 5 交易日 × 1000 次 | 统计显著性 |

**验收阈值**：

| 阶段 | Precision@20 | Lift@20 | CI 下界 |
|------|---------------|---------|---------|
| MVP 工程转正 | 仅记录 | 仅记录 | - |
| V1.0 第二阶段 | ≥ base_rate × 1.05 | ≥ 1.05 | ≥ 1.0 |
| 策略转正 | ≥ base_rate × 1.10 | ≥ 1.10 | ≥ 1.05 |
| V2.0 | ≥ base_rate × 1.20 | ≥ 1.20 | ≥ 1.10 |

### 5.6 Purged Walk-Forward + Block Bootstrap

```
1. 时间序列分块: 2 年历史按月切为 24 个块
2. Purge 窗口: 训练块与验证块之间留 10 个交易日 gap
3. Embargo: 验证块之后留 5 个交易日
4. 滑窗: 训练 12 个月 → 验证 1 个月，共 12 折
5. Bootstrap: block size = 5 交易日，重采样 1000 次
```

**通过标准**：12 折 Lift@20 均值 ≥ 1.10、中位数 ≥ 1.05、CI 下界 ≥ 1.05、Lift@20 < 1.0 的折数 ≤ 3。

### 5.7 回测与实盘对账

**偏差容忍范围**：

| 指标 | 允许偏差 | 观察期 | 超出后动作 |
|------|----------|--------|------------|
| 胜率 | 回测 ± 8% | 连续 20 交易日 | 触发调查 |
| 平均收益率 | 回测 ± 2% | 连续 20 交易日 | 触发调查 |
| 最大回撤 | 回测 × 1.5 以内 | 任意时刻 | 立即告警 |
| 夏普比率 | 回测 × 0.7 以上 | 滚动 30 天 | 触发调查 |

---

## 6. 自我进化机制

> **MVP 阶段策略**：完全冻结全部 13 因子权重（参数版本号 v1.0.0），仅启用快速层 CUSUM 监控做保护性告警，不调权。慢速层权重调整延后至工程转正后启用（环境变量 `EVOLUTION_WEIGHT_ADJUST_ENABLED` 控制）。

### 6.1 双层进化反馈

#### 6.1.1 快速层（CUSUM 监控）

**触发**：每日 T+8 标签回填后立即触发

```
对每个 active 因子 f:
S_t = max(0, S_{t-1} + IC_t - μ_IC - k)
其中 μ_IC = 滚动 90 天 IC 均值，k = 0.005，CUSUM 阈值 h = 0.05
若 S_t > h → 触发告警
```

**触发动作（不调权）**：

| 级别 | 条件 | 行为 |
|------|------|------|
| L1 告警 | 单因子 CUSUM 触发 | 飞书告警，记录 `factor_health_daily` |
| L2 暂停因子 | 单因子 5 日内触发 ≥ 2 次 | active → degraded（权重冻结） |
| L3 全局降级 | 当日 ≥ 5 个因子同时触发 | 启用低活跃期模式 + critical 告警 |

#### 6.1.2 慢速层（30 天权重调整，V1.0 第二阶段启用）

**触发频率**：每两周（biweekly_evolution，月初 1 日 + 月中 15 日 05:30 UTC）

| 动作 | 触发条件 |
|------|----------|
| 权重微调 | 30 天 CUSUM 累计 ≥ 5 次 + 30 天 IC 较入选时下降 ≥ 30% |
| 因子降级 | 30 天内 5 次 CUSUM 未恢复 + 30 天 IC < 0 |
| 因子重启 | degraded 因子 30 天 IC > 0 且 CUSUM 连续 15 天无告警 |
| 进化暂停 | Lift@20 < 1.0 持续 30 天 |

#### 6.1.3 Strategy Review Agent Prompt（慢速层执行体）

慢速层进化由 LLM Strategy Review Agent 驱动，每两周（biweekly_evolution）触发一次，输出可执行的权重调整 / 因子增删提案。Agent 输出仅作为**提案**，必须经人工 approve 后由系统执行。

**输入数据契约**：

```json
{
"window_days": 30,
"selection_history_summary": "<按日期/标的/评级/实际收益聚合, ≤ 800 tokens>",
"factor_contribution_json": {
"MOM_5D": {"ic_30d": 0.04, "cum_pnl_contrib": 0.012, "cusum_alerts_30d": 1},
"BB_SQUEEZE": {"ic_30d": 0.02, "cum_pnl_contrib": 0.005, "cusum_alerts_30d": 3}
},
"backtest_vs_live_deviation": {
"win_rate_gap": 0.05,
"return_gap": 0.012,
"deviation_trend": "stable",
"top_deviation_factors": [
{"factor": "MOM_5D", "ic_backtest": 0.05, "ic_live": 0.02},
{"factor": "BB_SQUEEZE", "ic_backtest": 0.06, "ic_live": 0.04}
],
"top_deviation_sectors": [
{"sector": "Tech", "win_rate_gap": -0.12}
],
"observation_window_days": 20,
"alert_triggered": false
},
"current_factor_config_summary": "v1.0.0: MOM_5D=14%, PTH=12%, ..."
}
```

**Prompt 模板**：

```
## Role
你是 AlphaScreener 的策略复盘 Agent。基于过去 N 天的选股回测与实盘对账数据，诊断当前策略的有效性，输出可执行的进化建议。

## Context (由系统注入)
- 复盘窗口: 过去 {window_days} 个交易日 (默认 30)
- 选股记录: {selection_history_summary} (按日期/标的/评级/实际收益聚合, ≤ 800 tokens)
- 因子贡献分析: {factor_contribution_json} (按因子的 IC 时序、累计收益贡献)
- 实盘对账: {backtest_vs_live_deviation} (见上方 schema)
- 当前因子配置: {current_factor_config_summary} (版本号 + 各因子权重)

## Task
基于以上数据，从以下 5 个维度生成诊断报告:
1. **失效因子识别**: 哪些 active 因子近 30 日 IC < 0.01? 是否触发 6.3 衰退条件?
2. **低估因子识别**: 哪些 probation/active 因子表现优于预期, 建议提升权重?
3. **权重调整方案**: 提供单次调整方案 (单因子 ±5% 内, 总和保持 100%)
4. **规则增删建议**: 是否需要新增/删除硬过滤条件? 给出依据
5. **市场 regime 判断**: 当前是否处于低活跃期 / 风格切换期? 是否需切换因子模板?

## Output Format (严格 JSON)
{
"diagnosis_summary": "MOM_5D 近 30 日 IC 衰减至 0.02, BB_SQUEEZE 表现稳健; Tech 板块实盘负偏差 -12%, 建议下调 MOM_5D 权重 2%",
"factor_actions": [
{"action": "weight_adjust", "factor": "MOM_5D", "from": 0.14, "to": 0.12, "reason": "30 天 IC 0.02 < 入选时 0.05 的 50%"},
{"action": "demote", "factor": "OLD_X", "from_status": "active", "to_status": "degraded", "reason": "连续 60 日 IC < 0.01"},
{"action": "propose_new", "factor_name": "NEW_X", "formula": "(Close - shift(Close, 3)) / ATR_5", "rationale": "短线动量与波动率联合信号"}
],
"rule_actions": [
{"action": "add_filter", "condition": "Volume > mean(Volume, 50) * 1.5", "phase": "Phase1"}
],
"regime_signal": "normal",
"confidence": 80,
"requires_human_review": true
}
```

**字段类型约束**：

| 字段 | 类型 | 约束 |
|------|------|------|
| `diagnosis_summary` | string | ≤ 150 字 |
| `factor_actions[].action` | enum | `weight_adjust` / `demote` / `promote` / `propose_new` / `retire` |
| `factor_actions[].from` / `to` | float | 单次调整 ±5% 内；所有权重总和误差 ≤ 0.001 |
| `factor_actions[].formula`（仅 `propose_new`） | string | 必须通过 6.4 因子 DSL 验证 |
| `rule_actions[].action` | enum | `add_filter` / `remove_filter` / `adjust_threshold` |
| `rule_actions[].phase` | enum | `Phase1` / `Phase2` |
| `regime_signal` | enum | `normal` / `low_activity` / `style_rotation` / `crisis` |
| `confidence` | int | 0-100 |
| `requires_human_review` | bool | true 时输出仅作为提案，需人工 approve |

**Token 与频率预算**：

| 维度 | 限额 |
|------|------|
| Input | ≤ 4000 tokens |
| Output | ≤ 1500 tokens |
| 调用频率 | 每两周 1 次（biweekly_evolution，月初 1 日 + 月中 15 日） |
| 单次调用成本 | ≈ $0.0015（GPT-4o-mini） |

**输出执行链**：

```
LLM 提案 (requires_human_review=true)
│
▼
写入 evolution_proposals 表 (status=pending)
│
▼
飞书推送审核卡片 (含 diff: 旧权重 vs 新权重)
│
├─ approve → 进入下次 biweekly 任务执行
├─ reject → 状态置 rejected, 记录拒绝原因
└─ 7 天无响应 → 自动 reject + 告警
```

### 6.2 自进化操作类型

| # | 操作 | 触发条件 | 启用版本 |
|---|------|----------|----------|
| 1 | 权重微调 | 因子 CUSUM 累计 + IC 下降 | V1.0 第二阶段 |
| 2 | 因子增加 | 人工提案（V1.0）/ Agent 自主（V1.5） | V1.0+ |
| 3 | 因子删除 | 连续 60 日 IC < 0.01 或数据源不可用 | V1.0 第二阶段 |
| 4 | 阈值调整 | 硬过滤过度/不足 | MVP（动态阈值） |
| 5 | 辩论策略调整 | LLM 评级与实际偏差大 | V1.0 第二阶段 |
| 6 | 市场环境适配 | 大盘风格切换 | V1.5+ |

### 6.3 因子生命周期

```
[proposed] → AST验证 → [probation] → 晋升 → [active]
↓ ↓
失败 → [rejected] 衰退 → [degraded] → [retired]
↓
恢复 → [active]
```

| 状态 | 权重范围 | 持续条件 |
|------|----------|----------|
| proposed | 0% | AST 验证 + 沙箱回测 |
| probation | 2-3% | 观察 30 个交易日 |
| active | 按模型分配 | IC 持续达标 |
| degraded | 冻结 | 观察 15 个交易日 |
| retired | 0% | 移入 archived |
| rejected | 0% | 60 天内不重试 |

**晋升条件**（probation → active，必须全部满足）：

| 维度 | 阈值 |
|------|------|
| 观察期 | ≥ 30 个交易日 |
| IC 均值 | > 0.03 |
| IC 标准差 | < 0.05 |
| 胜率贡献 | A/B 对比正向 |
| 数据 NaN 率 | < 1% |
| 计算时间变异系数 | < 30% |

**衰退触发**（满足任一）：

| ID | 规则 |
|----|------|
| D1 | 连续 15 个交易日 daily_IC < 0.01 |
| D2 | 滚动 30 天 IC 较入选时下降 > 50% |
| D3 | 与已有 active 因子相关性 > 0.8 |

### 6.4 因子公式 DSL 与 AST 验证

#### 6.4.1 DSL 白名单

**允许的数据字段**：

| 类别 | 字段 |
|------|------|
| 价格类 | `Close`, `Open`, `High`, `Low`, `AdjClose` |
| 成交量 | `Volume` |
| 移动均线 | `SMA_5`, `SMA_10`, `SMA_20`, `SMA_50`, `SMA_200`, `EMA_5`, `EMA_10`, `EMA_12`, `EMA_20`, `EMA_26`, `EMA_50` |
| 动量指标 | `RSI_14`, `RSI_7`, `MACD`, `MACD_Signal`, `MACD_Hist` |
| 波动率指标 | `BB_upper`, `BB_lower`, `BB_mid`, `ATR_5`, `ATR_14`, `ATR_20` |
| 资金流指标 | `MFI_14`, `CMF_21`, `OBV` |
| 衍生字段 | `Daily_Return`, `Log_Return`, `Typical_Price` |

**允许的操作符**：
- 算术：`+`, `-`, `*`, `/`, `**`
- 比较：`>`, `<`, `>=`, `<=`, `==`, `!=`
- 逻辑：`and`, `or`, `not`

**允许的函数**：

| 函数 | 说明 | 示例 |
|------|------|------|
| `abs(x)` | 绝对值 | `abs(Close - Open)` |
| `max(a, b)` | 取较大值 | `max(Close, SMA_20)` |
| `min(a, b)` | 取较小值 | `min(High - Low, ATR_14)` |
| `mean(x, n)` | 滚动均值 | `mean(Volume, 20)` |
| `std(x, n)` | 滚动标准差 | `std(Close, 20)` |
| `rolling(x, n)` | 滚动窗口 | `rolling(Close, 10)` |
| `shift(x, n)` | 时间偏移 | `shift(Close, 5)` |
| `rank(x)` | 截面排名 (0-1) | `rank(Volume)` |
| `log(x)` | 自然对数 | `log(Volume)` |
| `sqrt(x)` | 平方根 | `sqrt(ATR_14)` |

**禁止操作（硬性黑名单）**：

| 类别 | 禁用项 |
|------|--------|
| 模块导入 | `import`, `__import__`, `importlib` |
| 代码执行 | `exec`, `eval`, `compile` |
| 文件 I/O | `open`, `read`, `write`, `os.*`, `sys.*`, `pathlib`, `shutil` |
| 网络请求 | `requests`, `urllib`, `socket` |
| 反射/全局修改 | `globals()`, `locals()`, `setattr`, `delattr`, `getattr` |
| 子进程 | `subprocess` |

#### 6.4.2 6 步 AST 验证流程

```
1. 公式字符串 → ast.parse
└─ 解析失败 → reject("语法错误")
2. AST 节点白名单检查
└─ 仅允许：BinOp, Compare, Call, Name, Constant, UnaryOp, BoolOp, Attribute, Subscript
└─ 出现 Import / FunctionDef / Assign 等 → reject("非法节点: <type>")
3. 字段引用验证
└─ 遍历所有 Name 节点，必须在 6.4.1 字段表中
└─ 否则 → reject("字段不存在: <field_name>")
4. 函数调用验证
└─ 遍历所有 Call 节点，函数名必须在 6.4.1 函数表中
└─ 否则 → reject("禁止函数: <func_name>")
5. 复杂度检查
└─ AST 深度 ≤ 10，节点总数 ≤ 50
└─ 超出 → reject("复杂度超限")
6. 沙箱回测
└─ 30 个交易日 × 随机 100 只标的
└─ 单次执行 ≤ 5 秒，内存 ≤ 500 MB
└─ 异常 → reject("运行时错误: <msg>")
```

#### 6.4.3 安全边界

| 安全规则 | 处理方式 | 说明 |
|----------|----------|------|
| 除零保护 | 分母自动加 `epsilon = 1e-8` | `a / b` 编译为 `a / (b + 1e-8)` |
| NaN 处理 | 该因子当日得分 = 0 | 不影响其他因子评分 |
| Inf 处理 | 该因子当日得分 = 0 | 正负无穷均视为无效 |
| 性能约束 | 拒绝入库 | > 5s / 1000 标的 → "性能不达标" |
| 内存约束 | 拒绝入库 | > 500MB → "资源超限" |
| 数值范围 | 强制标准化 | 经 z-score 标准化到 [-3, +3] 截断后映射 [0, 100] |

**新因子标准化流程**（强制执行）：

```python
raw_value = formula(data)
raw_value[isnan(raw_value) | isinf(raw_value)] = 0 # Step 1
z = (raw_value - mean(raw_value, 30)) / (std(raw_value, 30) + 1e-8) # Step 2
z_capped = clip(z, -3, +3) # Step 3
score = 50 + z_capped * (50 / 3) # Step 4 → [0, 100]
```

#### 6.4.4 人工审核流程

```
LLM/人工提案
│
▼
6.4.2 AST 验证（自动，< 1 秒）
│ 通过
▼
沙箱回测（30 天 × 100 标的）
│ 通过 + IC > 0.02 + 胜率 > 52% + 与现有因子最大相关性 < 0.7
▼
生成"因子提案报告"（公式 + IC/胜率/换手率 + 相关性矩阵 + 风险评估）
│
▼
人工审核 (approve / reject)
│ approve
▼
probation 状态加入因子体系（初始 2-3% 权重，30 日观察）
```

**审核最低标准**：

| 维度 | 要求 |
|------|------|
| 30 日 IC | > 0.02 |
| 胜率 | > 52% |
| 与现有因子最大相关性 | < 0.7 |
| 30 日运行稳定性 | 无运行时错误 |
| 可解释性 | 因子逻辑有合理金融含义 |

#### 6.4.5 失败处理

| 阶段 | 失败处理 |
|------|----------|
| AST 验证失败 | 记录原因，回传提案方（CLI 反馈或 Agent 重试，最多 3 次） |
| 沙箱回测失败 | 记录到"失败因子日志"；连续 3 次失败 → 放弃该因子方向 |
| 人工 reject | 写入 `evolution_proposals.status = rejected`，60 天内不重试 |

### 6.5 版本管理

`MAJOR.MINOR.PATCH`：
- MAJOR：删除 > 3 因子或类别权重 > 30% 调整
- MINOR：新增/退役 active 因子
- PATCH：单因子权重 ≤ 5% 调整

**保留策略**：最近 10 版本完整保留；MAJOR 永久保留。支持一键回滚至任意历史版本（生成新 PATCH 版本）。

---

## 7. 数据流设计

### 7.1 数据源架构

| 层级 | 数据源 | 用途 | 成本 | 调用规模 |
|------|--------|------|------|----------|
| **主力** | yfinance | OHLCV / 基本面 / 新闻 / 基础内部人 | 免费 | ~2,000 股/日 |
| **备用 OHLCV** | Stooq Free（首选）/ Alpaca / Polygon | 交叉校验 + yfinance 失效降级 | 免费 | ~120 股/日 |
| **辅助** | FMP Free tier | 共识 EPS / 内部人交易细节 | 免费（≤ 250 req/日） | ~80 req/日 |
| **V1.5 扩展** | FMP Starter ($19/月) | 社交情绪 / 新闻情绪 | $19/月 | 按需 |

#### 7.1.1 备用 OHLCV 源选型对比

**优先级**：Stooq Free > Alpaca Market Data Free > Polygon.io Basic

| 数据源 | 免费 tier 限制 | 美股覆盖 | 数据延迟 | 选用理由 |
|--------|----------------|----------|----------|----------|
| **Stooq Free**（首选） | 无明确限速（约 100 req/min 软限） | SP500 + Russell 1000 完整覆盖 | T+1 收盘后 1-2 小时 | 完全免费、无 API key、URL 直连 CSV，最适合 4c8g 单机 |
| Alpaca Market Data Free | 200 req/min；最近 30 天数据 | NYSE/NASDAQ 全部 | 实时 | 需 API key + 注册；适合做实时校验 |
| Polygon.io Basic | 5 req/min（极严格） | 全美股 | 15 分钟延迟 | 仅作为最后备份；免费 tier 速率不够日常使用 |

#### 7.1.2 数据源 → 字段精确映射

| 数据需求 | MVP 数据源 | API / 属性 | 精筛阶段升级源 |
|----------|------------|------------|----------------|
| 股价 OHLCV | yfinance | `yf.download(tickers, period, group_by='ticker')` | Stooq CSV（校验） |
| 基本面（粗筛） | yfinance | `Ticker.info["forwardEps"]`, `Ticker.info["marketCap"]`, `Ticker.info["sector"]`, `Ticker.info["industry"]` | FMP `/v3/profile/{ticker}` |
| 共识 EPS（粗筛 ~2,000） | yfinance | `Ticker.info["forwardEps"]`（年化预期，近似） | — |
| 共识 EPS（精筛 Top 20） | FMP | `/v3/analyst-estimates/{ticker}?period=quarter` | — |
| 实际 EPS | yfinance（粗筛）/ FMP（精筛） | `Ticker.earnings_history` / `/v3/historical-earning-calendar/{ticker}` | — |
| 财报日期 | yfinance | `Ticker.earnings_dates`（用于 PEAD_FLAG） | FMP `/v3/earning_calendar` |
| 内部人交易 | yfinance（基础）/ FMP（详细） | `Ticker.insider_transactions` / `/v4/insider-trading?symbol={ticker}` | — |
| 分析师评级 | FMP（仅精筛 Top 20） | `/v3/grade/{ticker}` | — |
| 新闻 | yfinance | `Ticker.news`（标题 + URL） | FMP `/v3/stock_news?tickers={ticker}` |
| 社交情绪（V1.5） | FMP Starter | `/v4/social-sentiment?symbol={ticker}` | — |
| 退市/停牌状态 | yfinance | `Ticker.info["regularMarketTime"]` 校验 + SEC 退市公告 | — |

> **缺失数据降级**（详见 3.1.4）：FMP 不可用时，PEAD/INSIDER_BUY 回退到 yfinance；SOCIAL_SENT/NEWS_SENT 在 MVP 本就为 0% 权重不影响。

### 7.2 备用数据源交叉校验

```
1. yfinance 拉取 Top 100 + Top 20 的 T 日 OHLC
2. 备用源拉取相同标的（5 RPS 限速）
3. 字段级对比 Open/High/Low/Close 任一相对差异 > 0.5%
→ 写入 data_source_diff 表
→ 当日告警 ≥ 5 次 → 飞书告警
4. 差异标的：daily_scan 仍以 yfinance 为准，PM Prompt 注入 data_source_diff_flag = true
```

**全量降级**：yfinance 连续 3 个交易日失败率 > 30% → 自动切换备用源全量 + critical 告警。

### 7.3 yfinance 调用约束

| 维度 | 规则 |
|------|------|
| 批量下载 | 每批 ≤ 50 股，`threads=False` |
| 并发限速 | ≤ 5 RPS（asyncio.Semaphore） |
| 增量更新 | 每日仅拉增量（1-2 个交易日） |
| 重试策略 | tenacity 指数退避（初始 2s，最大 60s，3 次） |
| 失败降级 | 单股票连续 3 次失败 → 当日跳过 |
| 内存控制 | polars LazyFrame 流式处理 |

### 7.4 日频扫描流水线

```
23:00 UTC (美股盘后)
↓
1. 数据同步 (yfinance + 备用源校验)
↓
2. 因子计算 (polars 分块流式，4 批 × 500 标的)
↓
3. 粗筛 Phase 1 + Phase 2 (硬过滤 → 加权评分 → Top 30)
↓
4. 行业去重 (轻量规则 → Top 20)
↓
5. LLM 精筛 (4 位 Analyst + Bull/Bear/PM 风险审计)
↓
6. 最终输出 (signals_refined.parquet + 飞书推送)
↓
7. Paper Trading 记录
↓
8. 次日 11:00 UTC: daily_backtest_incremental
```

### 7.5 飞书消息推送

**触发**：每日扫描完成后自动触发（7.4 流水线第 6 步之后），1 次/日。

#### 7.5.1 配置

**环境变量**：
```bash
FEISHU_APP_ID=cli_xxxx # 飞书自建应用 App ID
FEISHU_APP_SECRET=xxxx # 飞书自建应用 App Secret
FEISHU_TARGET_OPENID=ou_xxxx # 接收消息的用户 Open ID
FEISHU_PUSH_ENABLED=true # 推送开关（可关闭，默认 true）
```

**应用权限**：`im:message:send_as_bot`（以应用身份发送消息）

#### 7.5.2 API 调用流程

```
Step 1: 获取 tenant_access_token (有效期 2 小时)
POST https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal
Body: {"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
Response: {"tenant_access_token": "t-xxxx", "expire": 7200}

Step 2: 发送交互卡片
POST https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id
Header: Authorization: Bearer {tenant_access_token}
Body: {
"receive_id": FEISHU_TARGET_OPENID,
"msg_type": "interactive",
"content": "{card_json_stringified}"
}
```

> **Token 缓存**：tenant_access_token 缓存到内存或 SQLite（key=`feishu_token`），过期前 5 分钟主动刷新。

#### 7.5.3 卡片 JSON 模板

```json
{
"msg_type": "interactive",
"card": {
"header": {
"title": "📊 AlphaScreener 每日报告 | {date}",
"template": "blue"
},
"elements": [
{
"tag": "markdown",
"content": "**扫描概览**\n全市场标的: {total} | 粗筛通过: {pass_count} | 精筛输出: {refine_count}"
},
{
"tag": "markdown",
"content": "**🏆 今日精筛 Top 标的**\n| 排名 | 标的 | 评级 | 置信度 | 主要催化剂 |\n|---|---|---|---|---|\n| 1 | {ticker1} | {rating1} | {confidence1}% | {catalyst1} |\n| 2 | {ticker2} | {rating2} | {confidence2}% | {catalyst2} |\n| ... | ... | ... | ... | ... |"
},
{
"tag": "markdown",
"content": "**📈 Alpha 验收（pure / llm 双轨）**\nPrecision@20: {p20_pure}% / {p20_llm}%\nLift@20: {lift_pure} / {lift_llm}\nbase_rate: {base_rate}%"
},
{
"tag": "markdown",
"content": "**📉 回测绩效（滚动 7 天）**\n胜率: {win_rate}% | 夏普: {sharpe} | 平均收益: {avg_return}%"
},
{
"tag": "markdown",
"content": "**💰 成本控制**\n今日 LLM 成本: ${daily_cost} | 本月累计: ${monthly_cost}/$100"
},
{"tag": "hr"},
{
"tag": "markdown",
"content": "**⚠️ 异常/告警**\n{alerts_summary}"
}
]
}
}
```

**字段映射**：

| 占位符 | 数据来源 | 备注 |
|--------|----------|------|
| `{date}` | 系统日期 | 扫描执行日期 |
| `{total}` / `{pass_count}` / `{refine_count}` | scan_engine 输出 | 标的数量 |
| `{ticker_i}` / `{rating_i}` / `{confidence_i}` / `{catalyst_i}` | PM Output | i = 1..N（默认 5，可配置） |
| `{p20_pure}` / `{p20_llm}` / `{lift_pure}` / `{lift_llm}` / `{base_rate}` | `alpha_acceptance_daily` 表 | Ablation 双轨展示 |
| `{win_rate}` / `{sharpe}` / `{avg_return}` | 回测模块滚动指标 | 7 天窗口 |
| `{daily_cost}` / `{monthly_cost}` | `llm_cost_daily` 表 | 美元计 |
| `{alerts_summary}` | `alerts` 表当日记录 | 无告警则 `"✅ 系统正常"` |

#### 7.5.4 失败处理

| 失败类型 | 处置 |
|----------|------|
| API 调用失败 | tenacity 重试 3 次，间隔 5s / 15s / 60s |
| Token 过期（401） | 自动刷新 tenant_access_token 后重试 |
| 连续 3 天推送失败 | 记录 ERROR 日志 + 进程内告警计数；不阻塞主流程 |
| 卡片渲染失败（400） | 降级为纯文本消息（保留 Top 5 标的列表） |

#### 7.5.5 MVP 阶段范围

- ✅ 每日精筛结果卡片推送
- ✅ Alpha 验收双轨指标 + 回测绩效 + 成本统计
- ✅ 失败重试与降级
- ❌ 卡片交互按钮（V1.0：如"查看详情"、"加入自选"）
- ❌ 多人推送 / 群推送（V1.0）

### 7.6 数据存储

#### 7.6.1 存储分层

| 层 | 介质 | 路径 |
|----|------|------|
| 元数据/配置 | SQLite | `~/.alphascreener/db/metadata.db` |
| 行情数据 | Parquet（按日分区） | `~/.alphascreener/data/ohlcv/dt=YYYY-MM-DD/` |
| 因子值 | Parquet（按日分区） | `~/.alphascreener/data/factors/dt=YYYY-MM-DD/` |
| 信号 | Parquet（按日分区） | `~/.alphascreener/data/signals/dt=YYYY-MM-DD/` |
| 回测/Paper Trading | Parquet（按月分区） | `~/.alphascreener/data/backtest/dt=YYYY-MM/` |
| LLM 日志 | JSONL 追加 | `~/.alphascreener/logs/llm_{YYYY-MM-DD}.jsonl` |

#### 7.6.2 SQLite 核心表

| 表名 | 用途 |
|------|------|
| `factor_versions` | 因子版本配置（与 6.5 因子 JSON 对应） |
| `paper_trades` | Paper Trading / 实盘虚拟交易记录 |
| `alerts` | 告警事件 |
| `llm_cost_daily` | LLM 成本日累计 |
| `pid_lock` | 进程互斥锁（全局串行执行） |
| `monitoring_samples` | 资源监控采样（RSS/CPU/FD） |
| `alpha_acceptance_daily` | Alpha 验收口径每日记录 |
| `data_source_diff` | 备用源交叉校验差异 |
| `factor_health_daily` | CUSUM 快速监控时序 |

##### 7.6.2.1 DDL 定义

```sql
-- 因子版本配置（与 6.5 因子 JSON 一一对应）
CREATE TABLE factor_versions (
version TEXT PRIMARY KEY, -- 如 "1.0.0"
released_at TIMESTAMP NOT NULL,
config_json TEXT NOT NULL, -- 完整因子配置 JSON
parent_version TEXT, -- 上一版本号
release_type TEXT CHECK(release_type IN ('MAJOR','MINOR','PATCH'))
);

-- Paper Trading / 实盘虚拟交易记录
CREATE TABLE paper_trades (
id INTEGER PRIMARY KEY AUTOINCREMENT,
signal_date DATE NOT NULL,
ticker TEXT NOT NULL,
rating TEXT NOT NULL, -- Strong Buy/Buy/Hold/Avoid
breakout_probability REAL NOT NULL,
entry_price REAL, -- T+1 开盘买入价
exit_price REAL, -- T+7 收盘价或止损价
exit_reason TEXT, -- 'time' / 'stop_loss' / 'halt'
pnl_pct REAL,
factor_version TEXT NOT NULL,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (factor_version) REFERENCES factor_versions(version)
);
CREATE INDEX idx_paper_trades_signal_date ON paper_trades(signal_date);

-- 告警事件
CREATE TABLE alerts (
id INTEGER PRIMARY KEY AUTOINCREMENT,
triggered_at TIMESTAMP NOT NULL,
severity TEXT CHECK(severity IN ('warning','critical')),
rule_name TEXT NOT NULL, -- 见 10.3
metric_value REAL,
notes TEXT,
resolved_at TIMESTAMP
);

-- LLM 成本日累计
CREATE TABLE llm_cost_daily (
cost_date DATE PRIMARY KEY,
total_usd REAL NOT NULL,
call_count INTEGER NOT NULL,
by_module_json TEXT -- {"refining": 0.05, "evolution": 0.01, ...}
);

-- 进程互斥锁（支撑 7.7.2 单机串行执行模型与全局任务排队）
CREATE TABLE pid_lock (
lock_name TEXT PRIMARY KEY, -- 通常为 'global'（全局互斥）；也支持任务名做细粒度锁
pid INTEGER NOT NULL, -- 持锁进程 PID（psutil 校验存活）
task_id TEXT NOT NULL, -- 任务标识（如 'daily_scan' / 'monthly_full_backtest'）
acquired_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
expires_at TIMESTAMP NOT NULL, -- 锁超时时间（acquired_at + 任务超时上限 + 10 分钟缓冲）
meta_json TEXT -- {"started_by": "apscheduler", "trigger_time": "..."}
);
CREATE INDEX idx_pid_lock_expires ON pid_lock(expires_at);

-- 资源监控采样（任务期间 RSS/CPU/FD 采样）
CREATE TABLE monitoring_samples (
id INTEGER PRIMARY KEY AUTOINCREMENT,
task_id TEXT NOT NULL, -- 哪个任务（与 pid_lock.task_id 对齐）
sampled_at TIMESTAMP NOT NULL,
rss_mb REAL NOT NULL, -- 进程驻留内存（MB），psutil 采集
cpu_percent REAL NOT NULL, -- CPU 占用百分比（4 核归一化到 0-400%）
open_fd_count INTEGER, -- 打开文件描述符数（监控泄漏）
thread_count INTEGER, -- 进程线程数
notes TEXT -- 异常说明（如 "exceed_budget"）
);
CREATE INDEX idx_monitoring_task_time ON monitoring_samples(task_id, sampled_at);
-- 数据保留：仅保留最近 30 天（每日 health_check 任务清理超期记录）

-- Alpha 验收口径每日记录（5.7 alpha 验收统一口径）
CREATE TABLE alpha_acceptance_daily (
metric_date DATE PRIMARY KEY,
base_rate REAL NOT NULL,
precision_at_20_pure REAL, precision_at_20_llm REAL,
precision_at_10_pure REAL, precision_at_10_llm REAL,
lift_at_20_pure REAL, lift_at_20_llm REAL,
ic_pure REAL, ic_llm REAL,
bootstrap_ci_lower_pure REAL, bootstrap_ci_upper_pure REAL,
bootstrap_ci_lower_llm REAL, bootstrap_ci_upper_llm REAL,
sample_size INTEGER NOT NULL
);

-- 备用源交叉校验差异记录（7.1 数据源交叉校验）
CREATE TABLE data_source_diff (
id INTEGER PRIMARY KEY AUTOINCREMENT,
metric_date DATE NOT NULL,
ticker TEXT NOT NULL,
field TEXT NOT NULL CHECK(field IN ('open','high','low','close','volume')),
yfinance_value REAL NOT NULL,
fallback_value REAL NOT NULL,
fallback_source TEXT NOT NULL CHECK(fallback_source IN ('stooq','alpaca','polygon')),
diff_pct REAL NOT NULL,
alerted BOOLEAN DEFAULT 0
);
CREATE INDEX idx_data_source_diff_date ON data_source_diff(metric_date);

-- CUSUM 快速监控时序（6.4.1 快速层因子健康监控）
CREATE TABLE factor_health_daily (
metric_date DATE NOT NULL,
factor_name TEXT NOT NULL,
daily_ic REAL,
rolling_ic_mean_90d REAL,
cusum_value REAL,
cusum_alert BOOLEAN DEFAULT 0,
consecutive_alerts INTEGER DEFAULT 0,
PRIMARY KEY (metric_date, factor_name)
);
CREATE INDEX idx_factor_health_factor_date ON factor_health_daily(factor_name, metric_date);
```

> **WAL 模式**：以上所有表均启用 WAL 模式（`PRAGMA journal_mode=WAL`）。数据保留策略：`alpha_acceptance_daily` 永久保留（小表）；`data_source_diff` / `factor_health_daily` 保留 365 天后归档至冷备份。

#### 7.6.3 数据保留策略

| 数据类型 | 保留期 | 总占用 |
|----------|--------|--------|
| 行情 OHLCV | 2 年 | ≤ 3 GB |
| 因子值 | 1 年 | ≤ 1 GB |
| 信号 | 2 年 | ≤ 500 MB |
| 回测/Paper Trading | 2 年 | ≤ 1 GB |
| LLM 日志 | 90 天 | ≤ 500 MB |
| SQLite metadata | 永久（含 WAL） | ≤ 200 MB |
| 应用日志 | 30 天滚动 | ≤ 500 MB |
| **总计** | - | **≤ 9.2 GB** |

超期数据归档为 zstd Parquet 移至 `~/.alphascreener/archive/`。

### 7.7 调度系统

**调度器**：APScheduler `BlockingScheduler`，单机模式，UTC 时区，`SQLAlchemyJobStore` 持久化。

#### 7.7.1 任务清单（严格分时调度，单机串行执行）

| 任务 ID | Cron (UTC) | 描述 | 耗时 | 内存预算 |
|---------|------------|------|------|----------|
| `monthly_cost_reset` | `0 0 1 * *` | 月度成本计数器重置 | < 1 分钟 | < 50 MB |
| `monthly_full_backtest` | `5 0 1 * *` | 全量历史回测（2 年窗口） | ≤ 4 小时 | ≤ 2 GB |
| `monthly_isoforest_retrain` | `0 5 1 * *` | IsolationForest 重训练（V1.5） | ≤ 10 分钟 | ≤ 1.5 GB |
| `biweekly_evolution` | `30 5 1,15 * *` | 进化 Agent 复盘 | ≤ 15 分钟 | ≤ 1.5 GB |
| `monthly_universe_refresh` | `0 8 1 * *` | 指数白名单更新 | ≤ 5 分钟 | < 200 MB |
| `daily_backtest_incremental` | `0 11 * * 2-6` | 日频增量回测 | ≤ 30 分钟 | ≤ 1 GB |
| `daily_health_check` | `0 12 * * *` | 数据源连通性 + 缓存清理 | ≤ 5 分钟 | < 200 MB |
| `daily_scan` | `0 23 * * 1-5` | 美股盘后日频扫描全流程 | ≤ 30 分钟 | ≤ 4 GB |

#### 7.7.2 单机串行执行约束

- **进程级互斥**：`pid_lock` 表 `lock_name='global'` + APScheduler `max_instances=1` 双重保险
- **任务排队**：上一任务未结束时，新任务等待 ≤ 2 小时；超时跳过本次执行并告警
- **死锁恢复**：进程启动时校验持锁 PID 是否存活（psutil），不存活则强制释放
- **资源预留**：任意时刻最多 1 个任务运行，峰值内存不叠加

---

## 8. 外部框架集成（TradingAgents）

### 8.1 集成原则

1. 不侵入上游源码
2. 通过公共 API 调用（`import tradingagents.xxx`）
3. 适配器层隔离 API 变化
4. git+commit 锁定版本

### 8.2 适配器层

| 适配器 | 封装的 TradingAgents 组件 |
|--------|-----------------------------|
| `llm_adapter.py` | `create_llm_client` |
| `analyst_adapter.py` | `create_market_analyst` 等 |
| `debate_adapter.py` | Bull/Bear/Research Manager |
| `dataflow_adapter.py` | `YFinanceDataFlow` 等 |
| `graph_adapter.py` | `TradingAgentsGraph` |

### 8.3 包结构

```
alpha-screener/
├── alphascreener/
│ ├── adapters/ # 适配器层
│ ├── core/ # 扫描引擎、因子库、评分
│ ├── backtest/ # 回测引擎
│ ├── evolution/ # 进化 Agent
│ └── cli.py
├── pyproject.toml # tradingagents @ git+...@<commit_sha>
├── tests/
└── README.md
```

### 8.4 依赖声明（pyproject.toml）

```toml
[project]
name = "alpha-screener"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
# 上游 AI Agent 框架（git+commit 锁定）
"tradingagents @ git+https://github.com/TauricResearch/TradingAgents.git@<commit_sha>",

# 核心数值与数据处理
"numpy>=1.26,<2.0",
"polars>=0.20", # 因子计算流式引擎（4c8g 内存优化）
"pandas>=2.1", # 兼容/输出（部分上游库依赖）
"pyarrow>=14", # Parquet 列存储

# 机器学习 / 聚类（V1.5 起使用）
"scikit-learn>=1.4", # IsolationForest 异常检测
"hdbscan>=0.8.33", # 聚类去重（V1.5）
"faiss-cpu>=1.7", # Breakout Analyst 案例库检索

# 回测
"backtrader>=1.9.78", # 4c8g 内存友好（替代 vectorbt）

# 数据源
"yfinance>=0.2.40", # 主力数据源
"requests>=2.31", # FMP / 飞书 / Stooq HTTP 调用
"httpx>=0.26", # 异步 HTTP（LLM/数据源并发）

# 调度与存储
"apscheduler>=3.10", # 任务调度
"sqlalchemy>=2.0", # SQLite ORM
"alembic>=1.13", # 数据库迁移

# 系统支持
"tenacity>=8.2", # 重试与退避
"psutil>=5.9", # 进程资源监控
"pydantic>=2.5", # BreakoutAssessment schema 校验
"python-dotenv>=1.0", # .env 加载
"click>=8.1", # CLI 框架
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-asyncio>=0.23", "ruff>=0.2", "mypy>=1.8"]

[project.scripts]
alphascreener = "alphascreener.cli:main"
```

#### 8.4.1 .env 模板

```bash
# ====== LLM API ======
OPENAI_API_KEY=sk-xxxx
LLM_MODEL=gpt-4o-mini
LLM_RPS=5 # 客户端限速（4.6.4）

# ====== 数据源 ======
FMP_API_KEY=your_fmp_api_key # FMP Free tier
# yfinance 无需 API key
STOOQ_BASE_URL=https://stooq.com/q/d/l/

# ====== 飞书推送 ======
FEISHU_APP_ID=cli_xxxx
FEISHU_APP_SECRET=xxxx
FEISHU_TARGET_OPENID=ou_xxxx
FEISHU_PUSH_ENABLED=true

# ====== 系统行为开关 ======
EVOLUTION_WEIGHT_ADJUST_ENABLED=false # MVP 冻结权重；工程转正后置 true
LLM_ABLATION_ENABLED=true # 强制双轨记录
COST_BUDGET_MONTHLY_USD=100

# ====== 数据存储路径 ======
ALPHASCREENER_HOME=~/.alphascreener
```

#### 8.4.2 config.py 默认值

```python
# alphascreener/config.py
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
# 数据源
primary_data_source: str = "yfinance"
fallback_ohlcv_source: str = "stooq" # stooq / alpaca / polygon
fmp_tier: str = "free" # free / starter
fmp_daily_budget: int = 250

# LLM
llm_model: str = "gpt-4o-mini"
llm_rps: int = 5
llm_batch_size: int = 3 # 4c8g 资源约束（4.6.1）
llm_max_concurrent_stage1: int = 6 # 3 标的 × Bull/Bear

# 成本熔断阈值（4.6.3）
cost_l1_warning_daily_usd: float = 0.80
cost_l2_degrade_daily_usd: float = 1.00
cost_l3_savings_monthly_usd: float = 80.0
cost_l4_circuit_monthly_usd: float = 95.0

# 粗筛阈值（3.2.1）
mom_5d_min: float = 0.0
atr_ratio_max: float = 0.8
rsi_range: tuple = (25, 75)
mfi_min_or_vol_anomaly: float = 40.0

# 行业去重（3.3）
sector_cap: int = 3
industry_cap: int = 2

# 路径
home: Path = Path("~/.alphascreener").expanduser()

class Config:
env_file = ".env"

settings = Settings()
```

### 8.5 CLI

```bash
alphascreener screen --market US --top 20 # 全市场扫描
alphascreener backtest --start 2023-01-01 # 回测
alphascreener evolve --review-last 30 # 进化复盘
alphascreener evolve --propose-factor "<formula>" # 人工提案因子
alphascreener walk-forward --version <new_version> # 因子升级前验证
```

---

## 9. 可观测性与运维

### 9.1 日志规范

**日志级别**：ERROR / WARN / INFO / DEBUG

**JSON 结构化日志格式**：
```json
{
"timestamp": "ISO8601",
"level": "INFO",
"module": "screening|refining|backtesting|evolution",
"event": "scan_completed",
"data": { ... },
"cost_usd": 0.003
}
```

### 9.2 核心指标

#### 业务指标

| 指标 | 类型 | 频率 |
|------|------|------|
| daily_scan_count | Gauge | 日 |
| screening_pass_count | Gauge | 日 |
| refining_output_count | Gauge | 日 |
| win_rate_7d | Gauge | 周 |
| sharpe_ratio_30d | Gauge | 月 |

#### 系统指标

| 指标 | 类型 | 频率 |
|------|------|------|
| llm_cost_daily / monthly | Counter | 日 / 月 |
| llm_latency_p95 | Histogram | 每次调用 |
| api_error_rate | Rate | 小时 |
| scan_duration_sec | Histogram | 日 |
| factor_nan_count | Counter | 日 |

#### 资源指标（4c8g 适配）

| 指标 | 频率 |
|------|------|
| process_rss_mb / peak_mb | 每分钟 / 每任务结束 |
| cpu_percent | 每分钟 |
| disk_usage_gb | 每日 |
| task_queue_wait_sec | 每任务排队 |
| oom_kill_count | 每次发生 |

### 9.3 告警规则

| 告警 | 条件 | 级别 | 动作 |
|------|------|------|------|
| 预算预警 | 月成本 > $80 | ⚠️ | 进入节约模式 |
| 预算熔断 | 月成本 > $95 | 🔴 | 暂停 LLM |
| 胜率下降 | win_rate_7d < 45% 连续 3 天 | ⚠️ | 触发进化诊断 |
| 胜率严重下降 | win_rate_7d < 35% | 🔴 | 暂停选股 |
| API 故障 | error_rate > 30% 持续 15min | 🔴 | 启动降级 |
| 内存超阈 | process_rss_mb > 7000 | 🔴 | 强制 kill 任务 |
| 内存预警 | process_rss_mb > 5500 持续 5min | ⚠️ | 评估降级 |
| CPU 持续满载 | cpu_percent > 380 持续 10min | ⚠️ | 检查任务卡死 |
| 磁盘紧张 | disk_usage_gb > 80 | 🔴 | 触发归档 |
| 任务排队超时 | task_queue_wait_sec > 7200 | ⚠️ | 跳过本次 |
| OOM 发生 | oom_kill_count > 0 | 🔴 | 停机自检 |
| 数据源差异 | data_source_diff 当日 ≥ 5 次 | ⚠️ | 飞书告警 |
| yfinance 连续失效 | 失败率 > 30% 持续 3 天 | 🔴 | 切换备用源全量 |
| Lift@20 低于 base | 滚动 30 天 < 1.0 持续 7 天 | ⚠️ | 触发慢速进化 |
| Alpha critical | 滚动 30 天 Lift@20 < 0.9 持续 14 天 | 🔴 | 暂停 LLM 修正层 |

### 9.4 MVP 阶段实现范围

- ✅ JSON 结构化日志、每日成本累计、预算熔断
- ❌ 实时 Dashboard（V1.0）、告警推送（V1.0，MVP 通过日志人工巡检）

---

## 10. 资源约束与性能预算（4c8g 单机）

### 10.1 部署目标

| 维度 | 规格 |
|------|------|
| CPU | 4 vCPU (x86_64) |
| 内存 | 8 GB RAM（硬上限） |
| 磁盘 | 100 GB SSD |
| 网络 | 100 Mbps |
| OS | Ubuntu 22.04 LTS（推荐） |

### 10.2 内存预算

```
8 GB 总内存
├─ OS + 基础进程: 800 MB
├─ Python runtime + 依赖库: 600 MB
├─ APScheduler + SQLite WAL: 200 MB
├─ ─────────────────────────────────
├─ 任务工作内存 (单任务): ≤ 4 GB
│ └─ daily_scan: ≤ 4 GB（峰值任务）
├─ ─────────────────────────────────
└─ 系统缓冲 + 应急: ≥ 2.4 GB
```

### 10.3 升级触发条件

| 触发条件 | 推荐方案 |
|----------|----------|
| 连续 7 天 process_rss_peak_mb > 7000 | 升级 8c16g，恢复 vectorbt |
| 月度 oom_kill_count > 0 | 立即升级 |
| daily_scan 平均耗时 > 60 分钟 | 评估 8c16g |
| disk_usage_gb > 80 持续 14 天 | 200 GB SSD 或冷数据迁移对象存储 |
| 实时盘中扫描需求（V2.0） | 16c32g（多进程 + Redis） |
| 港股扩展总标的数 > 3,000 | 8c16g 或拆分双实例 |

---

## 11. 里程碑规划

| 阶段 | 周期 | 关键交付 | 成功标准 |
|------|------|----------|----------|
| **MVP（工程转正）** | 4 周 | 美股扫描引擎、13 因子粗筛、轻量行业去重、LLM 风险审计 + Ablation 双轨、Alpha 验收口径、备用 OHLCV 源、Paper Trading 双层转正 | 60 天系统稳定（无 L3/L4 熔断、调度成功率 ≥ 95%、NaN < 5%）；日扫描 ≤ 30 分钟，峰值内存 ≤ 4GB；**MVP 完成 ≠ alpha 已被证明** |
| **V1.0 第二阶段** | +4 周 | 慢速层权重进化、完整回测系统、CLI、LLM Ablation 月报、Walk-forward 月度验证 | Lift@20 滚动 60 天均值 ≥ 1.10 + bootstrap CI 下界 ≥ 1.0；夜间增量回测 ≤ 30 分钟，月度全量 ≤ 4 小时 |
| **V1.5** | +4 周 | 港股支持、LLM 自主因子发现、IsolationForest、HDBSCAN 聚类去重 | 进化后 Lift@20 比初始提升 ≥ 0.10；4c8g 单机仍可承载，无 OOM |
| **策略转正** | +4-8 周 | ≥ 2 年 walk-forward 验证、≥ 6 个月 live shadow | Walk-forward 12 折 Lift@20 均值 ≥ 1.10、CI 下界 ≥ 1.05；live shadow 60 天达标；LLM ΔLift@20 ≥ 0.05 |
| **V2.0** | +8 周 | 实时盘中扫描、多策略并行、API 服务化、告警推送 | 盘中信号延迟 < 5 分钟；Lift@20 ≥ 1.20；需升级 8c16g |

### 11.1 港股启动条件（V1.5）

| 前置条件 | 验证方式 |
|----------|----------|
| 美股 MVP 胜率 > 55%（连续 30 天） | Paper Trading / 回测 |
| 美股 V1.0 夏普 > 1.0（滚动 30 天） | 回测报告 |
| 港股数据源可用性验证 | yfinance `.HK` + FMP 覆盖率 |
| 港股因子适配评估 | 去除美股特有因子（SHORT_INTEREST 等） |
| 港股测试数据集 ≥ 60 个交易日 | 历史回测 |
| LLM 预算余量 ≥ $30/月 | 成本报表 |

**港股特有调整**：交易时段 09:30-16:00 HKT；最小单位"手"；新增南下资金流向 / AH 股溢价考虑。

---

## 12. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM 调用成本高 | 月预算超支 | 分批串行 + 摘要压缩 + 4 级成本熔断 |
| 数据延迟/缺失 | FMP 配额耗尽 | 自动降级到 yfinance，保留 89% 因子权重 |
| 过拟合 | 因子优化拟合历史噪声 | 滚动回测 + 样本外验证 + 正则化 |
| 市场结构变化 | 策略突然失效 | 进化回路快速响应 + 人工熔断 |
| LLM 幻觉 | 辩论生成虚假论据 | 结构化输出 + 数据源引用 + 人工抽检 |
| 4c8g 资源紧张 | 内存接近 8GB 上限 OOM | 严格分时调度 + 启动前内存检查 + polars 流式 + 标的池预过滤 |
| backtrader 回测耗时长 | 全量回测可能 > 4 小时 | 增量回测优先 + 全量仅月度 + 接受夜间 4 小时窗口 |
| 指数白名单滞后 | 新晋热门股遗漏 | monthly_universe_refresh 月度更新 + 重大事件手动触发 |
| 因子 IC 样本外衰减 | 已发表因子 IC 平均衰减 58% | Alpha 验收强制 Lift@20 + bootstrap CI；walk-forward 验证；MVP 不调权 |
| 13 因子有效维度仅 4-6 | 组合 IC ≈ 0.04 | 因子相关性周计算（D3 衰退）；新因子相关性 < 0.7 |
| LLM 无信息优势 | 强制概率预测失去区分度 | 重定位为风险审计层；Ablation 强制 ΔLift@20 ≥ 0.05 |
| 过拟合风险 | 25 超参 + 60 天 Paper Trading 严重过拟合 | MVP 冻结全部权重；策略转正必须 walk-forward + bootstrap；进化拆双层 |
| yfinance 单点依赖 | 3 年 4 次 breaking change + 2 次封禁 | 引入备用 OHLCV（Stooq/Alpaca）+ 交叉校验 + 全量降级 |

---

## 13. 跨模块契约总览

### 13.1 数据契约

| 契约名 | 类型 | 生产者 | 消费者 |
|--------|------|--------|--------|
| `factors.parquet` | Parquet | 因子计算引擎 | 粗筛、回测、进化 Agent |
| `signals_refined.parquet` | Parquet | LLM 精筛 / PM Output | 飞书推送、回测对账、进化 Agent |
| `paper_trades` | SQLite | 回测引擎 | 进化 Agent、Dashboard |
| `factor_versions` | SQLite | 进化 Agent | 因子计算引擎、回测 |
| `alerts` | SQLite | 监控系统 | 飞书推送、运维 |
| `llm_cost_daily` | SQLite | LLM 适配器 | 成本熔断、月报 |
| `pid_lock` | SQLite | APScheduler 装饰器 | 任务排队、互斥锁 |
| `monitoring_samples` | SQLite | 资源监控 | 告警系统、容量评估 |
| `BreakoutAssessment` | Pydantic | LLM PM Output | 评级映射、回测、飞书卡片 |
| `case_library/cases.parquet` | Parquet | 每日新增正样本 | Breakout Analyst 检索 |
| `alpha_acceptance_daily` | SQLite | Alpha 验收口径计算 | 飞书卡片、月度 walk-forward |
| `data_source_diff` | SQLite | 备用 OHLCV 校验 | 告警系统 |
| `factor_health_daily` | SQLite | CUSUM 快速监控 | 慢速层进化、告警 |

### 13.2 接口契约

| 契约名 | 类型 | 调用方 |
|--------|------|--------|
| Bull / Bear / PM Researcher Prompt | LLM Prompt | 精筛流水线 |
| Strategy Review Prompt | LLM Prompt | 进化 Agent（双周触发） |
| 因子公式 DSL | Python AST 子集 | 进化 Agent / 人工 CLI |
| `alphascreener` CLI | Bash | 用户 / cron |
| 飞书 Open API | HTTP/JSON | 推送适配器 |
| TradingAgents 适配器 | Python API | adapters/ 五类 |
| LLM 适配器 | Python API | 业务代码（tenacity + 5 RPS） |
| yfinance 适配器 | Python API | dataflow_adapter（50 股/批 + 5 RPS） |
| FMP 适配器 | HTTP/JSON | dataflow_adapter（Free tier ≤ 250 req/日） |

---

## 14. 附录

### 14.1 关键术语

| 术语 | 英文 | 定义 |
|------|------|------|
| Alpha | Alpha | 策略超越基准（如 SPY）的超额收益部分 |
| DSL | Domain-Specific Language | 领域特定语言，本系统中指因子公式的安全表达规范 |
| IC | Information Coefficient | 信息系数，衡量因子预测能力。IC > 0.03 通常认为有预测价值 |
| Paper Trading | Paper Trading | 模拟交易，使用真实市场数据但不实际执行订单 |
| Regime | Market Regime | 市场运行模式（牛/熊/震荡），不同 regime 下因子表现可能截然不同 |
| 爆发增长 | Breakout Growth | T+1 开盘至 T+7 收盘期间，股价涨幅 ≥ 10% |
| 成本熔断 | Cost Circuit Breaker | LLM 成本达预设阈值时自动降级或暂停 |
| 多空辩论 | Bull-Bear Debate | Bull / Bear 研究员的对抗性分析 |
| 滑点 | Slippage | 预期成交价与实际成交价的差异 |
| 回撤 | Drawdown | 从历史最高点到最低点的最大跌幅 |
| 适配器层 | Adapter Layer | 封装第三方库调用，隔离 API 变化 |
| 胜率 | Win Rate | 盈利交易占总交易数的比例 |
| 夏普比率 | Sharpe Ratio | 风险调整后收益，> 1.0 为良好 |
| 因子衰退 | Factor Decay | 因子预测能力随时间下降的现象 |
| Lift@K | Lift@K | Precision@K / base_rate，选股精度相对市场基础概率的提升倍数 |
| Walk-forward | Walk-forward | 滚动训练-验证的样本外检验方法 |
| Block Bootstrap | Block Bootstrap | 时序数据的重采样统计推断方法 |
| CUSUM | Cumulative Sum | 累积和控制图，监控时序信号偏离 |

### 14.2 参考资源

- [TradingAgents](https://github.com/TauricResearch/TradingAgents)
- backtrader 回测框架
- 学术参考：George & Hwang (2024), Chen et al. (2025), Lopez de Prado (Walk-forward), McLean & Pontiff (2016, 因子衰减)

### 14.3 Canonical 端到端示例（AAPL @ 2025-12-15）

| 参数 | 取值 |
|------|------|
| 标的 | AAPL（SP500 成分股） |
| 选股日 T | 2025-12-15（周一） |
| T+1 开盘 | 2025-12-16（周二） |
| T+7 收盘 | 2025-12-24（周三，跨圣诞 12-25 休市） |
| 因子配置版本 | 1.0.0（MVP） |
| 系统时区 | UTC（daily_scan 触发于 23:00 UTC） |

**端到端流程**（按 7.4 流水线）：
1. 调度触发 → 抢占 pid_lock(global)
2. 数据同步 → yfinance + 备用源校验，输出 `ohlcv/dt=2025-12-15/AAPL.parquet`
3. 因子计算 → polars 分块流式，输出 `factors/dt=2025-12-15/factors.parquet`（13 active 行）
4. 粗筛 Phase 1 + Phase 2 → AAPL 通过硬过滤，Breakout_Score 加权排序进 Top 30
5. 行业去重 → Sector="Technology", Industry="Consumer Electronics"，假设 AAPL 通过为 Top 20 #5
6. LLM 精筛 → Batch 2，Bull/Bear 并行 + PM 风险审计，输出 BreakoutAssessment
7. 飞书推送 + Paper Trading 写入 `paper_trades` 表
8. 次日 11:00 UTC daily_backtest_incremental → backtrader 7 天持仓回测

---

**文档结束**
