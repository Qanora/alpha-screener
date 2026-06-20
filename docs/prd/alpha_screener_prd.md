# AlphaScreener PRD

| 项目 | 内容 |
|------|------|
| 产品定位 | 美股短期爆发标的筛选工具（CLI） |
| 文档版本 | v2.0 |
| 预测窗口 | T+1 开盘 → T+7 收盘 |
| 目标变量 | `y = 1 if (Close_T+7 / Open_T+1 - 1) ≥ 10% else 0` |
| 运行环境 | 单机，Python 3.11+，~500MB 内存 |
| 交互方式 | 纯 CLI（`alphascreener`），所有结果输出到终端 |

---

## 1. 产品概述

AlphaScreener 是一个**纯 CLI 美股短期爆发筛选工具**。

输入：美股 OHLCV 历史数据
输出：Top N 候选标的 + 回测验证结果

```
alphascreener                  # 筛选+回测（数据过期时自动更新）
alphascreener --top 20         # 自定义数量
alphascreener optimize         # 自进化：多轮回测寻找最优因子权重
alphascreener sync             # 手动强制更新 OHLCV（通常不需要）
alphascreener backtest AAPL    # 独立回测：查任意标的的历史表现
```

### optimize — 因子权重自进化

通过滚动窗口回测迭代，自动寻找最优因子权重组合。严格遵循 walk-forward 验证规范，避免过拟合。

**算法**：

```
输入: 初始权重 W₀ (等权 1/13)，历史 OHLCV 数据

对每个滚动窗口 (train 2年 / test 6个月):
  train 期: 用当前权重 W 跑筛选+回测 → 计算每个因子的边际贡献
  test 期:  用调整后权重跑筛选+回测 → 记录 Precision@K, Lift@K, Sharpe
  → 调权: 贡献大的因子权重↑，贡献小的↓（学习率递减退火）
  → 窗口前移 6 个月，重复

收敛条件: |W_new - W_old| < 1% 或 达到最大窗口数
输出: 最优权重 + 滚动窗口绩效报告 + Bootstrap 95% CI
```

**验证指标（每个 test 窗口）**：

| 指标 | 含义 | 来源 |
|------|------|------|
| Precision@20 | Top 20 命中率 | alpha_acceptance |
| Lift@20 | Precision / Base Rate (>1 = 有预测力) | alpha_acceptance |
| Base Rate | 全市场 T+7≥10% 的比例 | alpha_acceptance |
| Sharpe Ratio | 风险调整后收益 | backtrader |
| Max Drawdown | 最大回撤 | backtrader |
| Bootstrap CI | 95% 置信区间 (1000 resamples) | alpha_acceptance |

**输出示例**：

```
Factor Weight Evolution — 10 windows, converged in 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Factor       Initial   Final     Δ
  MOM_5D       0.077    0.142   +0.065  ↑
  MOM_21D      0.077    0.089   +0.012
  RSI_14       0.077    0.031   -0.046  ↓
  ...

  Window  Train        Test      P@20   Lift   Sharpe  MaxDD
  1       2022-2024    2024H1    0.35   2.1    1.42    -12%
  2       2022H2-24H2  2024H2    0.38   2.3    1.55    -10%
  ...

  Final: P@20=0.36±0.04  Lift=2.2±0.2  Sharpe=1.48±0.15
  (bootstrap 95% CI, 1000 resamples)
```

```bash
alphascreener optimize                    # 默认: 2y train / 6m test 滚动
alphascreener optimize --train 3y         # 3 年训练窗口
alphascreener optimize --step 3m          # 3 个月步长
```

### backtest — 独立回测

默认流程只对筛选出的 Top 5 做回测。但你可能想独立查询某个标的：
- 在别处看到一个代码（比如 NVDA），想看看它的历史回测数据
- 指定自定义日期范围（比如只看 2024 年表现）

```bash
alphascreener backtest NVDA --start 2024-01-01
```

---

## 2. 筛选管道

```
OHLCV 数据 → 13 因子计算 → Phase 1 硬过滤 → Phase 2 加权评分+去重 → 输出 Top N
                                    ↓
                              自动回测验证
```

### 2.1 因子清单（13 因子）

| # | 因子 | 含义 | 最少数据 |
|---|------|------|----------|
| 1 | MOM_5D | 5 日动量 | 5 行 |
| 2 | MOM_21D | 21 日动量 | 21 行 |
| 3 | RSI_14 | 相对强弱指标 | 14 行 |
| 4 | MFI_14 | 资金流量指标 | 14 行 |
| 5 | BB_SQUEEZE | 布林带挤压 | 60 行 |
| 6 | PTH | 价格通道突破 | 63 行 |
| 7 | ATR_RATIO | 波动率比率 | 20 行 |
| 8 | CMF_21 | Chaikin 资金流 | 21 行 |
| 9 | VOL_ANOMALY | 成交量异常 | 50 行 |
| 10 | SMA_50 | 50 日均线偏离 | 50 行 |
| 11 | SMA_200 | 200 日均线偏离 | 200 行 |
| 12 | REV_ACCEL | 营收增速加速度 | 季报 |
| 13 | PEAD_FLAG | 财报后漂移标记 | 财报 |

### 2.2 Phase 1 — 硬过滤

| 条件 | 默认阈值 | 说明 |
|------|----------|------|
| MOM_5D | > 0.0 | 短期正动量 |
| RSI | ∈ [25, 75] | 非超买/超卖 |
| MFI_14 | > 40 | 资金流入 |
| ATR_RATIO | < 0.80 | 波动率可控 |

若通过率低于 2%，自动放宽阈值 3 步（每次放宽 10%）。

### 2.3 Phase 2 — 加权评分 + 去重

每个因子有预设权重，加权得到 `breakout_score`。按行业去重（同行业最多 3 只，同细分最多 2 只），输出 Top N。

---

## 3. 回测

默认回测：对 Top 5 候选标的，过去 2 年日频回测。

| 指标 | 含义 |
|------|------|
| Total Return | 总收益率 |
| Annualized Return | 年化收益 |
| Sharpe Ratio | 夏普比率 |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 |
| SPY Benchmark | 同期标普 500 对比 |

```bash
alphascreener backtest AAPL --start 2023-01-01 --end 2024-12-31
```

---

## 4. 数据

### 4.1 数据源

yfinance（主力），获取 SP500 + Russell 1000 成分股日线 OHLCV。

### 4.2 存储

Parquet 文件，Hive 分区：`~/.alphascreener/data/ohlcv/dt=YYYY-MM-DD/`

### 4.3 自动更新

`alphascreener` 运行时自动检查数据新鲜度。超过 1 天未更新 → 自动拉取最新日线（增量 ~2,000 行）。首次运行需要 ~2,000 只 × 2 年 ≈ 100 万行。

---

## 5. 工程约束

### 5.1 CLI 设计原则

- `alphascreener`（无参数）= 完整筛选+回测流程
- 所有输出通过 rich 渲染到 stdout
- 日志写入文件，不污染终端
- 无内置调度器（用户用 cron 自行编排）
- 无外部推送（飞书/邮件/API 等）

### 5.2 代码规模目标

| 指标 | 目标 |
|------|------|
| 总代码量 | < 5,000 行 |
| 文件数 | < 20 个 |
| 外部依赖 | yfinance, polars, rich, click, backtrader |

### 5.3 不做什么

- ❌ LLM 多空辩论（删除 TradingAgents 集成）
- ❌ 纸交易系统
- ❌ 飞书/API 推送
- ❌ CUSUM 监控告警
- ❌ 调度器常驻进程
- ❌ 港股/其他市场

---

## 6. 目录结构（计划）

```
alphascreener/
├── cli.py              # CLI 入口
├── display.py          # rich 终端渲染
├── config.py           # 配置（因子权重、阈值）
├── backtrader.py       # 回测引擎
├── acceptance.py       # 验收指标 (Precision@K, Lift, Bootstrap CI)
├── optimize.py         # 权重自进化（滚动窗口 walk-forward）
├── data/
│   ├── io.py           # Parquet 读写
│   └── sync.py         # yfinance 数据同步
├── factors/
│   ├── engine.py       # 因子计算引擎
│   └── formulas.py     # 13 因子公式
├── screening/
│   ├── phase1.py       # 硬过滤
│   ├── phase2.py       # 加权评分 + 去重
│   └── threshold.py    # 阈值管理
└── logging/
    └── logger.py       # 文件日志
```

---

## 7. 使用示例

```bash
# 日常使用 — 一行搞定（数据自动更新）
alphascreener

# 权重自进化（周末跑，寻找最优配置）
alphascreener optimize --rounds 100

# 独立查某个标的
alphascreener backtest NVDA --start 2024-01-01

# 每天定时（cron — 盘后自动跑）
0 23 * * 1-5 cd /path/to/project && alphascreener --top 20
```
