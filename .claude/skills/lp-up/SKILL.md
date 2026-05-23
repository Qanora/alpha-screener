---
name: lp-up
description: 第一层——执行引擎+分析运行时数据，持续发现架构/实现/算法缺陷，通过 subagent 启动 lp-ms 驱动迭代改进
---

# LP-UP（第一层 · 持续改进引擎）

主动执行引擎、观察运行过程、发现缺陷、提出 milestone，通过 **subagent** 启动 lp-ms 驱动整个飞轮迭代。

```
┌─────────────────────────────────────────────────────┐
│                    lp-up (第一层)                     │
│  执行 → 观察 → 分析 → 报告 → subagent:lp-ms（自动）   │
└────────┬────────────────────────────────────────────┘
         │ milestone (via subagent)
         ▼
    lp-ms (需求拆解 → issue)
         │ issue
         ▼
    lp-mr (MR 生命周期)
         │ branch
         ▼
    lp-dev (写代码)
         │ merge
         ▼
    lp-up (再执行 → 验证修复 → 发现新问题 → ...)
```

## 调用方式

```text
/lp-up                           # 纯分析模式（不执行，只分析已有数据）
/lp-up --run quick               # 快速轮：screen + evolve review-last
/lp-up --run full                # 完整轮：screen → backtest → cusum → evolve → walk-forward
/lp-up --focus <area>            # 聚焦：performance | cost | accuracy | reliability
/lp-up --since <date>            # 只分析指定日期之后的数据
/lp-up --resume                  # 从中断恢复，继续上一轮未完成的改进循环
```

## 核心概念：持续改进循环

lp-up 不是一次性工具，而是一个**闭环迭代引擎**：

```text
Round N:   执行 → 观察 → 发现 F₁, F₂, F₃ → subagent:lp-ms → 飞轮实现
Round N+1: 执行 → 观察 → 验证 F₁ 已修复 ✓, F₂ 部分改善 ~, F₃ 未改善 ✗
                    → 发现新问题 F₄ → subagent:lp-ms → 飞轮实现
Round N+2: ...
```

每一轮都在上一轮的基础上推进，既验证历史修复效果，又发现新的改进空间。

## 流程

### 阶段 A：执行引擎（--run 模式）

若用户指定了 `--run`，先主动运行引擎管道，在运行过程中实时观察。

#### A.1 Quick Round

```bash
# 1. 全市场粗筛
alphascreener screen --top 20

# 2. Alpha 接受度回顾
alphascreener evolve review-last --days 30
```

#### A.2 Full Round

```bash
# 1. 全市场粗筛
alphascreener screen --top 20

# 2. 增量回测（最近 30 天）
alphascreener backtest --start $(date -d '30 days ago' +%Y-%m-%d)

# 3. 因子健康检查
python -c "
from alphascreener.scheduler.tasks import daily_cusum_check
daily_cusum_check()
"

# 4. Alpha 接受度回顾
alphascreener evolve review-last --days 30

# 5. Walk-forward 验证
alphascreener walk-forward --version v0.1.0
```

#### A.3 执行期间实时观察

每个命令执行时同步采集：

| 观察维度     | 采集方式                                                    |
| ------------ | ----------------------------------------------------------- |
| 退出码       | `$?`                                                        |
| 耗时         | `time` 包裹                                                 |
| stdout/stderr | 完整捕获                                                    |
| 资源峰值     | 执行前后各采样一次 `psutil`：RSS、CPU%、open FDs            |
| 错误计数     | stderr 行数 + 日志中 ERROR 级别行数                         |

```bash
# 示例：包裹执行并采集运行时指标
python -c "
import time, psutil, os, sys, subprocess, json

pid = os.getpid()
before = {'rss_mb': psutil.Process(pid).memory_info().rss / 1024**2}

t0 = time.monotonic()
result = subprocess.run(sys.argv[1:], capture_output=True, text=True)
elapsed = time.monotonic() - t0

after = {'rss_mb': psutil.Process(pid).memory_info().rss / 1024**2}

print(json.dumps({
    'exit_code': result.returncode,
    'elapsed_s': round(elapsed, 1),
    'rss_before_mb': round(before['rss_mb'], 1),
    'rss_after_mb': round(after['rss_mb'], 1),
    'rss_delta_mb': round(after['rss_mb'] - before['rss_mb'], 1),
    'stdout_lines': len(result.stdout.splitlines()),
    'stderr_lines': len(result.stderr.splitlines()),
}))
" -- alphascreener screen --top 20
```

### 阶段 B：数据采集

无论 `--run` 还是纯分析模式，都执行数据采集。采集来源分两类：

#### B.1 本轮执行数据（仅 --run 模式）

- 各命令的退出码、耗时、stdout/stderr
- 资源采样（RSS delta、CPU 峰值）
- 执行期间新产生的日志行

#### B.2 历史运行时数据（所有模式）

**结构化日志**（`~/.alphascreener/logs/*.log`）：
```bash
# 按级别和模块统计
cat ~/.alphascreener/logs/*.log | jq -r '[.level, .module] | @tsv' | sort | uniq -c | sort -rn

# 提取 ERROR（最近 30 天）
find ~/.alphascreener/logs/ -name "*.log" -mtime -30 | xargs cat | jq 'select(.level == "ERROR")'

# 提取各阶段耗时
cat ~/.alphascreener/logs/screening.log | jq 'select(.data.elapsed_s != null) | {event, elapsed_s: .data.elapsed_s}'
```

**SQLite 运行时指标**（`~/.alphascreener/data/alphascreener.db`）：

| 表名                        | 分析目标                     |
| --------------------------- | ---------------------------- |
| `monitoring_samples`        | 内存泄漏、FD 泄漏、CPU 异常  |
| `factor_health_daily`       | 因子 IC 衰减、CUSUM 触发频率 |
| `alpha_acceptance_daily`    | P@K/Lift@K 退化趋势          |
| `llm_cost_daily`            | 成本效率、circuit breaker 频率 |
| `paper_trades`              | 模拟交易 P&L、胜率           |
| `alerts`                    | 告警频率、类型分布、解决率   |
| `data_source_diff`          | 数据源偏差趋势               |

**Parquet 批量数据**（`~/.alphascreener/data/`）：
- `ohlcv/dt=*/` — OHLCV 覆盖度和完整性
- `factors/dt=*/` — 因子 NaN 率、极值分布
- `backtest/dt=*/` — 回测结果趋势

### 阶段 C：多维度分析

#### C.1 架构缺陷

**A1. 内存泄漏**
```sql
SELECT date(ts) as day, max(rss_mb) as peak_rss
FROM monitoring_samples
WHERE ts >= date('now', '-30 days')
GROUP BY date(ts)
ORDER BY day
```
判定：7 日 RSS 线性回归斜率 > 50MB/天 且 工作负载（线程数/处理量）持平 → 疑似泄漏。

**A2. FD 泄漏**
```sql
SELECT date(ts) as day, max(open_fds) as peak_fds
FROM monitoring_samples
WHERE ts >= date('now', '-30 days')
GROUP BY date(ts)
ORDER BY day
```
判定：7 日 FD 计数斜率 > 10/天 → 疑似泄漏。

**A3. 管道瓶颈**
从日志提取各阶段 P50/P95/P99 耗时，P95 > 2× 历史中位数 → 瓶颈。

**A4. 数据源退化**
```sql
SELECT date(dt) as day, avg(diff_pct) as avg_diff
FROM data_source_diff
WHERE dt >= date('now', '-30 days')
GROUP BY date(dt)
ORDER BY day
```
判定：连续 3 天 `avg_diff > 0.3` → 数据源显著退化。

**A5. 调度可靠性**
检查 cron 任务是否按预期执行——交易日无 monitoring_samples 记录 → 调度可能未运行。

#### C.2 实现缺陷

**I1. 错误聚类**
```bash
cat ~/.alphascreener/logs/*.log | jq -r 'select(.level == "ERROR") | "\(.module) | \(.event)"' | sort | uniq -c | sort -rn | head -10
```
判定：单一 error > 10 次/天 → 系统性 bug。

**I2. 因子 NaN 率**
```sql
SELECT factor_name, date(dt) as day, nan_rate
FROM factor_health_daily
WHERE dt >= date('now', '-30 days') AND nan_rate > 0.01
ORDER BY nan_rate DESC
```
判定：`nan_rate > 0.01` → 公式边界条件 bug。

**I3. 数据缺口**
检查 OHLCV Parquet 分区覆盖——某交易日记录数 < 中位数 50% → 同步不完整。

**I4. Phase 1 过滤质量**
从 screen 输出提取 `Phase 1 pass` 数量和比例。
判定：`pass_rate < 0.05` → 阈值过严；`> 0.80` → 阈值过松。

**I5. Phase 2 去重质量**
从 screen 输出提取去重前后数量比。
判定：`dedup_ratio > 0.7` → 行业集中度过高或 cap 过严。

**I6. 本轮执行异常（仅 --run 模式）**
- 任一命令退出码 ≠ 0 → 立即标记为 CRITICAL
- stderr 非空 → 提取关键错误信息
- RSS delta > 200MB（单次 screen 执行）→ 内存异常

#### C.3 算法缺陷

**G1. 因子 IC 衰减**
```sql
SELECT factor_name, date(dt) as day, ic_spearman
FROM factor_health_daily
WHERE dt >= date('now', '-30 days')
ORDER BY factor_name, day
```
判定：20 日滚动 IC 斜率 < -0.01/天 → 因子信号衰减。

**G2. Alpha 接受度退化**
```sql
SELECT date(dt) as day, precision_at_k, lift_at_k, ic_spearman
FROM alpha_acceptance_daily
WHERE dt >= date('now', '-30 days')
ORDER BY day
```
判定：20 日滚动 P@K 斜率 < -0.005/天，或连续 5 天低于历史中位数 50% → 严重退化。

**G3. 回测表现退化**
从 Parquet 或 backtest 输出提取 Sharpe、max_drawdown、excess_return。
判定：Sharpe < 1.0、max DD > 20%、excess_return < 0（vs SPY）→ 策略虚弱。

**G4. 成本效率**
```sql
SELECT date(dt) as day, total_cost_usd
FROM llm_cost_daily WHERE dt >= date('now', '-30 days') ORDER BY day
```
结合 paper_trades P&L：
```sql
SELECT date(entry_dt) as day, sum(realized_pnl) as daily_pnl
FROM paper_trades WHERE entry_dt >= date('now', '-30 days')
GROUP BY date(entry_dt)
```
判定：日均 cost > $0.80，或 `total_cost / n_successful_trades > $0.50`，或 cost 增长但 P&L 不增长 → LLM 边际收益递减。

**G5. 过拟合信号**
```sql
SELECT date(dt) as day, pure_lift_at_k, llm_lift_at_k
FROM alpha_acceptance_daily WHERE dt >= date('now', '-30 days') ORDER BY day
```
判定：`pure_lift_at_k / llm_lift_at_k > 2.0`（持续）→ LLM 精筛未产生增量价值。

### 阶段 D：发现分类

每个发现包含：

| 字段       | 说明                                                     |
| ---------- | -------------------------------------------------------- |
| 严重度     | `CRITICAL`（影响正确性/资金风险）、`WARNING`（影响效率）、`INFO`（优化建议） |
| 类别       | `ARCHITECTURE`、`IMPLEMENTATION`、`ALGORITHM`             |
| 证据       | 具体数据点 + 时间戳 + 统计量                              |
| 根因假设   | 基于证据的推断                                           |
| 建议范围   | 涉及模块/文件                                            |

### 阶段 E：报告生成

```text
## LP-UP 分析报告 — Round <N>

**分析时间**: <ISO timestamp>
**执行模式**: quick | full | passive
**数据范围**: <start> → <end>
**上一轮**: Round <N-1> 于 <date>，共 <M> 个发现，<X> 个已修复

---

### 上一轮修复验证（仅 Round ≥ 2）

| 发现 | 状态 | 证据 |
|------|------|------|
| #1 内存泄漏 | ✓ 已修复 | RSS 7 日斜率从 +45MB/天 降至 +3MB/天 |
| #2 MOM_5D IC 衰减 | ~ 部分改善 | IC 斜率从 -0.015 回升至 -0.005 |
| #3 Phase 1 阈值 | ✗ 未改善 | pass_rate 仍为 2.8% |

---

### 本轮发现汇总

| # | 严重度 | 类别 | 简述 | 建议 milestone |
|---|--------|------|------|----------------|
| 1 | CRITICAL | IMPLEMENTATION | screen 命令 exit code=1，stderr: "KeyError: breakout_score" | 修复 screen 命令 KeyError |
| 2 | WARNING | ALGORITHM | MOM_5D IC 20 日下降 0.2 | 重新评估动量因子公式 |
| 3 | INFO | ARCHITECTURE | screen 执行耗时 48s，P95 超过历史中位数 2.3x | 排查管道瓶颈 |

---

### 详细发现

#### 发现 #1: [CRITICAL][IMPLEMENTATION] screen 命令 KeyError

**证据**:
- 退出码: 1
- stderr: "KeyError: 'breakout_score'"
- 发生时间: 2026-05-23T08:00:12Z

**根因假设**: phase2_pipeline 返回的 DataFrame 缺少 breakout_score 列，可能是权重配置变更导致。

**建议 milestone**: [lp-up][IMPLEMENTATION] 修复 screen 命令 KeyError: phase2_pipeline 返回缺少 breakout_score 列

---
```

### 阶段 F：自动派发 lp-ms

报告生成后**自动推进，不询问用户**。派发规则：

| 严重度 | 动作 |
|--------|------|
| CRITICAL | **自动推进** — 立即通过 subagent 启动 lp-ms |
| WARNING | **自动推进** — 立即通过 subagent 启动 lp-ms |
| INFO | **自动跳过** — 记录到 findings.json，下一轮若升级则推进 |

对每个自动推进的发现，通过 `Agent` 工具以 subagent 模式启动 lp-ms：

```text
Agent(
  description: "lp-ms: <简述>",
  subagent_type: "lp-ms",
  prompt: "<milestone 描述>"
)
```

每个 milestone 描述格式：

```text
[lp-up][<类别>] <简述>

**来源**: lp-up Round <N> 分析报告
**严重度**: CRITICAL | WARNING | INFO
**类别**: ARCHITECTURE | IMPLEMENTATION | ALGORITHM
**证据摘要**: <关键数据点>
**根因假设**: <分析判断>
**预期收益**: <修复后的改善>
**建议范围**: <涉及模块/文件>
```

**串行派发**：按严重度排序（CRITICAL → WARNING），依次启动 subagent。每个 subagent 完成后启动下一个。派发完成后输出推进摘要。

### 阶段 G：状态持久化

每轮结束后保存状态：

```text
.claude/state/lp-up/
  round.md           # 当前 round 编号、最后分析日期、执行模式
  findings.json      # 历史发现追踪：[{id, title, severity, category, status, milestone_url, round_discovered, round_resolved}]
```

**findings.json 结构**：
```json
{
  "round": 3,
  "last_run": "2026-05-23T08:00:00Z",
  "findings": [
    {
      "id": "F-001",
      "title": "RSS 7 日增长 +320MB，疑似内存泄漏",
      "severity": "CRITICAL",
      "category": "ARCHITECTURE",
      "status": "resolved",
      "milestone_url": "https://github.com/Qanora/alpha-screener/milestone/5",
      "round_discovered": 1,
      "round_resolved": 2
    }
  ]
}
```

### 阶段 H：下一轮预告

```text
## 本轮总结

- 发现总数: 3
- 已推进: 2 个 milestone（通过 lp-ms subagent）
- 跳过: 1 个

## 下一轮建议

建议在 milestone 实现合并后（预计 1-3 天），运行：
  /lp-up --run quick

重点验证:
  - F-001 修复效果（内存泄漏）
  - F-002 修复效果（screen KeyError）
```

## 约束

- **执行权限**：`--run` 模式需要 Bash 权限运行 CLI 命令；纯分析模式只读
- **只读分析**：除 `--run` 中的引擎执行外，不修改任何代码、配置或数据
- **自动推进**：CRITICAL 和 WARNING 发现自动通过 subagent 派发 lp-ms，不询问用户；INFO 记录到 findings.json 供后续跟踪
- **串行派发**：多个 milestone 按严重度排序依次派发，不并行
- **数据采样上限**：单次分析不超过 30 天数据或 10 万行日志
- **证据驱动**：每个发现必须有可追溯的数据证据
- **增量分析**：若存在上一轮状态文件，默认只分析上次报告之后的新数据；同时验证上一轮发现的修复状态

## 数据不足处理

若某维度数据不足，标注跳过：

```text
⏭ [ARCHITECTURE] 内存泄漏检测 — 跳过：monitoring_samples 仅有 3 天数据，需至少 7 天
⏭ [ALGORITHM] 回测表现退化 — 跳过：无回测记录
```

不做强行推断。

---

## 附录 A：快速查询参考

### 日志快速诊断

```bash
# 最近 7 天 ERROR 分布
find ~/.alphascreener/logs/ -name "*.log" -mtime -7 | xargs cat | jq -r 'select(.level == "ERROR") | "\(.module) | \(.event)"' | sort | uniq -c | sort -rn

# 各模块日志量分布
find ~/.alphascreener/logs/ -name "*.log" -mtime -7 | xargs cat | jq -r '.module' | sort | uniq -c | sort -rn

# 最近 7 天 cost_usd 合计
find ~/.alphascreener/logs/ -name "*.log" -mtime -7 | xargs cat | jq -r '.cost_usd // 0' | paste -sd+ | bc
```

### SQLite 快速诊断

```bash
DB=~/.alphascreener/data/alphascreener.db

# 各表行数概览
for t in monitoring_samples factor_health_daily alpha_acceptance_daily llm_cost_daily paper_trades alerts data_source_diff; do
  echo -n "$t: "
  sqlite3 "$DB" "SELECT count(*) FROM $t;"
done

# 最近告警
sqlite3 "$DB" "SELECT triggered_at, severity, rule_name, metric_value FROM alerts WHERE triggered_at >= date('now', '-7 days') ORDER BY triggered_at DESC LIMIT 20;"

# CUSUM 触发统计
sqlite3 "$DB" "SELECT factor_name, count(*) as n_triggers FROM factor_health_daily WHERE cusum > 0.05 AND dt >= date('now', '-30 days') GROUP BY factor_name ORDER BY n_triggers DESC;"
```

### Parquet 快速诊断

```bash
# OHLCV 分区覆盖
python -c "
import polars as pl
from pathlib import Path
dts = sorted(p.name for p in Path('~/.alphascreener/data/ohlcv/').expanduser().glob('dt=*'))
print(f'OHLCV 分区数: {len(dts)}')
if dts: print(f'最早: {dts[0]}, 最晚: {dts[-1]}')
"

# 最近因子 NaN 率
python -c "
import polars as pl
df = pl.read_parquet('~/.alphascreener/data/factors/dt=2026-05-*/')
nan_rates = df.select([(pl.col(c).is_nan().sum() / pl.count()).alias(c) for c in df.columns if c not in ('ticker','dt')])
print(nan_rates)
"
```

---

## 附录 B：状态恢复（--resume）

中断后恢复继续：

```text
/lp-up --resume
```

恢复逻辑：
1. 读取 `.claude/state/lp-up/round.md` 和 `findings.json`
2. 检查上一轮派发的 milestone 状态（通过 `gh api` 查询 milestone 下 issues 的完成情况）
3. 对于已完成的 milestone，标记对应 finding 为 `resolved`
4. 对于仍在进行的 milestone，跳过（等待下一轮）
5. 重新执行分析（不重复执行引擎命令）
6. 继续未完成的派发

---

## 附录 C：报告持久化

每轮报告保存至：

```text
~/.alphascreener/reports/lp-up-round-<N>-<YYYY-MM-DD>.md
```

保留最近 10 份报告，自动清理更早的。
