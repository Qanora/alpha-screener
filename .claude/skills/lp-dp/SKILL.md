---
name: lp-dp
description: 诊断+打磨——分析四层飞轮执行上下文，发现流程偏差、可自动化环节、冗余交互，通过 lp-ms 持续优化飞轮自身
---

# LP-DP（Diagnose & Polish · 飞轮自检）

分析四层飞轮（lp-up / lp-ms / lp-mr / lp-dev）的实际执行情况，对照各 skill 的规范要求，发现：
- **流程偏差**：实际执行跳过了哪些步骤
- **可自动化点**：哪些手动环节可以自动完成
- **冗余交互**：哪些向用户询问的问题不需要
- **设计缺陷**：skill 定义中哪些流程可以优化

通过 lp-ms 驱动飞轮自身的持续进化。

```
┌──────────────────────────────────────────────────┐
│                   lp-dp（飞轮自检）                 │
│  观察飞轮执行 → 对照规范 → 发现偏差/冗余/机会        │
│  → 生成优化 milestone → subagent:lp-ms             │
└────────┬─────────────────────────────────────────┘
         │ 优化 milestone
         ▼
    lp-ms → lp-mr → lp-dev → merge → 飞轮更优
```

## 调用方式

```text
/lp-dp                          # 分析当前会话上下文，检查飞轮执行质量
/lp-dp --skill <name>           # 聚焦单个 skill 的规范符合度
/lp-dp --since <date>           # 分析指定日期以来的飞轮执行记录
/lp-dp --audit                  # 完整审计：逐条对照每个 skill 的流程规范
```

## 与 lp-up 的区别

| 维度 | lp-up | lp-dp |
|------|-------|-------|
| 分析对象 | 引擎运行时数据 | 飞轮自身执行行为 |
| 数据来源 | 日志/DB/Parquet/指标 | 会话上下文/Git 历史/state 文件 |
| 发现类型 | 引擎架构/实现/算法缺陷 | 飞轮流程偏差/自动化机会/冗余交互 |
| 改进目标 | 引擎质量 | 飞轮效率和质量 |

## 流程

### 1. 采集飞轮执行上下文

#### 1.1 当前会话观察

分析当前对话上下文中飞轮的执行痕迹：

- 哪些 skill 被调用了？调用顺序是否符合规范？
- 每个 skill 执行了哪些步骤？是否跳过了某些步骤？
- 向用户提了哪些问题？哪些是可自动决策的？
- 有哪些重复操作或低效模式？

#### 1.2 历史执行记录

```bash
# 近期飞轮相关 commit
git log --since="7 days ago" --oneline --grep="lp-ms\|lp-mr\|lp-dev\|lp-up\|lp-dp" --all

# 近期 issue/milestone 状态
gh issue list --repo Qanora/alpha-screener --state all --limit 30 --json number,state,title,labels

# State 文件分析 — 哪些 issue 经历了多轮 fix？哪些被 blocked？
find .claude/state/ -name "*.fix_round" -o -name "*.status" -o -name "*.close_reopen_count" | xargs cat 2>/dev/null

# lp-up 分析报告
ls ~/.alphascreener/reports/lp-up-*.md 2>/dev/null && cat ~/.alphascreener/reports/lp-up-round-*.md 2>/dev/null | head -200
```

#### 1.3 Skill 规范提取

读取每个 skill 的 SKILL.md，提取关键规范要求作为检查清单：
- 规定了哪些步骤？
- 哪些步骤是"阻塞步骤"？哪些是 optional？
- 哪些地方允许 subagent 调用？
- 哪些地方定义了与用户的交互点？

### 2. 对照检查

#### 2.1 流程完整性检查

对每个 skill 的每次调用，检查实际步骤是否完整：

| Skill | 预期步骤 | 实际执行 | 偏差 |
|-------|---------|---------|------|
| lp-mr | 1a→1b→2→3→4→... | 1a→1b→2→**停止** | 跳过步骤 3（监控） |

**检查方法**：追踪会话中的 tool call 序列，与 SKILL.md 定义的流程对照，找出跳过的步骤。

#### 2.2 自动化机会发现

| 模式 | 当前做法 | 可自动化 |
|------|---------|---------|
| 询问"是否继续" | 等待用户确认 | 若非破坏性操作，直接继续 |
| 查询状态 | 手动 gh pr view | 脚本化自动轮询 |
| 创建 issue/milestone | 手动逐个创建 | 批量创建 |
| cleanup | 手动切分支/删除 | 合并为单一脚本 |

**检查方法**：识别所有 `AskUserQuestion` 调用和 `gh` 手动查询，判断是否可用脚本/skill 替代。

#### 2.3 冗余交互发现

分析所有与用户的交互点：

- 哪些问题有明确的默认答案？（如 "是否继续" → 总是继续）
- 哪些信息可以从上下文推断？（如 issue 依赖关系 → 从 milestone 推断）
- 哪些确认是多余的？（如 auto-merge 后的"确认合入"）

**判定标准**：如果用户在过去 N 次交互中对某问题总是选择同一选项，该问题应改为自动决策。

#### 2.4 Skill 定义规范性检查

对照 skill 自身定义，检查是否存在设计缺陷：

- 流程是否包含不可达步骤？
- 错误处理是否覆盖所有信号类型？
- 重试上限是否合理？（当前是否有 issue 频繁触发上限？）
- 跨 skill 的接口信号（HANDOFF 格式）是否一致？

### 3. 发现分类

| 类别 | 含义 | 示例 |
|------|------|------|
| `DEVIATION` | 实际执行偏离规范 | lp-mr 跳过监控步骤 |
| `AUTOMATION` | 可自动化的手动操作 | "是否继续" 改为自动 |
| `REDUNDANCY` | 不必要的用户交互 | 重复确认可推断的信息 |
| `DESIGN` | Skill 定义本身可改进 | 流程缺少检查点机制 |

严重度沿用 lp-up 体系：`CRITICAL` / `WARNING` / `INFO`

### 4. 报告生成

```text
## LP-DP 飞轮审计报告

**审计时间**: <ISO timestamp>
**审计范围**: 当前会话 + 7 天历史

---

### 飞轮健康度

| Skill | 流程符合度 | 自动化率 | 冗余交互 | 评分 |
|-------|-----------|---------|---------|------|
| lp-up | 100% | 高 | 0 | A |
| lp-ms | 95% | 中 | 1 | B+ |
| lp-mr | 80% | 低 | 2 | C+ |
| lp-dev | 100% | 高 | 0 | A |

---

### 发现汇总

| # | 类别 | 严重度 | Skill | 简述 |
|---|------|--------|-------|------|
| 1 | DEVIATION | WARNING | lp-mr | 步骤 2→3 过渡时多次跳过监控 |
| 2 | AUTOMATION | INFO | lp-ms | "是否继续" 确认可改为自动 |
| 3 | REDUNDANCY | INFO | lp-mr | cleanup 可合并为单一脚本 |

---

### 详细发现

#### 发现 #1: [DEVIATION][WARNING] lp-mr 步骤 2→3 过渡跳过监控

**证据**: 当前会话中 lp-mr #169 执行时，步骤 2 完成后直接停止，未启动步骤 3 watch-pr.sh。后续 #173、#174 已修复。
**建议 milestone**: [lp-dp][DEVIATION] lp-mr 增加步骤检查点强制机制 — 已在 memory 中记录，属于 lp-up F-005，已通过 #171 修复。本轮验证修复效果。

---
```

### 5. 自动派发 lp-ms

与 lp-up 相同：CRITICAL + WARNING 自动通过 subagent 启动 lp-ms，INFO 记录到 state 文件。

```text
Agent(
  description: "lp-ms: <简述>",
  subagent_type: "lp-ms",
  prompt: "[lp-dp][<类别>] <简述> ..."
)
```

### 6. 状态持久化

```text
.claude/state/lp-dp/
  audit.md           # 最近一次审计报告
  findings.json      # 历史发现追踪
  skill_scores.json  # 各 skill 健康度评分趋势
```

## 约束

- **只分析飞轮，不分析引擎**：引擎问题由 lp-up 负责
- **以 skill 规范为基准**：对照 SKILL.md 而非主观判断
- **自动派发**：同 lp-up，CRITICAL + WARNING 自动推进
- **证据驱动**：每个发现指向具体的会话步骤或 git 记录
- **增量分析**：若存在上一轮审计记录，重点验证已修复项

## 与其他 skill 的关系

```
lp-dp（飞轮自检）
  ├── 观察 lp-up 的执行是否完整、数据采集是否充分
  ├── 观察 lp-ms 的拆解是否合理、确认环节是否必要
  ├── 观察 lp-mr 的流程是否完整、cleanup 是否执行
  └── 观察 lp-dev 的 HANDOFF 格式是否规范、验证是否通过
```
