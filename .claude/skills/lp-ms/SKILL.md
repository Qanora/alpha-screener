---
name: lp-ms
description: 第二层飞轮——需求拆解、Issue 创建、依赖分析、批次编排、进度追踪
---

# LP-MS（第二层）

Issue 生命周期管理。只负责 issue 层面编排，不直接操作代码或 MR。

## 调用方式

```text
/lp-ms <需求描述>
/lp-ms --resume <milestone-number>
```

**`--resume`**: 从 milestone 恢复中断的飞轮执行。查询 milestone 下所有 issues，根据状态表推断恢复动作。

## 流程

### 1. 分析需求

分析需求范围和边界。

### 2. 拆解 Issue

将需求拆成独立、可独立验证的 issue。拆解完成后**自动创建，不询问用户确认**：

```text
## 需求拆解
### Issue #1: <标题>
- 描述: <1-2句话>
- 类型: feature / fix
- 依赖: 无 / 依赖 #N

### Issue #2: ...
---
正在创建 Issue 并开始执行...
```

> **300 行约束（soft）**：单个 issue 预计改动超过 300 行时，在 issue body 中标注 `⚠️ large diff`。不强制拆分，不阻塞执行。

### 3. 创建 Issue + Milestone

**每个需求对应一个 milestone**，issue 创建时关联 milestone：

```bash
# 创建或获取 milestone
gh api repos/Qanora/alpha-screener/milestones --paginate --jq '.[] | select(.title == "<需求标题>").number' | head -1 || \
  gh api repos/Qanora/alpha-screener/milestones -f title="<需求标题>" -f state="open" --jq '.number'

# 创建 issue 并关联 milestone
gh issue create --repo Qanora/alpha-screener --title "<title>" --body "<body>" --label "<bug|enhancement>" --milestone "<milestone-number>"
```

### 4. 依赖分析 + 批次规划

#### 4.1 构建依赖图

从 issue 拆解结果中提取依赖关系，构建有向图：

```text
节点 = Issue 编号
边 A → B = Issue A 依赖 Issue B
```

#### 4.2 环检测

使用 DFS 检测依赖环。若发现环，**立即停止**并报告：

```text
❌ 检测到依赖环:
   Issue #A → Issue #B → Issue #C → Issue #A

请重新拆解需求，消除循环依赖。
```

#### 4.3 批次规划（无环时）

- 无依赖 → 第 1 批次
- 仅依赖第 1 批的 → 第 2 批次
- 以此类推

### 5. 派发执行

批次内 **串行执行**，每个 issue 依次交给第三层：

```text
/lp-mr <issue-number>
```

### 6. 查看进度

```bash
gh issue list --repo Qanora/alpha-screener --state open --limit 20
gh pr list --repo Qanora/alpha-screener --state open
```

### 7. 关闭 Milestone

检查 milestone 下所有 issues 是否已合入，若全部完成则关闭 milestone。

```bash
# 1. 获取 milestone 下所有 issues
gh issue list --state all --milestone "<N>" --json number,state

# 2. 若全部 closed，关闭 milestone
gh api -X PATCH repos/Qanora/alpha-screener/milestones/<N> -f state=closed
```

### 8. 交付报告

```text
## 交付报告
- 需求: <原始需求>
- Issue 总数: <N>
- 已合入: <list>
- MR 列表: <list>
- Milestone: #<N> (closed)
```

## 约束

- 仅做 issue 层面编排和追踪
- 不写代码（第四层负责）
- 不操作 MR（第三层负责）

---

## 附录 A: Issue Tracker 操作

Issues live as GitHub issues on `Qanora/alpha-screener`。使用 `gh` CLI 进行所有操作。

| 操作            | 命令                                                                |
| --------------- | ------------------------------------------------------------------- |
| 创建 issue      | `gh issue create --title "..." --body "..."`                        |
| 查看 issue      | `gh issue view <number> --comments`                                 |
| 列出 issues     | `gh issue list --state open --json number,title,body,labels`        |
| 评论 issue      | `gh issue comment <number> --body "..."`                            |
| 添加/删除 label | `gh issue edit <number> --add-label "..."` / `--remove-label "..."` |
| 关闭 issue      | `gh issue close <number> --comment "..."`                           |

---

## 附录 B: Triage Labels

五种 triage 标签：

| Label             | 含义                       |
| ----------------- | -------------------------- |
| `needs-triage`    | Maintainer 需要评估        |
| `needs-info`      | 等待更多信息               |
| `ready-for-agent` | 完整定义，可交给 AFK agent |
| `ready-for-human` | 需要人工实现               |
| `wontfix`         | 不处理                     |

---

## 附录 C: 状态恢复机制

Session 中断后，从 GitHub 反推当前状态继续执行。

### 恢复入口

```text
/lp-ms --resume <milestone-number>
```

### 状态恢复策略

查询 milestone 下所有 issues，根据 GitHub 状态推断恢复动作：

| Issue 状态  | MR 状态           | 恢复动作                                 |
| ----------- | ----------------- | ---------------------------------------- |
| open, 无 MR | —                 | 启动 `/lp-mr <issue>`                  |
| open, 有 MR | CI_FAILURE        | 收集 CI log → `/lp-mr --resume`        |
| open, 有 MR | PENDING           | 继续监控 (`watch-pr.sh`)                 |
| closed      | MR merged         | 跳过                                     |
| closed      | 无 MR             | 跳过                                     |

### 恢复流程

```bash
# 1. 获取 milestone 下的所有 issues
gh issue list --repo Qanora/alpha-screener --state all --milestone "<milestone-number>" --json number,state,title

# 2. 对每个 issue，查询关联的 MR
gh pr list --repo Qanora/alpha-screener --state all --json number,headRefName,state,statusCheckRollup

# 3. 根据状态表决定恢复动作
```

### 恢复示例

```text
## 恢复报告: Milestone #<N>

### Issue 状态分析
| Issue | MR | 状态 | 恢复动作 |
|-------|-----|------|----------|
| #26 | #27 | CI_FAILURE | /lp-mr 26 --resume |
| #28 | — | open | /lp-mr 28 |

### 执行计划
1. 恢复 #26: 修复 CI 失败
2. 启动 #28: 新建 MR
```

---

## 附录 D: Milestone 操作

Milestones 通过 `gh api` 操作。

| 操作      | 呇令                                                          |
| --------- | ------------------------------------------------------------- |
| 创建      | `gh api repos/{repo}/milestones -f title="..."`               |
| 列出      | `gh api repos/{repo}/milestones --jq '.[]\|.number'`          |
| 关闭      | `gh api -X PATCH repos/{repo}/milestones/N -f state=closed`   |
| 查 issues | `gh issue list --milestone N --state all --json number,state` |

注：`{repo}` = `Qanora/alpha-screener`
