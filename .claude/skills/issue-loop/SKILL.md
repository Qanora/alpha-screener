---
name: issue-loop
description: 第一层飞轮——需求拆解、Issue 创建、依赖分析、批次编排、进度追踪
---

# Issue Loop（第一层）

Issue 生命周期管理。只负责 issue 层面编排，不直接操作代码或 MR。

## 调用方式

```text
/issue-loop <需求描述>
```

## 流程

### 1. 分析需求

分析需求范围和边界。

### 2. 拆解 Issue

将需求拆成独立、可独立验证的 issue。展示给用户确认：

```text
## 需求拆解
### Issue #1: <标题>
- 描述: <1-2句话>
- 类型: feature / fix
- 依赖: 无 / 依赖 #N

### Issue #2: ...
---
确认后我将创建这些 Issue 并开始执行。
```

### 3. 创建 Issue

```bash
gh issue create --repo Qanora/alpha-screener --title "<title>" --body "<body>" --label "<bug|enhancement>"
```

### 4. 依赖分析 + 批次规划

- 无依赖 → 第 1 批次并发
- 仅依赖第 1 批的 → 第 2 批次并发
- 以此类推

### 5. 派发执行

对每个批次中的每个 issue，交给第二层：

在当前上下文直接执行第二层（无需 subagent）：

```text
/mr-loop <issue-number>
```

### 6. 查看进度

```bash
gh issue list --repo Qanora/alpha-screener --state open --limit 20
gh pr list --repo Qanora/alpha-screener --state open
```

### 7. 交付报告

```text
## 交付报告
- 需求: <原始需求>
- Issue 总数: <N>
- 已合入: <list>
- MR 列表: <list>
```

## 约束

- 仅做 issue 层面编排和追踪
- 不写代码（第三层负责）
- 不操作 MR（第二层负责）

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
