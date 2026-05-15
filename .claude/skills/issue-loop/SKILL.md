---
name: issue-loop
description: 第一层飞轮——需求拆解、Issue 创建、依赖分析、批次编排、进度追踪
---

# Issue Loop（第一层）

Issue 生命周期管理。只负责 issue 层面编排，不直接操作代码或 PR。

## 调用方式

```text
/issue-loop <需求描述>
```

## 流程

### 1. 分析需求

读取 `CONTEXT.md` 和 `docs/adr/`，理解需求在领域模型中的位置。

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
/pr-loop <issue-number>
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
- PR 列表: <list>
```

## 约束

- 仅做 issue 层面编排和追踪
- 不写代码（第三层负责）
- 不操作 PR（第二层负责）
