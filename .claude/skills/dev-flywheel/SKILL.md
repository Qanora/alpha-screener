---
name: dev-flywheel
description: 需求分析→拆解 issue→依赖分组→批次并发→追踪交付的外层开发飞轮
---

# Dev Flywheel

外层 loop：从用户需求到所有 issue 合入的全流程编排。

## 调用方式

```
/dev-flywheel <需求描述>
```

## 流程

### 1. 分析需求

读取 `CONTEXT.md` 和 `docs/adr/`，理解需求在领域模型中的位置。澄清模糊点后进入拆解。

### 2. 拆解 Issue

将需求拆成独立、可独立验证的 issue。每个 issue：
- 有明确的验收标准
- 不依赖未创建的 issue（除非明确标注依赖）
- 大小适中（一个 Agent 在一轮对话中能完成）

**展示列表给用户确认**，格式：

```
## 需求拆解
### Issue #1: <标题>
- 描述: <1-2句话>
- 类型: feature / fix
- 依赖: 无 / 依赖 #N
- 涉及文件: <预估>

### Issue #2: ...
---
确认后我将创建这些 Issue 并开始执行。
```

用户确认后再创建 Issue。

### 3. 创建 Issue

```bash
gh issue create --repo Qanora/alpha-screener --title "<title>" --body "<body>" --label "<bug|enhancement>"
```

记录每个 issue 的编号。

### 4. 依赖分析 + 批次规划

分析代码依赖，确定执行顺序。

**规则**：
- 无依赖 → 第 1 批次并发
- 仅依赖第 1 批的 → 第 2 批次并发
- 以此类推

展示批次计划：

```
## 执行计划
### 批次 1 (并发 3 个)
- #1: <标题> (无依赖)
- #2: <标题> (无依赖)
- #3: <标题> (无依赖)

### 批次 2 (并发 2 个)
- #4: <标题> (依赖 #1, #2)
- #5: <标题> (依赖 #3)

### 批次 3
- #6: <标题> (依赖 #4, #5)
---
确认后开始执行。
```

### 5. 执行批次

对每个批次的每个 issue，启动独立子 Agent：

```bash
# 每个 issue 用一个 Agent 开发
# 使用 Agent tool 启动 issue-dev skill
```

**并发启动**：同一批次的所有 issue 可以在同一轮中启动多个 Agent。

等待所有子 Agent 完成后，检查输出中的 `PR_URL=...`。

### 6. 内层循环监控

每个 PR 创建后，监控其状态：

```bash
bash scripts/watch-pr.sh <pr-number>
```

**响应不同的退出码**：

| watch-pr 退出码 | 含义 | 动作 |
|---|---|---|
| 0 | merged | 清理 worktree，issue done |
| 1 | CI failure | 启动修复 Agent: `/issue-dev <N> --fix <PR>` |
| 2 | stuck/timeout | `close-reopen.sh <PR> <branch>` 重新触发 |
| 4 | changes requested | 启动修复 Agent: `/issue-dev <N> --fix <PR>` |

**修复 Agent 启动时**：
1. 读取 worktree 里的 `.handoff-issue-<N>.md` 作为上下文
2. 传入 `--fix <pr-number>` flag
3. 修复 Agent 退出后重新运行 watch-pr

### 7. 批次完成

所有 issue merge 后：
- `git worktree prune` 清理 worktree
- 输出交付报告：

```
## 交付报告
- 需求: <原始需求>
- Issue 总数: <N>
- 已合入: <list>
- PR 列表: <list>
```

### 8. 批量进度查看

当用户询问"查看 issue 状态"时：

```bash
gh issue list --repo Qanora/alpha-screener --state open --limit 20
# 或查看特定 issue 的关联 PR
gh issue view <N> --repo Qanora/alpha-screener
```

分析 issue 之间的依赖关系，判断哪些 ready 可以开始，哪些阻塞。

---

## Agent 交接机制

- 开发 Agent 退出时在 worktree 根目录写入 `.handoff-issue-<N>.md`
- 修复 Agent 由主控传入 handoff 内容 + fetch-review 结果
- 主控负责维护 issue → worktree → PR 的映射关系

## 约束

- 不修改 `.claude/` 配置文件
- 不在主控中直接写业务代码，代码由子 Agent 在 worktree 中完成
- 每次 `/dev-flywheel` 调用对应一次完整交付，不跨会话
