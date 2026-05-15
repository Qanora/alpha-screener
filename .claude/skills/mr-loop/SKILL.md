---
name: mr-loop
description: 第二层飞轮——MR 全生命周期管理：提交、创建、监控、收集 review、分配修复、追踪轮次
---

# MR Loop（第二层）

MR (Merge Request) 生命周期管理。负责所有 git 和 MR 操作，不直接写代码。

## 调用方式

```text
/mr-loop <issue-number>
```

## 流程

### 1. 启动开发

**通过 subagent 调用**第三层 `/dev-loop <N>`，避免污染 mr-loop 的上下文：

```text
Agent(subagent_type="general-purpose", description="Dev issue #<N>", prompt="/dev-loop <N>")
```

subagent 退出后读取 worktree 里的 `.handoff-issue-<N>.md` 获取 branch 名。

### 2. commit + push + 创建 MR

第三层退出后（代码已在 worktree 中就绪），由第二层执行 git 操作：

```bash
cd .claude/worktrees/issue-<N>
git add -A
git commit -m "<type>: <description> (closes #<N>)"
# 分开执行，不用 &&
git push origin <BRANCH>
gh pr create --repo Qanora/alpha-screener --title "<type>: <description> (closes #<N>)" --body "$(cat <<'EOF'
Closes #<N>

## Summary

## Test plan

- [ ] ruff check passes
- [ ] python -m pytest tests/ -v passes
EOF
)" --base master
```

### 3. 监控 MR

```bash
bash scripts/watch-pr.sh <mr-number>
```

`round=1`（初始创建算第 1 轮）。

### 4. 响应状态

| 退出码 | 含义 | 动作 |
|---|---|---|
| 0 | merged | 清理 worktree → `ISSUE_DONE=<N>` |
| 1 | CI failure | 收集 CI 日志 → **交给 dev-loop**（见步骤 5） |
| 4 | changes requested | 收集 review → **交给 dev-loop**（见步骤 5） |
lose-reopen.sh <mr-number> <branch>` → 重置 round=0/a/
**重要**：exit code 1 (CI failure) 和 exit code 4 (changes requested) 都需要**通过 dev-loop** 修复，不得在 mr-loop 中直接修改代码。

**重要**：exit code 1 (CI failure) 和 exit code 4 (changes requested) 都需要**通过 dev-loop** 修复，不得在 mr-loop 中直接修改代码。

### 5. 收集 Review + 分配修复

fetch-review.sh **默认只返回针对当前 head commit 的评论**（fresh 模式），避免重复修复旧评论：

```bash
bash scripts/fetch-review.sh <mr-number>
gh pr view <mr-number> --repo Qanora/alpha-screener --json statusCheckRollup --jq '
  [.statusCheckRollup[] | select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"))] |
  .[] | "\(.name): \(.conclusion)"
'
```

将 fetch-review + CI 信息打包，**通过 subagent** 调用第三层修复：

```text
Agent(subagent_type="general-purpose", description="Fix MR #<mr>", prompt="/dev-loop <N> --fix <mr-number>

## CodeRabbit 评论
<fetch-review 输出>

## CI 失败
<CI log>")
```

**⚠️ 等待 dev-loop 完成**：subagent 退出后，必须检查终端输出包含 `FIX_DONE=<branch>`，确认修复完成后再进入步骤 6。

### 6. commit fix + push 同一分支

**确认 dev-loop 完成后**（检测到 FIX_DONE 信号），由第二层执行：

```bash
cd .claude/worktrees/issue-<N>
git add -A
git commit -m "fix: address review findings (#<N>)"
git push origin <BRANCH>
/
**注意**：修复 commit 只关联 issue（`#<N>`），不包含 `closes`，避免重复关闭。
```

**注意**：修复 commit 只关联 issue（`#<N>`），不包含 `closes`，避免重复关闭。

push 后 CodeRabbit 自动 re-review（`auto_pause_after_reviewed_commits: 10`）。

### 7. 回到监控

`round=round+1`，回到步骤 3。

### 8. 超过 6 轮

`round >= 6` 且仍 CHANGES_REQUESTED：

```bash
bash scripts/close-reopen.sh <mr-number> <branch>
round=0
```

回到步骤 3。

## 状态机

```
[开始] → /dev-loop → commit+push+mr create
    → watch-pr
        ├─ merged → [issue done]
        ├─ CI fail / changes → fetch-review(fresh) → /dev-loop --fix → [WAIT: FIX_DONE] → commit+push → watch-pr
        ├─ stuck → close-reopen
        └─ round>=6 → close-reopen
```

## 约束

- 负责**所有** git 操作（add/commit/push）和 gh 操作（mr create/close-reopen）
- **禁止直接修改代码**：不得使用 Edit、Write、NotebookEdit 工具；所有代码修改必须通过 `/dev-loop` subagent 完成
- **CI failure 也交给 dev-loop**：任何代码修复（包括 CI 失败）都必须调用 `/dev-loop <N> --fix`
- 修复始终 push 同一 MR 分支，不每轮 close-reopen
- 维护映射: issue → worktree → branch → MR → round
- **步骤 5→6 衔接**：必须等待 dev-loop 完成（检测 FIX_DONE 信号），否则跳过步骤 6
- fetch-review 默认 fresh 模式，防止重复修复旧评论