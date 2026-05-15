---
name: pr-loop
description: 第二层飞轮——PR 全生命周期管理：提交、创建、监控、收集 review、分配修复、追踪轮次
---

# PR Loop（第二层）

PR 生命周期管理。负责所有 git 和 PR 操作，不直接写代码。

## 调用方式

```text
/pr-loop <issue-number>
```

## 流程

### 1. 启动开发

**通过 subagent 调用**第三层 `/dev-loop <N>`，避免污染 pr-loop 的上下文：

```text
Agent(subagent_type="general-purpose", description="Dev issue #<N>", prompt="/dev-loop <N>")
```

subagent 退出后读取 worktree 里的 `.handoff-issue-<N>.md` 获取 branch 名。

### 2. commit + push + 创建 PR

第三层退出后（代码已在 worktree 中就绪），由第二层执行 git 操作：

```bash
cd .claude/worktrees/issue-<N>
git add -A
git commit -m "<type>: <description> (#<N>)"
# 分开执行，不用 &&
git push origin <BRANCH>
gh pr create --repo Qanora/alpha-screener --title "<title>" --body "$(cat <<'EOF'
Closes #<N>

## Summary

## Test plan

- [ ] ruff check passes
- [ ] python -m pytest tests/ -v passes
EOF
)" --base master
```

### 3. 监控 PR

```bash
bash scripts/watch-pr.sh <pr-number>
```

`round=1`（初始创建算第 1 轮）。

### 4. 响应状态

| 退出码 | 含义 | 动作 |
|---|---|---|
| 0 | merged | 清理 worktree → `ISSUE_DONE=<N>` |
| 1 | CI failure | 收集信息 → 进入步骤 5 |
| 4 | changes requested | 收集 review → 进入步骤 5 |
| 2 | stuck | `scripts/close-reopen.sh <PR> <branch>` → 重置 round=0 |

### 5. 收集 Review + 分配修复

```bash
bash scripts/fetch-review.sh <pr-number>
gh pr view <pr-number> --repo Qanora/alpha-screener --json statusCheckRollup --jq '
  [.statusCheckRollup[] | select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"))] |
  .[] | "\(.name): \(.conclusion)"
'
```

将 fetch-review + CI 信息打包，**通过 subagent** 调用第三层修复：

```text
Agent(subagent_type="general-purpose", description="Fix PR #<pr>", prompt="/dev-loop <N> --fix <pr-number>

## CodeRabbit 评论
<fetch-review 输出>

## CI 失败
<CI log>")
```

### 6. commit fix + push 同一分支

第三层修复完毕后，由第二层执行：

```bash
cd .claude/worktrees/issue-<N>
git add -A
git commit -m "fix: address review findings (#<N>)"
git push origin <BRANCH>
```

push 后 CodeRabbit 自动 re-review（`auto_pause_after_reviewed_commits: 10`）。

### 7. 回到监控

`round=round+1`，回到步骤 3。

### 8. 超过 6 轮

`round >= 6` 且仍 CHANGES_REQUESTED：

```bash
bash scripts/close-reopen.sh <pr-number> <branch>
round=0
```

回到步骤 3。

## 状态机

```
[开始] → /dev-loop → commit+push+pr create
    → watch-pr
        ├─ merged → [issue done]
        ├─ CI fail / changes → fetch-review → /dev-loop --fix → commit+push → watch-pr
        ├─ stuck → close-reopen
        └─ round>=6 → close-reopen
```

## 约束

- 负责**所有** git 操作（add/commit/push）和 gh 操作（pr create/close-reopen）
- 不直接写代码（第三层写）
- 修复始终 push 同一 PR 分支，不每轮 close-reopen
- 维护映射: issue → worktree → branch → PR → round
