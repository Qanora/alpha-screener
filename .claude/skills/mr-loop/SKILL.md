---
name: mr-loop
description: 第二层飞轮——MR 全生命周期管理：提交、创建、监控、收集 review、分配修复、追踪轮次
---

# MR Loop（第二层）

MR (Merge Request) 生命周期管理。负责所有 git 和 MR 操作，不直接写代码。

## 调用方式

```text
/mr-loop <issue-number>
/mr-loop <issue-number> --resume
```

**`--resume`**: 检查 `.status` 决定是否可恢复。可恢复状态：

- `BLOCKED_CI` 且 `fix_round < 3`
- `BLOCKED_REVIEW` 且 `close_reopen_count < 2`
- `CONFLICT` 或 `API_ERROR` 需人工介入，不可恢复

## 流程

### 1. 启动开发

**通过 subagent 调用**第三层 `/dev-loop <N>`，避免污染 mr-loop 的上下文：

```text
Agent(subagent_type="general-purpose", description="Dev issue #<N>", prompt="/dev-loop <N>")
```

subagent 退出后：

1. 读取 worktree 里的 `.handoff-issue-<N>.md` 获取 branch 名
2. 检查终端输出是否包含 `---HANDOFF---` ... `---HANDOFF_END---` 信号块
3. 解析信号：
   - `DEV_DONE=<branch>` → 继续步骤 2
   - `FAIL_DONE=<error-type>` → 根据 error type 处理（见错误处理章节）

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

| 退出码 | 含义              | 动作                                                         |
| ------ | ----------------- | ------------------------------------------------------------ |
| 0      | merged            | 写 `MERGED` → 清理 worktree → `ISSUE_DONE=<N>`               |
| 1      | CI failure        | 写 `BLOCKED_CI` → 收集 CI 日志 → **检查 fix_round**          |
| 2      | stuck             | **检查 close_reopen_count** → close-reopen.sh → 重置 round=0 |
| 4      | changes requested | 写 `BLOCKED_REVIEW` → 收集 review → **检查 fix_round**       |

**重要**：exit code 1 (CI failure) 和 exit code 4 (changes requested) 都需要**通过 dev-loop** 修复，不得在 mr-loop 中直接修改代码。

**重试上限检查**：

```bash
# 检查 fix_round 是否达到上限
if [ "$fix_round" -ge 3 ]; then
  echo "ERROR: fix_round 已达上限 (3 次)"
  echo "BLOCKED_CI" > .claude/worktrees/issue-<N>/.status
  # 需人工介入
  exit 1
fi
# 否则 fix_round++ 并继续步骤 5
```

**close-reopen 上限检查**：

```bash
# 检查 close_reopen_count 是否达到上限
if [ "$close_reopen_count" -ge 2 ]; then
  echo "ERROR: close_reopen_count 已达上限 (2 次)"
  echo "BLOCKED_REVIEW" > .claude/worktrees/issue-<N>/.status
  # 需人工介入
  exit 1
fi
# 否则 close_reopen_count++ 并执行 close-reopen
```

### 5. 收集 Review + 分配修复

**先检查 fix_round 上限**：

```bash
fix_round=$(cat .claude/worktrees/issue-<N>/.fix_round 2>/dev/null || echo 0)
if [ "$fix_round" -ge 3 ]; then
  echo "BLOCKED_CI" > .claude/worktrees/issue-<N>/.status
  echo "ERROR: fix_round 已达上限，需人工介入"
  exit 1
fi
```

fetch-review.sh **默认只返回针对当前 head commit 的评论**（fresh 模式），避免重复修复旧评论：

```bash
bash scripts/fetch-review.sh <mr-number>
gh pr view <mr-number> --repo Qanora/alpha-screener --json statusCheckRollup --jq '
  [.statusCheckRollup[] | select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"))] |
  .[] | "\(.name): \(.conclusion)"
'
```

**递增 fix_round**：

```bash
echo $((fix_round + 1)) > .claude/worktrees/issue-<N>/.fix_round
```

将 fetch-review + CI 信息打包，**通过 subagent** 调用第三层修复：

```text
Agent(subagent_type="general-purpose", description="Fix MR #<mr>", prompt="/dev-loop <N> --fix <mr-number>

## CodeRabbit 评论
<fetch-review 输出>

## CI 失败
<CI log>")
```

**等待 dev-loop 完成并解析信号**：

成功时：

```text
---HANDOFF---
FIX_DONE=<BRANCH>
---HANDOFF_END---
```

失败时（见错误处理章节）：

```text
---HANDOFF---
FAIL_DONE=<error-type>
---HANDOFF_END---
```

确认 `FIX_DONE` 信号后进入步骤 6。

### 6. commit fix + push 同一分支

**确认 dev-loop 完成后**（检测到 `FIX_DONE` 信号），由第二层执行：

```bash
cd .claude/worktrees/issue-<N>
# 清除阻塞状态文件
rm -f .status
# 重置 fix_round（修复成功 push 后重置）
echo 0 > .fix_round
git add -A
git commit -m "fix: address review findings (#<N>)"
git push origin <BRANCH>
```

**注意**：修复 commit 只关联 issue（`#<N>`），不包含 `closes`，避免重复关闭。

push 后 CodeRabbit 自动 re-review（`auto_pause_after_reviewed_commits: 10`）。

### 7. 回到监控

`round=round+1`，回到步骤 3。

### 8. 超过 6 轮

`round >= 6` 且仍 CHANGES_REQUESTED：

```bash
# 检查 close_reopen 上限
close_reopen_count=$(cat .claude/worktrees/issue-<N>/.close_reopen_count 2>/dev/null || echo 0)
if [ "$close_reopen_count" -ge 2 ]; then
  echo "BLOCKED_REVIEW" > .claude/worktrees/issue-<N>/.status
  echo "ERROR: close_reopen_count 已达上限，需人工介入"
  exit 1
fi
# 递增并执行 close-reopen
echo $((close_reopen_count + 1)) > .claude/worktrees/issue-<N>/.close_reopen_count
bash scripts/close-reopen.sh <mr-number> <branch>
round=0
```

回到步骤 3。

## 状态机

```text
[开始] → /dev-loop → commit+push+mr create
    → watch-pr
        ├─ merged → 写 MERGED → 清理 worktree → [issue done]
        ├─ CI fail → 写 BLOCKED_CI → 检查 fix_round < 3? → fetch-review → /dev-loop --fix → [WAIT: FIX_DONE] → 清除状态 → commit+push → watch-pr
        │         └─ fix_round >= 3 → 写 BLOCKED_CI → [人工介入]
        ├─ changes → 写 BLOCKED_REVIEW → 检查 fix_round < 3? → fetch-review → /dev-loop --fix → [WAIT: FIX_DONE] → 清除状态 → commit+push → watch-pr
        │         └─ fix_round >= 3 → 写 BLOCKED_CI → [人工介入]
        ├─ stuck → 检查 close_reopen_count < 2? → close-reopen
        │        └─ close_reopen_count >= 2 → 写 BLOCKED_REVIEW → [人工介入]
        └─ round>=6 → 同 stuck 流程
```

## 重试上限

| 计数器         | 上限 | 触发条件                  | 超限状态         | 重置时机           |
| -------------- | ---- | ------------------------- | ---------------- | ------------------ |
| `fix_round`    | 3    | 每次 dev-loop --fix 调用  | `BLOCKED_CI`     | 修复成功 push 后   |
| `close_reopen` | 2    | 每次 close-reopen.sh 执行 | `BLOCKED_REVIEW` | 不重置（全局累计） |

## 约束

- 负责**所有** git 操作和 gh 操作
- **禁止直接修改代码**：不得使用 Edit、Write、NotebookEdit 工具；所有代码修改必须通过 `/dev-loop` subagent 完成
- **CI failure 也交给 dev-loop**：任何代码修复（包括 CI 失败）都必须调用 `/dev-loop <N> --fix`
- 修复始终 push 同一 MR 分支，不每轮 close-reopen
- 维护映射: issue → worktree → branch → MR → round
- **步骤 5→6 衔接**：必须等待 dev-loop 完成（检测 FIX_DONE 信号），否则跳过步骤 6
- fetch-review 默认 fresh 模式，防止重复修复旧评论

## Worktree 生命周期管理

每个 issue 对应一个 worktree，通过状态标记文件管理生命周期。

### 状态标记文件

`.claude/worktrees/issue-<N>/.status`：

| 状态           | 含义             | 清理 | 可恢复条件               |
| -------------- | ---------------- | ---- | ------------------------ |
| MERGED         | MR 已合入        | 可   | 不可恢复                 |
| CONFLICT       | 合并冲突无法解决 | 否   | 不可恢复，需人工介入     |
| ABANDONED      | Issue 关闭无 MR  | 可   | 不可恢复                 |
| BLOCKED_CI     | CI 失败阻塞      | 否   | `fix_round < 3`          |
| BLOCKED_REVIEW | Review 阻塞      | 否   | `close_reopen_count < 2` |
| API_ERROR      | gh API 调用失败  | 否   | 不可恢复，需人工介入     |

### 计数器文件

| 文件                  | 用途                     | 初始值 |
| --------------------- | ------------------------ | ------ |
| `.fix_round`          | dev-loop --fix 调用次数  | 0      |
| `.close_reopen_count` | close-reopen.sh 执行次数 | 0      |

### --resume 逻辑

```bash
status=$(cat .claude/worktrees/issue-<N>/.status 2>/dev/null || echo "")
fix_round=$(cat .claude/worktrees/issue-<N>/.fix_round 2>/dev/null || echo 0)
close_reopen_count=$(cat .claude/worktrees/issue-<N>/.close_reopen_count 2>/dev/null || echo 0)

case "$status" in
  BLOCKED_CI)
    if [ "$fix_round" -lt 3 ]; then
      echo "可恢复：fix_round=$fix_round < 3"
      # 继续步骤 5
    else
      echo "不可恢复：fix_round=$fix_round >= 3，需人工介入"
      exit 1
    fi
    ;;
  BLOCKED_REVIEW)
    if [ "$close_reopen_count" -lt 2 ]; then
      echo "可恢复：close_reopen_count=$close_reopen_count < 2"
      # 继续步骤 5
    else
      echo "不可恢复：close_reopen_count=$close_reopen_count >= 2，需人工介入"
      exit 1
    fi
    ;;
  CONFLICT|API_ERROR)
    echo "不可恢复：$status，需人工介入"
    exit 1
    ;;
  MERGED|ABANDONED)
    echo "已完成：$status"
    exit 0
    ;;
  *)
    echo "无阻塞状态或未知状态：$status"
    # 继续正常流程
    ;;
esac
```

### 清理策略

| 场景                    | 动作                                     |
| ----------------------- | ---------------------------------------- |
| MR merged               | 写 `MERGED` → 清理 worktree              |
| Issue 手动关闭无 MR     | 写 `ABANDONED` → 清理 worktree           |
| Merge conflict 无法解决 | 写 `CONFLICT` → 保留 worktree，人工介入  |
| CI 失败需修复           | 写 `BLOCKED_CI` → 保留 worktree          |
| Review 阻塞             | 写 `BLOCKED_REVIEW` → 保留 worktree      |
| gh API 调用失败         | 写 `API_ERROR` → 保留 worktree，人工介入 |
| Dev-loop 异常退出       | 不写状态 → 保留 worktree                 |

### 清理命令

```bash
# 写入状态
echo "MERGED" > .claude/worktrees/issue-<N>/.status

# 执行清理（仅当状态为 MERGED 或 ABANDONED）
STATUS=$(cat .claude/worktrees/issue-<N>/.status 2>/dev/null || echo "")
if [ "$STATUS" = "MERGED" ] || [ "$STATUS" = "ABANDONED" ]; then
  git worktree remove .claude/worktrees/issue-<N> --force
  git branch -D <branch> 2>/dev/null || true
fi
```

### 状态写入时机

- **步骤 4 (merged)**：写 `MERGED`，执行清理
- **步骤 5 (CI failure)**：写 `BLOCKED_CI`
- **步骤 5 (changes requested)**：写 `BLOCKED_REVIEW`
- **错误处理 (CONFLICT_UNRESOLVABLE)**：写 `CONFLICT`
- **错误处理 (API_ERROR)**：写 `API_ERROR`
- **外部事件 (Issue 关闭无 MR)**：写 `ABANDONED`，执行清理

## 错误处理

当 dev-loop 返回 `FAIL_DONE=<error-type>` 信号时：

| Error type            | 含义                           | 处理方式                     |
| --------------------- | ------------------------------ | ---------------------------- |
| CR_UNFIXABLE          | cr review 有 findings 无法修复 | 人工介入，记录到 issue       |
| CONFLICT_UNRESOLVABLE | merge conflict 无法解决        | 写 `CONFLICT`，人工介入      |
| ENV_ERROR             | 环境问题（cr CLI 未安装等）    | 检查环境配置，安装依赖后重试 |
| API_ERROR             | gh API 调用失败                | 写 `API_ERROR`，人工介入     |
| UNKNOWN               | 其他异常                       | 记录日志，人工介入           |

处理流程：

1. 解析 `---HANDOFF---` 块中的 error type
2. 根据 error type 选择处理方式
3. 如需人工介入，在 issue 上添加 comment 说明情况
