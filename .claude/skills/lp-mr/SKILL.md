---
name: lp-mr
description: 第三层飞轮——MR 全生命周期管理：提交、创建、监控、收集 review、分配修复、追踪轮次
---

# LP-MR（第三层）

MR (Merge Request) 生命周期管理。负责所有 git 和 MR 操作，不直接写代码。

**顺序开发模式**：配合 lp-dev 的顺序开发，不使用 worktree。

## 调用方式

```text
/lp-mr <issue-number>
/lp-mr <issue-number> --resume
```

**`--resume`**: 从 GitHub 状态恢复。查询 PR 状态、round、fix_round、close_reopen_count，决定恢复动作。

可恢复状态：

- `BLOCKED_CI` 且 `fix_round < 3`
- `CONFLICT` 或 `API_ERROR` 需人工介入，不可恢复
- 无 `.status` 文件时，从 GitHub PR 状态推断

## 流程

### 1. 启动开发

**1a. 准备干净的开发基址**：始终从 `origin/master` 最新 commit 创建 feature 分支，
避免依赖 issue squash-merge 后产生 add/add 冲突（参见 [#95 冲突案例]）。

    git checkout master
    git fetch origin master
    git pull --ff-only origin master
    # 检查依赖 issue 是否已合入
    for dep in <依赖 issue 编号列表>; do
      state=$(gh issue view "$dep" --repo Qanora/alpha-screener --json state --jq '.state')
      if [ "$state" != "CLOSED" ]; then
        echo "WARNING: dependency #$dep is still $state"
      fi
    done
    git checkout -b feature/issue-<N>

> **规则**：feature 分支**只能**从 `origin/master` 创建，禁止从其他 feature 分支创建，
> 也禁止在 stale master 上直接开分支。

**1b. 通过 subagent 调用**第四层 `/lp-dev <N>`，避免污染 lp-mr 的上下文。
prompt 中需显式告知当前分支名以免 lp-dev 切回 master：

```text
Agent(subagent_type="general-purpose", description="Dev issue #<N>", prompt="/lp-dev <N>

当前开发分支：feature/issue-<N>（已从 origin/master 创建）。请在此分支上开发，不要切回 master。")
```

subagent 退出后：

1. 检查终端输出是否包含 `---HANDOFF---` ... `---HANDOFF_END---` 信号块
2. 解析信号：
   - `DEV_DONE=<branch>` → 继续步骤 2
   - `FAIL_DONE=<error-type>` → 根据 error type 处理（见错误处理章节）

### 2. commit + push + 创建 MR

第四层退出后（代码已在主仓库的 feature 分支中就绪），由第三层执行 git 操作：

```bash
BRANCH="feature/issue-<N>"
git add -A
git commit -m "<type>: <description> (closes #<N>)"
git push origin "$BRANCH"
MR_URL=$(gh pr create --repo Qanora/alpha-screener --title "<type>: <description> (closes #<N>)" --body "$(cat <<'EOF'
Closes #<N>

## Summary

## Test plan

- [ ] ruff check passes
EOF
)" --base master)
# 从 URL 提取 MR number 并启用 auto-merge
MR_NUMBER=$(echo "$MR_URL" | grep -oE '[0-9]+$')
# 不等待 review — watch-pr.sh 检测 CI green 后直接 merge
bash scripts/watch-pr.sh "$MR_NUMBER" && gh pr merge "$MR_NUMBER" --squash --delete-branch
```

### 3. 监控 MR

```bash
bash scripts/watch-pr.sh <mr-number>
```

`round=1`（初始创建算第 1 轮）。

### 4. 响应状态

| 退出码 | 含义       | 动作                                                       |
| ------ | ---------- | ---------------------------------------------------------- |
| 0      | CI green   | 执行 `gh pr merge --squash --delete-branch` → 清理本地分支 → `ISSUE_DONE=<N>` |
| 1      | CI failure | 写 `BLOCKED_CI` → 收集 CI 日志 → **检查 fix_round**        |
| 2      | timeout    | 检查 CI 状态：若 CI green → 合入；否则 → 人工介入          |

**CodeRabbit review 为顾问模式**：watch-pr.sh 不等待 review approve。
CodeRabbit 评论异步到达，可在 issue 中通过 `fetch-review.sh` 查看。
合入后 lp-up 会扫描已合入 MR 的 review 评论，发现未处理建议 → 创建新 issue。

**重试上限检查**：

```bash
# 检查 fix_round 是否达到上限
if [ "$fix_round" -ge 3 ]; then
  echo "ERROR: fix_round 已达上限 (3 次)"
  echo "BLOCKED_CI" > .claude/state/issue-<N>.status
  # 需人工介入
  exit 1
fi
# 否则 fix_round++ 并继续步骤 5
```

**超时处理**：

```bash
# watch-pr.sh exit 2 (timeout): 检查 CI 状态
# 若 CI green → 正常合入；否则 → 人工介入排查 CI 卡住原因
```

### 5. 收集 CI 日志 + 分配修复

**先检查 fix_round 上限**：

```bash
fix_round=$(cat .claude/state/issue-<N>.fix_round 2>/dev/null || echo 0)
if [ "$fix_round" -ge 3 ]; then
  echo "BLOCKED_CI" > .claude/state/issue-<N>.status
  echo "ERROR: fix_round 已达上限，需人工介入"
  exit 1
fi
```

拉取 CI 失败日志：

```bash
gh pr view <mr-number> --repo Qanora/alpha-screener --json statusCheckRollup --jq '
  [.statusCheckRollup[] | select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"))] |
  .[] | "\(.name): \(.conclusion)"
'
```

（可选）拉取 CodeRabbit 评论作为修复参考：
```bash
bash scripts/fetch-review.sh <mr-number>
```

**递增 fix_round**：

```bash
echo $((fix_round + 1)) > .claude/state/issue-<N>.fix_round
```

将 CI 日志 + 可选 CodeRabbit 评论打包，**通过 subagent** 调用第四层修复：

```text
Agent(subagent_type="general-purpose", description="Fix MR #<mr>", prompt="/lp-dev <N> --fix <mr-number>

## CI 失败
<CI log>

## CodeRabbit 评论（顾问参考）
<fetch-review 输出，无则省略>")
```

**等待 lp-dev 完成并解析信号**：

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

**确认 lp-dev 完成后**（检测到 `FIX_DONE` 信号），由第三层执行：

```bash
BRANCH="feature/issue-<N>"
# 清除阻塞状态文件
rm -f .claude/state/issue-<N>.status
# 重置 fix_round（修复成功 push 后重置）
echo 0 > .claude/state/issue-<N>.fix_round
git add -A
git commit -m "fix: address review findings (#<N>)"
git push origin "$BRANCH"
```

**注意**：修复 commit 只关联 issue（`#<N>`），不包含 `closes`，避免重复关闭。

push 后 CodeRabbit 自动 re-review。

### 7. 回到监控

`round=round+1`，回到步骤 3。

### 8. CodeRabbit review（顾问模式，不阻塞）

CodeRabbit review 不阻塞合入。`watch-pr.sh` 退出码 0 时直接 squash-merge。

合入后，lp-mr 可异步拉取 CodeRabbit 评论存入 issue comment：

```bash
bash scripts/fetch-review.sh <mr-number> | gh issue comment <N> --body-file -
```

lp-up 在下一轮分析中会扫描已合入 MR 的 review 建议，必要时创建新 issue。

## 状态机

```text
[开始] → 确保从 origin/master 开分支 → /lp-dev → commit+push+mr create
    → watch-pr
        ├─ CI green → gh pr merge --squash → 切回 master → 删除本地分支 → [issue done]
        ├─ CI fail → 写 BLOCKED_CI → 检查 fix_round < 3? → 收集 CI 日志 → /lp-dev --fix → [WAIT: FIX_DONE] → 清除状态 → commit+push → watch-pr
        │         └─ fix_round >= 3 → 写 BLOCKED_CI → [人工介入]
        ├─ timeout → 检查 CI: green → 合入; 否则 → 人工介入
        └─ CodeRabbit review (advisory, 异步到达，不阻塞合入)
```

## 重试上限

| 计数器      | 上限 | 触发条件                 | 超限状态     | 重置时机         |
| ----------- | ---- | ------------------------ | ------------ | ---------------- |
| `fix_round` | 3    | 每次 CI-failure lp-dev --fix 调用 | `BLOCKED_CI` | 修复成功 push 后 |

## 约束

- 负责**所有** git 操作和 gh 操作
- **禁止直接修改代码**：不得使用 Edit、Write、NotebookEdit 工具；所有代码修改必须通过 `/lp-dev` subagent 完成
- **CI failure 交给 lp-dev**：CI 失败时调用 `/lp-dev <N> --fix` 修复
- 修复 commit push 到同一分支，watch-pr 检测到 CI green 后自动合入
- 维护映射: issue → branch → MR → round
- **步骤 5→6 衔接**：必须等待 lp-dev 完成（检测 FIX_DONE 信号），否则跳过步骤 6
- **分支基址**：feature 分支必须从 `origin/master` 创建（步骤 1a），禁止从其他 feature 分支派生
- **CodeRabbit 顾问模式**：review 评论不阻塞合入，合入后 lp-up 会扫描并创建后续 issue

## 状态管理

状态文件存储在 `.claude/state/`。

### 状态标记文件

`.claude/state/issue-<N>.status`：

| 状态        | 含义             | 可恢复条件         |
| ----------- | ---------------- | ------------------ |
| MERGED      | MR 已合入        | 不可恢复           |
| CONFLICT    | 合并冲突无法解决 | 不可恢复，需人工介入 |
| ABANDONED   | Issue 关闭无 MR  | 不可恢复           |
| BLOCKED_CI  | CI 失败阻塞      | `fix_round < 3`    |
| API_ERROR   | gh API 调用失败  | 不可恢复，需人工介入 |

### 计数器文件

| 文件                  | 用途                     | 初始值 |
| --------------------- | ------------------------ | ------ |
| `.fix_round`          | lp-dev --fix 调用次数  | 0      |


### --resume 逻辑

#### 从 GitHub 恢复状态

```bash
# 1. 获取 issue 关联的 PR
PR_NUMBER=$(gh pr list --repo Qanora/alpha-screener --state all --json number,headRefName --jq ".[] | select(.headRefName == \"feature/issue-<N>\") | .number" | head -1)

# 2. 获取 PR 状态
PR_STATE=$(gh pr view "$PR_NUMBER" --repo Qanora/alpha-screener --json state,reviewDecision,statusCheckRollup --jq '{state, reviewDecision, checks: [.statusCheckRollup[] | select(.status == "COMPLETED" and .conclusion == "FAILURE")]}')

# 3. 推断当前状态
if [ -z "$PR_NUMBER" ]; then
  echo "NO_MR"
elif [ "$(echo "$PR_STATE" | jq -r '.state')" = "MERGED" ]; then
  echo "MERGED"
elif [ "$(echo "$PR_STATE" | jq -r '.reviewDecision')" = "CHANGES_REQUESTED" ]; then
  echo "BLOCKED_REVIEW"
elif [ "$(echo "$PR_STATE" | jq '.checks | length')" -gt 0 ]; then
  echo "BLOCKED_CI"
else
  echo "PENDING"
fi
```

#### 从 GitHub 恢复计数器

```bash
# 推断 fix_round: 计算 PR 上的修复 commit 数量
FIX_ROUND=$(gh pr view "$PR_NUMBER" --repo Qanora/alpha-screener --json commits --jq '[.commits[].messageHeadline | select(startswith("fix:"))] | length')


```

### 清理流程

当 MR merged 时执行清理（原子化脚本，绕过 guardrails 复合命令限制）：

```bash
bash scripts/lp-mr-cleanup.sh <N> feature/issue-<N>
```

该脚本原子化执行以下步骤：
1. 切回 master 并 `git reset --hard origin/master`
2. 删除远程残留分支（如有）
3. 删除本地 feature 分支
4. 写入 `MERGED` 状态，清理 `.fix_round` 文件

### 批量清理残留分支

定期或发现分支残留时，使用 `scripts/cleanup-merged-branches.sh`：

```bash
# Dry-run 查看待清理分支
bash scripts/cleanup-merged-branches.sh --dry-run

# 执行清理
bash scripts/cleanup-merged-branches.sh
```

## 错误处理

当 lp-dev 返回 `FAIL_DONE=<error-type>` 信号时：

| Error type            | 含义                           | 处理方式                     |
| --------------------- | ------------------------------ | ---------------------------- |
| SIMPLIFY_UNFIXABLE    | simplify 发现有问题无法修复    | 人工介入，记录到 issue       |
| CONFLICT_UNRESOLVABLE | merge conflict 无法解决        | 写 `CONFLICT`，人工介入      |
| UNKNOWN               | 其他异常                       | 记录日志，人工介入           |

处理流程：

1. 解析 `---HANDOFF---` 块中的 error type
2. 根据 error type 选择处理方式
3. 如需人工介入，在 issue 上添加 comment 说明情况