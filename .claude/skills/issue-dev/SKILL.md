---
name: issue-dev
description: 单个 issue 的完整开发→PR→修复循环。支持开发模式和修复模式。
---

# Issue Dev

内层 loop：单 issue 从开发到合入的完整生命周期。

## 调用方式

```
/issue-dev <issue-number> [--fix <pr-number>]
```

## 开发模式 (无 `--fix` flag)

### 1. 获取需求

```bash
gh issue view <N> --repo Qanora/alpha-screener
```

理解 issue 的验收标准（acceptance criteria）。

### 2. 环境准备

```bash
# 同步 master
git fetch origin master

# 创建 worktree（如果还不存在）
WT=".claude/worktrees/issue-<N>"
if [ ! -d "$WT" ]; then
  git worktree add "$WT" origin/master
fi
cd "$WT"

# 创建 feature 分支
BRANCH="feature/issue-<N>-<slug>"
git checkout -b "$BRANCH"
```

### 3. 实现

- 读取项目根目录的 `CONTEXT.md` 和 `docs/adr/` 理解领域模型
- 实现代码，遵循现有代码风格
- 如有测试要求，编写测试

### 4. 本地验证

```bash
ruff check . && ruff format --check .
if [ -d alphascreener ]; then mypy alphascreener/ tests/; fi
if [ -d tests ]; then python -m pytest tests/ -v; fi
```

修复所有 lint / type / test 错误后再继续。

### 4.5 本地 CodeRabbit 评审

```bash
# 本地触发 CodeRabbit review，提前发现问题
cr review --agent --base origin/master
```

- 查看输出，修复所有 Actionable comments
- 重复运行 `cr review ...` 直到 0 actionable findings
- **只有在本地 cr review 干净后才 push + 创建 PR**

> 注意: cr CLI 需要 `coderabbit-cli` 已安装。若不可用则跳过此步，走 PR 后 fetch-review 修复循环。

### 5. 提交 + 推送 + 创建 PR

```bash
git add <files>
git commit -m "<type>: <description> (#<N>)"
# 注意: push 不要和 && 链接，分开执行
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

### 6. 输出 Handoff

在 worktree 根目录写入 `.handoff-issue-<N>.md`:

```markdown
## Handoff: Issue #<N>

### 技术方案

<3-5 句话：做了什么、为什么这个方案、关键取舍>

### 文件变更

<git diff --stat origin/master 的输出>

### PR

<PR URL>

### 已知限制

<如有非本次 scope 的限制，列出>
```

终端输出必须包含 PR URL，格式: `PR_URL=<url>`，方便主控解析。

---

## 修复模式 (`--fix <pr-number>`)

修复循环：**同一 PR 上反复 push → CodeRabbit 自动 re-review（auto_pause: 10）→ 直到 approve。最多 6 轮，超过才 close-reopen。**

### 1. 进入 worktree

```bash
cd .claude/worktrees/issue-<N>
git fetch origin master
```

### 2. 同步分支（使用 merge 避免 force push 冲突）

```bash
git checkout <BRANCH>
git fetch origin master
if ! git merge origin/master --no-edit; then
  echo "ERROR: merge conflict — resolve manually before continuing"
  exit 1
fi
```

### 3. 获取本轮评审意见

```bash
bash scripts/fetch-review.sh <pr-number>
gh pr view <pr-number> --repo Qanora/alpha-screener --json statusCheckRollup --jq '
  [.statusCheckRollup[] | select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT"))] |
  .[] | "\(.name): \(.conclusion)"
'
```

### 4. 修复

- 逐条过所有 CodeRabbit 行内评论，全部修复
- 修复所有 CI 失败
- 不改代码之外的逻辑（不新增功能，不重构）

### 5. 提交 + 推送到**同一分支**

```bash
git add -A
git commit -m "fix: address review findings (#<N>)"
# 推送到同一个 PR 的分支，CodeRabbit 自动 re-review（auto_pause: 10）
git push origin <BRANCH>
```

### 6. 等待 re-review

push 后 CodeRabbit 会自动重新 review（因为 `auto_pause_after_reviewed_commits: 10`）。运行 `bash scripts/watch-pr.sh <pr-number>` 等待结果：
- **APPROVED + CI 绿** → 等待 GitHub 自动 merge → 完成
- **CHANGES_REQUESTED** → 回到步骤 1 继续下一轮修复
- **CI 失败** → 修复后 push 同一分支

### 7. 超过 6 轮仍未 approve

如果同一 PR 上修复超过 **6 轮**（即 push 了 6 次修复 commit）CodeRabbit 仍未 approve：

```bash
bash scripts/close-reopen.sh <pr-number> <BRANCH>
```

这会创建全新 PR，触发一次干净的 review。然后回到步骤 3 继续循环。

### 8. 输出修复摘要

```
## 修复摘要: PR #<pr-number> (第 <N> 轮)
- [x] <评论1简述>
- [x] <评论2简述>
- CI: <pass/fail 状态>
```

---

## 约束

- 始终在 worktree 内工作
- commit message 必须包含 `#<N>` issue reference
- push 后不自动 merge，由外层 loop 或用户处理
- 修复模式只修 issue，不新增功能
- 不修改 `.claude/` 配置文件
