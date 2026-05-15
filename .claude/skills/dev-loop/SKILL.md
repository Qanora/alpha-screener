---
name: dev-loop
description: 第三层飞轮——纯本地开发：实现/修复 → 本地验证 → 本地 cr review。不做任何 git 或 PR 操作。
---

# Dev Loop（第三层）

纯本地开发管理。只负责写代码和验证，**不做 commit/push/PR 等任何 git 操作**（全部由第二层负责）。

## 调用方式

```text
/dev-loop <issue-number> [--fix <pr-number>]
```

## 开发模式 (无 `--fix` flag)

### 1. 获取需求

```bash
gh issue view <N> --repo Qanora/alpha-screener
```

### 2. 环境准备

```bash
git fetch origin master
WT=".claude/worktrees/issue-<N>"
if [ ! -d "$WT" ]; then
  git worktree add "$WT" origin/master
fi
cd "$WT"
BRANCH="feature/issue-<N>-<slug>"
git checkout -b "$BRANCH"
```

### 3. 实现

- 读取 `CONTEXT.md` 和 `docs/adr/` 理解领域模型
- 实现代码，遵循现有风格
- 如有测试要求，编写测试

### 4. 本地验证

```bash
ruff check . && ruff format --check .
if [ -d alphascreener ]; then mypy alphascreener/ tests/; fi
if [ -d tests ]; then python -m pytest tests/ -v; fi
prettier --check "**/*.md" && markdownlint-cli2 "**/*.md"
```

### 5. 本地 cr review（必须，阻塞步骤）

```bash
if ! command -v cr &> /dev/null; then
  echo "ERROR: cr CLI (coderabbit-cli) 未安装。请运行: npm install -g coderabbit-cli"
  exit 1
fi
cr review --agent --base origin/master
```

修复所有 Actionable comments，重复直到 **0 actionable findings**。

> cr CLI 未装时 `exit 1`，不继续。

### 6. 输出 Handoff

在 worktree 根目录写入 `.handoff-issue-<N>.md`：

```markdown
## Handoff: Issue #<N>

### 技术方案
<3-5 句话>

### 文件变更
<git diff --stat origin/master>

### branch
<BRANCH>
```

**终端输出**: `DEV_DONE=<branch>` — 通知第二层代码就绪，可以 commit + push + 创建 PR。

---

## 修复模式 (`--fix <pr-number>`)

由第二层 `/pr-loop` 调用。获取上下文 → 修复 → 验证 → cr review → 退出。**不 commit，不 push。**

### 1. 进入 worktree + 同步

```bash
cd .claude/worktrees/issue-<N>
git checkout <BRANCH>
git fetch origin master
if ! git merge origin/master --no-edit; then
  echo "ERROR: merge conflict — resolve manually"
  exit 1
fi
```

### 2. 获取评审意见

第二层已通过 `fetch-review.sh` + CI log 收集好。直接读取上下文修复。

### 3. 修复

- 逐条过所有 CodeRabbit 行内评论，全部修复
- 修复所有 CI 失败
- 不新增功能，不重构

### 4. 本地验证（同开发模式步骤 4）

### 5. 本地 cr review（同开发模式步骤 5，必须）

```bash
cr review --agent --base origin/master
```

修复所有新发现的 comments，重复直到 0 findings。

### 6. 输出修复摘要

```text
## 修复摘要: PR #<pr-number>
- [x] <评论1简述>
- [x] <评论2简述>
```

终端输出 `FIX_DONE=<branch>` — 通知第二层可以 commit + push。

---

## 约束

- 始终在 worktree 内工作
- **不做任何 git 操作**：不 add、不 commit、不 push（全部由第二层负责）
- 只做本地开发：写代码 + 验证 + cr review
- 修复模式只修问题，不新增功能
- 不修改 `.claude/` 配置
- 步骤 5（cr review）是阻塞步骤
