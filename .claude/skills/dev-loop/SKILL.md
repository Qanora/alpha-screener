---
name: dev-loop
description: 第三层飞轮——纯本地开发：实现/修复 → 本地验证 → simplify。不做任何 git 或 MR 操作。
---

# Dev Loop（第三层）

纯本地开发管理。只负责写代码和验证，**不做 commit/push/MR 等任何 git 操作**（全部由第二层负责）。

## 调用方式

```text
/dev-loop <issue-number> [--fix <mr-number>]
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
BRANCH="feature/issue-<N>"
git checkout -b "$BRANCH"
```

### 3. 实现

- 实现代码，遵循现有风格
- 如有测试要求，编写测试

### 4. 300 行约束检查

```bash
git diff --shortstat origin/master
```

若改动超过 300 行，输出警告（soft constraint，不阻塞）：

```text
⚠️ 当前改动超过 300 行，建议考虑拆分为多个 issue
```

### 5. 本地验证

```bash
ruff check . && ruff format --check .
prettier --check "**/*.md" && markdownlint-cli2 "**/*.md"
```

### 6. Simplify（启动新 agent）

启动一个新的 claude agent 执行 simplify skill：

```
Agent(subagent_type="claude", prompt="执行 /simplify 对当前改动进行代码审查")
```

修复所有发现的问题，重复直到 simplify 返回无问题。

### 7. 输出 Handoff

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

**终端输出**（标准化信号格式）：

成功时：

```text
---HANDOFF---
DEV_DONE=<BRANCH>
---HANDOFF_END---
```

失败时：

```text
---HANDOFF---
FAIL_DONE=<error-type>
---HANDOFF_END---
```

Error types：

| Error type            | 含义                           |
| --------------------- | ------------------------------ |
| SIMPLIFY_UNFIXABLE    | simplify 发现有问题无法修复    |
| CONFLICT_UNRESOLVABLE | merge conflict 无法解决        |
| UNKNOWN               | 其他异常                       |

---

## 修复模式 (`--fix <mr-number>`)

由第二层 `/mr-loop` 调用。获取上下文 → 修复 → 验证 → simplify → 退出。**不 commit，不 push。**

### 1. 进入 worktree + 同步

```bash
cd .claude/worktrees/issue-<N>
git checkout <BRANCH>
git fetch origin master
git merge origin/master --no-edit
```

若 merge 成功，继续步骤 2。

**若 merge 失败（冲突）**，自动解决：

1. 查看冲突文件：

   ```bash
   git status --porcelain | grep "^UU\|^AA\|^DD"
   ```

2. 逐个 Read 冲突文件，识别 `<<<<<<<`, `=======`, `>>>>>>>` 标记
3. 使用 Edit 解决冲突（保留正确的代码片段，移除冲突标记）
4. 冲突全部解决后：

   ```bash
   git add .
   git merge --continue
   ```

5. 若无法解决冲突，输出 `FAIL_DONE=CONFLICT_UNRESOLVABLE` 并退出

### 2. 获取评审意见

第二层已通过 `fetch-review.sh` + CI log 收集好。直接读取上下文修复。

### 3. 修复

- 逐条过所有 CodeRabbit 行内评论，全部修复
- 修复所有 CI 失败
- 不新增功能，不重构

### 4. 本地验证（同开发模式步骤 5）

### 5. Simplify（启动新 agent，同开发模式步骤 6）

启动一个新的 claude agent 执行 simplify skill，修复所有发现的问题。

### 6. 输出修复摘要

```text
## 修复摘要: MR #<mr-number>
- [x] <评论1简述>
- [x] <评论2简述>
```

**终端输出**（标准化信号格式）：

成功时：

```text
---HANDOFF---
FIX_DONE=<BRANCH>
---HANDOFF_END---
```

失败时：

```text
---HANDOFF---
FAIL_DONE=<error-type>
---HANDOFF_END---
```

Error types 同开发模式。

---

## 约束

- 始终在 worktree 内工作
- **不做任何 git 操作**：不 add、不 commit、不 push（全部由第二层负责）
- 只做本地开发：写代码 + 验证 + simplify
- 修复模式只修问题，不新增功能
- 步骤 6（simplify）是阻塞步骤
