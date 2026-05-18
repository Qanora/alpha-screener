---
name: lp-dev
description: 第三层飞轮——纯本地开发：实现/修复 → 本地验证 → simplify。不做任何 git 或 MR 操作。
---

# LP-DEV（第三层）

纯本地开发管理。只负责写代码和验证，**不做 commit/push/MR 等任何 git 操作**（全部由第二层负责）。

**顺序开发模式**：一次只处理一个 issue，在主仓库内直接开发，不使用 worktree。

## 调用方式

```text
/lp-dev <issue-number> [--fix <mr-number>]
```

## 开发模式 (无 `--fix` flag)

### 1. 获取需求

```bash
gh issue view <N> --repo Qanora/alpha-screener
```

### 2. 强制同步 master（阻塞步骤）

**必须先更新本地 master，确保基于最新代码开发**：

```bash
cd /root/workspace/alpha-screener
git fetch origin master
git checkout master
git reset --hard origin/master
# 验证同步成功
git log -1 --oneline origin/master
```

**验证点**：本地 master 的 HEAD 必须等于 origin/master 的 HEAD。

### 3. 检查分支冲突（阻塞步骤）

**检查目标分支是否已存在**：

```bash
BRANCH="feature/issue-<N>"
# 检查本地
if git branch | grep -q "$BRANCH"; then
  echo "ERROR: 本地已存在分支 $BRANCH"
  echo "请先删除旧分支: git branch -D $BRANCH"
  exit 1
fi
# 检查远程
if git branch -r | grep -q "origin/$BRANCH"; then
  echo "ERROR: 远程已存在分支 $BRANCH"
  echo "请先删除远程分支: gh api repos/Qanora/alpha-screener/git/refs/heads/$BRANCH -X DELETE"
  exit 1
fi
```

若任何检查失败，输出错误并退出，等待人工处理。

### 4. 创建新分支

**从最新的 master 创建全新分支**：

```bash
BRANCH="feature/issue-<N>"
git checkout -b "$BRANCH"
# 验证分支起点
git log -1 --oneline
```

分支起点必须是当前 master 的 HEAD。

### 5. 实现

- 实现代码，遵循现有风格
- 如有测试要求，编写测试

### 6. 300 行约束检查

```bash
git diff --shortstat origin/master
```

若改动超过 300 行，输出警告（soft constraint，不阻塞）：

```text
⚠️ 当前改动超过 300 行，建议考虑拆分为多个 issue
```

### 7. 本地验证

```bash
ruff check . && ruff format --check .
prettier --check "**/*.md" && markdownlint-cli2 "**/*.md"
```

### 8. Simplify（启动新 agent）

启动一个新的 claude agent 执行 simplify skill：

```
Agent(subagent_type="claude", prompt="执行 /simplify 对当前改动进行代码审查")
```

修复所有发现的问题，重复直到 simplify 返回无问题。

### 9. 输出 Handoff

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

## 修复模式 (`--fix <mr-number>)

由第二层 `/lp-mr` 调用。获取上下文 → 修复 → 验证 → simplify → 退出。**不 commit，不 push。**

### 1. 切换到已有分支 + 同步 master

```bash
BRANCH="feature/issue-<N>"
git checkout "$BRANCH"
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

### 4. 本地验证（同开发模式步骤 7）

### 5. Simplify（启动新 agent，同开发模式步骤 8）

启动一个新的 claude agent 执行 simplify skill，修复所有发现的问题。

### 6. 输出修复摘要

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

- **顺序开发**：一次只处理一个 issue，在主仓库直接开发
- **不做任何 git 操作**：不 add、不 commit、不 push（全部由第二层负责）
- 只做本地开发：写代码 + 验证 + simplify
- 修复模式只修问题，不新增功能
- 步骤 8（simplify）是阻塞步骤
- **强制同步 master**：开发前必须 `git reset --hard origin/master`，确保基于最新代码
- **禁止复用分支**：每次开发必须从 master 创建全新 feature 分支，若已存在同名分支则报错退出