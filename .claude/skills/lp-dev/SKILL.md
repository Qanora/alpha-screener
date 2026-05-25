---
name: lp-dev
description: 第四层飞轮——纯本地开发：实现/修复 → 本地验证 → simplify。不做任何 git 或 MR 操作。
---

# LP-DEV（第四层）

纯本地开发管理。只负责写代码和验证，**不做 commit/push/MR 等任何 git 操作**（全部由第三层负责）。

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

### 3. 检查分支冲突（阻塞步骤，自动清理）

**检查目标分支是否已存在，若存在则自动删除**：

```bash
BRANCH="feature/issue-<N>"
# 自动清理本地旧分支
if git branch | grep -q "$BRANCH"; then
  git branch -D "$BRANCH"
fi
# 自动清理远程旧分支
if git branch -r | grep -q "origin/$BRANCH"; then
  gh api repos/Qanora/alpha-screener/git/refs/heads/$BRANCH -X DELETE
fi
```

### 4. 创建新分支

**从最新的 master 创建全新分支**：

```bash
BRANCH="feature/issue-<N>"
git checkout -b "$BRANCH"
# 验证分支起点
git log -1 --oneline
```

分支起点必须是当前 master 的 HEAD。

### 5. TDD 实现（红 → 绿循环）

严格遵循 TDD 模式：

**5.1 写失败的测试（RED）**：
- 根据 issue 需求，先编写测试用例
- 测试需覆盖正常路径、边界条件、异常情况
- 运行 `python -m pytest tests/ -v -k "<test_name>"` 验证测试失败

**5.2 写实现（GREEN）**：
- 编写最小实现代码使测试通过
- 运行 `python -m pytest tests/ -v` 全量验证所有测试通过
- 不通过则继续修复，直到全绿

**5.3 重构（REFACTOR）**：
- 消除重复、改善可读性，保持测试全绿

### 6. 300 行约束检查

```bash
git diff --shortstat origin/master
```

若改动超过 300 行，输出警告（soft constraint，不阻塞）：

```text
⚠️ 当前改动超过 300 行，建议考虑拆分为多个 issue
```

### 7. 本地验证（全部阻塞步骤）

必须全部通过才能继续：

```bash
# 代码质量
ruff check . && ruff format --check .

# 全量测试（阻塞 — 失败则修复后重试）
python -m pytest tests/ -v
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

由第三层 `/lp-mr` 调用。获取上下文 → 修复 → 验证 → simplify → 退出。**不 commit，不 push。**

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

第三层已通过 CI log 收集好。直接读取上下文修复。

### 3. 修复

- 修复所有 CI 失败
- 不新增功能，不重构
- 每条修复先更新对应测试（保证测试覆盖修复点），再修改代码

### 4. 本地验证（同开发模式步骤 7，全部阻塞）

### 5. Simplify（同开发模式步骤 8）

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
- **不做任何 git 操作**：不 add、不 commit、不 push（全部由第三层负责）
- 只做本地开发：写代码 + 验证 + simplify
- 修复模式只修问题，不新增功能
- **TDD 开发**：严格先写测试 → 测试失败 → 实现 → 测试通过 → 重构
- **全量测试阻塞**：`pytest -v` 必须全部通过才能输出 HANDOFF
- **Simplify 阻塞**：simplify 发现的问题必须全部修复
- **强制同步 master**：开发前必须同步到最新 origin/master
- **自动清理分支**：同名旧分支自动删除，不询问
- **零人工中断**：遇到可自动处理的问题直接处理（分支冲突、测试失败重试等），只有真正需要人类决策的才停止