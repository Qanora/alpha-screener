# Alpha Screener 项目配置

## 三层开发飞轮

从需求到合入的全流程自动化。

| 层 | Skill | 职责 | Git 操作 |
|---|---|---|---|
| 1 | `/issue-loop` | Issue 生命周期：拆解、创建、依赖、批次、追踪 | 无 |
| 2 | `/mr-loop` | MR 全生命周期：commit、push、监控、review、修复 | 全部 git 操作 |
| 3 | `/dev-loop` | 纯本地开发：实现/修复 → 验证 → cr review | 无 |

**协作流程**：
```
issue-loop → mr-loop → dev-loop（写代码）→ mr-loop（commit+push）→ watch → dev-loop（修复）→ mr-loop（commit+push）→ merge
```

## Scripts

| 脚本 | 用途 |
|---|---|
| `scripts/watch-pr.sh <N>` | 轮询 MR 状态直到 merge |
| `scripts/fetch-review.sh <N> [--all]` | 拉取 CodeRabbit 评论（默认 fresh，`--all` 显示历史） |
| `scripts/close-reopen.sh <N> <branch>` | 关旧开新，触发重新 review |
| `scripts/commit-msg` | 校验 commit message 含 issue reference |

## Git 规范

- **commit**: 必须关联 issue（如 `#3`、`closes #3`）
- **push**: 只允许 `git push [-u] origin feature/<name>`（master/force push 被 guardrails 拦截）
- **流程**: feature 分支 → MR → squash merge
- **门禁**: CodeRabbit approve + CI 通过 → GitHub auto-merge

## Issue Tracker

GitHub Issues + `gh` CLI。详见 `/issue-loop` 附录。

## Triage Labels

`needs-triage` | `needs-info` | `ready-for-agent` | `ready-for-human` | `wontfix`