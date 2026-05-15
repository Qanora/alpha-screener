## Agent skills

### Development flywheel

两层自动化开发飞轮，从需求到合入全流程驱动。

- **`/dev-flywheel`** — 外层 loop：需求分析 → 拆解 issue → 依赖分组 → 批次并发派活 → 追踪交付
- **`/issue-dev`** — 内层 loop：单个 issue 的开发 + CI/CodeRabbit 修复循环

### Issue tracker

Issues live as GitHub Issues on Qanora/alpha-screener, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

All five triage roles use the default label names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Workflow

- 每个 commit 必须在 message 中关联 issue (e.g. `#3`, `closes #3`)
- `git push` 只允许 `git push [-u] origin feature/<name>`（master/force push 被 guardrails 拦截）
- 所有代码变更走 feature 分支 → PR → squash merge 流程
- 合入门禁: CodeRabbit approve + CI 全部通过
- GitHub 原生 auto-merge 在满足门禁后自动 squash merge

## Scripts

| 脚本                                           | 用途                                                      |
| ---------------------------------------------- | --------------------------------------------------------- |
| `scripts/watch-pr.sh <N>`                      | 纯状态监控，轮询 PR 的 review + CI 状态直到 merge         |
| `scripts/fetch-review.sh <N> [--all]`          | 拉取所有 CodeRabbit 评论，按 review_id 分组，默认最新一轮 |
| `scripts/close-reopen.sh <old-N> <old-branch>` | 关旧开新，触发 CodeRabbit 重新 review                     |
| `scripts/commit-msg`                           | 校验 commit message 含 issue reference                    |

## Git 约束

- commit 必须关联 `#N`
- `git push` 只允许 `git push [-u] origin feature/<name>`
- feature 分支 → PR → squash merge
- 禁止 force push；不要 `&&` 连接 push 和建 PR
