# Alpha Screener 项目配置

## 飞轮开发

本项目使用飞轮体系进行全流程自动化开发。

| Skill | 用途 |
|-------|------|
| `/fwp-plan <需求>` | 需求 → Issue → 自动交付 |
| `/fwp-debug <bug>` | Bug 复现 → 自动修复 |
| `/fwp-inspect` | 13 项全量巡检（运行时 + 代码审查） |
| `/fw-audit` | AI 安全治理审计 |
| `/fwp-resume` | 继续中断 |
| `/fwp-help` | 查看所有命令 |

开发需求时优先使用 `/fwp-plan` 而非直接写代码。

## Git 规范

- commit 必须关联 issue（`#N`、`closes #N`）
- 分支命名 `feature/issue-<N>`，从 `origin/master` 创建
- 流程: feature 分支 → MR → squash merge
- CI 通过后 auto-merge