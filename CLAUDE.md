# Alpha Screener 项目配置

## 五层开发飞轮

从需求到合入的全流程自动化。

| 层  | Skill     | 职责                                            | Git 操作      |
| --- | --------- | ----------------------------------------------- | ------------- |
| 0a  | `/lp-up`  | 引擎观察：执行+分析运行时数据，发现引擎缺陷       | 无            |
| 0b  | `/lp-dp`  | 飞轮自检：分析飞轮执行上下文，发现流程偏差/冗余   | 无            |
| 1   | `/lp-ms`  | Issue 生命周期：拆解、创建、依赖、批次、追踪    | 无            |
| 2   | `/lp-mr`  | MR 全生命周期：commit、push、监控、修复          | 全部 git 操作 |
| 3   | `/lp-dev` | 纯本地开发：实现/修复 → 验证 → simplify         | 无            |

**协作流程**：

```
lp-up（引擎观察）  lp-dp（飞轮自检）
         ↘          ↙
         lp-ms（需求拆解 → issue）
           ↓
         lp-mr（MR 生命周期）
           ↓
         lp-dev（写代码）
           ↓ merge
         lp-up（再执行 → 验证修复）
         lp-dp（再审计 → 优化飞轮）
```

## Scripts

| 脚本                                   | 用途                                                 |
| -------------------------------------- | ---------------------------------------------------- |
| `scripts/watch-pr.sh <N>`            | 轮询 MR CI 状态直到 merge                        |
| `scripts/commit-msg`                 | 校验 commit message 含 issue reference            |
| `scripts/cleanup-merged-branches.sh` | 清理已合并但残留的 feature 分支                   |

## Git 规范

- **commit**: 必须关联 issue（如 `#3`、`closes #3`）
- **push**: 只允许 `git push [-u] origin feature/<name>`（master/force push 被 guardrails 拦截）
- **流程**: feature 分支 → MR → squash merge（auto-merge 自动删除远程分支）
- **门禁**: CI 通过 → GitHub auto-merge
- **清理**: MR merged 后 lp-mr 自动删除本地分支；远程分支由 auto-merge 删除

## Issue Tracker

GitHub Issues + `gh` CLI。详见 `/lp-ms` 附录。

## Triage Labels

`needs-triage` | `needs-info` | `ready-for-agent` | `ready-for-human` | `wontfix`