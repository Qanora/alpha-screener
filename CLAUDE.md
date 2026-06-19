# Alpha Screener 项目配置

## 五层开发飞轮

从需求到合入的全流程自动化。基于用户级 `fwp-*` / `fw-*` skill。

| 层  | Skill            | 职责                                            | Git 操作      |
| --- | ---------------- | ----------------------------------------------- | ------------- |
| 0a  | `fwp-inspect`    | 引擎观察：执行+分析运行时数据，发现引擎缺陷       | 无            |
| 0b  | `fw-audit`       | 飞轮自检：分析飞轮执行上下文，发现流程偏差/冗余   | 无            |
| 1   | `fwp-plan`       | Issue 生命周期：拆解、创建、依赖、批次、追踪    | 无            |
| 2   | `fwp-ship`       | MR 全生命周期：commit、push、监控、修复          | 全部 git 操作 |
| 3   | `fwp-build`      | 纯本地开发：实现/修复 → 验证 → simplify         | 无            |

**协作流程**：

```
fwp-inspect（引擎观察）  fw-audit（飞轮自检）
         ↘                ↙
         fwp-plan（需求拆解 → issue）
           ↓
         fwp-ship（MR 生命周期）
           ↓
         fwp-build（写代码）
           ↓ merge
         fwp-inspect（再执行 → 验证修复）
         fw-audit（再审计 → 优化飞轮）
```

## 辅助 Skill

| Skill | 用途 |
|-------|------|
| `fwp-debug` | Bug 修复入口：复现→收集证据→创建 issue→派发 |
| `fwp-resume` | 恢复中断：自动检测未完成的 milestone/issue，继续飞轮执行 |
| `fwp-setup` | 项目初始化：创建仓库、CI/CD、分支保护、模板、标签 |
| `diagnose` | 深度诊断：Reproduce→Minimise→Hypothesise→Instrument→Fix |
| `code-review` | AI 代码审查（CodeRabbit） |
| `security-review` | 安全审查：当前分支待处理更改 |
| `triage` | Issue 分诊：状态机驱动的 triage 流程 |
| `handoff` | 压缩当前会话为 handoff 文档供下一个 agent 恢复 |

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
- **清理**: MR merged 后 fwp-ship 自动删除本地分支；远程分支由 auto-merge 删除

## Issue Tracker

GitHub Issues + `gh` CLI。详见 `fwp-plan`。

## Triage Labels

`needs-triage` | `needs-info` | `ready-for-agent` | `ready-for-human` | `wontfix`