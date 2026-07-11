# Alpha Screener 项目约定

## 1) 你现在面对的这个仓库是谁

这是业务仓库，不是飞轮能力仓库。

- `/fwp-*` 功能由独立 skills 仓库提供。
- 本仓库只维护：业务代码、配置、测试和交付产物。

## 2) 飞轮能力绑定（每次会话先确认）

首次或切换环境后执行：

```bash
cd /root/workspace/alpha-screener
FLYWHEEL_SOURCE_DIR=/root/workspace/skills bash install.sh
```

- 若已有其他路径，改成你的实际路径。
- 命令执行后建议重开一次 Codex/Claude 会话。

## 3) 本仓库内开发动作

- 需求拆解：`/fwp-plan`
- Bug 复现与交接：`/fwp-debug`
- 巡检：`/fwp-inspect`
- 风险检查：`/fwp-audit`

- 快速入口：
  - `./scripts/dev-plan.sh`
  - `./scripts/dev-inspect.sh`
  - `./scripts/asc.sh`

## 4) Git 与 PR 规则

- 分支：`feature/issue-<N>`（从 `origin/master` 创建）
- 提交必须绑定 issue：`#N`、`closes #N`、`fixes #N`
- PR 合并前：`./scripts/dev-inspect.sh`

## 5) 变更优先级

- 默认以 [CODEX.md](/root/workspace/alpha-screener/CODEX.md) 约定为主。
- 对应能力/执行入口以 `install.sh` 和 `skills/project-meta` 为单一真实来源。
