# Alpha Screener Codex 执行约定

## 1) 约定优先级

- 先遵循 [CLAUDE.md](/root/workspace/alpha-screener/CLAUDE.md) 的仓库运行约定。
- 本文件聚焦 Codex/Claude 的开发执行规则。

## 2) 本仓库交互入口

- 需求拆解与排期：`/fwp-plan`
- Bug 报告与复现：`/fwp-debug`
- 一次性巡检：`/fwp-inspect`
- 风险检查：`/fwp-audit`

项目本地快速入口：
- `./scripts/dev-plan.sh`
- `./scripts/dev-inspect.sh`
- `./scripts/asc.sh`

## 3) 飞轮能力绑定（必做）

本仓库不包含飞轮 skill 实现。首次/重建环境请先执行：

```bash
cd /root/workspace/alpha-screener
FLYWHEEL_SOURCE_DIR=/root/workspace/skills bash install.sh
```

执行后重开 Codex/Claude 会话以刷新技能发现。

## 4) 分支与提交规范

- 分支：`feature/issue-<N>`（从 `origin/master` 创建）
- Commit 必须带 issue 关联：`#N` / `closes #N` / `fixes #N`

## 5) PR 前最小验收（全部通过后提交）

- `ruff check alphascreener tests`
- `pytest`
- `./scripts/dev-inspect.sh`
- PR 描述包含：变更范围、影响评估、回滚方案、验收方式

## 6) 失败与交接规则

- 每一步失败必须输出可复现的证据（日志/命令/输出路径）。
- 变更提交前必须保留问题上下文与回归验证记录。
