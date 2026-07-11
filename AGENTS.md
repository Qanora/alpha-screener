# Alpha Screener AI Coding Guide

## 项目

Alpha Screener 是一个面向美股的命令行筛选工具，唯一目标是找出未来 14 日内最可能出现爆发性增长的标的。

- 对用户的唯一交付接口是 `asc`。运行它应自动分析近 60 日可获得的数据，并输出未来 14 日最有可能爆发性增长的美股候选标的。
- 方法不受限：可使用数据、因子、模型、筛选规则或其他手段；实现决策以提升这一结果为准，而不是维持某种既定技术路线。
- Python 3.11+；依赖与工具配置以 `pyproject.toml` 为准。业务代码在 `alphascreener/`，测试在 `tests/`。
- 不要猜测业务规则：先阅读相关模块、测试和配置；对外部数据与密钥保持谨慎，勿提交 `.env`。

## 开发方式

- 先理解现有实现和测试，再做最小、聚焦的改动；避免顺手重构无关代码。
- 为行为变更补充或更新测试。优先运行受影响的测试，再运行完整测试集。
- 仓库使用 `.venv`；如尚未创建，可执行 `uv sync --extra dev`。
- 常用命令：
  - `ruff check alphascreener tests`
  - `pytest`
  - `./scripts/dev-inspect.sh`：快速巡检
  - `./scripts/dev-plan.sh`：完整本地验证
  - `./scripts/asc.sh --help`：CLI smoke test

## 交付约定

- 默认分支为 `master`；功能分支命名为 `feature/issue-<N>`，并从 `origin/master` 创建。
- 提交和 PR 必须关联 Issue，例如 `closes #<N>`；提交前运行与改动相称的验证。
- PR 描述应说明变更范围、影响、回滚方式和验证结果。
- 工作区可能包含他人未提交的改动；只修改本任务涉及的文件，不覆盖或清理无关变更。
- `install.sh`、项目脚本和当前配置是运行方式的事实来源；文档与它们冲突时，以实际实现为准并同步修正文档。

<!-- flywheel:begin -->
## Flywheel 开发脚手架

本仓库使用本机安装的 Flywheel skills，兼容 Claude Code 与 Codex。优先按以下入口处理工作：

- 新需求或改进：`/fwp-plan <需求>`
- 缺陷：`/fwp-debug <问题>`
- 质量检查：`/fwp-inspect`
- 恢复中断交付：`/fwp-resume`

`fwp-ship` 负责分支、提交、PR 与 CI；`fwp-build` 只负责实现和本地验证。可委托时使用下一层 skill；否则在当前会话按相同职责边界顺序执行。

GitHub Issue/PR 是交付真值，`/tmp/fw-flywheel/<owner_repo>/` 仅是可重建的临时交接目录。创建 Issue、分支、PR、push 或 merge 前必须遵守用户授权和本仓库现有规则。
<!-- flywheel:end -->
