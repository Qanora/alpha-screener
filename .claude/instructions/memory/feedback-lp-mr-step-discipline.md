---
name: feedback-lp-mr-step-discipline
description: lp-mr 步骤执行纪律——执行完步骤 2 后必须显式推进到步骤 3
metadata:
  type: feedback
---

lp-mr 每个步骤结束后必须显式输出检查点（"步骤 X 完成，进入步骤 Y"），不得依赖隐式假设跳过步骤。

**Why**: Issue #169 — lp-mr 在步骤 2（commit+push+MR+auto-merge）成功后直接停止，跳过了步骤 3（watch-pr.sh 监控）。根因是 auto-merge 启用后产生了"已完成"的错觉，流程短路。

**How to apply**: 执行 lp-mr 时，每完成一个步骤后，必须输出该步骤的完成确认才能进入下一步。尤其是步骤 2→3 过渡绝不能跳过——即使 auto-merge 已启用，也必须启动 watch-pr.sh 并等待 merge 完成。
