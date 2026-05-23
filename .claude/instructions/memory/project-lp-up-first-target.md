---
name: lp-up-first-improvement-target
description: lp-up 首个改进目标——为 lp-mr 增加步骤检查点强制机制
metadata:
  type: project
---

lp-up 发现的首个缺陷：lp-mr 缺少步骤检查点强制机制，导致步骤 2 完成后可能跳过步骤 3（监控 MR 直到合入）。

**发现**: 执行 Issue #169 的 lp-mr 流程时，步骤 2（commit+push+MR create+auto-merge）成功后流程短路为"已完成"，跳过了步骤 3（watch-pr.sh）。

**建议 milestone**: [lp-up][IMPLEMENTATION] lp-mr 增加步骤检查点强制机制：每步完成后输出确认标记，防止流程短路。关键约束点：步骤 2→3（auto-merge 不是 MR 终点，必须 watch 到 merged）。

**关联**: [[feedback-lp-mr-step-discipline]]
