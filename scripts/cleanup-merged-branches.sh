#!/bin/bash
# 清理已合并但残留的 feature 分支
# 用法: ./cleanup-merged-branches.sh [--dry-run]
set -euo pipefail

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=true
  echo "=== DRY RUN - 不执行实际删除 ==="
fi

REPO="${REPO:-Qanora/alpha-screener}"

# 获取已合并 PR 的分支列表
echo "Fetching merged PR branches..."
MERGED_BRANCHES=$(gh pr list --state merged --limit 1000 --json headRefName --jq '.[].headRefName' | sort -u)

# 获取当前本地 feature 分支（去除前导空格和 *）
LOCAL_BRANCHES=$(git branch | grep -E "^[\* ]+feature/" | sed 's/^[\* ]*//' | sort || true)
if [ -z "$LOCAL_BRANCHES" ]; then
  LOCAL_BRANCHES=""  # 确保空值时不会报错
fi

# 获取当前远程 feature 分支（刷新后）
git fetch --prune
REMOTE_BRANCHES=$(git branch -r | grep "origin/feature/" | sed 's/.*origin\///' | sort || true)
if [ -z "$REMOTE_BRANCHES" ]; then
  REMOTE_BRANCHES=""
fi

echo ""
echo "=== 已合并的 PR 分支 (来自 GitHub) ==="
echo "$MERGED_BRANCHES" | tr '\n' ' '
echo ""

echo ""
echo "=== 本地残留的 feature 分支 ==="
echo "$LOCAL_BRANCHES" | tr '\n' ' '
echo ""

echo ""
echo "=== 远程残留的 feature 分支 ==="
echo "$REMOTE_BRANCHES" | tr '\n' ' '
echo ""

# 计算需要清理的分支
if [ -n "$LOCAL_BRANCHES" ]; then
  LOCAL_TO_CLEAN=$(comm -12 <(echo "$MERGED_BRANCHES") <(echo "$LOCAL_BRANCHES"))
else
  LOCAL_TO_CLEAN=""
fi
if [ -n "$REMOTE_BRANCHES" ]; then
  REMOTE_TO_CLEAN=$(comm -12 <(echo "$MERGED_BRANCHES") <(echo "$REMOTE_BRANCHES"))
else
  REMOTE_TO_CLEAN=""
fi

echo ""
echo "=== 需要清理的本地分支 (已合并但未删除) ==="
if [ -n "$LOCAL_TO_CLEAN" ]; then
  echo "$LOCAL_TO_CLEAN"
  COUNT=$(echo "$LOCAL_TO_CLEAN" | wc -l | tr -d ' ')
  echo "共 $COUNT 个"
else
  echo "无"
fi

echo ""
echo "=== 需要清理的远程分支 (已合并但未删除) ==="
if [ -n "$REMOTE_TO_CLEAN" ]; then
  echo "$REMOTE_TO_CLEAN"
  COUNT=$(echo "$REMOTE_TO_CLEAN" | wc -l | tr -d ' ')
  echo "共 $COUNT 个"
else
  echo "无"
fi

if $DRY_RUN; then
  echo ""
  echo "=== DRY RUN 完成 - 运行不带 --dry-run 参数执行实际删除 ==="
  exit 0
fi

# 执行清理
echo ""
echo "=== 开始清理 ==="

if [ -n "$LOCAL_TO_CLEAN" ]; then
  echo "$LOCAL_TO_CLEAN" | while read -r branch; do
    echo "删除本地分支: $branch"
    git branch -D "$branch" 2>/dev/null || echo "  跳过 (不存在或已删除)"
  done
else
  echo "无本地分支需要清理"
fi

if [ -n "$REMOTE_TO_CLEAN" ]; then
  echo "$REMOTE_TO_CLEAN" | while read -r branch; do
    echo "删除远程分支: origin/$branch"
    gh api repos/Qanora/alpha-screener/git/refs/heads/"$branch" -X DELETE 2>/dev/null || echo "  跳过 (不存在或已删除)"
  done
else
  echo "无远程分支需要清理"
fi

echo ""
echo "=== 清理完成 ==="
echo "建议执行: git fetch --prune && git branch -a | grep feature/"