#!/bin/bash
# Close old PR and reopen with squashed 1-commit fresh start.
# Usage: ./close-reopen.sh <old_pr_number> <old_branch> [new_suffix]
set -euo pipefail

for cmd in git gh; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha-screener}"
OLD_PR="${1:?}"
OLD_BRANCH="${2:?}"

# 参数校验：OLD_PR 必须是数字，OLD_BRANCH 必须以 feature/ 开头
if ! echo "$OLD_PR" | grep -qE '^[0-9]+$'; then
  echo "ERROR: OLD_PR must be a numeric PR id, got: $OLD_PR"
  exit 2
fi
if ! echo "$OLD_BRANCH" | grep -q '^feature/'; then
  echo "ERROR: OLD_BRANCH must start with 'feature/', got: $OLD_BRANCH"
  exit 2
fi

# 检查旧分支是否存在
if ! git rev-parse --verify "$OLD_BRANCH" >/dev/null 2>&1; then
  echo "ERROR: branch '$OLD_BRANCH' does not exist"
  exit 2
fi

# 检查工作区是否干净（含 untracked 文件），避免未提交变更被 squash 丢失
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "ERROR: working tree is dirty — commit or stash changes first"
  echo "$DIRTY"
  exit 1
fi

# 版本号提取：从分支名末尾提取 v<N>，纯 bash 参数展开，不依赖 sed
if [[ "$OLD_BRANCH" =~ v([0-9]+)$ ]]; then
  CURRENT_VERSION=${BASH_REMATCH[1]}
else
  CURRENT_VERSION=0
fi
NEXT_VERSION=$((CURRENT_VERSION + 1))
SUFFIX="${3:-$NEXT_VERSION}"

if [[ "$OLD_BRANCH" =~ v[0-9]+$ ]]; then
  NEW_BRANCH="${OLD_BRANCH/%v*/v$SUFFIX}"
else
  NEW_BRANCH="${OLD_BRANCH}-v${SUFFIX}"
fi

echo "Creating $NEW_BRANCH from squashed $OLD_BRANCH..."

# 检查新分支名是否已存在
if git rev-parse --verify "$NEW_BRANCH" >/dev/null 2>&1; then
  echo "ERROR: branch '$NEW_BRANCH' already exists"
  exit 2
fi

# 检查本地 master 是否偏离 origin/master，避免误覆盖本地提交
git fetch origin master
LOCAL_MASTER=$(git rev-parse master 2>/dev/null || echo "")
REMOTE_MASTER=$(git rev-parse origin/master)
if [ -n "$LOCAL_MASTER" ] && [ "$LOCAL_MASTER" != "$REMOTE_MASTER" ]; then
  echo "WARNING: local master ($LOCAL_MASTER) differs from origin/master ($REMOTE_MASTER)"
  echo "  If errors occur, local master will be reset to origin/master during cleanup"
fi

# 保存原始 ref，以便冲突时恢复
if ORIG_REF=$(git symbolic-ref --quiet --short HEAD 2>/dev/null); then
  : # got branch name
else
  ORIG_REF=$(git rev-parse HEAD)
fi

# 先将旧分支 merge 到最新 master（不用 rebase 避免 force push 冲突）
git checkout "$OLD_BRANCH"
if ! git merge origin/master --no-edit; then
  echo "ERROR: merge conflict — resolve manually then re-run, or: git merge --abort"
  git checkout "$ORIG_REF" 2>/dev/null || true
  exit 1
fi

# 从 origin/master 创建新分支，squash merge 已 rebase 的旧分支
git checkout -b "$NEW_BRANCH" origin/master
if ! git merge --squash "$OLD_BRANCH"; then
  echo "ERROR: squash merge conflict — resolve manually then re-run"
  git merge --abort 2>/dev/null || git reset --merge 2>/dev/null || true
  git checkout "$ORIG_REF" 2>/dev/null || true
  git branch -D "$NEW_BRANCH" 2>/dev/null || true
  exit 1
fi

# 检测 squash 是否产生了变更（可能旧分支内容已完全合并）
if git diff --cached --quiet && git diff --quiet; then
  echo "ERROR: squash produced no changes — branch may already be merged"
  git checkout "$ORIG_REF" 2>/dev/null || true
  git branch -D "$NEW_BRANCH" 2>/dev/null || true
  exit 1
fi

# 恢复 guardrails hook（squash merge 可能丢失此文件）
if git checkout "$OLD_BRANCH" -- .claude/hooks/block-dangerous-git.sh 2>/dev/null; then
  echo "Restored guardrails hook from $OLD_BRANCH"
fi

COMMIT_MSG=$(git log -1 --pretty=format:'%s' "$OLD_BRANCH")
git commit -m "$COMMIT_MSG"

echo "Pushing $NEW_BRANCH..."
git push origin "$NEW_BRANCH"

# 先创建新 PR，成功后再关闭旧 PR 和删除旧分支
TITLE="$COMMIT_MSG"
echo "Creating new PR..."
NEW_PR=$(gh pr create --repo "$REPO" --title "$TITLE" --body "Replaces PR #$OLD_PR." --base master)
echo "New PR: $NEW_PR"
echo "Monitor with: bash scripts/watch-pr.sh ${NEW_PR##*/}"

echo "Closing old PR #$OLD_PR..."
gh pr close "$OLD_PR" --repo "$REPO" || echo "  (failed to close old PR — please close manually)"

echo "Deleting old remote branch $OLD_BRANCH..."
git push origin --delete "$OLD_BRANCH" 2>/dev/null || echo "  (already deleted or never pushed)"

echo "Deleting old local branch $OLD_BRANCH..."
git branch -D "$OLD_BRANCH" 2>/dev/null || echo "  (already deleted)"

echo "Done."
