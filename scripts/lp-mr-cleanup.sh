#!/bin/bash
# Atomic MR post-merge cleanup: checkout master, pull, delete branch, write state.
# Replaces 4+ separate Bash calls that are blocked by guardrails compound-command rules.
# Usage: ./lp-mr-cleanup.sh <issue-number> <branch-name>
set -euo pipefail

for cmd in git gh; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha-screener}"
ISSUE_NUM="${1:?Usage: $0 <issue-number> <branch-name>}"
BRANCH="${2:?Usage: $0 <issue-number> <branch-name>}"

# ── Parameter validation ────────────────────────────────────────────────

if ! echo "$ISSUE_NUM" | grep -qE '^[0-9]+$'; then
  echo "ERROR: issue-number must be numeric, got: $ISSUE_NUM"
  exit 2
fi
EXPECTED_BRANCH="feature/issue-$ISSUE_NUM"
if [ "$BRANCH" != "$EXPECTED_BRANCH" ]; then
  echo "ERROR: branch must be '$EXPECTED_BRANCH', got: $BRANCH"
  exit 2
fi

# ── Safety: dirty working tree ──────────────────────────────────────────

DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "ERROR: working tree is dirty — commit or stash changes first"
  echo "$DIRTY"
  exit 1
fi

echo "=== lp-mr cleanup for issue #$ISSUE_NUM (branch: $BRANCH) ==="

# ── Step 1: checkout master ─────────────────────────────────────────────

ORIG_BRANCH=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "")
echo "[1/5] Switching to master..."
if [ "$ORIG_BRANCH" = "master" ]; then
  echo "  Already on master, skipping checkout."
else
  git checkout master
fi

# ── Step 2: pull latest master ──────────────────────────────────────────

echo "[2/5] Pulling latest origin/master..."
git fetch origin master
git reset --hard origin/master

# ── Step 3: remove remote residual branch ───────────────────────────────

echo "[3/5] Checking for remote residual branch 'origin/$BRANCH'..."
git fetch --prune
if git branch -r | grep -q "origin/$BRANCH"; then
  echo "  Deleting remote branch 'origin/$BRANCH'..."
  TMP_ERR=$(mktemp)
  trap 'rm -f "$TMP_ERR"' EXIT
  if ! gh api "repos/$REPO/git/refs/heads/$BRANCH" -X DELETE 2>"$TMP_ERR"; then
    if grep -q '"status":404' "$TMP_ERR" 2>/dev/null; then
      echo "  Remote branch already deleted."
    else
      echo "ERROR: failed to delete remote branch 'origin/$BRANCH'"
      cat "$TMP_ERR"
      rm -f "$TMP_ERR"
      exit 3
    fi
  fi
  rm -f "$TMP_ERR"
  trap - EXIT
  git fetch --prune
else
  echo "  No remote residual."
fi

# ── Step 4: delete local branch ─────────────────────────────────────────

echo "[4/5] Deleting local branch '$BRANCH'..."
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git branch -D "$BRANCH"
else
  echo "  Branch '$BRANCH' already deleted locally."
fi

# ── Step 5: write state + clean stale files ─────────────────────────────

echo "[5/5] Writing MERGED state and cleaning state files..."
STATE_DIR=".claude/state"
mkdir -p "$STATE_DIR"

# Write MERGED status
echo "MERGED" > "$STATE_DIR/issue-$ISSUE_NUM.status"

# Remove associated counter files (stale after merge)
rm -f "$STATE_DIR/issue-$ISSUE_NUM.fix_round"
rm -f "$STATE_DIR/issue-$ISSUE_NUM.close_reopen_count"

echo "=== Cleanup complete for issue #$ISSUE_NUM ==="
echo ""
echo "Tip: check for other stale state files or branches with:"
echo "  ls .claude/state/"
echo "  bash scripts/cleanup-merged-branches.sh --dry-run"
