#!/bin/bash
# Fetch all CodeRabbit reviews + inline comments for a PR, grouped by review_id.
# Default: only fresh comments (for current head commit). Use --all for full history.
# Usage: ./fetch-review.sh <pr_number> [--all]
set -euo pipefail

for cmd in gh jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha-screener}"
PR="${1:?Usage: $0 <pr_number> [--all]}"
ALL_FLAG="${2:-}"

# Get PR head commit SHA for fresh filtering
HEAD_COMMIT=$(gh pr view "$PR" --repo "$REPO" --json commits --jq '.commits[-1].oid')

# Fetch all CodeRabbit reviews (sorted oldest first)
REVIEWS=$(gh api --paginate "repos/$REPO/pulls/$PR/reviews" --jq '[.[] | select(.user.login == "coderabbitai[bot]")] | sort_by(.submitted_at)')
REVIEW_COUNT=$(echo "$REVIEWS" | jq 'length')

if [ "$REVIEW_COUNT" -eq 0 ]; then
  echo "No CodeRabbit reviews found on PR #$PR."
  exit 0
fi

if [ "$ALL_FLAG" != "--all" ]; then
  REVIEWS=$(echo "$REVIEWS" | jq '[last]')
  echo "==> Fresh CodeRabbit comments on PR #$PR (only for current head commit $HEAD_COMMIT)"
  echo "    Use --all for full history including stale comments"
else
  echo "==> All $REVIEW_COUNT CodeRabbit review(s) on PR #$PR (including stale comments)"
  echo "    Head commit: $HEAD_COMMIT"
fi
echo ""

# Iterate over reviews
echo "$REVIEWS" | jq -c '.[]' | while read -r review; do
  REVIEW_ID=$(echo "$review" | jq -r '.id // "N/A"')
  REVIEW_STATE=$(echo "$review" | jq -r '.state // "N/A"')
  REVIEW_BODY=$(echo "$review" | jq -r '.body // ""')
  REVIEW_COMMIT=$(echo "$review" | jq -r 'if .commit_id then .commit_id[:7] else "N/A" end')
  SUBMITTED_AT=$(echo "$review" | jq -r '.submitted_at // "N/A"')

  # Mark if review is for current head
  REVIEW_FULL_COMMIT=$(echo "$review" | jq -r '.commit_id // ""')
  IS_FRESH=""
  if [ "$REVIEW_FULL_COMMIT" = "$HEAD_COMMIT" ]; then
    IS_FRESH=" ✓ fresh"
  fi

  echo "──────────────────────────────────────────"
  echo "## Review ID: $REVIEW_ID  |  State: $REVIEW_STATE  |  Commit: $REVIEW_COMMIT$IS_FRESH  |  $SUBMITTED_AT"
  echo ""

  # Review summary body
  if [ -n "$REVIEW_BODY" ] && [ "$REVIEW_BODY" != "null" ]; then
    echo "### Review Summary"
    echo "$REVIEW_BODY" | head -60
    if [ "$(echo "$REVIEW_BODY" | wc -l)" -gt 60 ]; then
      echo "... (truncated, full body available via gh pr review $PR --repo $REPO)"
    fi
    echo ""
  fi

  # Fetch inline comments for this review
  COMMENTS=$(gh api --paginate "repos/$REPO/pulls/$PR/comments" --jq '
    [.[] | select(.pull_request_review_id == '"$REVIEW_ID"') | {
      path: .path,
      line: (.line // .original_line // "?"),
      body: .body,
      commit_id: (.commit_id // "")
    }]
  ' 2>/dev/null || echo "[]")

  # Apply fresh filtering (default) - only keep comments for current head commit
  if [ "$ALL_FLAG" != "--all" ]; then
    COMMENTS=$(echo "$COMMENTS" | jq '[.[] | select(.commit_id == "'"$HEAD_COMMIT"'")]')
  fi

  COMMENT_COUNT=$(echo "$COMMENTS" | jq 'length')
  if [ "$COMMENT_COUNT" -gt 0 ]; then
    echo "### Inline Comments ($COMMENT_COUNT)"

    # Group by file
    echo "$COMMENTS" | jq -c 'group_by(.path)[]' | while read -r group; do
      file=$(echo "$group" | jq -r '.[0].path')
      echo "  $file:"
      echo "$group" | jq -c '.[]' | while read -r comment; do
        line=$(echo "$comment" | jq -r '.line')
        body=$(echo "$comment" | jq -r '.body')
        echo "    - [L$line] $body"
      done
    done
  else
    echo "  (no inline comments for this review)"
  fi
  echo ""
done
