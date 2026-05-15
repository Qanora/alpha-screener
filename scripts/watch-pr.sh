#!/bin/bash
# PR status monitor — pure state polling, no severity analysis.
# Usage: ./watch-pr.sh <pr_number> [timeout_rounds]
# Exit: 0=merged, 1=CI failure, 2=stuck/timeout, 4=changes requested (fresh), 5=missing tools
set -euo pipefail

for cmd in gh jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha-screener}"
PR="${1:?Usage: $0 <pr_number> [timeout_rounds]}"
TIMEOUT="${2:-40}"

case "$TIMEOUT" in
  ''|*[!0-9]*|0) echo "ERROR: timeout_rounds must be a positive integer, got: $TIMEOUT"; exit 6 ;;
esac

ROUND=0
STDERR_FILE=$(mktemp)
trap 'rm -f "$STDERR_FILE"' EXIT INT TERM

while true; do
  ROUND=$((ROUND + 1))
  : > "$STDERR_FILE"

  # Get PR state: CI checks, review decision, merged status, AND head commit
  if ! RESULT=$(gh pr view "$PR" --repo "$REPO" \
    --json statusCheckRollup,reviewDecision,mergedAt,commits \
    --jq '{
      failing: [(.statusCheckRollup // [])[] |
        select(.status == "COMPLETED" and (.conclusion == "FAILURE" or .conclusion == "TIMED_OUT" or .conclusion == "CANCELLED" or .conclusion == "ACTION_REQUIRED" or .conclusion == "STARTUP_FAILURE")) |
        "\(.name):\(.conclusion)"
      ],
      pending: [(.statusCheckRollup // [])[] |
        select(.status != "COMPLETED" and .status != null) |
        .name
      ],
      review: .reviewDecision,
      merged: .mergedAt,
      head: (.commits[-1].oid // "")
    }' 2>"$STDERR_FILE"); then
    echo "[$ROUND] $(date +%H:%M:%S) gh pr view failed"
    cat "$STDERR_FILE"
    exit 3
  fi

  REVIEW=$(echo "$RESULT" | jq -r '.review')
  MERGED=$(echo "$RESULT" | jq -r '.merged')
  FAILING=$(echo "$RESULT" | jq -r '.failing | join(",")')
  PENDING=$(echo "$RESULT" | jq -r '.pending | join(",")')
  HEAD_COMMIT=$(echo "$RESULT" | jq -r '.head')

  echo "[$ROUND] $(date +%H:%M:%S) review=$REVIEW pending=${PENDING:-none} failing=${FAILING:-none}"

  # Terminal: merged
  if [ "$MERGED" != "null" ]; then
    echo "=== MERGED at $MERGED ==="
    exit 0
  fi

  # Terminal: CI failure
  if [ -n "$FAILING" ]; then
    echo "=== CI FAILURES: $FAILING ==="
    echo "Actions: fix issues → commit → push → re-run monitor"
    exit 1
  fi

  # CHANGES_REQUESTED: verify it's fresh (review is for current head commit), not stale
  if [ "$REVIEW" = "CHANGES_REQUESTED" ]; then
    # Get latest CodeRabbit review's commit_id from REST API
    REVIEW_COMMIT=$(gh api "repos/$REPO/pulls/$PR/reviews" \
      --jq '[.[] | select(.user.login == "coderabbitai[bot]")] | last | .commit_id // ""' 2>/dev/null || echo "")

    if [ -n "$REVIEW_COMMIT" ] && [ -n "$HEAD_COMMIT" ] && [ "$REVIEW_COMMIT" = "$HEAD_COMMIT" ]; then
      echo "=== CHANGES_REQUESTED (fresh — for current head) — run fetch-review.sh $PR ==="
      exit 4
    else
      echo "=== CHANGES_REQUESTED (stale — review is for $REVIEW_COMMIT, head is $HEAD_COMMIT) — waiting for re-review ==="
    fi
  fi

  # Happy path: APPROVED + CI green
  if [ "$REVIEW" = "APPROVED" ] && [ -z "$PENDING" ] && [ -z "$FAILING" ]; then
    echo "=== APPROVED + CI green — ready for merge ==="
  fi

  # COMMENTED: CodeRabbit left comments but didn't request changes (chill profile).
  if [ "$REVIEW" = "COMMENTED" ]; then
    echo "=== COMMENTED (chill) — check with fetch-review.sh $PR if needed ==="
  fi

  if [ "$ROUND" -ge "$TIMEOUT" ]; then
    echo "=== STUCK after ${ROUND} rounds — consider close-reopen ==="
    exit 2
  fi

  sleep 30
done
