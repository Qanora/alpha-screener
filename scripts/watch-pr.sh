#!/bin/bash
# PR status monitor — pure state polling, no severity analysis.
# Usage: ./watch-pr.sh <pr_number>
# Exit: 0=CI green (ready to merge), 1=CI failure, 2=stuck/timeout, 5=missing tools
set -euo pipefail

for cmd in gh jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required but not installed"
    exit 5
  fi
done

REPO="${REPO:-Qanora/alpha-screener}"
PR="${1:?Usage: $0 <pr_number>}"
STDERR_FILE=$(mktemp)
trap 'rm -f "$STDERR_FILE"' EXIT INT TERM

# Calculate dynamic timeout based on PR diff size
# Base: 20 rounds for up to 300 lines, +5 rounds per additional 100 lines, max 60
ADDITIONS=0
DELETIONS=0
calculate_timeout() {
  local pr_meta total_lines
  if ! pr_meta=$(gh pr view "$PR" --repo "$REPO" --json additions,deletions 2>"$STDERR_FILE"); then
    echo "[INIT] $(date +%H:%M:%S) gh pr view failed while calculating timeout"
    cat "$STDERR_FILE"
    exit 3
  fi
  ADDITIONS=$(echo "$pr_meta" | jq -r '.additions // 0')
  DELETIONS=$(echo "$pr_meta" | jq -r '.deletions // 0')
  total_lines=$((ADDITIONS + DELETIONS))

  if [ "$total_lines" -le 300 ]; then
    TIMEOUT=20
  else
    local extra_lines=$((total_lines - 300))
    local extra_rounds=$((((extra_lines + 99) / 100) * 5))
    local timeout=$((20 + extra_rounds))
    if [ "$timeout" -gt 60 ]; then
      TIMEOUT=60
    else
      TIMEOUT="$timeout"
    fi
  fi
}

calculate_timeout
echo "[INFO] Dynamic timeout: $TIMEOUT rounds ($ADDITIONS additions, $DELETIONS deletions)"

ROUND=0

while true; do
  ROUND=$((ROUND + 1))
  : > "$STDERR_FILE"

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

  # Terminal: CI failure (failing checks exist)
  if [ -n "$FAILING" ]; then
    echo "=== CI FAILURES: $FAILING ==="
    exit 1
  fi

  # CodeRabbit review is advisory (non-blocking). Log but don't block merge.
  if [ "$REVIEW" = "CHANGES_REQUESTED" ]; then
      echo "=== CHANGES_REQUESTED (advisory — run fetch-review.sh $PR for details) ==="
  fi

  if [ "$REVIEW" = "COMMENTED" ] || [ "$REVIEW" = "APPROVED" ]; then
      echo "=== $REVIEW (advisory) ==="
  fi

  # Ready to merge: CI green + no pending checks
  if [ -z "$PENDING" ] && [ -z "$FAILING" ]; then
      echo "=== CI green — ready for merge ==="
      exit 0
  fi

  if [ "$ROUND" -ge "$TIMEOUT" ]; then
    echo "=== TIMEOUT after ${ROUND} rounds (max $TIMEOUT) ==="
    if [ -z "$FAILING" ]; then
      echo "=== CI is green despite timeout — exiting as ready ==="
      exit 0
    fi
    exit 2
  fi

  sleep 30
done
