#!/bin/bash

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Patterns that are always blocked
DANGEROUS_PATTERNS=(
  "git reset --hard"
  "git clean -fd"
  "git clean -f"
  "git checkout \."
  "git restore \."
  "push --force"
  "reset --hard"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qE "$pattern"; then
    echo "BLOCKED: '$COMMAND' matches dangerous pattern '$pattern'. The user has prevented you from doing this." >&2
    exit 2
  fi
done

# git branch -D: only allow feature/ branches (not master/main)
# Normalize command first (strip path, git -c/-C flags) like push check
BRANCH_DEL_CMD=$(echo "$COMMAND" | sed 's/ *2>&1 *$//; s/ *>[^ ]* *$//; s|^.*/git |git |; s/^git\( -[cC] [^ ]*\)\+/git /')
if echo "$BRANCH_DEL_CMD" | grep -qE 'git[[:space:]]+branch[[:space:]]+-D([[:space:]]|$)'; then
  # Require exact format: git branch -D feature/<name> [feature/<name> ...]
  if ! echo "$BRANCH_DEL_CMD" | grep -qE '^git[[:space:]]+branch[[:space:]]+-D([[:space:]]+feature/[A-Za-z0-9._/-]+)+[[:space:]]*$'; then
    echo "BLOCKED: invalid git branch -D format. Only plain feature/<name> branches are allowed." >&2
    exit 2
  fi
  if echo "$BRANCH_DEL_CMD" | grep -qE '(&&|;|\|\||\|)'; then
    echo "BLOCKED: command chaining with git branch -D is not allowed." >&2
    exit 2
  fi
  # Extract branch names after -D flag
  BRANCHES=$(echo "$BRANCH_DEL_CMD" | sed -n 's/.*git[[:space:]]\+branch[[:space:]]\+-D[[:space:]]\+//p')
  for branch in $BRANCHES; do
    case "$branch" in
      master|main)
        echo "BLOCKED: cannot delete master/main branch." >&2
        exit 2
        ;;
      feature/*) ;;
      *)
        echo "BLOCKED: 'git branch -D' only allowed for feature/<name> branches, got '$branch'." >&2
        exit 2
        ;;
    esac
  done
fi

# Only check git push commands (strip leading path and git config flags)
PUSH_CMD=$(echo "$COMMAND" | sed 's/ *2>&1 *$//; s/ *>[^ ]* *$//; s|^.*/git |git |; s/^git\( -[cC] [^ ]*\)\+/git /')
if echo "$PUSH_CMD" | grep -qE '^git push'; then

  if echo "$PUSH_CMD" | grep -qE '(&&|;)\s*git\s+push|git\s+push.*(&&|;)'; then
    echo "BLOCKED: command chaining with git push is not allowed." >&2
    exit 2
  fi

  if echo "$PUSH_CMD" | grep -qE 'git[[:space:]]+push.*(--force|-f)'; then
    echo "BLOCKED: git push --force is forbidden." >&2
    exit 2
  fi

  if ! echo "$PUSH_CMD" | grep -qE 'git[[:space:]]+push([[:space:]]+-u)?[[:space:]]+origin[[:space:]]+feature/[^[:space:]:]+$'; then
    echo "BLOCKED: only 'git push origin feature/<name>' is allowed." >&2
    exit 2
  fi

  if echo "$PUSH_CMD" | grep -qE 'git[[:space:]]+push([[:space:]]+-u)?[[:space:]]+origin[[:space:]]+(master|main|refs/heads/master|refs/heads/main|([^[:space:]:]+:)?(master|main))$'; then
    echo "BLOCKED: git push to master/main is forbidden. Use feature branches + PR." >&2
    exit 2
  fi
fi

exit 0
