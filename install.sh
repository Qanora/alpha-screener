#!/bin/bash
# 飞轮 skill 安装脚本 — 目录级软链到可写的技能安装根（兼容 Claude/Codex）
# 用法: ./install.sh [--dry-run] [--uninstall] [--validate] [--install-hooks]
#      ./install.sh [--source <path>] [--sync-meta|--no-sync-meta]
# 脚本随 skill 目录自动可用，无需单独安装
set -euo pipefail

DRY_RUN=false
UNINSTALL=false
VALIDATE=false
INSTALL_HOOKS=false
SYNC_META="${FLYWHEEL_SYNC_META:-true}"
FLYWHEEL_SOURCE_INPUT="${FLYWHEEL_SOURCE_DIR:-${FLYWHEEL_SOURCE_ROOT:-${FLYWHEEL_SOURCE_REPO:-}}}"
FLYWHEEL_META_DIR_INPUT="${FLYWHEEL_META_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --uninstall)
      UNINSTALL=true
      shift
      ;;
    --validate)
      VALIDATE=true
      shift
      ;;
    --install-hooks)
      INSTALL_HOOKS=true
      shift
      ;;
    --source)
      if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
        echo "Missing --source value"
        exit 1
      fi
      FLYWHEEL_SOURCE_INPUT="$2"
      shift 2
      ;;
    --sync-meta)
      SYNC_META=true
      shift
      ;;
    --no-sync-meta)
      SYNC_META=false
      shift
      ;;
    *)
      echo "Unknown: $1"
      exit 1
      ;;
  esac
done

# Resolve flywheel source. Accept source root or explicit flywheel dir.
resolve_flywheel_source() {
  local candidate
  for candidate in "$@"; do
    [ -z "$candidate" ] && continue
    if [ -f "$candidate/scripts/flywheel-skills.txt" ]; then
      echo "$candidate"
      return 0
    fi
    if [ -f "$candidate/flywheel/scripts/flywheel-skills.txt" ]; then
      echo "$candidate/flywheel"
      return 0
    fi
  done
  return 1
}

if ! SKILL_SOURCE=$(resolve_flywheel_source \
  "$FLYWHEEL_SOURCE_INPUT" \
  "$SCRIPT_DIR/flywheel" \
  "$SCRIPT_DIR/../skills" \
  "$SCRIPT_DIR/../.." \
  "/root/workspace/skills"
); then
  echo "[ERROR] 未找到可用飞轮 skill 源"
  echo "支持目录:"
  echo "  - --source <path>"
  echo "  - FLYWHEEL_SOURCE_DIR"
  echo "  - 默认: <项目目录>/flywheel, <项目父目录>/skills, /root/workspace/skills"
  exit 1
fi

FLYWHEEL_SRC_ROOT="$(cd "$SKILL_SOURCE/.." && pwd)"
META_SOURCE_DIR="${FLYWHEEL_META_DIR_INPUT:-$FLYWHEEL_SRC_ROOT/project-meta}"
SKILL_LIST_FILE="$SKILL_SOURCE/scripts/flywheel-skills.txt"
SKILLS=()
if [ -f "$SKILL_SOURCE/scripts/flywheel-context.sh" ]; then
  # shellcheck disable=SC1091
  source "$SKILL_SOURCE/scripts/flywheel-context.sh"
fi
if declare -f fw_skill_roots >/dev/null; then
  mapfile -t DSTS < <(fw_skill_roots)
else
  data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
  DSTS=(
    "$data_home/codex/skills"
    "$data_home/claude/skills"
    "$HOME/.codex/skills"
    "$HOME/.claude/skills"
  )
fi

# 读取可安装 skill 列表（支持无脑新增新 skill）
if [ -f "$SKILL_LIST_FILE" ]; then
  # shellcheck disable=SC2207
  mapfile -t SKILLS < <(awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*#/ {next} /^[[:space:]]*$/ {next} {print $1}' "$SKILL_LIST_FILE")
else
  SKILLS=("fwp-inspect" "fw-audit" "fwp-plan" "fwp-ship" "fwp-build" "fwp-setup" "fwp-debug" "fwp-resume" "fwp-help")
fi

[ "${#SKILLS[@]}" -gt 0 ] || { echo "ERROR: empty skill list"; exit 1; }

# 卸载时也清理旧 lp-* 名称
OLD_SKILLS=("lp-up" "lp-dp" "lp-ms" "lp-mr" "lp-dev" "lp-init" "fw-setup" "fw-plan" "fw-debug" "fw-inspect" "fw-build" "fw-ship" "fw-resume")

ensure_dst_dir() {
  local dst="$1"
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] mkdir -p $dst"
    return
  fi
  mkdir -p "$dst"
}

install_one() {
  local dst="$1" skill="$2" src="$3" link="$dst/$2"

  if [ -L "$link" ] && [ "$(readlink "$link")" = "$src" ]; then
    echo "  [OK]    $dst/$skill (已正确链接)"
    return
  fi

  if [ -e "$link" ] || [ -L "$link" ]; then
    $DRY_RUN && echo "[DRY-RUN] rm -rf $link" || rm -rf "$link"
  fi

  if $DRY_RUN; then
    echo "[DRY-RUN] ln -s $src → $link"
  else
    ln -sf "$src" "$link"
    echo "  [LINK]  $dst/$skill → $src"
  fi
}

uninstall_one() {
  local dst="$1" skill="$2" link="$dst/$2"

  if [ -L "$link" ]; then
    $DRY_RUN && echo "[DRY-RUN] rm $link" || rm "$link"
    echo "[REMOVE] $link (目录级软链)"
  elif [ -d "$link" ]; then
    $DRY_RUN && echo "[DRY-RUN] rm -rf $link" || rm -rf "$link"
    echo "[REMOVE] $link (目录)"
  elif [ -f "$link/SKILL.md" ]; then
    $DRY_RUN && echo "[DRY-RUN] rm $link/SKILL.md && rmdir $link" || { rm "$link/SKILL.md"; rmdir "$link" 2>/dev/null || true; }
    echo "[REMOVE] $link (旧格式)"
  else
    echo "[SKIP]  $link (不存在)"
  fi
}

sync_project_meta() {
  if [ "$SYNC_META" != "true" ]; then
    return 0
  fi

  if [ ! -d "$META_SOURCE_DIR" ]; then
    echo "[SKIP] 元文件源不存在: $META_SOURCE_DIR"
    return 0
  fi

  for meta_file in CLAUDE.md CODEX.md; do
    local src_meta="$META_SOURCE_DIR/$meta_file"
    if [ -f "$src_meta" ]; then
      if $DRY_RUN; then
        echo "[DRY-RUN] cp $src_meta -> $SCRIPT_DIR/$meta_file"
      else
        cp "$src_meta" "$SCRIPT_DIR/$meta_file"
        echo "  [SYNC]  $meta_file ← $src_meta"
      fi
    fi
  done
}

# ── 卸载 ──
if $UNINSTALL; then
  echo "=== 卸载飞轮 skills ==="
  ALL=("${SKILLS[@]}" "${OLD_SKILLS[@]}")
  for dst in "${DSTS[@]}"; do
    echo "==> 清理 $dst"
    for skill in "${ALL[@]}"; do
      uninstall_one "$dst" "$skill"
    done
  done
  exit 0
fi

# ── 检查源目录 ──
echo "=== 飞轮 skill 安装 ==="
echo "源: $SKILL_SOURCE (meta=$META_SOURCE_DIR)"
echo "目标: ${DSTS[*]} (目录级软链)"
if [ "${#DSTS[@]}" -eq 0 ]; then
  echo "[ERROR] 未发现可写的 skill 安装根（可设置 FLYWHEEL_SKILL_ROOTS 覆盖，格式: /path/a:/path/b）"
  echo "[ERROR] 无可写安装目录，安装无法执行"
  exit 1
fi
echo ""

for skill in "${SKILLS[@]}"; do
  if [ ! -d "$SKILL_SOURCE/$skill" ]; then
    echo "[ERROR] 源目录不存在: $SKILL_SOURCE/$skill"
    exit 1
  fi
done

# 同步项目级元文件（可选）
sync_project_meta

# ── 安装目录级软链 ──
for dst in "${DSTS[@]}"; do
  echo "==> 安装到 $dst"
  ensure_dst_dir "$dst"
  for skill in "${SKILLS[@]}"; do
    install_one "$dst" "$skill" "$SKILL_SOURCE/$skill"
  done
  echo ""
done

echo ""
if $VALIDATE; then
  echo "=== 额外校验职责边界 ==="
  bash "$SKILL_SOURCE/scripts/validate-contracts.sh"
  echo ""
fi

if $INSTALL_HOOKS; then
  echo "=== 安装提交前钩子 ==="
  bash "$SKILL_SOURCE/scripts/setup-git-hooks.sh"
  echo ""
fi

echo "=== 安装完成 ==="
echo ""
for dst in "${DSTS[@]}"; do
  echo "验证: ls -la $dst/fwp-*"
done
echo "测试: /fwp-help"
echo "建议: bash $SKILL_SOURCE/scripts/validate-contracts.sh"
echo "新项目必须: bash $SKILL_SOURCE/scripts/setup-git-hooks.sh"
