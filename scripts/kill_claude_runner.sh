#!/usr/bin/env bash
# kill_claude_runner.sh — Find and kill claude_night_runner.sh and its child processes
#
# Usage:
#   scripts/kill_claude_runner.sh          # Show processes, ask before killing
#   scripts/kill_claude_runner.sh -f       # Force kill without asking
#   scripts/kill_claude_runner.sh --dry    # Only show, don't kill

set -euo pipefail

FORCE=false
DRY_RUN=false

case "${1:-}" in
  -f|--force) FORCE=true ;;
  --dry)      DRY_RUN=true ;;
  -h|--help)
    echo "Usage: $0 [-f|--force] [--dry]"
    echo "  -f, --force   Kill without confirmation"
    echo "  --dry         Show matching processes only"
    exit 0
    ;;
esac

# Find all related processes
PIDS=()
PROC_INFO=""

# 1. claude_night_runner.sh itself
while IFS= read -r line; do
  pid=$(echo "$line" | awk '{print $2}')
  PIDS+=("$pid")
  PROC_INFO+="$line"$'\n'
done < <(ps aux | grep 'claude_night_runner' | grep -v grep | grep -v "kill_claude_runner" || true)

# 2. claude CLI spawned by the runner (claude -p --dangerously-skip-permissions)
while IFS= read -r line; do
  pid=$(echo "$line" | awk '{print $2}')
  PIDS+=("$pid")
  PROC_INFO+="$line"$'\n'
done < <(ps aux | grep -E 'claude.*dangerously-skip-permissions' | grep -v grep || true)

# 3. tee processes writing to claude logs
while IFS= read -r line; do
  pid=$(echo "$line" | awk '{print $2}')
  PIDS+=("$pid")
  PROC_INFO+="$line"$'\n'
done < <(ps aux | grep 'tee.*logs/claude/' | grep -v grep || true)

# Deduplicate PIDs
UNIQUE_PIDS=($(printf '%s\n' "${PIDS[@]}" | sort -u))

if [[ ${#UNIQUE_PIDS[@]} -eq 0 ]]; then
  echo "[INFO] No claude_night_runner processes found."
  exit 0
fi

echo "Found ${#UNIQUE_PIDS[@]} related process(es):"
echo "---------------------------------------------------"
echo "$PROC_INFO" | grep -v '^$'
echo "---------------------------------------------------"
echo "PIDs: ${UNIQUE_PIDS[*]}"

if $DRY_RUN; then
  echo "[DRY] Would kill: ${UNIQUE_PIDS[*]}"
  exit 0
fi

if ! $FORCE; then
  printf "Kill all? [y/N] "
  read -r answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

kill -9 "${UNIQUE_PIDS[@]}" 2>/dev/null || true
echo "[OK] Killed ${#UNIQUE_PIDS[@]} process(es): ${UNIQUE_PIDS[*]}"
