#!/usr/bin/env bash
# ==============================================================================
# claude_night_runner.sh — Unattended overnight runner for Claude Code CLI
#
# Purpose:
#   Runs `claude -p --dangerously-skip-permissions` in a loop, mimicking the
#   behaviour of manually opening Claude interactive mode and typing the task
#   file path.  Each cycle:
#     1. Sends the task file path as the prompt (identical to manual usage).
#     2. Claude reads the file, follows its instructions, and exits.
#     3. Subsequent cycles use `--continue` to resume the same conversation,
#        preserving context so Claude remembers previous fixes.
#     4. Every COMPACT_INTERVAL runs the conversation is reset (fresh start)
#        to keep context size manageable — equivalent to `/compact`.
#     5. Optionally runs ruff format + git auto-commit between cycles.
#
# Why the original script behaved differently from manual usage:
#   1. The prompt was a verbose instruction instead of just the file path.
#      In interactive mode you only type the file path; Claude reads and
#      follows it automatically.
#   2. No --max-turns was set, so -p mode may stop after fewer agentic turns
#      than interactive mode allows.
#   3. No context compaction — the conversation grew unbounded.
#
# Prerequisites:
#   - Claude Code CLI installed and authenticated
#   - Python environment with fincore installed in dev mode
#   - tmux or screen recommended (so the session survives terminal close)
#
# Usage:
#   chmod +x scripts/claude_night_runner.sh
#
#   # Simplest (defaults):
#   scripts/claude_night_runner.sh
#
#   # Custom:
#   scripts/claude_night_runner.sh \
#     --task-file docs/0037-不断修复bug并完善项目/readme.md \
#     --restart-delay 10 --max-runs 20 --compact-interval 3
#
#   # In tmux (recommended):
#   tmux new -s claude-night
#   scripts/claude_night_runner.sh
#   # Ctrl+b d  to detach;  tmux attach -t claude-night  to reattach
#
# Options:
#   --task-file <path>       Task markdown (relative to workdir)
#                            (default: docs/0037-不断修复bug并完善项目/readme.md)
#   --workdir <path>         Project root (default: script's parent dir)
#   --log-dir <path>         Log directory (default: <workdir>/logs/claude)
#   --restart-delay <s>      Seconds between cycles (default: 10)
#   --max-runs <n>           Max cycles, 0 = infinite (default: 0)
#   --compact-interval <n>   Start a fresh conversation every N runs to
#                            manage context size (simulates /compact).
#                            0 = never reset (default: 5)
#   --max-turns <n>          Max agentic turns per cycle (default: 200).
#                            Set high to match interactive-mode behaviour.
#   --model <name>           Model to use, e.g. sonnet, opus (default: CLI default)
#   --extra-args <args>      Extra arguments passed to claude CLI
#   --skip-format            Skip post-run ruff format + git commit
#   --skip-health            Skip pre/post health checks
#   -h, --help               Show this help
# ==============================================================================

set -euo pipefail

# ---- Resolve defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKDIR="$DEFAULT_WORKDIR"
TASK_FILE="docs/0037-不断修复bug并完善项目/readme.md"
LOG_DIR=""
MODEL=""
RESTART_DELAY=10
MAX_RUNS=0
COMPACT_INTERVAL=5
MAX_TURNS=200
EXTRA_ARGS=""
SKIP_FORMAT=false
SKIP_HEALTH=false
RUN_COUNT=0

print_help() {
  sed -n '/^# ==/,/^# ==/p' "$0" | sed 's/^# *//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-file)         TASK_FILE="${2:?--task-file requires a value}";         shift 2 ;;
    --workdir)           WORKDIR="${2:?--workdir requires a value}";             shift 2 ;;
    --log-dir)           LOG_DIR="${2:?--log-dir requires a value}";             shift 2 ;;
    --model)             MODEL="${2:?--model requires a value}";                 shift 2 ;;
    --restart-delay)     RESTART_DELAY="${2:?--restart-delay requires a value}"; shift 2 ;;
    --max-runs)          MAX_RUNS="${2:?--max-runs requires a value}";           shift 2 ;;
    --compact-interval)  COMPACT_INTERVAL="${2:?--compact-interval requires a value}"; shift 2 ;;
    --max-turns)         MAX_TURNS="${2:?--max-turns requires a value}";         shift 2 ;;
    --extra-args)        EXTRA_ARGS="${2:-}";                                    shift 2 ;;
    --skip-format)       SKIP_FORMAT=true;                                       shift   ;;
    --skip-health)       SKIP_HEALTH=true;                                       shift   ;;
    -h|--help)           print_help; exit 0 ;;
    *)                   echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

# ---- Validate ----
TASK_FULL_PATH="$WORKDIR/$TASK_FILE"
if [[ ! -f "$TASK_FULL_PATH" ]]; then
  echo "Error: task file not found: $TASK_FULL_PATH"; exit 1
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$WORKDIR/logs/claude"
fi
mkdir -p "$LOG_DIR"

# Health-check CSV
HEALTH_CSV="$LOG_DIR/health_history.csv"
if [[ ! -f "$HEALTH_CSV" ]]; then
  echo "run,phase,timestamp,tests_passed,tests_failed,tests_skipped,ruff_issues,coverage_pct" > "$HEALTH_CSV"
fi

if ! command -v claude &>/dev/null; then
  echo "Error: 'claude' not found. Install Claude Code CLI first."
  echo "  npm install -g @anthropic-ai/claude-code"
  exit 1
fi

# ---- Health check function ----
collect_health() {
  local run_num="$1"
  local phase="$2"   # "pre" or "post"
  local log_file="$3"

  local ts
  ts="$(date '+%F %T')"

  local passed=0 failed=0 skipped=0 ruff_count=0 cov_pct="N/A"

  echo "  [HEALTH] Collecting $phase-run health snapshot..." | tee -a "$log_file"

  # --- pytest ---
  local pytest_output
  pytest_output=$(cd "$WORKDIR" && python -m pytest tests --tb=no -q --no-header 2>&1 || true)
  if echo "$pytest_output" | grep -qE '[0-9]+ passed'; then
    passed=$(echo "$pytest_output" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')
  fi
  if echo "$pytest_output" | grep -qE '[0-9]+ failed'; then
    failed=$(echo "$pytest_output" | grep -oE '[0-9]+ failed' | head -1 | grep -oE '[0-9]+')
  fi
  if echo "$pytest_output" | grep -qE '[0-9]+ skipped'; then
    skipped=$(echo "$pytest_output" | grep -oE '[0-9]+ skipped' | head -1 | grep -oE '[0-9]+')
  fi

  # --- ruff ---
  local ruff_output
  ruff_output=$(cd "$WORKDIR" && ruff check fincore/ tests/ --statistics 2>&1 || true)
  if echo "$ruff_output" | grep -qE 'Found [0-9]+ error'; then
    ruff_count=$(echo "$ruff_output" | grep -oE 'Found [0-9]+ error' | grep -oE '[0-9]+')
  elif echo "$ruff_output" | grep -qi 'All checks passed'; then
    ruff_count=0
  fi

  # --- coverage (only for post-run, as it's slow) ---
  if [[ "$phase" == "post" ]]; then
    local cov_output
    cov_output=$(cd "$WORKDIR" && python -m pytest tests --cov=fincore --cov-report=term --tb=no -q --no-header 2>&1 || true)
    local cov_line
    cov_line=$(echo "$cov_output" | grep -E '^TOTAL' | tail -1 || true)
    if [[ -n "$cov_line" ]]; then
      cov_pct=$(echo "$cov_line" | grep -oE '[0-9]+%' | tail -1 | tr -d '%')
    fi
  fi

  # --- Log and CSV ---
  {
    echo "  [HEALTH] $phase | tests: ${passed}p/${failed}f/${skipped}s | ruff: ${ruff_count} | cov: ${cov_pct}%"
  } | tee -a "$log_file"

  echo "${run_num},${phase},${ts},${passed},${failed},${skipped},${ruff_count},${cov_pct}" >> "$HEALTH_CSV"
}

# ---- Build base claude args ----
# -p = non-interactive print mode (process prompt, use tools, exit)
# --dangerously-skip-permissions = auto-approve all tool/command usage
# --max-turns = allow enough agentic turns to match interactive behaviour
CLAUDE_BASE_ARGS=(-p --dangerously-skip-permissions --verbose --output-format stream-json)
[[ "$MAX_TURNS" -gt 0 ]] && CLAUDE_BASE_ARGS+=(--max-turns "$MAX_TURNS")
[[ -n "$MODEL" ]] && CLAUDE_BASE_ARGS+=(--model "$MODEL")
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && CLAUDE_BASE_ARGS+=($EXTRA_ARGS)

# ---- Build the prompt ----
# KEY FIX: Use the exact same input as manual interactive usage — just the
# task file path. Claude reads the file and follows its instructions.
PROMPT="$TASK_FILE"

# Continuation prompt: re-send the task file path so Claude picks up where
# it left off within the same conversation.
CONTINUE_PROMPT="$TASK_FILE"

# ---- Print config ----
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  claude_night_runner.sh — Overnight Mode (fincore)           ║"
echo "╠════════════════════════════════════════════════════════════════╣"
printf "║ %-20s %s\n" "Workdir:"          "$WORKDIR"
printf "║ %-20s %s\n" "Task file:"        "$TASK_FILE"
printf "║ %-20s %s\n" "Model:"            "${MODEL:-default}"
printf "║ %-20s %s\n" "Log dir:"          "$LOG_DIR"
printf "║ %-20s %ss\n" "Restart delay:"   "$RESTART_DELAY"
printf "║ %-20s %s\n" "Max runs:"         "$( [[ $MAX_RUNS -eq 0 ]] && echo 'infinite' || echo $MAX_RUNS )"
printf "║ %-20s %s\n" "Compact interval:" "$( [[ $COMPACT_INTERVAL -eq 0 ]] && echo 'disabled' || echo "every ${COMPACT_INTERVAL} runs" )"
printf "║ %-20s %s\n" "Max turns/cycle:"  "$MAX_TURNS"
printf "║ %-20s %s\n" "Post-format:"      "$( $SKIP_FORMAT && echo 'disabled' || echo 'enabled' )"
printf "║ %-20s %s\n" "Health checks:"    "$( $SKIP_HEALTH && echo 'disabled' || echo 'enabled' )"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "[INFO] Press Ctrl+C to stop."
echo

# ---- Main loop ----
while true; do
  RUN_COUNT=$((RUN_COUNT + 1))

  if [[ $MAX_RUNS -gt 0 && $RUN_COUNT -gt $MAX_RUNS ]]; then
    echo "[DONE] Reached max runs ($MAX_RUNS). Exiting."
    break
  fi

  ts="$(date +%F_%H-%M-%S)"
  log_file="$LOG_DIR/claude_run${RUN_COUNT}_${ts}.log"

  # Decide whether to --continue or start fresh (compact).
  # First run is always fresh. After that, use --continue unless we hit
  # the compact interval boundary.
  USE_CONTINUE=false
  if [[ $RUN_COUNT -gt 1 ]]; then
    if [[ $COMPACT_INTERVAL -gt 0 ]] && (( (RUN_COUNT - 1) % COMPACT_INTERVAL == 0 )); then
      echo "[COMPACT] Run #$RUN_COUNT: starting fresh conversation (context reset)." | tee -a "$log_file"
      USE_CONTINUE=false
    else
      USE_CONTINUE=true
    fi
  fi

  # Build args for this run
  CLAUDE_ARGS=("${CLAUDE_BASE_ARGS[@]}")
  CURRENT_PROMPT="$PROMPT"
  if $USE_CONTINUE; then
    CLAUDE_ARGS+=(--continue)
    CURRENT_PROMPT="$CONTINUE_PROMPT"
  fi

  {
    echo "==================================================================="
    echo "[RUN  ] #$RUN_COUNT  $(if $USE_CONTINUE; then echo '(--continue)'; else echo '(fresh)'; fi)"
    echo "[START] $(date '+%F %T')"
    echo "[TASK ] $TASK_FILE"
    echo "[CMD  ] claude ${CLAUDE_ARGS[*]} \"$CURRENT_PROMPT\""
    echo "==================================================================="
  } | tee -a "$log_file"

  # ---- Pre-run health check ----
  if ! $SKIP_HEALTH; then
    collect_health "$RUN_COUNT" "pre" "$log_file"
  fi

  # ---- Run claude ----
  # Use stream-json output format for real-time streaming (like interactive mode).
  # A Python filter extracts human-readable content and displays it while saving
  # the raw JSON events to the log file.
  set +e
  (
    cd "$WORKDIR"
    claude "${CLAUDE_ARGS[@]}" "$CURRENT_PROMPT" 2>&1
  ) | LOG_FILE="$log_file" python3 -u -c '
import os, sys, json

log_path = os.environ["LOG_FILE"]
with open(log_path, "a") as log:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        log.write(line)
        log.flush()
        s = line.strip()
        if not s:
            continue
        try:
            d = json.loads(s)
            msg_type = d.get("type", "")
            content = d.get("content", "")
            tool = d.get("tool", "")
            if msg_type == "assistant" and content:
                print(content, flush=True)
            elif msg_type == "tool" and tool:
                print(f"[Tool: {tool}]", flush=True)
            elif msg_type == "result" and content:
                print(content, flush=True)
        except (json.JSONDecodeError, KeyError):
            print(s, flush=True)
'
  exit_code=${PIPESTATUS[0]}
  set -e

  {
    echo
    echo "==================================================================="
    echo "[END  ] $(date '+%F %T')"
    echo "[CODE ] $exit_code"
    echo "==================================================================="
  } | tee -a "$log_file"

  if [[ $exit_code -eq 0 ]]; then
    echo "[OK  ] Run #$RUN_COUNT finished successfully."
  else
    echo "[WARN] Run #$RUN_COUNT exited with code $exit_code."
  fi

  # ---- Post-run health check ----
  if ! $SKIP_HEALTH; then
    collect_health "$RUN_COUNT" "post" "$log_file"
  fi

  # ---- Post-run formatting + git commit ----
  if ! $SKIP_FORMAT; then
    echo "[INFO] Running post-run formatting..." | tee -a "$log_file"
    (
      cd "$WORKDIR"
      ruff format fincore/ tests/ 2>&1 || true
      ruff check fincore/ tests/ --fix --exit-zero --ignore F401,F811 2>&1 || true
    ) | tee -a "$log_file"

    # Git auto-commit
    (
      cd "$WORKDIR"
      if git diff --quiet && git diff --cached --quiet; then
        echo "[GIT ] No changes to commit." | tee -a "$log_file"
      else
        git add -A
        git commit -m "auto(0037): claude night runner cycle #${RUN_COUNT} — $(date '+%F %T')" \
          2>&1 | tee -a "$log_file" || true
      fi
    )
  fi

  echo "[INFO] Next run in ${RESTART_DELAY}s... (Ctrl+C to stop)"
  sleep "$RESTART_DELAY"
done
