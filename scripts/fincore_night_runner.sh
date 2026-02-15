#!/usr/bin/env bash
# ==============================================================================
# fincore_night_runner.sh — Unattended overnight runner for OpenAI Codex CLI
#
# Purpose:
#   Reads the 0037 task file, feeds its content as the prompt to
#   `codex exec --full-auto`, and automatically restarts when each session
#   finishes — so it keeps making progress all night long.
#
#   Before each Codex run the script collects a health snapshot (pytest count,
#   ruff issues, coverage %) and appends it to the log so you can track
#   improvement across cycles.
#
# Prerequisites:
#   - codex CLI installed and logged in (`codex login`)
#   - Python environment with fincore installed in dev mode
#   - tmux or screen (recommended, so the session survives terminal close)
#
# Usage:
#   1) Make executable (first time only):
#        chmod +x scripts/fincore_night_runner.sh
#
#   2) Simplest run (uses defaults for this project):
#        scripts/fincore_night_runner.sh
#
#   3) Run with custom options:
#        scripts/fincore_night_runner.sh \
#          --task-file docs/0037-不断修复bug并完善项目/readme.md \
#          --workdir /Users/yunjinqi/Documents/source_code/fincore \
#          --model o3 \
#          --restart-delay 15 \
#          --max-runs 20
#
#   4) Overnight in tmux (recommended):
#        tmux new -s fincore-night
#        scripts/fincore_night_runner.sh
#        # Detach:   Ctrl+b  then  d
#        # Reattach: tmux attach -t fincore-night
#
# Options:
#   --task-file <path>     Task markdown file relative to workdir
#                          (default: docs/0037-不断修复bug并完善项目/readme.md)
#   --workdir <path>       Project root (default: this script's parent dir)
#   --log-dir <path>       Log directory (default: <workdir>/logs/codex)
#   --model <model>        Codex model to use (default: auto / codex CLI default)
#   --restart-delay <s>    Seconds between restarts (default: 15)
#   --max-runs <n>         Max restart cycles, 0 = infinite (default: 0)
#   --skip-health          Skip pre-run health checks (faster restarts)
#   --extra-args <args>    Extra arguments passed verbatim to codex exec
#   -h, --help             Show this help
#
# How it works:
#   Each cycle:
#     1. (optional) Collect health snapshot: pytest, ruff, coverage
#     2. cat <task-file> | codex exec --full-auto -C <workdir> -m <model> -
#     3. (optional) Collect post-run health snapshot
#     4. Wait --restart-delay seconds, then loop
#
# Notes:
#   - --full-auto = automatic command approval + workspace-write sandbox.
#   - Logs are written to timestamped files under --log-dir for review.
#   - A summary CSV (health_history.csv) tracks metrics across runs.
#   - Press Ctrl+C to stop the loop.
# ==============================================================================

set -euo pipefail

# ---- Resolve defaults relative to this script's location ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKDIR="$DEFAULT_WORKDIR"
TASK_FILE="docs/0037-不断修复bug并完善项目/readme.md"
LOG_DIR=""
MODEL=""
RESTART_DELAY=15
MAX_RUNS=0
SKIP_HEALTH=false
EXTRA_ARGS=""
RUN_COUNT=0

print_help() {
  sed -n '/^# ==/,/^# ==/p' "$0" | sed 's/^# *//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-file)      TASK_FILE="${2:?--task-file requires a value}";    shift 2 ;;
    --workdir)        WORKDIR="${2:?--workdir requires a value}";        shift 2 ;;
    --log-dir)        LOG_DIR="${2:?--log-dir requires a value}";        shift 2 ;;
    --model)          MODEL="${2:?--model requires a value}";            shift 2 ;;
    --restart-delay)  RESTART_DELAY="${2:?--restart-delay requires a value}"; shift 2 ;;
    --max-runs)       MAX_RUNS="${2:?--max-runs requires a value}";      shift 2 ;;
    --skip-health)    SKIP_HEALTH=true;                                  shift   ;;
    --extra-args)     EXTRA_ARGS="${2:-}";                               shift 2 ;;
    -h|--help)        print_help; exit 0 ;;
    *)                echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

# ---- Validate ----
TASK_FULL_PATH="$WORKDIR/$TASK_FILE"
if [[ ! -f "$TASK_FULL_PATH" ]]; then
  echo "Error: task file not found: $TASK_FULL_PATH"
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$WORKDIR/logs/codex"
fi
mkdir -p "$LOG_DIR"

HEALTH_CSV="$LOG_DIR/health_history.csv"
if [[ ! -f "$HEALTH_CSV" ]]; then
  echo "run,phase,timestamp,tests_passed,tests_failed,tests_skipped,ruff_issues,coverage_pct" > "$HEALTH_CSV"
fi

if ! [[ "$RESTART_DELAY" =~ ^[0-9]+$ ]]; then
  echo "Error: --restart-delay must be a non-negative integer"; exit 1
fi
if ! [[ "$MAX_RUNS" =~ ^[0-9]+$ ]]; then
  echo "Error: --max-runs must be a non-negative integer"; exit 1
fi

if ! command -v codex &>/dev/null; then
  echo "Error: 'codex' command not found. Install it first: npm install -g @openai/codex"
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

  # --- pytest (quick, no parallelism to avoid overhead) ---
  local pytest_output
  pytest_output=$(cd "$WORKDIR" && python -m pytest tests --tb=no -q --no-header 2>&1 || true)
  # Parse "N passed, M failed, K skipped" from last line
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

# ---- Print config ----
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    fincore_night_runner.sh — Overnight Mode (fincore)        ║"
echo "╠════════════════════════════════════════════════════════════════╣"
printf "║ %-16s %s\n" "Workdir:"        "$WORKDIR"
printf "║ %-16s %s\n" "Task file:"      "$TASK_FILE"
printf "║ %-16s %s\n" "Model:"          "${MODEL:-auto (codex default)}"
printf "║ %-16s %s\n" "Log dir:"        "$LOG_DIR"
printf "║ %-16s %ss\n" "Restart delay:" "$RESTART_DELAY"
printf "║ %-16s %s\n" "Max runs:"       "$( [[ $MAX_RUNS -eq 0 ]] && echo 'infinite' || echo $MAX_RUNS )"
printf "║ %-16s %s\n" "Health checks:"  "$( $SKIP_HEALTH && echo 'disabled' || echo 'enabled' )"
[[ -n "$EXTRA_ARGS" ]] && printf "║ %-16s %s\n" "Extra args:" "$EXTRA_ARGS"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# ---- Build codex exec arguments ----
CODEX_ARGS=(exec --full-auto -C "$WORKDIR")
[[ -n "$MODEL" ]] && CODEX_ARGS+=(-m "$MODEL")
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && CODEX_ARGS+=($EXTRA_ARGS)
CODEX_ARGS+=(-)

# ---- Main loop ----
while true; do
  RUN_COUNT=$((RUN_COUNT + 1))

  if [[ $MAX_RUNS -gt 0 && $RUN_COUNT -gt $MAX_RUNS ]]; then
    echo "[DONE] Reached max runs ($MAX_RUNS). Exiting."
    break
  fi

  ts="$(date +%F_%H-%M-%S)"
  log_file="$LOG_DIR/fincore_run${RUN_COUNT}_${ts}.log"

  {
    echo "==================================================================="
    echo "[RUN  ] #$RUN_COUNT"
    echo "[START] $(date '+%F %T')"
    echo "[TASK ] $TASK_FILE"
    echo "[MODEL] ${MODEL:-auto}"
    echo "[CMD  ] codex ${CODEX_ARGS[*]}"
    echo "==================================================================="
  } | tee -a "$log_file"

  # ---- Pre-run health check ----
  if ! $SKIP_HEALTH; then
    collect_health "$RUN_COUNT" "pre" "$log_file"
  fi

  # ---- Run codex ----
  set +e
  cat "$TASK_FULL_PATH" | codex "${CODEX_ARGS[@]}" \
    2>&1 | tee -a "$log_file"
  exit_code=${PIPESTATUS[1]}
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

  # ---- Post-run code formatting (keep codebase tidy between runs) ----
  echo "[INFO] Running post-run formatting (ruff format + ruff fix)..." | tee -a "$log_file"
  (
    cd "$WORKDIR"
    ruff format fincore/ tests/ 2>&1 || true
    ruff check fincore/ tests/ --fix --exit-zero --ignore F401,F811 2>&1 || true
  ) | tee -a "$log_file"

  # ---- Git auto-commit (optional, only if there are changes) ----
  (
    cd "$WORKDIR"
    if git diff --quiet && git diff --cached --quiet; then
      echo "[GIT ] No changes to commit." | tee -a "$log_file"
    else
      git add -A
      git commit -m "auto(0037): night runner cycle #${RUN_COUNT} — $(date '+%F %T')" \
        2>&1 | tee -a "$log_file" || true
    fi
  )

  echo "[INFO] Next run in ${RESTART_DELAY}s... (Ctrl+C to stop)"
  sleep "$RESTART_DELAY"
done
