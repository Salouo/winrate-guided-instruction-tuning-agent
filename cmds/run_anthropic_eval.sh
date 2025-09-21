set -euo pipefail

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
banner(){ echo -e "\n==== [$(ts)] $* ====\n"; }
run_with_retries(){
  local attempts="$1"; shift
  local sleep_s="$1"; shift
  local title="$1"; shift
  local i=1
  while :; do
    echo "[TRY $i/$attempts] $title"
    if "$@"; then echo "[OK] $title"; return 0; fi
    if (( i>=attempts )); then echo "[FAIL] $title after $attempts attempts"; return 1; fi
    echo "[RETRY] $title -> sleep ${sleep_s}s"; sleep "$sleep_s"; ((i++))
  done
}

# ===== Models (can be overridden by env) =====
ANTHROPIC_EXECUTOR_MODEL="${ANTHROPIC_EXECUTOR_MODEL:-claude-3-5-haiku-latest}"
OPENAI_SCORER_MODEL="${OPENAI_SCORER_MODEL:-gpt-4.1}"
OPENAI_OPTIMIZER_MODEL="${OPENAI_OPTIMIZER_MODEL:-gpt-5}"

# ===== Fixed core settings =====
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"

# ===== Eval settings =====
ANTHROPIC_EVAL_CONCURRENCY="${ANTHROPIC_EVAL_CONCURRENCY:-8}"
EXEC_TEMPERATURE="${EXEC_TEMPERATURE:-0}"
SCOR_TEMPERATURE="${SCOR_TEMPERATURE:-0}"
EVAL_METRIC="${EVAL_METRIC:-auto}"   # auto/em/f1/llm
EVAL_Z="${EVAL_Z:-1.96}"
EXTRA_EVAL_FLAGS="${EXTRA_EVAL_FLAGS:-}"

# ===== Run control =====
RETRIES="${RETRIES:-40}"
RETRY_SLEEP="${RETRY_SLEEP:-10}"
DS_GLOB="${DS_GLOB:-}"
LOG_DIR="${LOG_DIR:-}"

# ===== Dataset dir & initial enumeration =====
DATASET_DIR="${1:-datasets/test}"
shift || true

declare -a ALL_DATASETS=()
shopt -s nullglob
for f in "$DATASET_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  ds="${f##*/}"; ds="${ds%.*}"
  ALL_DATASETS+=("$ds")
done
shopt -u nullglob

(( ${#ALL_DATASETS[@]} > 0 )) || { echo "No dataset *.json in $DATASET_DIR"; exit 1; }

# ===== Build candidate list from CLI args or DS_GLOB or interactive =====
declare -a SELECTED=()

if (( $# >= 1 )); then
  SELECTED=("$@")
else
  if [[ -n "$DS_GLOB" ]]; then
    IFS=', ' read -r -a _globs <<< "$DS_GLOB"
    for d in "${ALL_DATASETS[@]}"; do
      for g in "${_globs[@]}"; do
        [[ -z "$g" ]] && continue
        if [[ "$d" == $g ]]; then SELECTED+=("$d"); break; fi
      done
    done
    if (( ${#SELECTED[@]} > 1 )); then
      declare -a _dedup=()
      for x in "${SELECTED[@]}"; do
        dup=0; for y in "${_dedup[@]:-}"; do [[ "$x" == "$y" ]] && { dup=1; break; }; done
        ((dup==0)) && _dedup+=("$x")
      done
      SELECTED=("${_dedup[@]}")
    fi
  fi

  if (( ${#SELECTED[@]} == 0 )); then
    banner "Datasets in $DATASET_DIR"
    for i in "${!ALL_DATASETS[@]}"; do
      printf "%3d) %s\n" "$((i+1))" "${ALL_DATASETS[$i]}"
    done
    echo
    read -rp "Select dataset indices (e.g. '1 3 5') or 'all' [all]: " sel
    sel="${sel:-all}"
    sel="${sel//,/ }"
    if [[ "$sel" == "all" ]]; then
      SELECTED=("${ALL_DATASETS[@]}")
    else
      for idx in $sel; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx>=1 && idx<=${#ALL_DATASETS[@]} )); then
          SELECTED+=("${ALL_DATASETS[$((idx-1))]}")
        else
          echo "Skip invalid index: $idx"
        fi
      done
    fi
  fi
fi

(( ${#SELECTED[@]} > 0 )) || { echo "No dataset selected."; exit 1; }

[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"

extra_eval_conc=""
[[ -n "${ANTHROPIC_EVAL_CONCURRENCY:-}" && "$EXTRA_EVAL_FLAGS" != *"--concurrency"* ]] && extra_eval_conc="--concurrency $ANTHROPIC_EVAL_CONCURRENCY"

banner "Datasets to EVAL:"
printf " - %s\n" "${SELECTED[@]}"

declare -a EVAL_FAILS=()

for dataset in "${SELECTED[@]}"; do
  eval_dir="${ANTHROPIC_EXECUTOR_MODEL}_${OPENAI_SCORER_MODEL}_${OPENAI_OPTIMIZER_MODEL}"
  eval_desc="EVAL [executor=anthropic | scorer=openai | optimizer=openai | $eval_dir | $dataset]"

  if [[ -n "$LOG_DIR" ]]; then
    eval_cmd=(bash -c "uv run python -u -m src.eval \
      --executor-vendor anthropic --executor-model '$ANTHROPIC_EXECUTOR_MODEL' \
      --scorer-vendor   openai    --scorer-model   '$OPENAI_SCORER_MODEL' \
      --optimizer-model '$OPENAI_OPTIMIZER_MODEL' \
      --test-set '$dataset' --batch-size '$BATCH_SIZE' --max-rounds '$MAX_ROUNDS' \
      $extra_eval_conc \
      --exec-temperature '$EXEC_TEMPERATURE' --max-hist 1 --scor-temperature '$SCOR_TEMPERATURE' \
      --z '$EVAL_Z' \
      ${EXTRA_EVAL_FLAGS} 2>&1 | tee \"${LOG_DIR}/anthropic-openai-openai__${eval_dir}__${dataset}__eval.log\"")
  else
    eval_cmd=(uv run python -u -m src.eval
      --executor-vendor anthropic --executor-model "$ANTHROPIC_EXECUTOR_MODEL"
      --scorer-vendor   openai    --scorer-model "$OPENAI_SCORER_MODEL"
      --optimizer-model "$OPENAI_OPTIMIZER_MODEL"
      --test-set "$dataset" --batch-size "$BATCH_SIZE" --max-rounds "$MAX_ROUNDS"
      $extra_eval_conc
      --exec-temperature "$EXEC_TEMPERATURE" --max-hist 1 --scor-temperature "$SCOR_TEMPERATURE"
      --z "$EVAL_Z"
      ${EXTRA_EVAL_FLAGS})
  fi

  banner "$eval_desc"
  if ! run_with_retries "$RETRIES" "$RETRY_SLEEP" "$eval_desc" "${eval_cmd[@]}"; then
    EVAL_FAILS+=("anthropic-openai-openai:$eval_dir:$dataset:eval")
  fi
done

banner "All evals done."
echo "Eval failures: ${#EVAL_FAILS[@]}"
for x in "${EVAL_FAILS[@]:-}"; do echo " - $x"; done

# Non-zero exit if any failure
if (( ${#EVAL_FAILS[@]} > 0 )); then
  exit 2
fi
