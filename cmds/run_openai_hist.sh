set -euo pipefail

echo "[SCRIPT] run_openai_duel_hist_sweep.sh :: train->eval (OpenAI only, sweeping max_hist)"

TRAIN_DIR="${1:-datasets/train}"
SWEEP_HISTS="${SWEEP_HISTS:-12 24 48 72}"

# ===== Models (can be overridden by env) =====
OPENAI_EXECUTOR_MODEL="${OPENAI_EXECUTOR_MODEL:-gpt-4o}"
OPENAI_SCORER_MODEL="${OPENAI_SCORER_MODEL:-gpt-4.1}"
OPENAI_OPTIMIZER_MODEL="${OPENAI_OPTIMIZER_MODEL:-gpt-5}"

# ===== Fixed core settings (override via env if needed) =====
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"

# ===== Train settings =====
OPENAI_CONCURRENCY="${OPENAI_CONCURRENCY:-8}"   # default 8
EXEC_TEMPERATURE="${EXEC_TEMPERATURE:-0}"
SCOR_TEMPERATURE="${SCOR_TEMPERATURE:-0}"
EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"

# ===== Eval settings =====
OPENAI_EVAL_CONCURRENCY="${OPENAI_EVAL_CONCURRENCY:-$OPENAI_CONCURRENCY}"
EXTRA_EVAL_FLAGS="${EXTRA_EVAL_FLAGS:-}"
EVAL_Z="${EVAL_Z:-1.96}"

# ===== Run control =====
RETRIES="${RETRIES:-40}"
RETRY_SLEEP="${RETRY_SLEEP:-10}"
DS_GLOB="${DS_GLOB:-}"          # optional filter; space/comma separated patterns
LOG_DIR="${LOG_DIR:-}"           # if set, save logs into this dir

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

# ===== Enumerate datasets (strip .json) =====
declare -a DATASETS=()
shopt -s nullglob
for f in "$TRAIN_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  ds="${f##*/}"; ds="${ds%.*}"
  DATASETS+=("$ds")
done
shopt -u nullglob

# ===== Optional filter =====
if [[ -n "${DS_GLOB:-}" ]]; then
  IFS=', ' read -r -a _globs <<< "$DS_GLOB"
  _filtered=()
  for d in "${DATASETS[@]}"; do
    for g in "${_globs[@]}"; do
      [[ -z "$g" ]] && continue
      if [[ "$d" == $g ]]; then _filtered+=("$d"); break; fi
    done
  done
  # dedup
  DATASETS=()
  for x in "${_filtered[@]}"; do
    dup=0; for y in "${DATASETS[@]}"; do [[ "$x" == "$y" ]] && { dup=1; break; }; done
    ((dup==0)) && DATASETS+=("$x")
  done
fi
(( ${#DATASETS[@]} > 0 )) || { echo "No dataset *.json in $TRAIN_DIR"; exit 1; }

[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"

# inject concurrency only if not present in extra flags
extra_train_conc=""
[[ -n "${OPENAI_CONCURRENCY:-}" && "$EXTRA_TRAIN_FLAGS" != *"--concurrency"* ]] && extra_train_conc="--concurrency $OPENAI_CONCURRENCY"

extra_eval_conc=""
[[ -n "${OPENAI_EVAL_CONCURRENCY:-}" && "$EXTRA_EVAL_FLAGS" != *"--concurrency"* ]] && extra_eval_conc="--concurrency $OPENAI_EVAL_CONCURRENCY"

banner "Datasets:"
printf " - %s\n" "${DATASETS[@]}"
echo "Sweep max_hist values: ${SWEEP_HISTS}"

# =========================
#         SWEEP
# =========================
for HIST in ${SWEEP_HISTS}; do
  banner ">>> Sweep max_hist = ${HIST}"

  declare -a TRAIN_FAILS=()
  declare -a TRAIN_OKS=()

  # ---------- TRAIN ----------
  for dataset in "${DATASETS[@]}"; do
    eval_dir="${OPENAI_EXECUTOR_MODEL}_${OPENAI_SCORER_MODEL}_${OPENAI_OPTIMIZER_MODEL}"
    train_desc="TRAIN [openai | $eval_dir | $dataset | mh=${HIST}]"

    if [[ -n "$LOG_DIR" ]]; then
      train_cmd=(bash -c "uv run python -u -m src.train \
        --executor-vendor openai   --executor-model '$OPENAI_EXECUTOR_MODEL' \
        --scorer-vendor   openai   --scorer-model   '$OPENAI_SCORER_MODEL' \
        --optimizer-vendor openai  --optimizer-model '$OPENAI_OPTIMIZER_MODEL' \
        --training-set '$dataset' --batch-size '$BATCH_SIZE' --max-rounds '$MAX_ROUNDS' \
        $extra_train_conc --exec-temperature '$EXEC_TEMPERATURE' --scor-temperature '$SCOR_TEMPERATURE' \
        ${EXTRA_TRAIN_FLAGS} \
        --max-hist '${HIST}' \
        2>&1 | tee \"${LOG_DIR}/openai__${eval_dir}__${dataset}__mh${HIST}__train.log\"")
    else
      train_cmd=(uv run python -u -m src.train
        --executor-vendor openai   --executor-model "$OPENAI_EXECUTOR_MODEL"
        --scorer-vendor   openai   --scorer-model "$OPENAI_SCORER_MODEL"
        --optimizer-vendor openai  --optimizer-model "$OPENAI_OPTIMIZER_MODEL"
        --training-set "$dataset" --batch-size "$BATCH_SIZE" --max-rounds "$MAX_ROUNDS"
        $extra_train_conc --exec-temperature "$EXEC_TEMPERATURE" --scor-temperature "$SCOR_TEMPERATURE"
        ${EXTRA_TRAIN_FLAGS}
        --max-hist "$HIST")
    fi

    banner "$train_desc"
    if run_with_retries "$RETRIES" "$RETRY_SLEEP" "$train_desc" "${train_cmd[@]}"; then
      TRAIN_OKS+=("$dataset")
    else
      TRAIN_FAILS+=("openai:$eval_dir:$dataset:mh${HIST}")
    fi
  done

  banner "[mh=${HIST}] Training done. Failures: ${#TRAIN_FAILS[@]}"
  for x in "${TRAIN_FAILS[@]:-}"; do echo " - $x"; done

  # ---------- EVAL ----------
  banner "[mh=${HIST}] Start evaluation for ${#TRAIN_OKS[@]} trained dataset(s)"

  declare -a EVAL_FAILS=()
  for dataset in "${TRAIN_OKS[@]}"; do
    eval_dir="${OPENAI_EXECUTOR_MODEL}_${OPENAI_SCORER_MODEL}_${OPENAI_OPTIMIZER_MODEL}"
    eval_desc="EVAL [openai | $eval_dir | $dataset | mh=${HIST}]"

    if [[ -n "$LOG_DIR" ]]; then
      eval_cmd=(bash -c "uv run python -u -m src.eval \
        --executor-vendor openai   --executor-model '$OPENAI_EXECUTOR_MODEL' \
        --scorer-vendor   openai   --scorer-model   '$OPENAI_SCORER_MODEL' \
        --optimizer-model '$OPENAI_OPTIMIZER_MODEL' \
        --test-set '$dataset' --batch-size '$BATCH_SIZE' --max-rounds '$MAX_ROUNDS' \
        $extra_eval_conc \
        --exec-temperature '$EXEC_TEMPERATURE' --scor-temperature '$SCOR_TEMPERATURE' \
        --z '$EVAL_Z' \
        ${EXTRA_EVAL_FLAGS} \
        --max-hist '${HIST}' \
        2>&1 | tee \"${LOG_DIR}/openai__${eval_dir}__${dataset}__mh${HIST}__eval.log\"")
    else
      eval_cmd=(uv run python -u -m src.eval
        --executor-vendor openai   --executor-model "$OPENAI_EXECUTOR_MODEL"
        --scorer-vendor   openai   --scorer-model "$OPENAI_SCORER_MODEL"
        --optimizer-model "$OPENAI_OPTIMIZER_MODEL"
        --test-set "$dataset" --batch-size "$BATCH_SIZE" --max-rounds "$MAX_ROUNDS"
        $extra_eval_conc
        --exec-temperature "$EXEC_TEMPERATURE" --scor-temperature "$SCOR_TEMPERATURE"
        --z "$EVAL_Z"
        ${EXTRA_EVAL_FLAGS}
        --max-hist "$HIST")
    fi

    banner "$eval_desc"
    if ! run_with_retries "$RETRIES" "$RETRY_SLEEP" "$eval_desc" "${eval_cmd[@]}"; then
      EVAL_FAILS+=("openai:$eval_dir:$dataset:mh${HIST}:eval")
    fi
  done

  banner "[mh=${HIST}] Sweep summary"
  echo "Train failures: ${#TRAIN_FAILS[@]}"
  for x in "${TRAIN_FAILS[@]:-}"; do echo " - $x"; done
  echo "Eval failures: ${#EVAL_FAILS[@]}"
  for x in "${EVAL_FAILS[@]:-}"; do echo " - $x"; done
  
  if (( ${#TRAIN_FAILS[@]} + ${#EVAL_FAILS[@]} > 0 )); then
    echo "[mh=${HIST}] Some failures occurred."
  fi
done

banner "All hist sweeps done."
