# train
uv run python -m src.train \
  --executor-vendor openai   --executor-model gpt-4o \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-vendor openai  --optimizer-model gpt-5 \
  --training-set bigbenchhard_ja_cot --exec-temperature 0 \
   --scor-temperature 0 --max-hist 1

uv run python -m src.train \
  --executor-vendor openai   --executor-model gpt-oss:20b \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-vendor openai  --optimizer-model gpt-5 --training-set bigbenchhard_ja_cot --exec-temperature 0 --scor-temperature 0
  
uv run python -m src.train \
  --executor-vendor anthropic   --executor-model claude-3-5-haiku-latest \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-vendor openai  --optimizer-model gpt-5 --training-set bigbenchhard_ja_cot --exec-temperature 0 --scor-temperature 0


# eval
uv run python -m src.eval \
  --executor-vendor openai   --executor-model gpt-4o \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-model gpt-5 --test-set bigbenchhard_ja_cot --exec-temperature 0 --scor-temperature 0 --max-hist 1

uv run python -m src.eval \
  --executor-vendor anthropic   --executor-model claude-3-5-haiku-latest \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-model gpt-5 --test-set bigbenchhard_ja_cot --exec-temperature 0 --scor-temperature 0 --max-hist 0



# Train and eval
bash cmds/run_openai.sh
bash cmds/run_anthropic.sh
bash cmds/run_google.sh
bash DS_GLOB="jsick" cmds/run_openai.sh
DS_GLOB="gsm8k commonsensemoralja jsick jsquad kuci bigbenchhard_ja_cot" cmds/run_anthropic_duel.sh

# Eval win rate
bash cmds/run_openai_eval.sh datasets/test gsm8k
bash cmds/run_anthropic_eval.sh datasets/test bigbenchhard_ja_cot

# Eval optimized instructions list
bash cmds/run_openai_eval_multi.sh
bash cmds/run_anthropic_eval_multi.sh

# Eval absolute acc
uv run -m src.eval_acc \
  --executor-vendor anthropic \
  --executor-model claude-3-5-haiku-latest \
  --optimizer-model gpt-5 \
  --batch-size 64 \
  --concurrency 6


# Iterate over length of history
DS_GLOB='bigbenchhard_ja_cot' cmds/run_openai_hist.sh
DS_GLOB='bigbenchhard_ja_cot' cmds/run_anthropic_hist.sh
