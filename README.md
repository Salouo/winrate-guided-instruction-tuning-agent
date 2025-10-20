# Winrate-Guided-Instruction-Tuning-Agent
![python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## ✨ Project Overview
This project develops a **Auto-Prompt-Tuning Agent** that searches for the best instruction to improve an LLM’s accuracy across different tasks (see datasets list below).  

You can mix and match models from multiple vendors for instruction optimization—including **OpenAI (GPT)**, **Google (Gemini)**, and **Anthropic (Claude)**—and set vendors/models independently for the **executor**, **scorer**, and **optimizer**.

### Purpose
In a typical assistant-user interaction, the prompt is composed of an **instruction** and a **user question**. This project optimizes the **instruction** component to improve the model’s problem-solving accuracy.

### Method
**Auto-Prompt-Tuning Agent** is composed of 3 unit agents:

- *Task Executor:* answers the questions.
- *Scorer:* evaluates the TaskExecutor’s answers.
- *Instruction Optimizer:* observes the <instruction, win rate> trajectory and updates the instruction accordingly.

The Instruction Optimizer act as an optimizer to adjust the instruction iteratively based on the observed trajectory and returns the current baseline (the most competitive one) instruction at the end of the iteration. 

### Project Duration
2025.8.18 - 2025.9.19  

<br>

## ⚙️ Environment
This project assumes it will be set up in a virtual environment created by `uv`.

| Language / Framework  | Version |
| --------------------- | ---------- |
| Python | ^3.11 |
| uv | 0.8.2 |
| openai | ^1.102.0 |
| google-cloud-aiplatform | ^1.111.0 |
| google-genai | ^1.32.0 |
| anthropic | ^0.64.0 |
| matplotlib | ^3.10.5 |
| langgraph | ^0.6.6 |
| dotenv | ^0.9.9 |
| argparse | ^1.4.0 |
| pathlib | ^1.0.1 |
| scikit-learn | ^1.7.1 |
| tenacity | ^9.1.2 |

For versions of other packages, please refer to `pyproject.toml`.

<br>

## 📁 Directory Structure
```
.
├── cmds
├── cut_trajectory.py
├── datasets
│   ├── raw
│   ├── test
│   ├── test_base
│   ├── train
│   └── train_base
├── datasets_processing
│   ├── count_samples.py
│   └── train_test_split.py
├── outputs
├── plot_accepted_insts.py
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── backends.py
│   ├── eval_acc.py
│   ├── eval.py
│   ├── logger.py
│   ├── train.py
│   ├── unit_agents_acc.py
│   ├── unit_agents.py
│   └── utils.py
└── uv.lock
```

<br>

## 🚀 How to Run
### Dataset pre-process
Use `train_test_split.py` to split raw datasets into training set and test set, which will be saved in *./datasets/trian* and *./datasets/test*.

### Execution

#### 1. Install dependencies

```bash
uv sync
```

This will install required packages in the project virtual environment.

#### 2. Set api keys
Create a `.env` file in the project root:
```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

VERTEXAI_PROJECT=your-gcp-project-id
VERTEXAI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service_account.json
```

This will make models provided by OpenAI, Google, Anthropic utilizable.

#### 3. Run `trian.py` to search for the optimized instruction

```bash
uv run python -m src.train \
  --executor-vendor openai   --executor-model gpt-4o \
  --scorer-vendor openai     --scorer-model gpt-4.1 \
  --optimizer-vendor openai  --optimizer-model gpt-5 \
  --training-set bigbenchhard_ja_cot
   
  # common settings
  --max-rounds 10 \
  --batch-size 20 \
  --exec-temperature 0 \
  --scor-temperature 0 \
  --max-hist 100 \
  --concurrency 10 \
  --out-dir outputs
```

- `--executor-vendor` / `--scorer-vendor` / `--optimizer-vendor`  

  Vendor which should choose from `openai`, `google`, `anthropic`. (default: openai)

- `--executor-model` / `--scorer-model` / `--optimizer-model`

  Model  under the chosen vendor. See examples below. (default: gpt-5)

- `--training-set`

​	Training set chosen from provided datasets. See datasets list below.  (default: bigbenchhard_jp_direct)

- `--max-rounds`

  Max optimized rounds. (default: 10)

- `--batch-size`

  Number of samples scored each round (default: 20)

- `--exec-temperature`

  Temperature of Task Executor (default: 0.0)

- `--scor-temperature`

  Temperature of Scorer (default: 0.0)

- `--max-hist`

​	Max trajectory length passed to the Instruction Optimizer. (default: 100)

- `--out-dir`

  Path to save the best instruction and the training histories. (default: ./outputs)

After running `train.py`, the optimized instruction and the training histories will be saved in `./outputs/`.

##### Vendors & Model Examples

- Openai: `gpt-5`,  `gpt-4o`, `gpt-oss:20b`
- Google: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- Anthropic: `claude-opus-4-1`, `claude-sonnet-4`, `claude-3-7-sonnet`

##### Datasets List

```text
Natural Language Inference
- jamp (train: 8955, test: 348)
- janli (train: 12312, test: 720)
- jnli (train: 18065, test: 2434)
- jsem (train: 12667, test: 3167)
- [jsick] (train: 4500, test: 4927)

Reading Comprehension
- jsquad (train: 56573, test: 4442)

Commonse Reasoning
- [jcommonsenseqa] (train: 8045, test: 1119)
- [commonsensemoralja] (train: 13975, test: 3992)
- [kuci] (train: 83127, test: 10391)
- winogrande_xl (train: 40398, test: 1267)

Human Evaluation
- mmmlu (train: 11233, test: 2809)
- jmmlu (train: 120, test: 30)
- mmlu_prox_ja (train: 63, test: 11759)

Entity Linking
- chabsa (train: 2572, test: 322)

Fine-grained Analysis
- wiki_ner (train: 881, test: 113)
- wiki_dependency (trian: 1517, test: 200)
- wiki_pas (train: 1514, test: 199)
- wiki_coreference (train: 1515, 200)
- wiki_reading (train: 512, test: 200)

Math Reasoning
- mawps (train: 16, test: 250)
- mgsm (train: 8, test:250)
- polymath-ja (train: 400, test: 100)
- [gsm8k] (train: 7473, test: 1319)


Machine Translation
- alt-e-to-j (train: 17972, test: 1010)
- wikicorpus-e-to-j (train: 28348, test: 7088)

Summarization
- xlsum_ja (train: 5320, test: 677)

Mixed Reasoning
- bigbenchhard_ja_direct (train: 1042, test: 261)
- [bigbenchhard_ja_cot] (train: 5208, test: 1303)
```

Datasets with `[]` are already evaluated in this project.

#### 4. Run `eval.py` to evaluate the best instruction that was found

```bash
uv run python -m src.eval \
  # vendor & model settings example
  --executor-vendor openai   --executor-model gpt-4o \
  --scorer-vendor openai     --scorer-model gpt-5 \
  --test-set bigbenchhard_ja_cot
  
  # common settings example
  --max-rounds 10 \
  --batch-size 8 \
  --exec-temperature 0 \
  --scor-temperature 0 \
  --test-set bigbenchhard_ja_direct \
  --concurrency 8
```

- `--executor-vendor` / `--scorer-vendor`
   Vendor for each component. *Choices:* `openai` | `google` | `anthropic`（default: openai）

- `--executor-model` / `--scorer-model`
   Model under the chosen vendor（default: gpt-5）

- `--max-rounds`
   Max evaluation rounds（default:100）

- `--batch-size`
   Number of samples evaluated per round（default: 8）

- `--exec-temperature`
  
   Temperature of Task Executor (default: 0.0)
   
- `--scor-temperature`

   Temperature of Scorer (default: 0.0)

- `--test-set`
   Dataset used for evaluation (split expected under `./datasets/test`). **Please choose the same dataset as in training.**（e.g., `bigbenchhard_ja_direct`）

- `--concurrency`
   Max in-flight async calls（default: `8`）

**ALL the evaluation results will be saved in `./outputs`.**



#### 5. Evaluate an instruction on datasets based on accuracy

After training, an optimized instruction can be obtained (or not found). Then the accuracy by using this instruction can be computed on the datasets.

**Utilization example**

```bash
uv run -m src.eval_acc \
  --executor-vendor anthropic \
  --executor-model claude-3-5-haiku-latest \
  --optimizer-model gpt-5 \
  --batch-size 64 \
  --concurrency 8
```

`eval_acc.py` will load the datasets in *datasets/test* and load the instruction you want to evaluate in *outputs/\<dataset_name\>\_\<batch_size\>\_\<length of given history\>\_\<your\_model\_settings>*. The results will be saved in the same directory.

<br>

## 📈 Reference Results
### Accuracy Comparison

We report **Exact Match (EM, %)** on each dataset using (1) the dataset’s **baseline instruction** and (2) our **optimized instruction** produced by this project.  


Evaluations are run with **GPT-4o** and **Claude 3.5 Haiku** across five benchmarks: **BBH-CoT**, **Moral**, **GSM8K**, **JSick**, and **Kuci**.  

**GPT-4o**

| Prompt Type        | BBH-CoT | Moral | GSM8K | JSick | Kuci |
|--------------------|:------:|:----:|:----:|:----:|:----:|
| Baseline           |  8.83  | 94.09| 57.62| 70.10| 81.44|
| Optimized (ours)   | **62.55** | — | **59.67** | **84.19** | **83.42** |

**Claude 3.5 Haiku**

| Prompt Type        | BBH-CoT | Moral | GSM8K | JSick | Kuci |
|--------------------|:------:|:----:|:----:|:----:|:----:|
| Baseline           | 12.20  | 51.45| 92.19|  0.00|  0.79|
| Optimized (ours)   | **53.75** | **83.44** | — | **76.76** | **71.03** |

*“—” indicates the optimized instruction was not applied for that model–dataset pair.*

### Instruction Optimization Examples

---

### GSM8K
**Baseline (EN)**
~~~text
Solve the given question. Return number only.
~~~

**Optimized (EN)**
~~~text
Accurately extract the quantitative relations in the word problem (sum, difference, multiple, ratio, “per-one”), and perform any necessary unit conversions. First, identify exactly one final quantity requested by the question and compute only that quantity. For discrete entities (items, books, bottles, boxes, groups, or “Y per X”), count only whole units; for patterns like “Y per X” and bundles/boxes/groups, use only the integer quotient after dividing by X and ignore any remainder unless it can be legitimately used. For money, do internal integer arithmetic in the smallest currency unit (e.g., cents) and convert back to the requested unit if needed. Perform all calculations internally and do not output intermediate steps. Output exactly one non-negative integer, with no units, symbols, commas, spaces, newlines, or extra characters. Before output, verify: (1) consistency of assumptions, quantitative relations, and units; (2) integerization due to discrete constraints; (3) if money, the value is an integer in the requested unit; and (4) absence of rounding or arbitrary treatment of fractions. If any check fails, revise the interpretation/calculation.
~~~

---

### JSick
**Baseline（JA）**
~~~text
前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。

制約：
- 前提が真であるとき仮説が必ず真になる場合はentailmentと出力
- 前提が真であるとき仮説が必ず偽になる場合はcontradictionと出力
- そのいずれでもない場合はneutralと出力
~~~

**Optimized（JA）**
~~~text
前提と仮説の関係を判定せよ。入力は「前提：...」「仮説：...」の2文。出力は entailment / contradiction / neutral のいずれか1語のみ（半角小文字、前後の空白・改行や他の語句・句読点は禁止）。
判定手順（優先順）：
1) 同時に真になり得ないなら contradiction（否定・数量・比較「一人も」「全て」「少なくとも」「より多い/少ない」「ちょうど」等に注意）。
2) 前提が仮説を必然的に真にするなら entailment（言い換え・包含関係は可）。ただし仮説が前提に新たな条件や詳細を付加しているだけなら entailment にはしない。
3) それ以外は neutral（前提にない事実の推測や常識補完はしない）
~~~

---

### KUCI
**Baseline（JA）**
~~~text
文脈と選択肢を入力として受け取り、選択肢から文脈の後に続く文として最も適切なものを選択してください。
なお、回答は選択肢の番号（例：0）でするものとします。
~~~

**Optimized（JA）**
~~~text
目的
- 文脈に最も自然に続き、S=文脈+候補 が完結するものの番号（0–9）を半角1字で返す。勝率下限を上げる。

入力
- 形式：文脈：<テキスト>　選択肢：0.<候補>,1.<候補>,…
- 候補の改変不可。番号は0–9。

出力
- 半角数字1字のみ（他文字・空白・改行なし）。

評価対象
- S＝文脈の直後に候補をそのまま続けた連続テキスト。
- 外部知識に依存しない。

失格（除外）
- 非文・未完・従属節ぶら下がり
- 明白な照応不能・矛盾・顕著な話題飛躍
- 文体の大不一致（常体と敬体の混在など）

接続整合（文脈末の形に応じる）
- 原因（ので/から/ため/だから）→ 結果・判断で終止
- 条件（と/たら/れば/なら/「と、」）→ 自然な結果・一般則
- 逆接（が/けど/しかし/だが/のに/ものの）→ 対比・譲歩で収束
- 列挙/追加（、/や/など/また/さらに/そして）→ 同系列の追加か簡潔な要約
- 体言・助詞止め（〜の/〜ため/〜について/〜は/〜が/〜で/〜に/〜から）→ 述部を補って終止
- 引用・疑問（〜と/「…」/？）→ それへの判断・応答
- 直前の助詞・活用との文法的つながりも確認

選定基準（優先順位）
1) 完結性（終止形で曖昧な省略なし）
2) 接続整合
3) 一貫性（主語・時制・語の再利用／新規導入最小）
4) 常識的妥当性（因果・語用の自然さ）
5) 簡潔性（短い）

同点時
- 新規内容語が少ない → 文字数が短い → 番号が小さい の順。

フォールバック
- 全失格なら、最も文法的に成立し話題拡張の小さい候補。判断不能なら最小番号。

手順
1) 各候補で S を作る。
2) 失格を除外。
3) 接続整合と完結性を最優先に上記基準で一つ選ぶ。
4) 同点・不確実時は同点規則・フォールバックを適用。

自己チェック
- 選んだ番号が選択肢に存在するか。
- 出力は半角数字1字のみ（改行・他文字なし）。
~~~



