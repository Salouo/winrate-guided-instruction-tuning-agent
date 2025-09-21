"""unit.agents.py

Composable building blocks for agent systems.

(Important) This module provides:
    TaskExecutor: runs tasks.
    Scorer: evaluates outputs with configurable metrics.
    InstructionOptimizer: iteratively refines instructions.
"""

import asyncio
import regex as re
import math

from typing import Literal, Sequence
from openai import RateLimitError, APIConnectionError, APIError
from google.api_core import exceptions as gexc
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from decimal import Decimal, InvalidOperation

from .utils import (
    TrainState,
    EvalState
)

RETRY_EXCS = (
    RateLimitError, APIConnectionError, APIError, asyncio.TimeoutError,
    gexc.GoogleAPIError, gexc.DeadlineExceeded, gexc.ServiceUnavailable
)
Outcome = Literal["A", "B", "TIE"]
MetricType = Literal["em", "math_em", "llm"]

_PUNCT_RE = re.compile(r'[\p{P}\p{S}]', re.UNICODE)
_WS_RE = re.compile(r'\s+', re.UNICODE)
_NUM_RE = re.compile(
    r'[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+'
)
_HASH_NUM = re.compile(
    r'####\s*([-\d,]*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+)',
    re.IGNORECASE
)
_FRAC_TEX = re.compile(r'\\d?frac\{([-\d]+)\}\{([-\d]+)\}')


class _LoopBoundSemaphore:
    """Create/refresh a Semaphore bound to the *current* running event loop."""
    def __init__(self, limit: int):
        self._limit = limit
        self._sem: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem = asyncio.Semaphore(self._limit)
            self._loop = loop
        return self._sem

    def reset(self) -> None:
        self._sem = None
        self._loop = None
# ========================================================================= #
#                                Task Executor                              #
# ========================================================================= #
class TaskExecutor:
    """TaskExecutor

    Executes the model against a batch of questions to produce the corresponding answers.

    Attributes:
        backend:
        temperature: Sampling temperature.
        concurrency:
    """

    def __init__(self, backend, temperature: float = 0.0, concurrency: int = 5):
        self.backend = backend
        self.temperature = temperature
        self._sem = _LoopBoundSemaphore(concurrency)

    @retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(6),
            retry=retry_if_exception_type(RETRY_EXCS),
            reraise=True
    )
    async def _execute_one(self, system_instruction: str, question: str) -> str:
        async with self._sem.get():
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ]
            content = await self.backend.chat(messages=messages, temperature=self.temperature)
        return content.strip()

    # ---- Duel Mode ----
    async def run_pair(
            self,
            state: TrainState | EvalState,
            baseline_instruction: str,
            candidate_instruction: str
    ) -> tuple[list[str], list[str]]:
        """
        Execute tasks based on both baseline instruction and candidate instruction
        """
        # Select the current batch
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        batch = state.samples[i:j]

        tasks_A = [
            asyncio.create_task(self._execute_one(baseline_instruction, b.question))
            for b in batch
        ]
        tasks_B = [
            asyncio.create_task(self._execute_one(candidate_instruction, b.question))
            for b in batch
        ]

        answers_A, answers_B = await asyncio.gather(
            asyncio.gather(*tasks_A),
            asyncio.gather(*tasks_B),
        )
        return answers_A, answers_B


# ========================================================================= #
#                                Scorer                                     #
# ========================================================================= #
class Scorer:
    def __init__(self, backend, temperature: float = 1.0, concurrency: int = 5):
        self.backend = backend
        self.temperature = temperature
        self._sem = _LoopBoundSemaphore(concurrency)

    def wilson_ci(self, wins: int, losses: int, z: float = 1.96) -> tuple[float, float, float]:
        """
        Computing Wilson confidence interval.

        Returns:
            p_hat, ci_low, ci_high
        """
        n = wins + losses
        if n == 0:
            return 0.5, 0.0, 1.0
        p_hat = wins / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2/(2*n)) / denom
        margin = (z * math.sqrt((p_hat*(1-p_hat)/n) + (z**2)/(4*n**2))) / denom
        return p_hat, max(0.0, center - margin), min(1.0, center + margin)

    def _parse_ab_tie(self, s: str) -> Outcome:
        """Extract result of duels."""
        s = s.strip().upper()
        if re.fullmatch(r"(A|B|TIE)", s):
            return "TIE" if s == "TIE" else s
        m = re.search(r"\b(A|B|TIE)\b", s)
        if m:
            t = m.group(1)
            assert t in ("A", "B", "TIE")
            return "TIE" if t == "TIE" else t
        raise ValueError(f"Unexpected judge output: {s!r}")

    # --------------------- Metrics --------------------- #
    # ---- EM ----
    def _normalize_em(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = _PUNCT_RE.sub(' ', s)
        s = _WS_RE.sub(' ', s).strip()
        return s

    def _exact_match(self, pred: str, ref: str | None) -> int:
        if ref is None:
            return 0
        p = self._normalize_em(pred)
        r = self._normalize_em(ref)
        return int(p == r)
    
    # ---- Math EM ----
    @staticmethod
    def _extract_final_number(text: str | None) -> str | None:
        if not text:
            return None
        m = _HASH_NUM.search(text)
        if m:
            return m.group(1)
        tex = _FRAC_TEX.findall(text)
        if tex:
            a, b = tex[-1]
            return f"{a}/{b}"
        nums = _NUM_RE.findall(text)
        return nums[-1] if nums else None
    
    @staticmethod
    def _to_decimal(num_str: str | None) -> Decimal | None:
        if not num_str:
            return None
        s = num_str.strip().rstrip('.,;')
        s = s.replace(',', '')
        if '/' in s and s.count('/') == 1:
            n, d = s.split('/')
            try:
                return Decimal(n) / Decimal(d)
            except (InvalidOperation, ZeroDivisionError):
                return None
        try:
            return Decimal(s)
        except InvalidOperation:
            try:
                return Decimal(s.replace('e', 'E'))
            except InvalidOperation:
                return None
    @classmethod
    def _exact_match_numeric(cls, pred: str | None, ref: str | None, eps: Decimal = Decimal("0")) -> int:
        """Extract number"""
        p = cls._to_decimal(cls._extract_final_number(pred))
        g = cls._to_decimal(cls._extract_final_number(ref))
        if p is None or g is None:
            return 0
        return int(abs(p - g) <= eps)
    
    async def _duel_one_sample(
        self,
        question: str,
        ans_A: str,
        ans_B: str,
        ground_truth: str | None,
        metric: MetricType,
    ) -> Outcome:
        """
        Judge the duel. Returns 'A' | 'B' | 'TIE'.
        metric:
          - 'em'  : exact match
          - 'llm' : LLM judge
        """
        # ---- EM ----
        if metric == "em":
            a = self._exact_match(ans_A, ground_truth)
            b = self._exact_match(ans_B, ground_truth)
            if a > b: 
                return "A"
            if b > a: 
                return "B"
            return "TIE"
        
        # ---- Math EM ----
        elif metric == "math_em":
            a = self._exact_match_numeric(ans_A, ground_truth)
            b = self._exact_match_numeric(ans_B, ground_truth)
            if a > b: 
                return "A"
            if b > a: 
                return "B"
            return "TIE"

        # ---- LLM metric ----
        sys = (
            "あなたは公平な審査員です。\n"
            "以下の質問に対する二つの回答 A と B を比較し、"
            "事実性・網羅性・形式遵守の3点に基づいて優劣を判定してください。\n"
            "出力は必ず A, B, TIE のいずれか1語のみ。説明や数字は書かないでください。\n"
            "僅差で同等と判断される場合は TIE を選びます。"
        )
        user = (
            f"質問:\n{question}\n\n"
            f"回答A:\n{ans_A}\n\n"
            f"回答B:\n{ans_B}\n\n"
        )
        if ground_truth:
            user += f"参考となる正解（あれば）:\n{ground_truth}\n\n"

        async with self._sem.get():
            content = await self.backend.chat(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0
            )
        if not content:
            raise ValueError("Model returned no content in pairwise judge.")
        return self._parse_ab_tie(content.strip())

    async def run_pairwise(
        self,
        state: TrainState | EvalState,
        answers_A: Sequence[str],
        answers_B: Sequence[str],
        metric: MetricType,
        z: float = 1.96,
    ) -> dict:
        """
        Judge the current batch in which questions are both answered based on the candidate instruction and the optimized instruction.

        Returns:
            {
              "wins": int, "losses": int, "ties": int,
              "n_effective": int, "p_hat": float,
              "ci_low": float, "ci_high": float
            }
        """
        # Select the current batch
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        batch_samples = state.samples[i:j]
        assert len(answers_A) == len(batch_samples) == len(answers_B), \
            f"length mismatch: A={len(answers_A)}, B={len(answers_B)}, batch={len(batch_samples)}"

        wins = losses = ties = 0
        tasks = []
        for k, b in enumerate(batch_samples):
            tasks.append(asyncio.create_task(
                self._duel_one_sample(
                    question=b.question,
                    ans_A=answers_A[k],
                    ans_B=answers_B[k],
                    ground_truth=b.ground_truth,
                    metric=metric,
                )
            ))
        outcomes = await asyncio.gather(*tasks)

        for o in outcomes:
            if o == "B":
                wins += 1
            elif o == "A":
                losses += 1
            else:
                ties += 1

        p_hat, ci_low, ci_high = self.wilson_ci(wins, losses, z=z)
        n_effective = wins + losses  # ties are omitted
        return {
            "wins": wins, "losses": losses, "ties": ties,
            "n_effective": n_effective, "p_hat": p_hat,
            "ci_low": ci_low, "ci_high": ci_high,
        }


# ========================================================================= #
#                             Instruction Optimizer                         #
# ========================================================================= #
class InstructionOptimizer:
    """InstructionOptimizer

    Optimizes instructions using the history of (instruction, score) pairs
    and a small set of exemplars.
    """

    def __init__(self, backend, max_hist: int, temperature: float = 1.0):
        self.backend = backend
        self.temperature = temperature
        self.max_hist = max_hist

    def run(self, state: TrainState) -> str:
        """Propose an optimized instruction."""
        history = state.history
        tail_histories = history[-self.max_hist:] if self.max_hist > 0 else []

        def fmt_one(idx: int, h: dict) -> str:
            instruction = h.get("instruction", "").strip().replace("\n", "\\n")
            win_rate = h.get("p_hat", 0.0)
            ci_low = h.get("ci_low", 0.0)
            ci_high = h.get("ci_high", 1.0)
            wins  = h.get("wins", 0)
            losses  = h.get("losses", 0)
            ties  = h.get("ties", 0)
            acc = h.get("accepted", None)
            acc_txt = "✅採用" if acc else "❌不採用" if acc is not None else ""
            return (f"- イテレーション {idx}: 指示: {instruction} | "
                    f"勝率: {win_rate:.3f} (CI {ci_low:.3f}–{ci_high:.3f}; W/L/T={wins}/{losses}/{ties}) {acc_txt}")

        if tail_histories:
            start_idx = len(history) - len(tail_histories) + 1
            history_lines = [fmt_one(start_idx + i, h) for i, h in enumerate(tail_histories)]
            log = "\n".join(history_lines)
        else:
            log = "(履歴なし：このラウンドは過去を見せない設定です)"

        def embed_ins_into_examples(examples):
            parts = []
            for ex in examples:
                parts.append(
                    "<INS>\n"
                    "入力:\n"
                    f"{ex.question}\n"
                    "出力:\n"
                    f"{ex.ground_truth}\n"
                )
            return "".join(parts)

        ins_embedding_examples = embed_ins_into_examples(state.examples)

        # ---- Step Adapter ----
        def choose_step_size(
            history: list[dict],
            curr_round: int,
            total_rounds: int,
            # Customizable thresholds per phase:
            # (consecutive rejects to reach "medium", consecutive rejects to reach "large")
            early_thresh: tuple[int, int] = (1, 3),   # first 50% of rounds
            mid_thresh:   tuple[int, int] = (2, 4),   # middle 30% of rounds
            late_thresh:  tuple[int, int] = (5, 8),   # last 20% of rounds
        ) -> str:
            """Decide step size (small/medium/large) from recent consecutive rejections and training phase."""

            # 1) Count consecutive rejections at the end of history (tail False streak).
            rej_streak = 0
            for h in reversed(history):
                acc = h.get("accepted", None)
                if acc is False:
                    rej_streak += 1
                else:
                    # Break on first non-rejection (True or None/missing) — streak ends.
                    break

            # 2) Phase by progress: first 50% / up to 80% / final 20%.
            prog = curr_round / max(1, total_rounds)
            if prog <= 0.50:
                med_th, large_th = early_thresh
            elif prog <= 0.80:
                med_th, large_th = mid_thresh
            else:
                med_th, large_th = late_thresh

            # 3) Map streak to step size via thresholds.
            if rej_streak >= large_th:
                return "large"
            if rej_streak >= med_th:
                return "medium"
            return "small"  # Little/no recent rejection → make small, incremental edits.

        step = choose_step_size(
            history=state.history,
            curr_round=state.round,
            total_rounds=state.max_rounds,
        )

        if step == "small":
            change_rule = "1) 変更は小さく具体的に（1〜2点）。"
        elif step == "medium":
            change_rule = "1) 中程度の変更（2〜4点）。構成・順序の見直しも可。"
        else:  # large
            change_rule = ("1) 大きめの変更（全面の再構成や手順の再設計を含む）。"
                        "冗長さを排し、出力形式と検証手順を明確化。")

        user_prompt = (
            "いくつかの指示文と、その評価履歴（対ベースラインの勝率と信頼区間）が与えられます。\n"
            "履歴を分析し、過去に改善に寄与した特徴を維持しつつ、さらに勝率が上がる新しい指示文を1つ提案してください。\n"
            "ここでの目標は、勝率の下限（95% CI の下界）を押し上げることです。"
            "編集の大きさは下記『要件』に従ってください。冗長性や形式揺れは避けてください。\n\n"
            "【評価履歴】\n"
            f"{log}\n\n"
            "【適用例（あなたの指示文がどう使われるか）】\n"
            "以下の <INS> をあなたが提案する指示文に置き換え、各入力に対して期待される出力を返すことを想定します。"
            "この例は説明用であり、実際の評価データには含まれません。\n"
        )
        user_prompt += ins_embedding_examples + "\n"
        user_prompt += (
            "要件:\n"
            f"{change_rule}\n"
            "2) 入力条件・出力形式・検証/自己チェックの明確化を優先。\n"
            "3) 冗長表現・曖昧表現・過度な禁止事項の乱発は避ける。\n"
            "4) 出力は角括弧 [ ] で囲んだ指示文のみ。説明文は不要。\n"
        )

        content = self.backend.chat(
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature
        )
        if content is None:
            raise ValueError("Model returned no content in InstructionOptimizer")

        text = content.strip()
        m = re.search(r"\[(.+?)\]", text, flags=re.S)
        ins = m.group(1).strip() if m else text
        current_optimized_instruction = f"[{ins}]"
        return current_optimized_instruction
