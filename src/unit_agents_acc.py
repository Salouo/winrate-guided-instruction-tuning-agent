"""unit_agents_acc.py

Minimal agent utilities for absolute accuracy (EM / math_em).

- Async TaskExecutor runs one instruction over a batch with retries & loop-bound semaphore.
- Scorer computes EM and math_em (hash-line/TeX fraction/numeric parsing via Decimal).
- Returns per-batch {'correct', 'total', 'acc'}; backend kept for ctor compatibility.

If accuracy computation is not required, this module is unnecessary.
"""

import asyncio
import regex as re
from typing import Literal, Sequence

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIConnectionError, APIError
from google.api_core import exceptions as gexc
from decimal import Decimal, InvalidOperation

from .utils import TrainState, EvalState

# ------------------------- retry / concurrency helpers ------------------------- #

RETRY_EXCS = (
    RateLimitError, APIConnectionError, APIError, asyncio.TimeoutError,
    gexc.GoogleAPIError, gexc.DeadlineExceeded, gexc.ServiceUnavailable
)

class _LoopBoundSemaphore:
    """Create/refresh a Semaphore bound to the current running event loop."""
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

# ================================== Executor ================================== #

class TaskExecutor:
    """Run one instruction over a batch slice and return answers."""
    def __init__(self, backend, temperature: float = 0.0, concurrency: int = 5):
        self.backend = backend
        self.temperature = temperature
        self._sem = _LoopBoundSemaphore(concurrency)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(RETRY_EXCS),
        reraise=True,
    )
    async def _execute_one(self, system_instruction: str, question: str) -> str:
        async with self._sem.get():
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": question},
            ]
            content = await self.backend.chat(messages=messages, temperature=self.temperature)
        return (content or "").strip()

    async def run_single(self, state: TrainState | EvalState, instruction: str) -> list[str]:
        """Execute a single instruction for the current batch slice of `state`."""
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        batch = state.samples[i:j]
        tasks = [asyncio.create_task(self._execute_one(instruction, b.question)) for b in batch]
        return await asyncio.gather(*tasks)

# =================================== Scorer =================================== #

MetricType = Literal["em", "math_em"]

# simple normalization for EM
_PUNCT_RE = re.compile(r'[\p{P}\p{S}]', re.UNICODE)
_WS_RE    = re.compile(r'\s+', re.UNICODE)

# numeric extraction for math_em
_NUM_RE   = re.compile(r'[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+')
_HASH_NUM = re.compile(r'####\s*([-\d,]*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+)', re.IGNORECASE)
_FRAC_TEX = re.compile(r'\\d?frac\{([-\d]+)\}\{([-\d]+)\}')

class Scorer:
    """Compute absolute accuracy for EM / math_em on a batch slice."""
    def __init__(self, backend=None, temperature: float = 0.0, concurrency: int = 5):
        # backend is unused for absolute accuracy; kept for constructor compatibility
        self.backend = backend

    # ---- EM helpers ----
    def _normalize_em(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = _PUNCT_RE.sub(" ", s)
        s = _WS_RE.sub(" ", s).strip()
        return s

    def _exact_match(self, pred: str, ref: str | None) -> int:
        if ref is None:
            return 0
        return int(self._normalize_em(pred) == self._normalize_em(ref))

    # ---- math_em helpers ----
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
        s = num_str.strip().rstrip(".,;").replace(",", "")
        if "/" in s and s.count("/") == 1:
            n, d = s.split("/")
            try:
                return Decimal(n) / Decimal(d)
            except (InvalidOperation, ZeroDivisionError):
                return None
        try:
            return Decimal(s)
        except InvalidOperation:
            try:
                return Decimal(s.replace("e", "E"))
            except InvalidOperation:
                return None

    @classmethod
    def _exact_match_numeric(cls, pred: str | None, ref: str | None) -> int:
        p = cls._to_decimal(cls._extract_final_number(pred))
        g = cls._to_decimal(cls._extract_final_number(ref))
        if p is None or g is None:
            return 0
        return int(p == g)

    # ---- public API ----
    def compute_accuracy(
        self,
        state: TrainState | EvalState,
        answers: Sequence[str],
        metric: MetricType,
    ) -> dict:
        """
        Compute absolute accuracy for the current batch slice.
        Returns: {'correct': int, 'total': int, 'acc': float}
        """
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        batch = state.samples[i:j]
        assert len(answers) == len(batch), f"len mismatch: answers={len(answers)} batch={len(batch)}"

        if metric not in ("em", "math_em"):
            raise ValueError(f"Unsupported metric for absolute accuracy: {metric}")

        correct = 0
        if metric == "em":
            for a, b in zip(answers, batch):
                correct += self._exact_match(a, getattr(b, "ground_truth", None))
        else:  # math_em
            for a, b in zip(answers, batch):
                correct += self._exact_match_numeric(a, getattr(b, "ground_truth", None))

        total = len(batch)
        acc = (correct / total) if total else 0.0
        return {"correct": int(correct), "total": int(total), "acc": acc}
