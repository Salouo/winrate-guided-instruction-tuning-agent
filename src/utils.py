"""utils.py

Shared utilities for the instruction-tuning pipeline.
"""

import operator
import json
import os
import random
from pathlib import Path
from typing import Optional, Literal, Any, List, Dict, Tuple

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .backends import (
    GPTAsyncBackend,
    GPTSyncBackend,
    GeminiAsyncBackend,
    GeminiSyncBackend,
    ClaudeAsyncBackend,
    ClaudeSyncBackend,
)

# ROOT/src
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------  Data model: a single QA sample  ------------------------------------ #
class Sample(BaseModel):
    """Single QA sample."""
    question: str = Field(..., description="Question to be answered")
    answer: str = Field(default="", description="(Optional) placeholder for the model's answer")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth / reference answer")


# ------------------------------------  State used in pairwise training  ------------------------------------ #
class TrainState(BaseModel):
    """Graph state for training."""

    # Instructions
    current_instruction: str = Field(..., description="Candidate instruction being evaluated this round")
    # Keep XML field optional for compatibility; duel pipeline does not use it.
    current_xml_instruction: str = Field(default="", description="(Optional) XML-format instruction, unused in duel")

    # Data
    dataset_name: str = Field(..., description="Name of the dataset")
    samples: list[Sample] = Field(default_factory=list, description="Shuffled training samples")
    examples: list[Sample] = Field(default_factory=list, description="Few-shot examples for the optimizer")

    # Models
    executor_model: str = Field(..., description="Model used by TaskExecutor")
    scorer_model: str = Field(..., description="Model used by LLMScorer")
    optimizer_model: str = Field(..., description="Model used by InstructionOptimizer")

    # Batching
    batch_size: int = Field(default=8, description="Number of samples per round")
    batch_start: int = Field(default=0, description="Start index of the current batch within samples")

    # Best-so-far baseline
    best_instruction: str = Field(default="", description="Current baseline instruction (best-so-far)")
    best_score: float = Field(default=-1.0, description="(Optional) reserved; not used in duel")

    # Pairwise history accumulator
    # Each record typically includes:
    #   round, instruction, wins, losses, ties, n_effective, p_hat, ci_low, ci_high, accepted, batch_start, batch_size
    history: Annotated[list[dict], operator.add] = Field(
        default_factory=list, description="Pairwise round logs (candidate metrics and metadata)"
    )
    max_hist: int = Field(default=100, description="Max number of shown histories")

    # Loop control
    round: int = Field(default=1, description="Current round (1-based)")
    max_rounds: int = Field(..., description="Maximum number of rounds before stopping")

    # Checkpoint flag
    load_ckpt: bool = Field(default=False, description="Whether current state was loaded from a checkpoint")


class EvalState(BaseModel):
    """Graph state for evaluation."""

    current_instruction: str = Field(..., description="Candidate instruction to evaluate (best from training)")

    dataset_name: str = Field(..., description="Name of the dataset")
    samples: list[Sample] = Field(default_factory=list, description="Evaluation samples")

    # Models
    executor_model: str = Field(..., description="Model used by TaskExecutor")
    scorer_model: str = Field(..., description="Model used by LLMScorer")
    optimizer_model: str = Field(..., description="Optimizer model used in training (for path consistency)")

    # Batching
    batch_size: int = Field(default=8, description="Number of samples per round")
    batch_start: int = Field(default=0, description="Start index of the current batch within samples")

    # Accumulate per-round pairwise win rates (p_hat)
    scores: Annotated[list[float], operator.add] = Field(default_factory=list, description="List of per-round win rates")

    # Loop control
    round: int = Field(default=1, description="Current round (1-based)")
    max_rounds: int = Field(default=5, description="Maximum number of rounds before stopping")

    # History settings
    max_hist: int = Field(default=100, description="Max number of shown histories")

    # Checkpoint flag
    load_ckpt: bool = Field(default=False, description="Whether current state was loaded from a checkpoint")

    # Mode label
    mode: str = Field(default="eval", description="Evaluation mode label (e.g., 'eval' or 'baseline')")


# ------------------------------------  Dataset helpers  ------------------------------------ #
def load_dataset(
    dataset_path: Path,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[Sample], str, str]:
    """
    (Function) Load samples, the initial instruction, and the metric from a dataset JSON file.

    Returns:
    - samples (list[Sample]): Parsed sample objects (optionally shuffled).
    - init_instruction (str): Initial instruction string from the dataset.
    - metric (str): Evaluation metric name (e.g., "em", "math_em", or "llm").
    """
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("samples", [])
    init_instruction = data.get("instruction", "")
    metric = data.get("metrics")

    all_samples: list[Sample] = []
    for item in items:
        question = item.get("input")
        ground_truth = item.get("output", None)
        all_samples.append(Sample(question=question, ground_truth=ground_truth))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    return all_samples, init_instruction, metric


def split_samples_and_examples(all_samples: list[Sample], n_examples: int = 3):
    """
    (Function) Split a dataset into (samples, examples), where the last n_examples
    are reserved as optimization exemplars showing to an InstructionOptimizer object.

    Returns:
    - samples (list[Sample]): Subset used for training/evaluation.
    - examples (list[Sample]): Tail subset used as exemplars for the optimizer.
    """
    n_examples = max(0, min(n_examples, len(all_samples)))
    samples = all_samples[:-n_examples] if n_examples > 0 else all_samples
    examples = all_samples[-n_examples:] if n_examples > 0 else []
    return samples, examples


# ------------------------------------  Checkpoint I/O  ------------------------------------ #
def save_train_ckpt(state: TrainState, overrides: dict[str, Any], mode: str = "train") -> Path:
    """
    (Function) Save a training checkpoint for the duel pipeline, summarizing
    best instruction, average win rate, and trajectory.

    Returns:
    - path (Path): Filesystem path to the written training checkpoint JSON.
    """
    ckpt_path = get_ckpt_path(
        dataset_name=state.dataset_name, 
        executor_model=state.executor_model, 
        scorer_model=state.scorer_model, 
        optimizer_model=state.optimizer_model, 
        batch_size=state.batch_size, 
        max_hist=state.max_hist, 
        mode=mode
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = overrides.get("history", [])
    best_score = overrides.get("best_score")  # kept for compatibility
    best_instruction = overrides.get("best_instruction", "")
    batch_start = overrides.get("batch_start", 0)
    metric = overrides.get("metric")

    # Average training win rate over recorded rounds; fallback to 'average_score' if present (compatibility).
    def _extract_wr(h: dict) -> Optional[float]:
        if "p_hat" in h:
            try:
                return float(h["p_hat"])
            except Exception:
                return None
        if "average_score" in h:  # legacy
            try:
                return float(h["average_score"])
            except Exception:
                return None
        return None

    wr_vals = [v for v in (_extract_wr(h) for h in history) if v is not None]
    train_avg_win_rate = (sum(wr_vals) / len(wr_vals)) if wr_vals else 0.0

    # Persist a compact trajectory (keep the key metrics)
    traj = []
    for h in history:
        traj.append({
            "round": int(h.get("round", 0)),
            "instruction": h.get("instruction", ""),
            "wins": int(h.get("wins", 0)),
            "losses": int(h.get("losses", 0)),
            "ties": int(h.get("ties", 0)),
            "n_effective": int(h.get("n_effective", h.get("wins", 0) + h.get("losses", 0))),
            "p_hat": float(h.get("p_hat", 0.0)),
            "ci_low": float(h.get("ci_low", 0.0)),
            "ci_high": float(h.get("ci_high", 0.0)),
            "accepted": bool(h.get("accepted", False)),
            "batch_start": int(h.get("batch_start", 0)),
            "batch_size": int(h.get("batch_size", 0)),
        })

    curr_ckpt = {
        "best_instruction": best_instruction,
        "best_score": float(best_score),
        "metric": metric,
        "max_hist": state.max_hist,
        "train_avg_win_rate": train_avg_win_rate,
        "n_rounds": traj[-1]["round"] if traj else 0,
        "trajectory": traj,
        "batch_start": int(batch_start),
    }

    ckpt_path.write_text(json.dumps(curr_ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    return ckpt_path


def load_train_ckpt(path: Path) -> Tuple[List[Dict[str, Any]], int, str, str]:
    """
    (Function) Load a training checkpoint and parse the stored trajectory
    and best instruction.

    Returns:
    - trajectory (list[dict]): Per-round records (ordered by round).
    - n_rounds (int): Number of rounds completed.
    - last_instruction (str): Instruction from the most recent round.
    - best_instruction (str): Best-so-far baseline instruction.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    trajectory = data.get("trajectory", [])
    trajectory = sorted(trajectory, key=lambda x: int(x.get("round", 0)))
    n_rounds = int(data.get("n_rounds", len(trajectory)))
    last_instruction = trajectory[-1]["instruction"] if trajectory else ""
    best_instruction = data.get("best_instruction", "")
    return trajectory, n_rounds, last_instruction, best_instruction


def save_eval_ckpt(state: EvalState, overrides: dict[str, Any], mode: str = "eval") -> Path:
    """
    (Function) Save an evaluation checkpoint that stores per-round win rates
    and summary statistics.

    Returns:
    - path (Path): Filesystem path to the written evaluation checkpoint JSON.
    """
    ckpt_path = get_ckpt_path(
        state.dataset_name, 
        state.executor_model, 
        state.scorer_model, 
        state.optimizer_model, 
        state.batch_size, 
        max_hist=state.max_hist,
        mode=mode
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    eval_scores = overrides.get("scores", [])
    eval_avg_win_rate = (sum(float(s) for s in eval_scores) / len(eval_scores)) if eval_scores else 0.0
    batch_start = overrides.get("batch_start", 0)
    instruction = overrides.get("instruction", state.current_instruction)
    metric = overrides.get("metric", "")

    curr_ckpt = {
        "instruction": instruction,  # stored for reference; not required by the loop
        "eval_avg_win_rate": eval_avg_win_rate,
        "metric": metric,
        "max_hist": state.max_hist,
        "n_eval_rounds": len(eval_scores),
        "scores": [float(s) for s in eval_scores],
        "batch_start": int(batch_start)
    }

    ckpt_path.write_text(json.dumps(curr_ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    return ckpt_path


def load_eval_ckpt(path: Path, mode: str = "eval") -> Tuple[List[float], int, str]:
    """
    (Function) Load an evaluation checkpoint and extract win-rate history
    and the stored candidate instruction.

    Returns:
    - scores (list[float]): Per-round win rates (p_hat values).
    - n_rounds (int): Number of evaluation rounds completed.
    - instruction (str): Candidate instruction recorded in the checkpoint.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = [float(s) for s in data.get("scores", [])]
    eval_n_rounds = int(data.get("n_eval_rounds", len(scores)))
    instruction = data.get("instruction", "")
    return scores, eval_n_rounds, instruction


def get_ckpt_path(
    dataset_name: str,
    executor_model: str,
    scorer_model: str,
    optimizer_model: str,
    batch_size: int,
    max_hist: int,
    mode: str,
) -> Path:
    """
    (Function) Build a standardized checkpoint path for a given dataset/config.

    Returns:
    - path (Path): Absolute path to the target checkpoint JSON (by mode).
    """
    dataset_key = f"{dataset_name}_{batch_size}"
    base = Path("outputs") / dataset_key
    save_dir = base / str(max_hist) / f"{executor_model}_{scorer_model}_{optimizer_model}"
    if mode == "train":
        return save_dir / "checkpoint_train.json"
    elif mode == "eval":
        return save_dir / "checkpoint_eval.json"
    elif mode == "baseline":
        return save_dir / "checkpoint_baseline.json"
    else:
        raise ValueError("Unknown mode for get_ckpt_path().")


# ----------------------- Backends factory (adapter over vendors) ----------------------- #
def _require_env(k: str) -> str:
    """
    (Function) Read a required environment variable or raise a helpful error.

    Returns:
    - value (str): The non-empty environment variable value.
    """
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Missing env var: {k}. Did you set it in .env?")
    return v


def make_backend(vendor: str, model: str, mode: Literal["async", "sync"]):
    """
    (Function) Create a backend instance for the given vendor/model and mode.

    Returns:
    - backend (object): A sync/async backend exposing `chat(messages, temperature) -> str`.
    """
    vendor = vendor.lower()
    if mode == "async":
        if vendor == "openai":
            return GPTAsyncBackend(api_key=_require_env("OPENAI_API_KEY"), model=model)
        elif vendor == "google":
            project = _require_env("VERTEXAI_PROJECT")
            location = _require_env("VERTEXAI_LOCATION")
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                raise RuntimeError("Vertex requires GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON.")
            return GeminiAsyncBackend(model=model, project=project, location=location)
        elif vendor == "anthropic":
            return ClaudeAsyncBackend(api_key=_require_env("ANTHROPIC_API_KEY"), model=model)
        else:
            raise ValueError(f"Unknown vendor: {vendor}")

    elif mode == "sync":
        if vendor == "openai":
            return GPTSyncBackend(api_key=_require_env("OPENAI_API_KEY"), model=model)
        elif vendor == "google":
            project = _require_env("VERTEXAI_PROJECT")
            location = _require_env("VERTEXAI_LOCATION")
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                raise RuntimeError("Vertex requires GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON.")
            return GeminiSyncBackend(model=model, project=project, location=location)
        elif vendor == "anthropic":
            return ClaudeSyncBackend(api_key=_require_env("ANTHROPIC_API_KEY"), model=model)
        else:
            raise ValueError(f"Unknown vendor: {vendor}")
    else:
        raise ValueError("mode must be 'async' or 'sync'")


def obtain_phase(curr_round: int, max_rounds: int) -> str:
    """
    (Function) Map training progress to a phase label: early / mid / late.

    Returns:
    - phase (str): One of {"early", "mid", "late"} based on round proportion.
    """
    total_rounds = max(1, max_rounds)
    proportion = curr_round / total_rounds

    if proportion <= 0.50: 
        return "early"
    elif proportion <= 0.80: 
        return "mid"
    else:  
        return "late"


def extract_accepted_instructions(train_ckpt_path: Path) -> List[Tuple[int, str]]:
    """
    (Function) Read a training checkpoint and collect unique accepted
    instructions in their first acceptance order.

    Returns:
    - accepted (list[tuple[int, str]]): (round, instruction) pairs sorted by round.
    """
    data = json.loads(train_ckpt_path.read_text(encoding="utf-8"))
    traj = data.get("trajectory") or []

    accepted = []
    seen = set()
    for t in traj:
        if t.get("accepted") is True:
            ins = (t.get("instruction") or "").strip()
            if ins and ins not in seen:
                seen.add(ins)
                accepted.append((int(t.get("round", 0)), ins))

    accepted.sort(key=lambda x: x[0])
    return accepted
