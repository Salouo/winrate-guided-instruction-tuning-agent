"""eval.py

Pairwise evaluator for optimized vs. baseline instructions provided by datasets.

- Runs batch-wise duels (A=baseline, B=candidate) to estimate win rate with Wilson CI.
- Supports single best-instruction or multi-eval over accepted candidates.
- Saves/loads eval checkpoints with per-round p_hat (every CKPT_EVERY rounds).
- Async, vendor-agnostic backends; configurable rounds, batch size, concurrency, temperatures, z.
- CLI: choose vendors/models/dataset, max_hist, accepted-only mode.
"""

import argparse
import asyncio
import sys
import json
from typing import Any, List
from pathlib import Path

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .utils import (
    EvalState,
    load_dataset,
    make_backend,
    get_ckpt_path,
    load_train_ckpt,
    load_eval_ckpt,
    save_eval_ckpt,
    extract_accepted_instructions
)
from .unit_agents import TaskExecutor, Scorer

# Load API keys
loaded = load_dotenv()

CKPT_EVERY = 3  # Save checkpoint every N rounds


# ========================================================================= #
#                                  Evaluator                                #
# ========================================================================= #
class Evaluator:
    """
    Pairwise evaluator:
      - For each round (batch):
        1) run executor twice (baseline vs candidate)
        2) pairwise judge -> win rate & CI
      - Append p_hat to state.scores
      - Stop when round > max_rounds
    """

    def __init__(
        self,
        task_executor: TaskExecutor,
        scorer: Scorer,
        baseline_instruction: str,
        candidate_instruction: str,
        metric: str = "llm",
        z: float = 1.96,
        save_ckpt: bool = True,
    ):
        """
        (Function) Initialize evaluator with executor, scorer, baseline/candidate instructions,
        and evaluation settings.

        Returns:
        - None
        """
        self.task_executor = task_executor
        self.scorer = scorer
        self.baseline_instruction = baseline_instruction
        self.candidate_instruction = candidate_instruction
        # metric ∈ {"em","llm"}
        self.metric = metric
        self.z = z
        self.save_ckpt = save_ckpt

    # ------------------------------ graph --------------------------------
    def _create_graph(self) -> StateGraph:
        """
        (Function) Build the evaluation state graph with a single node that
        executes and scores each batch until termination.

        Returns:
        - graph (StateGraph): Uncompiled LangGraph graph for evaluation.
        """
        graph = StateGraph(EvalState)
        graph.add_node("execute_and_score", self._execute_and_score)
        graph.set_entry_point("execute_and_score")

        def should_stop(state: EvalState) -> bool:
            return state.round > state.max_rounds

        graph.add_conditional_edges(
            "execute_and_score", should_stop, {True: END, False: "execute_and_score"}
        )
        return graph

    async def _execute_and_score(self, state: EvalState) -> dict[str, Any]:
        """
        (Function) Run a pairwise duel on the current batch (baseline vs candidate),
        compute win-rate and CI, log progress, and optionally checkpoint.

        Returns:
        - delta (dict): Partial state update containing:
          - "scores" (list[float]): p_hat for this batch.
          - "batch_start" (int): Next batch start index (or 0 if wrapped).
          - "round" (int): Next round index.
        """
        # Current batch slice
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        next_start = j if j < len(state.samples) else 0
        curr_round = state.round

        # Generate answers under baseline (A) and candidate (B)
        answers_A, answers_B = await self.task_executor.run_pair(
            state=state,
            baseline_instruction=self.baseline_instruction,
            candidate_instruction=self.candidate_instruction,
        )

        # Pairwise judge
        stats = await self.scorer.run_pairwise(
            state=state,
            answers_A=answers_A,
            answers_B=answers_B,
            metric=self.metric,  # "em" | "llm"
            z=self.z,
        )

        # Log
        print(
            f"[Round {curr_round}/{state.max_rounds}] batch {i}:{j} -> "
            f"p_hat={stats['p_hat']:.3f} "
            f"(CI {stats['ci_low']:.3f}-{stats['ci_high']:.3f}; "
            f"W/L/T={stats['wins']}/{stats['losses']}/{stats['ties']}; "
            f"n_eff={stats['n_effective']})"
        )

        # Prepare overrides for a single eval checkpoint
        overrides = {
            "scores": state.scores + [float(stats["p_hat"])],
            "batch_start": next_start,
            "instruction": self.candidate_instruction,  # candidate used in this evaluation
        }

        # Save eval checkpoint periodically or at the end (can be disabled)
        if self.save_ckpt and ((curr_round != 0 and curr_round % CKPT_EVERY == 0) or (curr_round == state.max_rounds)):
            ckpt_path = save_eval_ckpt(state=state, overrides=overrides)
            state.load_ckpt = True
            print(f"Eval checkpoint saved in {ckpt_path}")

        return {
            "scores": [float(stats["p_hat"])],
            "batch_start": next_start,
            "round": state.round + 1,
        }

    # ------------------------------ Run ---------------------------------- #
    async def run(self, init_state: EvalState) -> dict[str, Any]:
        """
        (Function) Compile and execute the evaluation graph starting from the
        provided initial state.

        Returns:
        - final_state (dict[str, Any]): Aggregated state produced by the graph.
        """
        compiled = self._create_graph().compile()
        recursion_limit = init_state.max_rounds * 3 + 5
        print("Running pairwise evaluation by rounds...")
        final_state: dict[str, Any] = await compiled.ainvoke(
            init_state, config={"recursion_limit": recursion_limit}
        )
        print("Finish!\n")
        return final_state


# -------------------------- helpers for accepted list -------------------------- #
def average(lst: List[float]) -> float:
    """
    (Function) Compute the arithmetic mean of a list of floats (safe on empty).

    Returns:
    - mean (float): Average value, or 0.0 if the list is empty.
    """
    return sum(lst) / len(lst) if lst else 0.0


# =================================== main =================================== #
async def main():
    """
    (Function) Parse CLI args, load dataset and checkpoints, construct backends
    and agents, run pairwise evaluation (single best or multiple accepted),
    and emit/save results.

    Returns:
    - None
    """
    # ------------------------- Argparse settings -------------------------
    parser = argparse.ArgumentParser(description="Pairwise evaluator")

    # Vendors
    parser.add_argument(
        "--executor-vendor", type=str, default="openai", choices=["openai", "google", "anthropic"]
    )
    parser.add_argument(
        "--scorer-vendor", type=str, default="openai", choices=["openai", "google", "anthropic"]
    )

    # Models
    """
    OpenAI: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4o
    Google: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-2.0-flash-lite
    Anthropic: claude-opus-4-1, claude-opus-4, claude-sonnet-4, claude-3-7-sonnet, claude-3-5-haiku, claude-3-haiku
    """
    parser.add_argument("--executor-model", type=str, default="gpt-4o", help="LLM for TaskExecutor")
    parser.add_argument("--scorer-model", type=str, default="gpt-4.1", help="LLM for Scorer")
    parser.add_argument("--optimizer-model", type=str, default="gpt-5", help="Optimizer model used in training")

    # Eval setup
    parser.add_argument("--test-set", type=str, default="bigbenchhard_ja_cot", help="Test set name")
    parser.add_argument("--max-hist", type=int, default=100, help="Only pass the last N history items to the optimizer (0 = no history).")
    parser.add_argument("--max-rounds", type=int, default=100, help="Max eval rounds")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per round")
    parser.add_argument("--concurrency", type=int, default=8, help="Async concurrency for execution/judge")
    parser.add_argument("--exec-temperature", type=float, default=0.0, help="Executor temperature")
    parser.add_argument("--scor-temperature", type=float, default=0.0, help="Scorer (LLM judge) temperature")
    parser.add_argument("--z", type=float, default=1.96, help="Z for Wilson CI (1.96≈95% CI)")

    # Multi-eval mode over accepted instructions
    parser.add_argument(
        "--accepted-only",
        default=False,
        action="store_true",
        help="Evaluate baseline vs EACH accepted instruction from training trajectory (ordered)",
    )

    args = parser.parse_args()

    # ------------------------- Load test set -------------------------
    dataset_path = Path("datasets") / "test" / f"{args.test_set}.json"
    test_samples, init_instruction, metric = load_dataset(
        dataset_path=dataset_path, shuffle=False
    )

    # ------------------------- Set backends -------------------------
    be_executor = make_backend(
        vendor=args.executor_vendor, model=args.executor_model, mode="async"
    )
    be_scorer = make_backend(
        vendor=args.scorer_vendor, model=args.scorer_model, mode="async"
    )

    task_executor = TaskExecutor(
        backend=be_executor, temperature=args.exec_temperature, concurrency=args.concurrency
    )
    scorer = Scorer(
        backend=be_scorer, temperature=args.scor_temperature, concurrency=args.concurrency
    )

    # ------------------------- Choose baseline -------------------------
    baseline_instruction = f"[{init_instruction}]"

    # Handle special alias for local model naming
    executor_model = "gpt-oss-20b" if args.executor_model == "gpt-oss:20b" else args.executor_model

    # Training checkpoint path (optimizer_model must match training)
    train_ckpt_path = get_ckpt_path(
        dataset_name=args.test_set,
        executor_model=executor_model,
        scorer_model=args.scorer_model,
        optimizer_model=args.optimizer_model,  # must match training
        batch_size=args.batch_size,
        max_hist=args.max_hist,
        mode="train",
    )
    if not train_ckpt_path.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {train_ckpt_path}\n"
            f"Please ensure --optimizer-model and --batch-size match those used in training."
        )

    # ------------------------- Multi-Accepted-Instructions Mode -------------------------
    if args.accepted_only:
        # Extract accepted instructions from train checkpoint
        accepted_list = extract_accepted_instructions(train_ckpt_path)
        if not accepted_list:
            print("No accepted instructions found in training trajectory.")
            sys.exit(0)

        print("\n=== Accepted instructions (ordered by first acceptance round) ===")
        for r, _ in accepted_list:
            print(f" - round {r}")
        print("=================================================================\n")

        # Compute rounds budget from data
        n_test = len(test_samples)
        total_rounds = max(1, n_test // args.batch_size)
        max_rounds = min(args.max_rounds, total_rounds)

        # Prepare output path (same dir as single eval ckpt, different file name)
        single_eval_ckpt = get_ckpt_path(
            dataset_name=args.test_set,
            executor_model=executor_model,
            scorer_model=args.scorer_model,
            optimizer_model=args.optimizer_model,
            batch_size=args.batch_size,
            max_hist=args.max_hist,
            mode="eval",
        )
        out_path = single_eval_ckpt.with_name("checkpoint_eval_accepted.json")

        results = []
        for idx, (round_no, cand_ins) in enumerate(accepted_list, start=1):
            if cand_ins == baseline_instruction:
                print(f"Candidate instruction and baseline instruction are the same.\n{cand_ins}")
                continue
            print(f"### [{idx}/{len(accepted_list)}] Evaluating accepted instruction from round {round_no} ###")
            print(f"Candidate (head): {cand_ins[:120]}{'...' if len(cand_ins)>120 else ''}")

            evaluator = Evaluator(
                task_executor=task_executor,
                scorer=scorer,
                baseline_instruction=baseline_instruction,
                candidate_instruction=cand_ins,
                metric=metric,
                z=args.z,
                save_ckpt=False,  # do not overwrite single-eval ckpt
            )

            init_state = EvalState(
                current_instruction=cand_ins,
                dataset_name=args.test_set,
                max_hist=args.max_hist,
                executor_model=executor_model,
                scorer_model=args.scorer_model,
                optimizer_model=args.optimizer_model,
                scores=[],
                round=1,
                samples=test_samples,
                batch_start=0,
                batch_size=args.batch_size,
                max_rounds=max_rounds,
                load_ckpt=False,
            )

            out = await evaluator.run(init_state=init_state)
            p_hats = [float(x) for x in out.get("scores", [])]
            avg_wr = average(p_hats)
            print(f"=> Average win rate for round {round_no} instruction: {avg_wr:.3f} over {len(p_hats)} round(s)\n")

            results.append({
                "round": round_no,
                "n_eval_rounds": len(p_hats),
                "avg_win_rate": avg_wr,
                "p_hats": p_hats,
                "instruction": cand_ins,
            })

        # Save a compact summary
        out_obj = {
            "dataset": args.test_set,
            "executor_vendor": args.executor_vendor,
            "executor_model": executor_model,
            "scorer_vendor": args.scorer_vendor,
            "scorer_model": args.scorer_model,
            "optimizer_model": args.optimizer_model,
            "batch_size": args.batch_size,
            "max_rounds": args.max_rounds,
            "metric": metric,
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVED] {out_path}")
        return

    # ------------------------- Single-Accepted-Best-Instructions Mode -------------------------
    _, _, _, best_instruction = load_train_ckpt(train_ckpt_path)
    if not best_instruction:
        raise RuntimeError("best_instruction not found in training checkpoint.")
    candidate_instruction = best_instruction

    # If candidate instruction and baseline instruction are the same,
    if candidate_instruction == baseline_instruction:
        print(f"Candidate instruction and baseline instruction are the same.\n{candidate_instruction}")
        return

    print("\n=== Pairwise Eval Setup ===")
    print(f"Baseline  : {baseline_instruction[:120]}{'...' if len(baseline_instruction)>120 else ''}")
    print(f"Candidate : {candidate_instruction[:120]}{'...' if len(candidate_instruction)>120 else ''}")
    print(f"Metric    : {metric}, Z={args.z}")
    print("===========================\n")

    # ------------------------- Prepare EvalState -------------------------
    batch_size = args.batch_size
    n_test = len(test_samples)
    total_rounds = max(1, n_test // batch_size)
    max_rounds = min(args.max_rounds, total_rounds)

    # Single eval checkpoint (store per-round p_hat and the candidate instruction used)
    eval_ckpt_path = get_ckpt_path(
        dataset_name=args.test_set,
        executor_model=executor_model,
        scorer_model=args.scorer_model,
        optimizer_model=args.optimizer_model,  # consistent with training
        max_hist=args.max_hist,
        batch_size=args.batch_size,
        mode="eval",
    )

    # If candidate instruction and baseline instruction are the same, return directly.
    if candidate_instruction == baseline_instruction:
        print(f"Candidate instruction and baseline instruction are the same.\n{candidate_instruction}")
        eval_ckpt_path.write_text(json.dumps({"default": "No optimized instruction found."}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if eval_ckpt_path.exists():
        print(f"Checkpoint exists. Load checkpoint from {eval_ckpt_path}...")
        print(f"Metric: {metric}")
        scores, eval_n_rounds, _ = load_eval_ckpt(eval_ckpt_path)
        completed_rounds = eval_n_rounds
        start_idx = completed_rounds * batch_size

        leftover_rounds = max(0, min(max_rounds, total_rounds - completed_rounds))
        if leftover_rounds == 0:
            print("All eval rounds already completed in a previous run. Nothing to do.")
            sys.exit(0)

        abs_max_rounds = completed_rounds + leftover_rounds

        init_state = EvalState(
            current_instruction=candidate_instruction,
            dataset_name=args.test_set,
            max_hist=args.max_hist,
            executor_model=executor_model,
            scorer_model=args.scorer_model,
            optimizer_model=args.optimizer_model,
            scores=scores,
            round=completed_rounds + 1,
            samples=test_samples,
            batch_start=start_idx,
            batch_size=batch_size,
            max_rounds=abs_max_rounds,
            load_ckpt=True,
        )
    else:
        print("No checkpoint exists, start from scratch.")
        print(f"Metric: {metric}")
        init_state = EvalState(
            current_instruction=candidate_instruction,
            dataset_name=args.test_set,
            max_hist=args.max_hist,
            executor_model=executor_model,
            scorer_model=args.scorer_model,
            optimizer_model=args.optimizer_model,
            scores=[],
            round=1,
            samples=test_samples,
            batch_start=0,
            batch_size=batch_size,
            max_rounds=max_rounds,
            load_ckpt=False,
        )

    # ---------------------- Run evaluator ----------------------
    evaluator = Evaluator(
        task_executor=task_executor,
        scorer=scorer,
        baseline_instruction=baseline_instruction,
        candidate_instruction=candidate_instruction,
        metric=metric,
        z=args.z,
        save_ckpt=True,   # keep original behavior in single mode
    )
    output = await evaluator.run(init_state=init_state)

    # ---------------------- Aggregate result ----------------------
    eval_scores = output["scores"]
    assert eval_scores is not None and len(eval_scores) > 0
    avg_win_rate = sum(float(s) for s in eval_scores) / len(eval_scores)
    print(f"Average win rate over {len(eval_scores)} round(s): {avg_win_rate:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
