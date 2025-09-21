"""train.py

Pairwise training to optimize instructions via win-rate improvements.

- Loops: optimize → duel (baseline vs candidate) → accept/reject → update.
- Uses Wilson CI and phase-aware n_eff gating for robust acceptance.
- Saves periodic train checkpoints; supports resume from ckpt.
- Async, vendor-agnostic backends; configurable rounds, batch size, temps, z.
- CLI: choose vendors/models/dataset, max_hist, accept_rule, min_effective_n.
"""

import time
import argparse
import asyncio
import sys

from langgraph.graph import StateGraph, END
from typing import Any
from dotenv import load_dotenv
from pathlib import Path

from .unit_agents import (
    TaskExecutor,
    Scorer,
    InstructionOptimizer
)
from .utils import (
    TrainState,
    load_dataset,
    split_samples_and_examples,
    make_backend,
    save_train_ckpt,
    load_train_ckpt,
    get_ckpt_path,
    obtain_phase
)

# Load environment variables (API keys, etc.)
loaded = load_dotenv()
CKPT_EVERY = 3


# ========================================================================= #
#                   Optimized Instruction Generation Agent                  #
# ========================================================================= #
class OptimizedInstructionGeneration:
    """OptimizedInstructionGeneration (pairwise)

    Wires TaskExecutor, LLMScorer, and InstructionOptimizer into a loop:
      optimize -> duel_and_update -> optimize -> ...
    """

    def __init__(
        self,
        task_executor: TaskExecutor,
        scorer: Scorer,
        instruction_optimizer: InstructionOptimizer,
        metric: str = "llm",
        z: float = 1.96,
        min_effective_n: int = 0,
        accept_rule: str = "ci>0.5",
    ):
        """
        (Function) Initialize the agent with executor, scorer, and optimizer plus evaluation settings.

        Returns:
        - None
        """
        self.task_executor = task_executor
        self.scorer = scorer
        self.instruction_optimizer = instruction_optimizer
        self.metric = metric  # "em" | "f1" | "llm"
        self.z = z
        self.min_effective_n = min_effective_n
        self.accept_rule = accept_rule  # "ci>0.5" | "p>0.5" | "always" | "never"

    # ------------------------------ Create graph --------------------------------
    def _create_graph(self) -> StateGraph[TrainState]:
        """
        (Function) Build the training state graph:
        entry 'optimize_instruction' → 'duel_and_update' → loop until round > max_rounds.

        Returns:
        - graph (StateGraph[TrainState]): Uncompiled LangGraph state graph for training.
        """
        graph = StateGraph(TrainState)
        graph.add_node("optimize_instruction", self._optimize_instruction)
        graph.add_node("duel_and_update", self._duel_and_update)

        # Always start by proposing a candidate (even when resuming)
        graph.set_entry_point("optimize_instruction")

        # optimize -> duel
        graph.add_edge("optimize_instruction", "duel_and_update")

        # stop or continue
        def should_stop(state: TrainState) -> bool:
            return state.round > state.max_rounds
        graph.add_conditional_edges("duel_and_update", should_stop, {True: END, False: "optimize_instruction"})

        return graph

    def _decide_accept(self, stats: dict, curr_round: int, total_rounds: int) -> bool:
        """
        (Function) Decide whether to accept the candidate over the baseline using:
        - Wilson CI lower bound (ci_low > 0.5) and
        - phase-aware effective sample size gating (wins+losses ≥ threshold).

        Returns:
        - accepted (bool): True if candidate passes n_eff gate and CI rule; False otherwise.
        """
        p_hat = float(stats.get("p_hat", 0.0))
        ci_low = float(stats.get("ci_low", 0.0))
        n_effective = int(stats.get("n_effective", 0))

        phase_to_delta = {"early": -4, "mid": -2, "late": 0}
        phase = obtain_phase(curr_round=curr_round, max_rounds=total_rounds)
        delta = phase_to_delta.get(phase)

        base = int(self.min_effective_n or 0)
        threshold = max(0, base + delta)

        # Record
        stats["n_eff_threshold"] = threshold
        stats["phase"] = phase

        if n_effective < threshold:
            return False
        # default
        return ci_low > 0.5

    async def _duel_and_update(self, state: TrainState) -> dict[str, Any]:
        """
        (Function) Run one duel round:
        1) Execute baseline vs candidate on the current batch,
        2) Compute pairwise win-rate + Wilson CI,
        3) Decide accept/reject,
        4) Log, checkpoint (periodically), and emit partial state updates.

        Returns:
        - delta (dict): Fields to update in reducer, including:
          - best_instruction (str)
          - best_score (float | None)
          - batch_start (int)
          - history (list[dict])  # current round record (to be appended)
          - round (int)           # next round index
        """
        # Difine batch slice
        i = state.batch_start
        j = min(i + state.batch_size, len(state.samples))
        batch = state.samples[i:j]
        next_start = j if j < len(state.samples) else 0

        # Define baseline & candidate
        # The current best instruction will be the baseline instruction
        baseline = state.best_instruction or state.current_instruction
        candidate = state.current_instruction

        # Generate answers
        answers_A, answers_B = await self.task_executor.run_pair(
            state=state,
            baseline_instruction=baseline,
            candidate_instruction=candidate
        )

        # Obtain pairwise stats
        stats = await self.scorer.run_pairwise(
            state=state,
            answers_A=answers_A,
            answers_B=answers_B,
            metric=self.metric,
            z=self.z,
        )
        stats["n_effective"] = int(stats.get("n_effective", stats.get("wins", 0) + stats.get("losses", 0)))

        # Judge acceptance
        accepted = self._decide_accept(stats, curr_round=state.round, total_rounds=state.max_rounds)
        stats["accepted"] = accepted

        # Print progress log
        curr_round = state.round
        thr = stats.get("n_eff_threshold", None)
        phase = stats.get("phase", "")
        gate_txt = ""
        if thr is not None:
            pass_fail = "PASS" if stats["n_effective"] >= thr else "FAIL"
            gate_txt = f" | gate n_eff≥{thr} ({phase}) -> {pass_fail}"

        print(
            f"[Iteration {curr_round}/{state.max_rounds}] "
            f"batch {i}:{j} -> winrate={stats['p_hat']:.3f} "
            f"(CI {stats['ci_low']:.3f}-{stats['ci_high']:.3f}; "
            f"W/L/T={stats['wins']}/{stats['losses']}/{stats['ties']}; "
            f"n_eff={stats['n_effective']})"
            f"{gate_txt} "
            f"=> {'ACCEPT' if accepted else 'REJECT'}"
        )

        # Record this round
        record = {
            "round": curr_round,
            "instruction": candidate,
            "wins": int(stats["wins"]),
            "losses": int(stats["losses"]),
            "ties": int(stats["ties"]),
            "n_effective": int(stats["n_effective"]),
            "p_hat": float(stats["p_hat"]),
            "ci_low": float(stats["ci_low"]),
            "ci_high": float(stats["ci_high"]),
            "accepted": bool(accepted),
            "batch_start": i,
            "batch_size": len(batch),
        }

        # Update best baseline
        best_instruction = candidate if accepted else (state.best_instruction or state.current_instruction)
        best_score = float(stats["p_hat"]) if accepted else state.best_score

        # Save checkpoint periodically or at the end
        overrides = {
            "best_instruction": best_instruction,
            "best_score": best_score,
            "metric": self.metric,
            "batch_start": next_start,
            "history": state.history + [record]
        }
        if (curr_round != 0 and curr_round % CKPT_EVERY == 0) or (curr_round == state.max_rounds):
            ckpt_path = save_train_ckpt(state=state, overrides=overrides)
            print(f"Train checkpoint saved in {ckpt_path}")

        # Return partial state to be reduced
        return {
            "best_instruction": best_instruction,
            "best_score": best_score,
            "batch_start": next_start,
            "history": [record],       # Appended to history
            "round": curr_round + 1,   # Next round
        }

    def _optimize_instruction(self, state: TrainState) -> dict[str, Any]:
        """
        (Function) Propose a new candidate instruction based on the current state/history.

        Returns:
        - update (dict): {"current_instruction": str} with the new candidate.
        """
        current_optimized_instruction = self.instruction_optimizer.run(state=state)
        return {"current_instruction": current_optimized_instruction}

    def run(self, init_state: TrainState) -> dict[str, Any]:
        """
        (Function) Execute the training loop by compiling and running the state graph.

        Returns:
        - final_state (dict[str, Any]): Aggregated state produced by the graph (includes 'history', 'best_instruction', etc.).
        """
        compiled = self._create_graph().compile()
        recursion_limit = init_state.max_rounds * 3 + 5
        print("Running pairwise training by rounds...")
        final_state: dict[str, Any] = asyncio.run(
            compiled.ainvoke(init_state, config={"recursion_limit": recursion_limit})
        )
        print("Finish!\n")
        return final_state


# -------------------------------------------------- Main -------------------------------------------------- #
def main():
    """
    (Function) Parse CLI args, build backends, load data, (resume and) run the training agent, then print a summary.

    Returns:
    - None
    """
    # ------------------------- Argparse settings -------------------------
    parser = argparse.ArgumentParser(description="Instruction-tuning agent runner (pairwise)")

    # Vendor choice
    parser.add_argument("--executor-vendor", type=str, default="openai", choices=["openai", "google", "anthropic"])
    parser.add_argument("--scorer-vendor",   type=str, default="openai", choices=["openai", "google", "anthropic"])
    parser.add_argument("--optimizer-vendor",type=str, default="openai", choices=["openai", "google", "anthropic"])

    # Models
    parser.add_argument("--executor-model", type=str, default="gpt-4o", help="LLM for TaskExecutor agent")
    parser.add_argument("--scorer-model", type=str, default="gpt-4.1", help="LLM for Scorer agent")
    parser.add_argument("--optimizer-model", type=str, default="gpt-5", help="LLM for Optimizer agent")

    # Hyper-parameters
    parser.add_argument("--max-rounds", type=int, default=100, help="Number of optimization rounds")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per round")
    parser.add_argument("--training-set", type=str, default="alt-e-to-j", help="Training set name (without .json)")
    parser.add_argument("--max-hist", type=int, default=100, help="Only pass the last N history items to the optimizer (0 = no history).")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Folder to save outputs")
    parser.add_argument("--concurrency", type=int, default=8, help="Async concurrency per stage")
    parser.add_argument("--exec-temperature", type=float, default=0.0, help="Executor temperature")
    parser.add_argument("--scor-temperature", type=float, default=0.0, help="Scorer temperature")

    # Pairwise settings
    parser.add_argument("--accept-rule", type=str, choices=["ci>0.5","p>0.5"], default="ci>0.5")
    parser.add_argument("--z", type=float, default=1.96, help="Z-score for Wilson CI")
    parser.add_argument("--min-effective-n", type=int, default=6, help="Minimum (wins+losses) required to allow accept")

    args = parser.parse_args()

    # ------------------------- Backends -------------------------
    be_executor  = make_backend(vendor=args.executor_vendor,  model=args.executor_model,  mode="async")
    be_scorer    = make_backend(vendor=args.scorer_vendor,    model=args.scorer_model,    mode="async")
    be_optimizer = make_backend(vendor=args.optimizer_vendor, model=args.optimizer_model, mode="sync")

    # ------------------------------ Data & init ------------------------------
    dataset_path = Path("datasets") / "train" / f"{args.training_set}.json"
    seed, n_examples = 42, 3
    all_samples, init_instruction, metric = load_dataset(
        dataset_path=dataset_path, shuffle=False, seed=seed
    )
    samples, examples_for_instruction_optimization = split_samples_and_examples(
        all_samples=all_samples, n_examples=n_examples
    )

    # Model alias normalization for filenames/paths
    executor_model = "gpt-oss-20b" if args.executor_model == "gpt-oss:20b" else args.executor_model

    # Checkpoint path
    ckpt_path = get_ckpt_path(
        dataset_name=args.training_set,
        executor_model=executor_model,
        scorer_model=args.scorer_model,
        optimizer_model=args.optimizer_model,
        batch_size=args.batch_size,
        max_hist=args.max_hist,
        mode="train"
    )
    batch_size = args.batch_size

    # If train checkpoint exists, resume from it
    if ckpt_path.exists():
        print(f"Checkpoint exists. Load checkpoint from {ckpt_path}...")
        print(f"Metric: {metric}")
        trajectory, n_rounds, last_instruction, best_instruction = load_train_ckpt(ckpt_path)

        completed_rounds = n_rounds
        start_idx = completed_rounds * batch_size

        total_rounds = len(samples) // batch_size
        leftover_rounds = max(0, min(args.max_rounds, total_rounds - completed_rounds))
        if leftover_rounds == 0:
            print("All training rounds already completed in a previous run. Nothing to do.")
            sys.exit(0)

        abs_max_rounds = completed_rounds + leftover_rounds

        # If the checkpoint lacks best_instruction, use last_instruction as the baseline
        baseline_ins = best_instruction or last_instruction or f"[{init_instruction}]"

        init_state = TrainState(
            current_instruction=last_instruction or baseline_ins,
            best_instruction=baseline_ins,
            dataset_name=args.training_set,
            executor_model=executor_model,
            scorer_model=args.scorer_model,
            optimizer_model=args.optimizer_model,
            history=trajectory,
            max_hist=args.max_hist,
            # begin next round
            round=completed_rounds + 1,
            samples=samples,
            examples=examples_for_instruction_optimization,
            batch_start=start_idx,
            batch_size=batch_size,
            max_rounds=abs_max_rounds,
            load_ckpt=True
        )
    else:
        print("No checkpoint exists, start from scratch.")
        print(f"Metric: {metric}")
        total_rounds   = len(samples) // batch_size
        actual_rounds  = min(args.max_rounds, total_rounds)

        baseline_ins = f"[{init_instruction}]"
        init_state = TrainState(
            dataset_name=args.training_set,
            executor_model=executor_model,
            scorer_model=args.scorer_model,
            optimizer_model=args.optimizer_model,
            current_instruction=baseline_ins,   # The first optimize step will propose a candidate based on this
            best_instruction=baseline_ins,      # Used as the baseline in pairwise duels
            history=[],
            max_hist=args.max_hist,
            round=1,
            samples=samples,
            examples=examples_for_instruction_optimization,
            batch_start=0,
            batch_size=batch_size,
            max_rounds=actual_rounds,
            load_ckpt=False
        )

    # ------------------------- Agents -------------------------
    task_executor = TaskExecutor(backend=be_executor, temperature=args.exec_temperature, concurrency=args.concurrency)
    scorer = Scorer(backend=be_scorer, temperature=args.scor_temperature, concurrency=args.concurrency)
    instruction_optimizer = InstructionOptimizer(backend=be_optimizer, temperature=1.0, max_hist=args.max_hist)

    agent = OptimizedInstructionGeneration(
        task_executor=task_executor,
        scorer=scorer,
        instruction_optimizer=instruction_optimizer,
        metric=metric,
        z=args.z,
        min_effective_n=args.min_effective_n,
        accept_rule=args.accept_rule,
    )

    # ---------------------- Run the agent ----------------------
    output = agent.run(init_state=init_state)

    # ---------------------- Summary ----------------------
    history = output["history"]
    avg_wr = sum(float(h["p_hat"]) for h in history) / len(history) if history else 0.0
    print("\nBest Instruction:\n", output["best_instruction"])
    print(f"\nAverage training win rate over {len(history)} round(s): {avg_wr:.3f}")


if __name__ == "__main__":
    time_start = time.perf_counter()
    main()
    time_end = time.perf_counter()
    running_time = time_end - time_start
    print(f"\n\nrunning time: {running_time:.2f}s")
