"""eval_acc.py

Absolute accuracy evaluator for baseline vs. best instruction.

- Computes EM / math_em on test sets in batches (async).
- Compares baseline (init instruction) vs. best-from-training.
- Saves results to `checkpoint_acc_compare.json` next to eval ckpts.
- Prints per-batch logs and a final summary table.
- Requires training ckpt (model/vendor/batch-size/max-hist must match).
- CLI: select datasets, models/vendors, batch size, temperatures, concurrency.
"""

import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from .utils import (
    EvalState,
    load_dataset,
    make_backend,
    get_ckpt_path,
    load_train_ckpt,
)
from .unit_agents_acc import TaskExecutor, Scorer

load_dotenv()


async def compute_accuracy_over_dataset(
    task_executor: TaskExecutor,
    scorer: Scorer,
    instruction: str,
    samples,
    batch_size: int,
    metric: str,
    *,
    verbose: bool = True,
) -> dict:
    """Iterate over all the datasets."""
    n = len(samples)
    total_correct = 0
    total_seen = 0
    round_idx = 1

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        state = EvalState(
            current_instruction=instruction,
            dataset_name="",
            executor_model="",
            scorer_model="",
            optimizer_model="",
            scores=[],
            round=round_idx,
            samples=samples,
            batch_start=start,
            batch_size=batch_size,
            max_rounds=round_idx,
            load_ckpt=False,
        )
        answers = await task_executor.run_single(state=state, instruction=instruction)
        stats = scorer.compute_accuracy(state=state, answers=answers, metric=metric)
        total_correct += stats["correct"]
        total_seen += stats["total"]
        if verbose:
            print(f"[ACC] batch {start}:{end} -> acc={stats['acc']:.3f} "
                  f"(correct={stats['correct']}/{stats['total']})")
        round_idx += 1

    overall = {
        "correct": int(total_correct),
        "total": int(total_seen),
        "acc": (total_correct / total_seen) if total_seen else 0.0,
    }
    print(f"[ACC] overall -> acc={overall['acc']:.3f} "
          f"(correct={overall['correct']}/{overall['total']})")
    return overall


async def run_one_dataset(args, dataset_name: str) -> dict | None:
    print(f"\n================ {dataset_name} ================")
    # Load datasets
    dataset_path = Path("datasets") / "test" / f"{dataset_name}.json"
    test_samples, init_instruction, metric = load_dataset(dataset_path=dataset_path, shuffle=False)
    if metric not in ("em", "math_em"):
        print(f"[SKIP] {dataset_name}: metric={metric} (only supports 'em' or 'math_em').")
        return None

    baseline_instruction = f"[{init_instruction}]"
    executor_model = "gpt-oss-20b" if args.executor_model == "gpt-oss:20b" else args.executor_model

    # Obtain path to eval checkpoint
    eval_ckpt_path = get_ckpt_path(
        dataset_name=dataset_name,
        executor_model=executor_model,
        scorer_model=args.scorer_model,
        optimizer_model=args.optimizer_model,
        batch_size=args.batch_size,
        max_hist=args.max_hist,
        mode="eval",
    )
    out_path = eval_ckpt_path.with_name("checkpoint_acc_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load best instruction
    train_ckpt_path = get_ckpt_path(
        dataset_name=dataset_name,
        executor_model=executor_model,
        scorer_model=args.scorer_model,
        optimizer_model=args.optimizer_model,
        batch_size=args.batch_size,
        max_hist=args.max_hist,
        mode="train",
    )
    if not train_ckpt_path.exists():
        print(f"[ERROR] Training checkpoint not found: {train_ckpt_path}\n"
              f"        Ensure --optimizer-model and --batch-size match training.")
        return None

    # Baseline
    print("=== Absolute accuracy: Baseline ===")
    baseline_acc = await compute_accuracy_over_dataset(
        task_executor=args._task_executor,
        scorer=args._scorer,
        instruction=baseline_instruction,
        samples=test_samples,
        batch_size=args.batch_size,
        metric=metric,
        verbose=True,
    )

    # Candidate（best instruction）
    _, _, _, best_instruction = load_train_ckpt(train_ckpt_path)
    if not best_instruction:
        print("[ERROR] best_instruction not found in training checkpoint.")
        return None

    print("\n=== Absolute accuracy: Best instruction (from training) ===")
    if best_instruction == baseline_instruction:
        print("Best equals baseline; reuse baseline accuracy.")
        cand_acc = baseline_acc
    else:
        cand_acc = await compute_accuracy_over_dataset(
            task_executor=args._task_executor,
            scorer=args._scorer,
            instruction=best_instruction,
            samples=test_samples,
            batch_size=args.batch_size,
            metric=metric,
            verbose=True,
        )

    payload = {
        "dataset": dataset_name,
        "executor_vendor": args.executor_vendor,
        "executor_model": executor_model,
        "scorer_vendor": args.scorer_vendor,
        "scorer_model": args.scorer_model,
        "optimizer_model": args.optimizer_model,
        "batch_size": args.batch_size,
        "metric": metric,
        "baseline_instruction": baseline_instruction,
        "best_instruction": best_instruction,
        "baseline": baseline_acc,
        "candidate": cand_acc,
        "delta_acc": cand_acc["acc"] - baseline_acc["acc"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return {
        "dataset": dataset_name,
        "baseline_acc": baseline_acc["acc"],
        "candidate_acc": cand_acc["acc"],
        "delta": payload["delta_acc"],
        "out": str(out_path),
    }


async def main():
    parser = argparse.ArgumentParser(description="Run acc compare for multiple datasets")
    parser.add_argument("--executor-vendor", type=str, default="openai",
                        choices=["openai", "google", "anthropic"])
    parser.add_argument("--scorer-vendor", type=str, default="openai",
                        choices=["openai", "google", "anthropic"])
    parser.add_argument("--executor-model", type=str, default="gpt-4o")
    parser.add_argument("--scorer-model", type=str, default="gpt-4.1")
    parser.add_argument("--optimizer-model", type=str, default="gpt-5",
                        help="Only used to locate the training checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-hist", type=int, default=100, help="Only pass the last N history items to the optimizer (0 = no history).")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--exec-temperature", type=float, default=0.0)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["bigbenchhard_ja_cot"],
        help="Datasets to evaluate (space-separated).",
    )
    args = parser.parse_args()

    be_executor = make_backend(vendor=args.executor_vendor, model=args.executor_model, mode="async")
    be_scorer   = make_backend(vendor=args.scorer_vendor, model=args.scorer_model, mode="async")
    args._task_executor = TaskExecutor(backend=be_executor, temperature=args.exec_temperature, concurrency=args.concurrency)
    args._scorer        = Scorer(backend=be_scorer,  temperature=0.0,                   concurrency=args.concurrency)

    results = []
    for name in args.datasets:
        try:
            r = await run_one_dataset(args, name)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[FAILED] {name}: {e}")

    if not results:
        print("\nNo results generated.")
        return

    # Print results
    print("\n================ SUMMARY ================")
    w = max(len(r["dataset"]) for r in results)
    print(f"{'dataset'.ljust(w)} | baseline  | candidate | delta")
    print("-" * (w + 32))
    for r in results:
        print(f"{r['dataset'].ljust(w)} | {r['baseline_acc']:.3f}    | {r['candidate_acc']:.3f}   | {r['delta']:+.3f}")
    print("=========================================\n")


if __name__ == "__main__":
    asyncio.run(main())
