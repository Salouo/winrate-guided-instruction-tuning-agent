"""plot_accepted_winrate.py

Visualize accepted-instruction win rates over training iterations.
"""

from __future__ import annotations
import os
import json
from typing import Union, Dict, Any, List
import matplotlib.pyplot as plt

def plot_accepted_winrate(
    data: Union[str, Dict[str, Any]],
    save_path: str | None = None,
    annotate: bool = True,
    tick_step: int | None = None,
):
    # Load file
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = data

    results: List[Dict[str, Any]] = payload.get("results", [])
    if not results:
        raise ValueError("No 'results' found in data.")

    # Obtain data points
    results = sorted(results, key=lambda r: r.get("round", 0))
    rounds = [int(r.get("round", 0)) for r in results]
    winrates_pct = [float(r.get("avg_win_rate", 0.0)) * 100.0 for r in results]

    # 
    ext_rounds = [0] + rounds
    ext_winrates = [50.0] + winrates_pct

    # Create a figure
    fig = plt.figure(figsize=(9, 5.2))
    ax = plt.gca()

    ax.plot(ext_rounds, ext_winrates, marker="o", label="Accepted Win Rate")

    # Add an red point
    ax.plot([0], [50], marker="o", color="tab:red")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 120)
    ax.set_yticks(list(range(0, 110, 20)))  # 0,20,40,60,80,100

    ds = payload.get("dataset", "")
    ax.set_title(f"{ds} – Accepted Instructions Win Rate" if ds else
                 "Accepted Instructions Win Rate")

    # Draw the 50% baseline
    ax.axhline(50, linestyle="--", linewidth=1, color="tab:red", label="Baseline (50%)")
    ax.grid(True, linestyle=":", linewidth=0.8)

    # Plot x axis
    max_rounds = int(payload.get("max_rounds", max(rounds)))
    ax.set_xlim(0, max_rounds + 0.5)

    step = tick_step if tick_step is not None else max(1, round(max_rounds / 10))
    ax.set_xticks(list(range(0, max_rounds + 1, step)))

    # Annotation the last point
    if annotate and len(ext_rounds) > 1:
        last_x, last_y = rounds[-1], winrates_pct[-1]
        ax.annotate(f"{last_y:.1f}%", xy=(last_x, last_y), xytext=(0, 8),
                    textcoords="offset points", ha="left", va="center")

    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    in_path = "/Users/junfeng.chen/Desktop/apt_duel/outputs/bigbenchhard_ja_cot_64/100/claude-3-5-haiku-latest_gpt-4.1_gpt-5/checkpoint_eval_accepted.json"
    out_path = "/Users/junfeng.chen/Desktop/apt_duel/imgs/img.png"
    plot_accepted_winrate(in_path, save_path=out_path, tick_step=10)
