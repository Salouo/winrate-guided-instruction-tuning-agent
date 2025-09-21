"""train_test_split.py

Split raw dataset JSONs into 80/20 train/test sets.
"""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    raw_dir = Path("./datasets/raw")
    out_base = Path("./datasets")
    out_train = out_base / "train"
    out_test  = out_base / "test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True,  exist_ok=True)

    json_files = sorted(raw_dir.glob("*.json"))
    total_train = total_test = 0

    for src in json_files:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples")
        if not isinstance(samples, list):
            print(f"[SKIP] {src.name}: missing 'samples' list.")
            continue

        train_samples, test_samples = train_test_split(
            samples, test_size=0.2, random_state=42
        )

        train_data = dict(data)
        train_data["samples"] = train_samples
        test_data = dict(data)
        test_data["samples"] = test_samples

        (out_train / src.name).write_text(
            json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_test / src.name).write_text(
            json.dumps(test_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"[OK] {src.name}: train={len(train_samples)}, test={len(test_samples)}")
        total_train += len(train_samples)
        total_test  += len(test_samples)

    print(f"Done. Files={len(json_files)}, total_train={total_train}, total_test={total_test}")
    print(f"Saved to: {out_train.resolve()} and {out_test.resolve()}")


if __name__ == "__main__":
    main()