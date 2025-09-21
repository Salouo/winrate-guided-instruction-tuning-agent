"""count_samples.py

Add a dataset-level summary to each JSON in ./datasets/train.

- Counts the number of items in "samples" and writes it as "n_samples".
- Preserves original key order, inserting "n_samples" immediately before "samples".
- Updates files in place (UTF-8).
"""

import json
from pathlib import Path
from collections import OrderedDict


def main():
    dataset_path = Path("./datasets/train")
    for file in dataset_path.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        n_samples = len(data["samples"])
        new_data = OrderedDict()
        
        for k, v in data.items():
            if k == "samples":
                new_data["n_samples"] = n_samples
            new_data[k] = v
        
        with file.open("w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2) 

if __name__ == "__main__":
    main()
