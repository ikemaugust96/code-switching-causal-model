"""
Inspect the raw dataset before preprocessing.

This script helps verify whether the raw dataset actually contains
multiple language pairs or whether the processed output is correctly
showing only one pair.
"""

from collections import Counter
from datasets import load_dataset


def normalize_pair(lang1, lang2):
    """Normalize a language pair into canonical form."""
    left = (lang1 or "unknown").strip().lower()
    right = (lang2 or "unknown").strip().lower()
    return "-".join(sorted([left, right]))


def main():
    dataset_name = "Shelton1013/SwitchLingua_text"

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    print("\nAvailable splits:")
    for split_name in ds.keys():
        print(f"  - {split_name}: {len(ds[split_name])} rows")

    pair_counter = Counter()
    lang1_counter = Counter()
    lang2_counter = Counter()

    total_rows = 0
    missing_pair_rows = 0

    for split_name, split_data in ds.items():
        for row in split_data:
            total_rows += 1

            lang1 = row.get("first_language")
            lang2 = row.get("second_language")

            if not lang1 or not lang2:
                missing_pair_rows += 1
                continue

            lang1_counter[str(lang1).strip().lower()] += 1
            lang2_counter[str(lang2).strip().lower()] += 1

            pair_id = normalize_pair(str(lang1), str(lang2))
            pair_counter[pair_id] += 1

    print("\nTotal rows scanned:", total_rows)
    print("Rows missing first_language or second_language:", missing_pair_rows)

    print("\nTop first_language values:")
    for name, count in lang1_counter.most_common(20):
        print(f"  {name}: {count}")

    print("\nTop second_language values:")
    for name, count in lang2_counter.most_common(20):
        print(f"  {name}: {count}")

    print("\nTop language pairs:")
    for pair_id, count in pair_counter.most_common(30):
        print(f"  {pair_id}: {count}")


if __name__ == "__main__":
    main()