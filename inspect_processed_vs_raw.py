"""
Compare raw dataset language-pair distribution with processed_data.json.
"""

import json
from collections import Counter
from datasets import load_dataset


def normalize_pair(lang1, lang2):
    """Normalize a language pair into canonical form."""
    left = (lang1 or "unknown").strip().lower()
    right = (lang2 or "unknown").strip().lower()
    return "-".join(sorted([left, right]))


def inspect_raw():
    """Inspect raw dataset pairs."""
    ds = load_dataset("Shelton1013/SwitchLingua_text")
    counter = Counter()

    for split_name, split_data in ds.items():
        for row in split_data:
            lang1 = row.get("first_language")
            lang2 = row.get("second_language")
            if lang1 and lang2:
                counter[normalize_pair(str(lang1), str(lang2))] += 1

    return counter


def inspect_processed():
    """Inspect processed dataset pairs."""
    with open("./data/processed/processed_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for row in data:
        lang1 = row.get("first_language", "unknown")
        lang2 = row.get("second_language", "unknown")
        counter[normalize_pair(lang1, lang2)] += 1

    return counter


def main():
    raw_counter = inspect_raw()
    processed_counter = inspect_processed()

    print("\n=== RAW DATASET PAIRS ===")
    for pair_id, count in raw_counter.most_common(20):
        print(f"{pair_id}: {count}")

    print("\n=== PROCESSED DATASET PAIRS ===")
    for pair_id, count in processed_counter.most_common(20):
        print(f"{pair_id}: {count}")

    print("\n=== PAIRS PRESENT IN RAW BUT MISSING IN PROCESSED ===")
    missing = []
    for pair_id, count in raw_counter.items():
        if pair_id not in processed_counter:
            missing.append((pair_id, count))

    missing.sort(key=lambda x: x[1], reverse=True)

    for pair_id, count in missing[:30]:
        print(f"{pair_id}: {count}")


if __name__ == "__main__":
    main()
