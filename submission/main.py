"""
Main Runner Script for Code-Switch Prediction Project

This script orchestrates:
1. Load existing processed data
2. Train/test split with shuffle
3. Baseline model training and evaluation
4. Pair-specific baseline evaluation
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter
import json
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        './data/cache',
        './data/processed',
        './figures',
        './results',
        './models'
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def filter_pair(data, lang1, lang2):
    """Filter dataset for a specific language pair."""
    filtered = []

    for ex in data:
        l1 = ex.get("first_language")
        l2 = ex.get("second_language")

        if {l1, l2} == {lang1, lang2}:
            filtered.append(ex)

    return filtered


def main(args):
    print("=" * 70)
    print(" CODE-SWITCH PREDICTION - BASELINE EVALUATION FROM PROCESSED DATA")
    print("=" * 70)

    # Step 0: Setup
    print("\n[Step 0] Setting up directory structure...")
    create_directory_structure()

    processed_path = "./data/processed/processed_data.json"

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            "processed_data.json not found. "
            "You already ran full preprocessing once, so this file should exist at "
            "./data/processed/processed_data.json"
        )

    # Step 1: Load existing processed data
    print("\n" + "=" * 70)
    print("[Step 1] LOAD EXISTING PROCESSED DATA")
    print("=" * 70)

    with open(processed_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"✓ Loaded processed data: {len(all_data)} examples")

    # Step 1.5: Language Pair Distribution
    print("\n" + "=" * 70)
    print("[Step 1.5] LANGUAGE PAIR DISTRIBUTION")
    print("=" * 70)

    pair_counter = Counter()

    for ex in all_data:
        lang1 = ex.get("first_language", "unknown")
        lang2 = ex.get("second_language", "unknown")
        pair = tuple(sorted([lang1, lang2]))
        pair_counter[pair] += 1

    total = sum(pair_counter.values())

    print("\nTop Language Pairs:")
    for pair, count in pair_counter.most_common(20):
        pct = count / total * 100 if total > 0 else 0
        print(f"{pair[0]} - {pair[1]} : {count} ({pct:.2f}%)")

    pair_stats = []
    for pair, count in pair_counter.most_common():
        pct = count / total * 100 if total > 0 else 0
        pair_stats.append({
            "language_1": pair[0],
            "language_2": pair[1],
            "count": count,
            "percentage": round(pct, 2)
        })

    with open("./results/language_pair_distribution.json", "w", encoding="utf-8") as f:
        json.dump(pair_stats, f, indent=2, ensure_ascii=False)

    print("✓ Saved language_pair_distribution.json")

    # Step 2: Shuffle + Train/Test Split
    print("\n" + "=" * 70)
    print("[Step 2] SHUFFLED TRAIN/TEST SPLIT")
    print("=" * 70)

    random.seed(args.seed)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    with open("./data/processed/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)

    with open("./data/processed/test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)

    print("✓ Saved shuffled train/test splits")

    # Step 3: Overall Baseline Evaluation
    print("\n" + "=" * 70)
    print("[Step 3] BASELINE EVALUATION (OVERALL)")
    print("=" * 70)

    from causal_baselines import (
        MajorityClassBaseline,
        NGramBaseline,
        LastLanguageBaseline,
        LogisticRegressionBaseline,
        BaselineEvaluator
    )

    evaluator = BaselineEvaluator()

    print("\n--- Baseline 1: Majority Class ---")
    majority_model = MajorityClassBaseline()
    majority_model.train(train_data)
    evaluator.evaluate_model(majority_model, "Majority Class", test_data)

    print("\n--- Baseline 2: Last Language ---")
    last_lang_model = LastLanguageBaseline()
    last_lang_model.train(train_data)
    evaluator.evaluate_model(last_lang_model, "Last Language", test_data)

    print(f"\n--- Baseline 3: {args.n_gram}-gram ---")
    ngram_model = NGramBaseline(n=args.n_gram)
    ngram_model.train(train_data)
    evaluator.evaluate_model(ngram_model, f"{args.n_gram}-gram", test_data)

    if not args.skip_lr:
        print("\n--- Baseline 4: Logistic Regression ---")
        lr_model = LogisticRegressionBaseline(context_window=5)
        lr_model.train(train_data)
        evaluator.evaluate_model(lr_model, "Logistic Regression", test_data)
    else:
        print("\n--- Baseline 4: Logistic Regression ---")
        print("Skipped (--skip_lr enabled)")

    evaluator.compare_models(save_path="./figures/baseline_comparison.png")
    evaluator.save_results(output_path="./results/baseline_results.json")

    # Step 4: Pair-Specific Baselines
    print("\n" + "=" * 70)
    print("[Step 4] PAIR-SPECIFIC BASELINES")
    print("=" * 70)

    top_pairs = pair_counter.most_common(args.top_k_pairs)
    pair_results = []

    for pair, count in top_pairs:
        lang1, lang2 = pair

        print(f"\nRunning baseline for pair: {lang1} - {lang2}")

        train_pair = filter_pair(train_data, lang1, lang2)
        test_pair = filter_pair(test_data, lang1, lang2)

        print(f"train size: {len(train_pair)} test size: {len(test_pair)}")

        if len(train_pair) == 0 or len(test_pair) == 0:
            print("Skipped (not enough data)")
            continue

        pair_evaluator = BaselineEvaluator()
        model = NGramBaseline(n=args.n_gram)
        model.train(train_pair)

        result = pair_evaluator.evaluate_model(
            model,
            f"{lang1}-{lang2}",
            test_pair
        )

        pair_results.append({
            "pair": f"{lang1}-{lang2}",
            "train_examples": len(train_pair),
            "test_examples": len(test_pair),
            "switch_accuracy": result["switch_accuracy"],
            "switch_f1": result["switch_f1"],
            "duration_accuracy": result["duration_accuracy"],
            "duration_f1_macro": result["duration_f1_macro"]
        })

    with open("./results/pair_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(pair_results, f, indent=2, ensure_ascii=False)

    print("\n✓ Saved pair_baseline_results.json")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Generated files:")
    print("results/language_pair_distribution.json")
    print("results/baseline_results.json")
    print("results/pair_baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Code-Switch Prediction: Baseline Evaluation from Processed Data'
    )

    parser.add_argument(
        '--n_gram',
        type=int,
        default=3,
        help='N for N-gram baseline model'
    )

    parser.add_argument(
        '--top_k_pairs',
        type=int,
        default=6,
        help='Number of language pairs for pair baseline'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffle'
    )

    parser.add_argument(
        '--skip_lr',
        action='store_true',
        help='Skip logistic regression baseline'
    )

    args = parser.parse_args()

    main(args)