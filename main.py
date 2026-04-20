"""
Main Runner Script for Code-Switch Prediction Project.

This script orchestrates:
1. Load existing processed data
2. Normalize language-pair metadata
3. Train/test split with shuffle
4. Overall baseline training and evaluation
5. Pair-specific baseline evaluation
6. Optional leave-one-pair-out universality evaluation
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        "./data/cache",
        "./data/processed",
        "./figures",
        "./results",
        "./models"
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def normalize_pair_id(lang1: str, lang2: str) -> str:
    """
    Normalize a language pair into a canonical pair ID.

    Example:
        ('es', 'en') -> 'en-es'
    """
    left = (lang1 or "unknown").strip().lower()
    right = (lang2 or "unknown").strip().lower()
    return "-".join(sorted([left, right]))


def split_pair_id(pair_id: str):
    """
    Split a normalized pair ID into its two component languages.
    """
    parts = pair_id.split("-", maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair_id, "unknown"


def attach_pair_metadata(data):
    """
    Ensure every example has a normalized pair_id field.
    """
    normalized = []

    for ex in data:
        lang1 = ex.get("first_language", "unknown")
        lang2 = ex.get("second_language", "unknown")
        ex["pair_id"] = normalize_pair_id(lang1, lang2)
        normalized.append(ex)

    return normalized


def filter_pair(data, pair_id: str):
    """
    Filter dataset examples for one normalized language pair.
    """
    return [ex for ex in data if ex.get("pair_id") == pair_id]


def save_json(obj, path: str):
    """
    Save a Python object to JSON with indentation.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_overall_baselines(train_data, test_data, n_gram: int, skip_lr: bool):
    """
    Train and evaluate the overall baseline models.
    """
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

    print(f"\n--- Baseline 3: {n_gram}-gram ---")
    ngram_model = NGramBaseline(n=n_gram)
    ngram_model.train(train_data)
    evaluator.evaluate_model(ngram_model, f"{n_gram}-gram", test_data)

    if not skip_lr:
        print("\n--- Baseline 4: Logistic Regression ---")
        lr_model = LogisticRegressionBaseline(context_window=5)
        lr_model.train(train_data)
        evaluator.evaluate_model(lr_model, "Logistic Regression", test_data)
    else:
        print("\n--- Baseline 4: Logistic Regression ---")
        print("Skipped (--skip_lr enabled)")

    evaluator.compare_models(save_path="./figures/baseline_comparison.png")
    evaluator.save_results(output_path="./results/baseline_results.json")


def run_pair_specific_baselines(train_data, test_data, pair_counter, top_k_pairs: int, n_gram: int):
    """
    Run pair-specific N-gram baselines for the top-K most frequent pairs.
    """
    from causal_baselines import NGramBaseline, BaselineEvaluator

    top_pairs = pair_counter.most_common(top_k_pairs)
    pair_results = []

    for pair_id, count in top_pairs:
        lang1, lang2 = split_pair_id(pair_id)

        print(f"\nRunning pair-specific baseline for: {lang1} - {lang2}")

        train_pair = filter_pair(train_data, pair_id)
        test_pair = filter_pair(test_data, pair_id)

        print(f"train size: {len(train_pair)} | test size: {len(test_pair)}")

        if len(train_pair) == 0 or len(test_pair) == 0:
            print("Skipped (not enough data)")
            continue

        pair_evaluator = BaselineEvaluator()
        model = NGramBaseline(n=n_gram)
        model.train(train_pair)

        result = pair_evaluator.evaluate_model(
            model=model,
            model_name=pair_id,
            test_data=test_pair
        )

        pair_results.append({
            "pair": pair_id,
            "train_examples": len(train_pair),
            "test_examples": len(test_pair),
            "raw_pair_count": count,
            "switch_accuracy": result["switch_accuracy"],
            "switch_f1": result["switch_f1"],
            "duration_accuracy": result["duration_accuracy"],
            "duration_f1_macro": result["duration_f1_macro"]
        })

    save_json(pair_results, "./results/pair_baseline_results.json")
    print("\n✓ Saved pair_baseline_results.json")


def main(args):
    print("=" * 70)
    print(" CODE-SWITCH PREDICTION - BASELINE + UNIVERSALITY PIPELINE")
    print("=" * 70)

    print("\n[Step 0] Setting up directory structure...")
    create_directory_structure()

    processed_path = "./data/processed/processed_data.json"

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            "processed_data.json not found. "
            "Please make sure it exists at ./data/processed/processed_data.json"
        )

    print("\n" + "=" * 70)
    print("[Step 1] LOAD EXISTING PROCESSED DATA")
    print("=" * 70)

    with open(processed_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    all_data = attach_pair_metadata(all_data)
    print(f"✓ Loaded processed data: {len(all_data)} examples")

    save_json(all_data, processed_path)
    print("✓ Normalized pair_id metadata in processed_data.json")

    print("\n" + "=" * 70)
    print("[Step 1.5] LANGUAGE PAIR DISTRIBUTION")
    print("=" * 70)

    pair_counter = Counter(ex["pair_id"] for ex in all_data)
    total = sum(pair_counter.values())

    print("\nTop Language Pairs:")
    for pair_id, count in pair_counter.most_common(20):
        lang1, lang2 = split_pair_id(pair_id)
        pct = count / total * 100 if total > 0 else 0.0
        print(f"{lang1} - {lang2}: {count} ({pct:.2f}%)")

    pair_stats = []
    for pair_id, count in pair_counter.most_common():
        lang1, lang2 = split_pair_id(pair_id)
        pct = count / total * 100 if total > 0 else 0.0
        pair_stats.append({
            "pair_id": pair_id,
            "language_1": lang1,
            "language_2": lang2,
            "count": count,
            "percentage": round(pct, 2)
        })

    save_json(pair_stats, "./results/language_pair_distribution.json")
    print("✓ Saved language_pair_distribution.json")

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

    save_json(train_data, "./data/processed/train_data.json")
    save_json(test_data, "./data/processed/test_data.json")
    print("✓ Saved shuffled train/test splits")

    print("\n" + "=" * 70)
    print("[Step 3] BASELINE EVALUATION (OVERALL)")
    print("=" * 70)

    run_overall_baselines(
        train_data=train_data,
        test_data=test_data,
        n_gram=args.n_gram,
        skip_lr=args.skip_lr
    )

    print("\n" + "=" * 70)
    print("[Step 4] PAIR-SPECIFIC BASELINES")
    print("=" * 70)

    run_pair_specific_baselines(
        train_data=train_data,
        test_data=test_data,
        pair_counter=pair_counter,
        top_k_pairs=args.top_k_pairs,
        n_gram=args.n_gram
    )

    if args.run_universality:
        print("\n" + "=" * 70)
        print("[Step 5] LEAVE-ONE-PAIR-OUT UNIVERSALITY EVALUATION")
        print("=" * 70)

        from universality_eval import run_universality_experiments

        run_universality_experiments(
            processed_path=processed_path,
            n_gram=args.n_gram,
            skip_lr=args.skip_lr,
            min_pair_examples=args.min_pair_examples,
            max_pairs=args.max_universality_pairs,
            seed=args.seed,
            gru_epochs=args.universality_epochs,
            gru_batch_size=args.universality_batch_size,
            gru_max_train_samples=args.universality_max_train_samples,
            gru_max_test_samples=args.universality_max_test_samples
        )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Generated files:")
    print("  results/language_pair_distribution.json")
    print("  results/baseline_results.json")
    print("  results/pair_baseline_results.json")
    if args.run_universality:
        print("  results/universality_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code-Switch Prediction: Baseline + Universality Evaluation"
    )

    parser.add_argument(
        "--n_gram",
        type=int,
        default=3,
        help="N for the N-gram baseline model"
    )

    parser.add_argument(
        "--top_k_pairs",
        type=int,
        default=6,
        help="Number of language pairs for pair-specific baseline evaluation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and training"
    )

    parser.add_argument(
        "--skip_lr",
        action="store_true",
        help="Skip the logistic regression baseline"
    )

    parser.add_argument(
        "--run_universality",
        action="store_true",
        help="Run leave-one-pair-out universality experiments"
    )

    parser.add_argument(
        "--min_pair_examples",
        type=int,
        default=50,
        help="Minimum number of examples required for a pair to be held out in universality evaluation"
    )

    parser.add_argument(
        "--max_universality_pairs",
        type=int,
        default=6,
        help="Maximum number of held-out pairs to evaluate in universality experiments"
    )

    parser.add_argument(
        "--universality_epochs",
        type=int,
        default=4,
        help="Number of GRU epochs for each held-out-pair universality run"
    )

    parser.add_argument(
        "--universality_batch_size",
        type=int,
        default=256,
        help="Batch size for universality GRU experiments"
    )

    parser.add_argument(
        "--universality_max_train_samples",
        type=int,
        default=400000,
        help="Maximum number of sampled position-level training instances per universality run"
    )

    parser.add_argument(
        "--universality_max_test_samples",
        type=int,
        default=100000,
        help="Maximum number of sampled position-level test instances per universality run"
    )

    args = parser.parse_args()
    main(args)