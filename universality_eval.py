"""
Leave-one-pair-out universality evaluation.

This script measures how well the models transfer to an unseen language pair.
For each held-out pair:
1. Train baselines on all other pairs
2. Test on the held-out pair
3. Train the proposed GRU on all other pairs
4. Test on the held-out pair
5. Save per-pair and aggregated results
"""

import json
import os
import random
from collections import Counter

from causal_baselines import (
    BaselineEvaluator,
    LastLanguageBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    NGramBaseline
)
from proposed_model import run_training_experiment


def normalize_pair_id(lang1: str, lang2: str) -> str:
    """Normalize a language pair into canonical form."""
    left = (lang1 or "unknown").strip().lower()
    right = (lang2 or "unknown").strip().lower()
    return "-".join(sorted([left, right]))


def ensure_pair_metadata(examples):
    """Ensure that every example has pair_id metadata."""
    normalized = []

    for ex in examples:
        if "pair_id" not in ex:
            ex["pair_id"] = normalize_pair_id(
                ex.get("first_language", "unknown"),
                ex.get("second_language", "unknown")
            )
        normalized.append(ex)

    return normalized


def save_json(obj, path: str):
    """Save a Python object to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_processed_examples(processed_path: str):
    """Load processed examples and normalize pair metadata."""
    with open(processed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ensure_pair_metadata(data)


def count_pairs(examples):
    """Count examples per language pair."""
    return Counter(ex["pair_id"] for ex in examples)


def select_held_out_pairs(all_data, min_pair_examples=50, max_pairs=6):
    """
    Select the most frequent language pairs that have enough examples
    to support leave-one-pair-out evaluation.
    """
    pair_counter = count_pairs(all_data)

    eligible = [
        (pair_id, count)
        for pair_id, count in pair_counter.most_common()
        if count >= min_pair_examples
    ]

    if max_pairs is not None:
        eligible = eligible[:max_pairs]

    return eligible


def split_by_held_out_pair(all_data, held_out_pair: str):
    """Split examples into train and test by holding out one pair."""
    train_data = [ex for ex in all_data if ex["pair_id"] != held_out_pair]
    test_data = [ex for ex in all_data if ex["pair_id"] == held_out_pair]
    return train_data, test_data


def evaluate_baselines_on_split(train_data, test_data, n_gram=3, skip_lr=False):
    """
    Evaluate baseline models on one leave-one-pair-out split.
    """
    rows = []

    baseline_specs = [
        ("Majority Class", MajorityClassBaseline()),
        ("Last Language", LastLanguageBaseline()),
        (f"{n_gram}-gram", NGramBaseline(n=n_gram)),
    ]

    if not skip_lr:
        baseline_specs.append(
            ("Logistic Regression", LogisticRegressionBaseline(context_window=5))
        )

    for model_name, model in baseline_specs:
        evaluator = BaselineEvaluator()

        try:
            model.train(train_data)
            result = evaluator.evaluate_model(model, model_name, test_data)

            rows.append({
                "model": model_name,
                "switch_accuracy": result["switch_accuracy"],
                "switch_f1": result["switch_f1"],
                "duration_accuracy": result["duration_accuracy"],
                "duration_f1_macro": result["duration_f1_macro"],
                "num_test_instances": result["num_test_instances"],
                "num_switches": result["num_switches"]
            })
        except Exception as exc:
            rows.append({
                "model": model_name,
                "error": str(exc),
                "switch_accuracy": 0.0,
                "switch_f1": 0.0,
                "duration_accuracy": 0.0,
                "duration_f1_macro": 0.0,
                "num_test_instances": 0,
                "num_switches": 0
            })

    return rows


def summarize_rows(rows):
    """
    Aggregate average metrics by model across held-out pairs.
    """
    grouped = {}

    for row in rows:
        model_name = row["model"]
        grouped.setdefault(model_name, {
            "switch_accuracy": [],
            "switch_f1": [],
            "duration_accuracy": [],
            "duration_f1_macro": [],
            "held_out_pairs": 0
        })

        grouped[model_name]["switch_accuracy"].append(row.get("switch_accuracy", 0.0))
        grouped[model_name]["switch_f1"].append(row.get("switch_f1", 0.0))
        grouped[model_name]["duration_accuracy"].append(row.get("duration_accuracy", 0.0))
        grouped[model_name]["duration_f1_macro"].append(row.get("duration_f1_macro", 0.0))
        grouped[model_name]["held_out_pairs"] += 1

    summary = []
    for model_name, stats in grouped.items():
        summary.append({
            "model": model_name,
            "held_out_pairs": stats["held_out_pairs"],
            "avg_switch_accuracy": sum(stats["switch_accuracy"]) / max(1, len(stats["switch_accuracy"])),
            "avg_switch_f1": sum(stats["switch_f1"]) / max(1, len(stats["switch_f1"])),
            "avg_duration_accuracy": sum(stats["duration_accuracy"]) / max(1, len(stats["duration_accuracy"])),
            "avg_duration_f1_macro": sum(stats["duration_f1_macro"]) / max(1, len(stats["duration_f1_macro"]))
        })

    summary.sort(key=lambda x: x["avg_switch_f1"], reverse=True)
    return summary


def run_universality_experiments(
    processed_path="./data/processed/processed_data.json",
    n_gram=3,
    skip_lr=False,
    min_pair_examples=50,
    max_pairs=6,
    seed=42,
    gru_epochs=4,
    gru_batch_size=256,
    gru_max_train_samples=400000,
    gru_max_test_samples=100000
):
    """
    Run leave-one-pair-out universality experiments.
    """
    random.seed(seed)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    all_data = load_processed_examples(processed_path)
    eligible_pairs = select_held_out_pairs(
        all_data=all_data,
        min_pair_examples=min_pair_examples,
        max_pairs=max_pairs
    )

    if not eligible_pairs:
        raise ValueError(
            "No eligible language pairs found for universality evaluation. "
            "Try lowering --min_pair_examples."
        )

    all_rows = []
    pair_manifest = []

    print("\nEligible held-out pairs:")
    for pair_id, count in eligible_pairs:
        print(f"  {pair_id}: {count} examples")

    for held_out_pair, pair_count in eligible_pairs:
        print("\n" + "=" * 70)
        print(f"HELD-OUT PAIR: {held_out_pair}")
        print("=" * 70)

        train_data, test_data = split_by_held_out_pair(all_data, held_out_pair)

        print(f"Train examples (all other pairs): {len(train_data)}")
        print(f"Test examples (held-out pair only): {len(test_data)}")

        pair_manifest.append({
            "held_out_pair": held_out_pair,
            "held_out_pair_example_count": pair_count,
            "num_train_examples": len(train_data),
            "num_test_examples": len(test_data)
        })

        baseline_rows = evaluate_baselines_on_split(
            train_data=train_data,
            test_data=test_data,
            n_gram=n_gram,
            skip_lr=skip_lr
        )

        for row in baseline_rows:
            row["held_out_pair"] = held_out_pair
            row["held_out_pair_example_count"] = pair_count
            row["train_examples"] = len(train_data)
            row["test_examples"] = len(test_data)
            all_rows.append(row)

        safe_pair_name = held_out_pair.replace("-", "_")

        gru_payload = run_training_experiment(
            train_data=train_data,
            test_data=test_data,
            model_output_path=f"./models/universality_{safe_pair_name}.pt",
            results_output_path=f"./results/universality_{safe_pair_name}_gru.json",
            seed=seed,
            epochs=gru_epochs,
            batch_size=gru_batch_size,
            max_train_samples=gru_max_train_samples,
            max_test_samples=gru_max_test_samples,
            experiment_name=f"leave_one_pair_out::{held_out_pair}"
        )

        best_results = gru_payload["best_results"]

        all_rows.append({
            "model": "Proposed GRU",
            "held_out_pair": held_out_pair,
            "held_out_pair_example_count": pair_count,
            "train_examples": len(train_data),
            "test_examples": len(test_data),
            "switch_accuracy": best_results["switch_accuracy"],
            "switch_f1": best_results["switch_f1"],
            "duration_accuracy": best_results["duration_accuracy"],
            "duration_f1_macro": best_results["duration_f1_macro"],
            "num_test_instances": best_results["num_position_samples"],
            "num_switches": best_results["num_duration_samples"]
        })

    summary = summarize_rows(all_rows)

    payload = {
        "config": {
            "processed_path": processed_path,
            "n_gram": n_gram,
            "skip_lr": skip_lr,
            "min_pair_examples": min_pair_examples,
            "max_pairs": max_pairs,
            "seed": seed,
            "gru_epochs": gru_epochs,
            "gru_batch_size": gru_batch_size,
            "gru_max_train_samples": gru_max_train_samples,
            "gru_max_test_samples": gru_max_test_samples
        },
        "held_out_pairs": pair_manifest,
        "rows": all_rows,
        "summary_by_model": summary
    }

    save_json(payload, "./results/universality_results.json")
    print("\n✓ Saved results/universality_results.json")
    return payload