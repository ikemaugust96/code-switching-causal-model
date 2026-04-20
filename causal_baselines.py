"""
Causal Baselines for Streaming Code-Switch Prediction

This module implements several baseline models that respect causal constraints:
- Majority Class Baseline
- N-gram Baseline (uses only historical context)
- Last Language Baseline
- Logistic Regression Baseline (learned features with causal constraints)

All baselines predict only using past information (positions 0 to t)
to predict what happens at position t+1.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


class MajorityClassBaseline:
    """
    Simplest baseline: always predict the majority class observed in training.
    """

    def __init__(self):
        self.switch_majority = 0
        self.duration_majority = 0
        self.name = "Majority Class Baseline"
        self.is_trained = False

    def train(self, training_data: List[Dict]):
        """
        Find the majority class in the training data.

        Args:
            training_data: List of processed examples with streaming labels
        """
        switch_labels = []
        duration_labels = []

        for example in training_data:
            for label_info in example["streaming_labels"]:
                switch_labels.append(int(label_info["switch_label"]))

                # Duration is only evaluated when a true switch occurs
                if int(label_info["switch_label"]) == 1:
                    duration_labels.append(int(label_info["duration_label"]))

        switch_counts = Counter(switch_labels)
        self.switch_majority = switch_counts.most_common(1)[0][0]

        if duration_labels:
            duration_counts = Counter(duration_labels)
            self.duration_majority = duration_counts.most_common(1)[0][0]
        else:
            self.duration_majority = 0

        self.is_trained = True

        print(f"\n{self.name} - Training Complete")
        print(
            f"  Switch majority class: {self.switch_majority} "
            f"({switch_counts[self.switch_majority]}/{len(switch_labels)} = "
            f"{switch_counts[self.switch_majority] / max(1, len(switch_labels)) * 100:.1f}%)"
        )
        print(f"  Duration majority class: {self.duration_majority}")

    def predict(self, prefix_tokens: List[str], prefix_langs: List[str] = None) -> Dict[str, int]:
        """
        Predict using the majority class baseline.

        Args:
            prefix_tokens: Tokens up to the current position
            prefix_langs: Language IDs up to the current position

        Returns:
            Dictionary with switch and duration predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return {
            "switch_label": int(self.switch_majority),
            "duration_label": int(self.duration_majority),
        }


class NGramBaseline:
    """
    N-gram baseline using historical language patterns only.

    Uses the past N language IDs to predict if the next token will switch.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.switch_probs = defaultdict(lambda: {"switch": 0, "no_switch": 0})
        self.duration_probs = defaultdict(lambda: {0: 0, 1: 0, 2: 0})
        self.global_switch_majority = 0
        self.global_duration_majority = 0
        self.name = f"{n}-gram Baseline"
        self.is_trained = False

    def _get_context_from_prefix(self, prefix_langs: List[str]) -> str:
        """
        Build an N-gram context from the visible language prefix.

        Args:
            prefix_langs: Language IDs visible to the model up to current position

        Returns:
            String representation of N-gram context
        """
        if not prefix_langs:
            return "<EMPTY>"

        context = prefix_langs[-self.n :]
        return "-".join(context)

    def train(self, training_data: List[Dict]):
        """
        Learn N-gram statistics from training data.

        Args:
            training_data: List of processed examples
        """
        switch_labels = []
        duration_labels = []

        for example in training_data:
            language_ids = example["language_ids"]

            for label_info in example["streaming_labels"]:
                position = int(label_info["position"])
                prefix_langs = language_ids[: position + 1]
                context = self._get_context_from_prefix(prefix_langs)

                switch_label = int(label_info["switch_label"])
                duration_label = int(label_info["duration_label"])

                switch_labels.append(switch_label)

                if switch_label == 1:
                    self.switch_probs[context]["switch"] += 1
                    self.duration_probs[context][duration_label] += 1
                    duration_labels.append(duration_label)
                else:
                    self.switch_probs[context]["no_switch"] += 1

        switch_counts = Counter(switch_labels)
        self.global_switch_majority = switch_counts.most_common(1)[0][0]

        if duration_labels:
            duration_counts = Counter(duration_labels)
            self.global_duration_majority = duration_counts.most_common(1)[0][0]
        else:
            self.global_duration_majority = 0

        self.is_trained = True

        print(f"\n{self.name} - Training Complete")
        print(f"  Unique {self.n}-gram contexts: {len(self.switch_probs)}")
        print(
            f"  Total observations: "
            f"{sum(v['switch'] + v['no_switch'] for v in self.switch_probs.values())}"
        )

    def predict(self, prefix_tokens: List[str], prefix_langs: List[str]) -> Dict[str, int]:
        """
        Predict using learned N-gram statistics.

        Args:
            prefix_tokens: Tokens up to the current position
            prefix_langs: Language IDs up to the current position

        Returns:
            Dictionary with switch and duration predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        context = self._get_context_from_prefix(prefix_langs)

        if context in self.switch_probs:
            stats = self.switch_probs[context]
            switch_pred = 1 if stats["switch"] > stats["no_switch"] else 0
        else:
            switch_pred = self.global_switch_majority

        if switch_pred == 1 and context in self.duration_probs:
            dur_stats = self.duration_probs[context]
            duration_pred = max(dur_stats, key=dur_stats.get)
        else:
            duration_pred = self.global_duration_majority

        return {
            "switch_label": int(switch_pred),
            "duration_label": int(duration_pred),
        }


class LastLanguageBaseline:
    """
    Simple baseline: always predict no switch.

    This corresponds to predicting that the next token stays in the same language
    as the current visible language.
    """

    def __init__(self):
        self.name = "Last Language Baseline"
        self.is_trained = False

    def train(self, training_data: List[Dict]):
        """
        No training is required for this baseline.
        """
        self.is_trained = True
        print(f"\n{self.name} - No training required")

    def predict(self, prefix_tokens: List[str], prefix_langs: List[str]) -> Dict[str, int]:
        """
        Predict that the next token will not switch language.

        Args:
            prefix_tokens: Tokens up to the current position
            prefix_langs: Language IDs up to the current position

        Returns:
            Dictionary with switch and duration predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return {
            "switch_label": 0,
            "duration_label": 0,
        }


class LogisticRegressionBaseline:
    """
    Learned baseline using Logistic Regression with hand-crafted causal features.
    """

    def __init__(self, context_window: int = 5):
        self.context_window = context_window
        self.switch_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )
        self.duration_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )
        self.lang_encoder = LabelEncoder()
        self.fitted = False
        self.name = "Logistic Regression Baseline"
        self.default_duration = 0

    def _extract_features(self, prefix_tokens: List[str], prefix_langs: List[str]) -> np.ndarray:
        """
        Extract causal features from the visible prefix only.

        Args:
            prefix_tokens: Tokens up to the current position
            prefix_langs: Language IDs up to the current position

        Returns:
            Numpy feature vector
        """
        if not prefix_langs:
            return np.zeros(self.context_window + 5, dtype=float)

        features = []
        position = len(prefix_langs) - 1
        current_lang = prefix_langs[-1]

        # Feature 1: current language
        current_lang_encoded = self.lang_encoder.transform([current_lang])[0]
        features.append(current_lang_encoded)

        # Feature 2..(context_window+1): previous languages
        for i in range(1, self.context_window + 1):
            if len(prefix_langs) - 1 - i >= 0:
                prev_lang = prefix_langs[-1 - i]
                prev_lang_encoded = self.lang_encoder.transform([prev_lang])[0]
                features.append(prev_lang_encoded)
            else:
                features.append(-1)

        # Feature: normalized position
        features.append(position / 100.0)

        # Feature: normalized prefix length
        features.append(min(len(prefix_langs), 10) / 10.0)

        # Feature: switch rate so far
        switches_so_far = 0
        for i in range(len(prefix_langs) - 1):
            if prefix_langs[i] != prefix_langs[i + 1]:
                switches_so_far += 1
        switch_rate = switches_so_far / max(1, len(prefix_langs) - 1)
        features.append(switch_rate)

        # Feature: stability length in current language
        stability = 0
        for i in range(len(prefix_langs) - 1, -1, -1):
            if prefix_langs[i] == current_lang:
                stability += 1
            else:
                break
        features.append(min(stability, 10) / 10.0)

        return np.array(features, dtype=float)

    def train(self, training_data: List[Dict]):
        """
        Train logistic regression models on extracted features.

        Args:
            training_data: List of processed examples with streaming labels
        """
        all_langs = []
        for example in training_data:
            all_langs.extend(example["language_ids"])

        unique_langs = sorted(list(set(all_langs)))
        self.lang_encoder.fit(unique_langs)

        print(f"\n{self.name} - Extracting features...")

        X_switch = []
        y_switch = []
        X_duration = []
        y_duration = []

        for example in training_data:
            tokens = example["tokens"]
            language_ids = example["language_ids"]

            for label_info in example["streaming_labels"]:
                position = int(label_info["position"])
                prefix_tokens = tokens[: position + 1]
                prefix_langs = language_ids[: position + 1]

                features = self._extract_features(prefix_tokens, prefix_langs)

                switch_label = int(label_info["switch_label"])
                duration_label = int(label_info["duration_label"])

                X_switch.append(features)
                y_switch.append(switch_label)

                if switch_label == 1:
                    X_duration.append(features)
                    y_duration.append(duration_label)

        X_switch = np.array(X_switch)
        y_switch = np.array(y_switch)

        print(f"  Training switch model on {len(X_switch)} instances...")
        self.switch_model.fit(X_switch, y_switch)

        if len(X_duration) > 0:
            X_duration = np.array(X_duration)
            y_duration = np.array(y_duration)
            print(f"  Training duration model on {len(X_duration)} switch instances...")
            self.duration_model.fit(X_duration, y_duration)
            self.default_duration = Counter(y_duration).most_common(1)[0][0]
        else:
            self.default_duration = 0

        self.fitted = True
        print(f"✓ {self.name} - Training Complete")
        print(f"  Features per instance: {X_switch.shape[1]}")
        print(f"  Unique languages: {len(unique_langs)}")

    def predict(self, prefix_tokens: List[str], prefix_langs: List[str]) -> Dict[str, int]:
        """
        Predict using the trained logistic regression models.

        Args:
            prefix_tokens: Tokens up to the current position
            prefix_langs: Language IDs up to the current position

        Returns:
            Dictionary with switch and duration predictions
        """
        if not self.fitted:
            raise ValueError("Model not trained. Call train() first.")

        features = self._extract_features(prefix_tokens, prefix_langs).reshape(1, -1)

        switch_pred = int(self.switch_model.predict(features)[0])

        if switch_pred == 1:
            duration_pred = int(self.duration_model.predict(features)[0])
        else:
            duration_pred = int(self.default_duration)

        return {
            "switch_label": switch_pred,
            "duration_label": duration_pred,
        }


class BaselineEvaluator:
    """
    Evaluate baseline models on streaming code-switch prediction.
    """

    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, model_name: str, test_data: List[Dict]) -> Dict:
        """
        Evaluate a baseline model on test data.

        Args:
            model: Trained baseline model with predict(prefix_tokens, prefix_langs)
            model_name: Name of the model for reporting
            test_data: List of processed test examples

        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "=" * 60)
        print(f"Evaluating: {model_name}")
        print("=" * 60)

        switch_true = []
        switch_pred = []
        duration_true = []
        duration_pred = []

        for example in test_data:
            tokens = example["tokens"]
            language_ids = example["language_ids"]
            streaming_labels = example["streaming_labels"]

            for label_info in streaming_labels:
                pos = int(label_info["position"])

                prefix_tokens = tokens[: pos + 1]
                prefix_langs = language_ids[: pos + 1]

                pred = model.predict(prefix_tokens, prefix_langs)

                true_switch = int(label_info["switch_label"])
                true_duration = int(label_info["duration_label"])

                pred_switch = int(pred.get("switch_label", 0))
                pred_duration = int(pred.get("duration_label", 0))

                switch_true.append(true_switch)
                switch_pred.append(pred_switch)

                if true_switch == 1 and true_duration != -1:
                    duration_true.append(true_duration)
                    duration_pred.append(pred_duration)

        if len(switch_true) == 0:
            switch_accuracy = 0.0
            switch_f1 = 0.0
        else:
            switch_accuracy = accuracy_score(switch_true, switch_pred)
            switch_f1 = f1_score(
                switch_true,
                switch_pred,
                average="binary",
                zero_division=0
            )

        print("\nSwitch Prediction (Binary):")
        print(f"  Accuracy: {switch_accuracy:.4f}")
        print(f"  F1-Score: {switch_f1:.4f}")

        print("\nClassification Report:")
        if len(switch_true) == 0:
            print("No switch samples available for this evaluation split.")
        else:
            print(classification_report(
                switch_true,
                switch_pred,
                labels=[0, 1],
                target_names=["No Switch", "Switch"],
                digits=4,
                zero_division=0
            ))

        if len(duration_true) == 0:
            duration_accuracy = 0.0
            duration_f1_macro = 0.0
        else:
            duration_accuracy = accuracy_score(duration_true, duration_pred)
            duration_f1_macro = f1_score(
                duration_true,
                duration_pred,
                average="macro",
                zero_division=0
            )

        print("\nDuration Prediction (3-class, on switches only):")
        print(f"  Accuracy: {duration_accuracy:.4f}")
        print(f"  F1-Score (Macro): {duration_f1_macro:.4f}")

        print("\nClassification Report:")
        if len(duration_true) == 0:
            print("No duration samples available for this evaluation split.")
        else:
            print(classification_report(
                duration_true,
                duration_pred,
                labels=[0, 1, 2],
                target_names=["Small", "Medium", "Large"],
                digits=4,
                zero_division=0
            ))

        results = {
            "model_name": model_name,
            "switch_accuracy": float(switch_accuracy),
            "switch_f1": float(switch_f1),
            "duration_accuracy": float(duration_accuracy),
            "duration_f1_macro": float(duration_f1_macro),
            "num_test_instances": int(len(switch_true)),
            "num_switches": int(sum(switch_true)),
            "num_duration_instances": int(len(duration_true)),
        }

        self.results[model_name] = results
        return results

    def compare_models(self, save_path: str = "./figures/baseline_comparison.png"):
        """
        Create a comparison visualization of all evaluated models.

        Args:
            save_path: Path to save comparison figure
        """
        if not self.results:
            print("No results to compare. Run evaluate_model() first.")
            return

        models = list(self.results.keys())
        switch_f1 = [self.results[m]["switch_f1"] for m in models]
        switch_acc = [self.results[m]["switch_accuracy"] for m in models]
        duration_acc = [self.results[m]["duration_accuracy"] for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(range(len(models)), switch_f1)
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha="right")
        axes[0].set_ylabel("F1-Score", fontsize=12)
        axes[0].set_title("Switch Prediction F1-Score", fontsize=13, fontweight="bold")
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis="y", alpha=0.3)

        for i, v in enumerate(switch_f1):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        axes[1].bar(range(len(models)), switch_acc)
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha="right")
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Switch Prediction Accuracy", fontsize=13, fontweight="bold")
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis="y", alpha=0.3)

        for i, v in enumerate(switch_acc):
            axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        axes[2].bar(range(len(models)), duration_acc)
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha="right")
        axes[2].set_ylabel("Accuracy", fontsize=12)
        axes[2].set_title("Duration Prediction Accuracy", fontsize=13, fontweight="bold")
        axes[2].set_ylim([0, 1.0])
        axes[2].grid(axis="y", alpha=0.3)

        for i, v in enumerate(duration_acc):
            axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Comparison plot saved to {save_path}")
        plt.close()

        print(f"\n{'=' * 80}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'Model':<25} {'Switch F1':>12} {'Switch Acc':>12} {'Duration Acc':>12}")
        print(f"{'-' * 80}")
        for model in models:
            r = self.results[model]
            print(
                f"{model:<25} {r['switch_f1']:>12.4f} "
                f"{r['switch_accuracy']:>12.4f} {r['duration_accuracy']:>12.4f}"
            )
        print(f"{'=' * 80}")

    def save_results(self, output_path: str = "./results/baseline_results.json"):
        """
        Save evaluation results to a JSON file.

        Args:
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Baseline Evaluation")
    print("=" * 60)

    print("\nLoading processed data...")
    with open("./data/processed/processed_data.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)

    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"  Training examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")

    evaluator = BaselineEvaluator()

    print("\n" + "=" * 60)
    print("Training Baseline 1: Majority Class")
    print("=" * 60)
    majority_model = MajorityClassBaseline()
    majority_model.train(train_data)
    evaluator.evaluate_model(majority_model, "Majority Class", test_data)

    print("\n" + "=" * 60)
    print("Training Baseline 2: Last Language")
    print("=" * 60)
    last_lang_model = LastLanguageBaseline()
    last_lang_model.train(train_data)
    evaluator.evaluate_model(last_lang_model, "Last Language", test_data)

    print("\n" + "=" * 60)
    print("Training Baseline 3: 3-gram")
    print("=" * 60)
    trigram_model = NGramBaseline(n=3)
    trigram_model.train(train_data)
    evaluator.evaluate_model(trigram_model, "3-gram", test_data)

    print("\n" + "=" * 60)
    print("Training Baseline 4: Logistic Regression")
    print("=" * 60)
    lr_model = LogisticRegressionBaseline(context_window=5)
    lr_model.train(train_data)
    evaluator.evaluate_model(lr_model, "Logistic Regression", test_data)

    evaluator.compare_models(save_path="./figures/baseline_comparison.png")
    evaluator.save_results(output_path="./results/baseline_results.json")

    print("\n" + "=" * 60)
    print("✓ Baseline evaluation complete!")
    print("=" * 60)