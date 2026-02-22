"""
Causal Baselines for Streaming Code-Switch Prediction

This module implements several baseline models that respect causal constraints:
- Majority Class Baseline
- N-gram Baseline (uses only historical context)
- Simple RNN Baseline (causal by design)

All baselines predict only using past information (positions 1 to t)
to predict what happens at position t+1.
"""

import numpy as np
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MajorityClassBaseline:
    """
    Simplest baseline: Always predict the majority class.
    
    This establishes a lower bound on model performance.
    """
    
    def __init__(self):
        self.switch_majority = None
        self.duration_majority = None
        self.name = "Majority Class Baseline"
    
    def train(self, training_data: List[Dict]):
        """
        Find the majority class in training data.
        
        Args:
            training_data: List of processed examples with streaming labels
        """
        switch_labels = []
        duration_labels = []
        
        for example in training_data:
            for label_info in example['streaming_labels']:
                switch_labels.append(label_info['switch_label'])
                if label_info['switch_label'] == 1:  # Only count duration when switch occurs
                    duration_labels.append(label_info['duration_label'])
        
        # Find majority classes
        switch_counts = Counter(switch_labels)
        self.switch_majority = switch_counts.most_common(1)[0][0]
        
        if duration_labels:
            duration_counts = Counter(duration_labels)
            self.duration_majority = duration_counts.most_common(1)[0][0]
        else:
            self.duration_majority = 0  # Default to small
        
        print(f"\n{self.name} - Training Complete")
        print(f"  Switch majority class: {self.switch_majority} "
              f"({switch_counts[self.switch_majority]}/{len(switch_labels)} = "
              f"{switch_counts[self.switch_majority]/len(switch_labels)*100:.1f}%)")
        print(f"  Duration majority class: {self.duration_majority}")
    
    def predict(self, test_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using majority class for all instances.
        
        Args:
            test_data: List of processed examples
            
        Returns:
            (switch_predictions, duration_predictions)
        """
        num_instances = sum(len(ex['streaming_labels']) for ex in test_data)
        
        switch_preds = np.full(num_instances, self.switch_majority)
        duration_preds = np.full(num_instances, self.duration_majority)
        
        return switch_preds, duration_preds


class NGramBaseline:
    """
    N-gram based baseline using historical language patterns.
    
    Uses the past N language IDs to predict if the next token will switch.
    Respects causal constraint: only uses past information.
    """
    
    def __init__(self, n: int = 3):
        """
        Args:
            n: Number of previous language IDs to consider (default=3)
        """
        self.n = n
        self.switch_probs = defaultdict(lambda: {'switch': 0, 'no_switch': 0})
        self.duration_probs = defaultdict(lambda: {0: 0, 1: 0, 2: 0})
        self.name = f"{n}-gram Baseline"
    
    def _get_context(self, language_ids: List[str], position: int) -> str:
        """
        Get the N-gram context for prediction at position.
        
        Args:
            language_ids: Full sequence of language IDs
            position: Current position (predicting position+1)
            
        Returns:
            String representation of N-gram context
        """
        start = max(0, position - self.n + 1)
        context = language_ids[start:position + 1]
        return '-'.join(context)
    
    def train(self, training_data: List[Dict]):
        """
        Learn N-gram statistics from training data.
        
        Args:
            training_data: List of processed examples
        """
        for example in training_data:
            language_ids = example['language_ids']
            
            for label_info in example['streaming_labels']:
                position = label_info['position']
                
                # Get N-gram context (only past information)
                context = self._get_context(language_ids, position)
                
                # Update switch statistics
                if label_info['switch_label'] == 1:
                    self.switch_probs[context]['switch'] += 1
                else:
                    self.switch_probs[context]['no_switch'] += 1
                
                # Update duration statistics (only when switch occurs)
                if label_info['switch_label'] == 1:
                    dur_class = label_info['duration_label']
                    self.duration_probs[context][dur_class] += 1
        
        print(f"\n{self.name} - Training Complete")
        print(f"  Unique {self.n}-gram contexts: {len(self.switch_probs)}")
        print(f"  Total observations: {sum(v['switch'] + v['no_switch'] for v in self.switch_probs.values())}")
    
    def predict(self, test_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using N-gram probabilities.
        
        Args:
            test_data: List of processed examples
            
        Returns:
            (switch_predictions, duration_predictions)
        """
        switch_preds = []
        duration_preds = []
        
        for example in test_data:
            language_ids = example['language_ids']
            
            for label_info in example['streaming_labels']:
                position = label_info['position']
                context = self._get_context(language_ids, position)
                
                # Predict switch
                if context in self.switch_probs:
                    stats = self.switch_probs[context]
                    # Predict the more frequent outcome
                    switch_pred = 1 if stats['switch'] > stats['no_switch'] else 0
                else:
                    # Unknown context: predict no switch (safe default)
                    switch_pred = 0
                
                switch_preds.append(switch_pred)
                
                # Predict duration
                if switch_pred == 1 and context in self.duration_probs:
                    dur_stats = self.duration_probs[context]
                    # Predict the most frequent duration class
                    duration_pred = max(dur_stats, key=dur_stats.get)
                else:
                    duration_pred = 0  # Default to small
                
                duration_preds.append(duration_pred)
        
        return np.array(switch_preds), np.array(duration_preds)


class LastLanguageBaseline:
    """
    Simple baseline: Predict that the next token will be the same language as current.
    
    This is equivalent to predicting "no switch" always, but contextually motivated.
    """
    
    def __init__(self):
        self.name = "Last Language Baseline"
    
    def train(self, training_data: List[Dict]):
        """No training needed for this baseline."""
        print(f"\n{self.name} - No training required")
    
    def predict(self, test_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict: next token will have same language as current token.
        
        Args:
            test_data: List of processed examples
            
        Returns:
            (switch_predictions, duration_predictions)
        """
        switch_preds = []
        duration_preds = []
        
        for example in test_data:
            for label_info in example['streaming_labels']:
                # Always predict no switch
                switch_preds.append(0)
                duration_preds.append(0)  # Duration doesn't matter if no switch
        
        return np.array(switch_preds), np.array(duration_preds)


class BaselineEvaluator:
    """
    Evaluate baseline models and compare their performance.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, 
                      model, 
                      model_name: str,
                      test_data: List[Dict]) -> Dict:
        """
        Evaluate a baseline model on test data.
        
        Args:
            model: Baseline model instance
            model_name: Name for reporting
            test_data: List of processed test examples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Get ground truth labels
        switch_true = []
        duration_true = []
        
        for example in test_data:
            for label_info in example['streaming_labels']:
                switch_true.append(label_info['switch_label'])
                # Only evaluate duration when switch actually occurs
                if label_info['switch_label'] == 1:
                    duration_true.append(label_info['duration_label'])
        
        # Get predictions
        switch_pred, duration_pred = model.predict(test_data)
        
        # Filter duration predictions to only switches
        duration_pred_filtered = []
        for i, switch in enumerate(switch_true):
            if switch == 1:
                duration_pred_filtered.append(duration_pred[i])
        
        # Calculate metrics for switch prediction
        switch_f1 = f1_score(switch_true, switch_pred, average='binary')
        switch_accuracy = accuracy_score(switch_true, switch_pred)
        
        print(f"\nSwitch Prediction (Binary):")
        print(f"  Accuracy: {switch_accuracy:.4f}")
        print(f"  F1-Score: {switch_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(switch_true, switch_pred, 
                                   target_names=['No Switch', 'Switch'],
                                   digits=4))
        
        # Calculate metrics for duration prediction
        if len(duration_true) > 0:
            duration_accuracy = accuracy_score(duration_true, duration_pred_filtered)
            duration_f1_macro = f1_score(duration_true, duration_pred_filtered, 
                                        average='macro')
            
            print(f"\nDuration Prediction (3-class, on switches only):")
            print(f"  Accuracy: {duration_accuracy:.4f}")
            print(f"  F1-Score (Macro): {duration_f1_macro:.4f}")
            print("\nClassification Report:")
            print(classification_report(duration_true, duration_pred_filtered,
                                       target_names=['Small', 'Medium', 'Large'],
                                       digits=4))
        else:
            duration_accuracy = 0.0
            duration_f1_macro = 0.0
            print("\nNo switches in test data - duration metrics not applicable")
        
        # Store results
        results = {
            'switch_accuracy': switch_accuracy,
            'switch_f1': switch_f1,
            'duration_accuracy': duration_accuracy,
            'duration_f1_macro': duration_f1_macro,
            'num_test_instances': len(switch_true),
            'num_switches': sum(switch_true)
        }
        
        self.results[model_name] = results
        
        return results
    
    def compare_models(self, save_path: str = "./figures/baseline_comparison.png"):
        """
        Create comparison visualization of all evaluated models.
        
        Args:
            save_path: Path to save comparison figure
        """
        if not self.results:
            print("No results to compare. Run evaluate_model() first.")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        switch_f1 = [self.results[m]['switch_f1'] for m in models]
        switch_acc = [self.results[m]['switch_accuracy'] for m in models]
        duration_acc = [self.results[m]['duration_accuracy'] for m in models]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Switch F1-Score
        axes[0].bar(range(len(models)), switch_f1, color='#3498db')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('F1-Score', fontsize=12)
        axes[0].set_title('Switch Prediction F1-Score', fontsize=13, fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(switch_f1):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Plot 2: Switch Accuracy
        axes[1].bar(range(len(models)), switch_acc, color='#2ecc71')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Switch Prediction Accuracy', fontsize=13, fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(switch_acc):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Plot 3: Duration Accuracy
        axes[2].bar(range(len(models)), duration_acc, color='#e74c3c')
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylabel('Accuracy', fontsize=12)
        axes[2].set_title('Duration Prediction Accuracy', fontsize=13, fontweight='bold')
        axes[2].set_ylim([0, 1.0])
        axes[2].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(duration_acc):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to {save_path}")
        plt.close()
        
        # Print summary table
        print(f"\n{'='*80}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<25} {'Switch F1':>12} {'Switch Acc':>12} {'Duration Acc':>12}")
        print(f"{'-'*80}")
        for model in models:
            r = self.results[model]
            print(f"{model:<25} {r['switch_f1']:>12.4f} "
                  f"{r['switch_accuracy']:>12.4f} {r['duration_accuracy']:>12.4f}")
        print(f"{'='*80}")
    
    def save_results(self, output_path: str = "./results/baseline_results.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Causal Baseline Evaluation")
    print("="*60)
    
    # Load processed data
    print("\nLoading processed data...")
    with open("./data/processed/processed_data.json", 'r') as f:
        all_data = json.load(f)
    
    # Split into train/test (80/20)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Training examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")
    
    # Initialize evaluator
    evaluator = BaselineEvaluator()
    
    # Baseline 1: Majority Class
    print("\n" + "="*60)
    print("Training Baseline 1: Majority Class")
    print("="*60)
    majority_model = MajorityClassBaseline()
    majority_model.train(train_data)
    evaluator.evaluate_model(majority_model, "Majority Class", test_data)
    
    # Baseline 2: Last Language (no switch)
    print("\n" + "="*60)
    print("Training Baseline 2: Last Language")
    print("="*60)
    last_lang_model = LastLanguageBaseline()
    last_lang_model.train(train_data)
    evaluator.evaluate_model(last_lang_model, "Last Language", test_data)
    
    # Baseline 3: 3-gram
    print("\n" + "="*60)
    print("Training Baseline 3: 3-gram")
    print("="*60)
    trigram_model = NGramBaseline(n=3)
    trigram_model.train(train_data)
    evaluator.evaluate_model(trigram_model, "3-gram", test_data)
    
    # Baseline 4: 5-gram
    print("\n" + "="*60)
    print("Training Baseline 4: 5-gram")
    print("="*60)
    fivegram_model = NGramBaseline(n=5)
    fivegram_model.train(train_data)
    evaluator.evaluate_model(fivegram_model, "5-gram", test_data)
    
    # Compare all baselines
    evaluator.compare_models(save_path="./figures/baseline_comparison.png")
    
    # Save results
    evaluator.save_results(output_path="./results/baseline_results.json")
    
    print("\n" + "="*60)
    print("✓ Baseline evaluation complete!")
    print("="*60)