"""
Main Runner Script for Code-Switch Prediction Project

This script orchestrates the complete pipeline:
1. Data loading and processing
2. Baseline model training and evaluation
3. Results visualization

"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules (assuming they are saved as separate files)
# If running as single script, comment out these imports
# from data_processing import SwitchLinguaProcessor
# from causal_baselines import (MajorityClassBaseline, NGramBaseline, 
#                                LastLanguageBaseline, BaselineEvaluator)


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


def main(args):
    """
    Main execution pipeline.
    
    Args:
        args: Command line arguments
    """
    print("="*70)
    print(" CODE-SWITCH PREDICTION - STREAMING DATA & CAUSAL BASELINES")
    print("="*70)
    
    # Step 0: Setup
    print("\n[Step 0] Setting up directory structure...")
    create_directory_structure()
    
    # Step 1: Data Processing
    print("\n" + "="*70)
    print("[Step 1] DATA PROCESSING")
    print("="*70)
    
    from data_processing import SwitchLinguaProcessor
    
    processor = SwitchLinguaProcessor(cache_dir="./data/cache")
    
    # Load dataset
    processor.load_dataset()
    
    # Process examples
    max_ex = args.max_examples if args.max_examples > 0 else None
    processor.process_examples(split='train', max_examples=max_ex)
    
    # Compute and visualize statistics
    stats = processor.compute_statistics()
    processor.visualize_statistics(save_dir="./figures")
    
    # Save processed data
    processor.save_processed_data(output_path="./data/processed")
    
    # Step 2: Train/Test Split
    print("\n" + "="*70)
    print("[Step 2] TRAIN/TEST SPLIT")
    print("="*70)
    
    import json
    
    with open("./data/processed/processed_data.json", 'r') as f:
        all_data = json.load(f)
    
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Training set: {len(train_data)} examples")
    print(f"  Test set: {len(test_data)} examples")
    
    # Save splits
    with open("./data/processed/train_data.json", 'w') as f:
        json.dump(train_data, f)
    with open("./data/processed/test_data.json", 'w') as f:
        json.dump(test_data, f)
    
    print(f"  ✓ Saved train_data.json and test_data.json")
    
    # Step 3: Baseline Evaluation
    print("\n" + "="*70)
    print("[Step 3] CAUSAL BASELINE EVALUATION")
    print("="*70)
    
    from causal_baselines import (MajorityClassBaseline, NGramBaseline,
                                   LastLanguageBaseline, BaselineEvaluator)
    
    evaluator = BaselineEvaluator()
    
    # Baseline 1: Majority Class
    print("\n--- Baseline 1: Majority Class ---")
    majority_model = MajorityClassBaseline()
    majority_model.train(train_data)
    evaluator.evaluate_model(majority_model, "Majority Class", test_data)
    
    # Baseline 2: Last Language
    print("\n--- Baseline 2: Last Language ---")
    last_lang_model = LastLanguageBaseline()
    last_lang_model.train(train_data)
    evaluator.evaluate_model(last_lang_model, "Last Language", test_data)
    
    # Baseline 3: N-gram (default n=3)
    print(f"\n--- Baseline 3: {args.n_gram}-gram ---")
    ngram_model = NGramBaseline(n=args.n_gram)
    ngram_model.train(train_data)
    evaluator.evaluate_model(ngram_model, f"{args.n_gram}-gram", test_data)
    
    # Step 4: Compare Results
    print("\n" + "="*70)
    print("[Step 4] RESULTS COMPARISON")
    print("="*70)
    
    evaluator.compare_models(save_path="./figures/baseline_comparison.png")
    evaluator.save_results(output_path="./results/baseline_results.json")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("[Step 5] SUMMARY")
    print("="*70)
    
    print("\n✓ Pipeline completed successfully!")
    print("\nGenerated files:")
    print("  Data:")
    print("    - ./data/processed/processed_data.json")
    print("    - ./data/processed/train_data.json")
    print("    - ./data/processed/test_data.json")
    print("    - ./data/processed/statistics.json")
    print("\n  Figures:")
    print("    - ./figures/duration_and_switch_distribution.png")
    print("    - ./figures/language_distribution.png")
    print("    - ./figures/sequence_length_distribution.png")
    print("    - ./figures/baseline_comparison.png")
    print("\n  Results:")
    print("    - ./results/baseline_results.json")
    
    print("\n" + "="*70)
    print("Ready for Project Update 1 presentation!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Code-Switch Prediction: Streaming Data & Causal Baselines'
    )
    
    parser.add_argument(
        '--max_examples',
        type=int,
        default=0,
        help='Maximum number of examples to process (0 = all). Default: 0'
    )
    
    parser.add_argument(
        '--n_gram',
        type=int,
        default=3,
        help='N for N-gram baseline model. Default: 3'
    )
    
    args = parser.parse_args()
    
    main(args)
