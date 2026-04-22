"""
SwitchLingua Data Processing Pipeline
Project: Predictive Multitask Learning for Streaming Code-Switching

This module handles:
1. Dataset loading from Hugging Face
2. Streaming data preprocessing with causal constraints
3. Predictive label generation (switch + duration)
4. Data statistics and visualization
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Lazy import for language detection
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SwitchLinguaProcessor:
    """
    Main class for processing SwitchLingua dataset with streaming and causal constraints.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the processor.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset = None
        self.processed_data = []
        self.statistics = {}
        
    def load_dataset(self, language_pairs: Optional[List[str]] = None):
        """
        Load SwitchLingua dataset from Hugging Face
        
        Args:
            language_pairs: List of language pairs to filter (e.g., ['en-es', 'en-hi'])
                          If None, loads all available pairs
        """
        print("Loading SwitchLingua dataset from Hugging Face...")
        
        try:
            # Load the full dataset
            self.dataset = load_dataset("Shelton1013/SwitchLingua_text", cache_dir=self.cache_dir)
            print(f"✓ Dataset loaded successfully!")
            print(f"  Available splits: {list(self.dataset.keys())}")
            
            # Display dataset structure
            if 'train' in self.dataset:
                print(f"\n  Sample from training set:")
                print(f"  Columns: {self.dataset['train'].column_names}")
                print(f"  Number of examples: {len(self.dataset['train'])}")
                
                # Show first example
                first_example = self.dataset['train'][0]
                print(f"\n  First example structure:")
                for key, value in first_example.items():
                    if isinstance(value, (list, str)):
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"    {key}: {preview}")
            
            # Filter by language pairs if specified
            if language_pairs:
                print(f"\n  Filtering for language pairs: {language_pairs}")
                self._filter_language_pairs(language_pairs)
                
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            print("  Please ensure you have access to the dataset.")
            print("  Try: huggingface-cli login")
            raise
    
    def _filter_language_pairs(self, language_pairs: List[str]):
        """
        Filter dataset to only include specified language pairs.
        
        Args:
            language_pairs: List of language pair codes (e.g., ['en-es', 'en-hi'])
        """
        # Implementation depends on dataset structure
        # This is a placeholder - adjust based on actual dataset schema
        print(f"  Note: Language pair filtering to be implemented based on dataset schema")
    
    def generate_streaming_labels(self, 
                                  tokens: List[str], 
                                  language_ids: List[str]) -> List[Dict]:
        """
        Generate predictive labels for streaming code-switch prediction.
        
        CAUSAL CONSTRAINT: For each position t, we only use information from positions 1 to t
        to predict what happens at position t+1.
        
        Args:
            tokens: List of tokens in the sequence
            language_ids: List of language IDs corresponding to each token
            
        Returns:
            List of dictionaries containing predictive labels for each position
        """
        streaming_labels = []
        
        # Process each position in the sequence (except the last one)
        for t in range(len(tokens) - 1):
            # Current token information (what the model can see)
            current_token = tokens[t]
            current_lang = language_ids[t]
            
            # Next token information (what we want to predict)
            next_token = tokens[t + 1]
            next_lang = language_ids[t + 1]
            
            # Task 1: Binary switch prediction
            # Does the language change at position t+1?
            is_switch = 1 if current_lang != next_lang else 0
            
            # Task 2: Duration prediction (only meaningful when switch occurs)
            duration_class = -1  # -1 means "ignore" (no switch)
            
            if is_switch:
                # Count how long the new language segment lasts
                burst_length = 1  # At least the next token
                
                # Look ahead to count burst length (for labeling purposes only)
                for future_idx in range(t + 2, len(tokens)):
                    if language_ids[future_idx] == next_lang:
                        burst_length += 1
                    else:
                        break
                
                # Classify burst length into 3 categories
                if burst_length <= 2:
                    duration_class = 0  # Small: 1-2 tokens
                elif burst_length <= 6:
                    duration_class = 1  # Medium: 3-6 tokens
                else:
                    duration_class = 2  # Large: 7+ tokens
            
            # Store the streaming label
            streaming_labels.append({
                'position': t,
                'current_token': current_token,
                'current_lang': current_lang,
                'context_length': t + 1,  # Number of tokens seen so far
                'switch_label': is_switch,
                'duration_label': duration_class,
                'next_token': next_token,  # For reference only
                'next_lang': next_lang  # For reference only
            })
        
        return streaming_labels
    
    def _parse_switchlingua_text(self, text_data: str) -> Tuple[List[str], List[str]]:
        """
        Parse SwitchLingua text data to extract tokens and language IDs.
        
        The data_generation_result field contains mixed-language text.
        We need to tokenize and identify languages.
        
        Args:
            text_data: Raw text or list of sentences
            
        Returns:
            (tokens, language_ids)
        """
        import re
        
        # If it's a list (multiple turns), join them
        if isinstance(text_data, list):
            text_data = ' '.join(text_data)
        
        # Simple tokenization (split by whitespace and punctuation)
        # This is a simplified version - you may need more sophisticated tokenization
        tokens = re.findall(r'\w+|[^\w\s]', text_data)
        
        # Hybrid language detection: fastText + Unicode fallback
        language_ids = []
        allowed_langs = {'en', 'es', 'hi', 'zh', 'ar', 'ru', 'ja', 'ko', 'fr', 'de', 'pt', 'it'}
        
        for token in tokens:
            lang = None
            
            # Try langdetect first
            if HAS_LANGDETECT and len(token) >= 3 and any(c.isalpha() for c in token):
                try:
                    detected_lang = detect(token)
                    if detected_lang in allowed_langs:
                        lang = detected_lang
                except Exception:
                    pass
            
            # Fallback to Unicode detection
            if lang is None:
                if any('\u0600' <= c <= '\u06FF' for c in token):  # Arabic
                    lang = 'ar'
                elif any('\u4e00' <= c <= '\u9fff' for c in token):  # Chinese
                    lang = 'zh'
                elif any('\u0400' <= c <= '\u04FF' for c in token):  # Cyrillic/Russian
                    lang = 'ru'
                elif any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' for c in token):  # Japanese
                    lang = 'ja'
                elif any('\uAC00' <= c <= '\uD7AF' for c in token):  # Korean
                    lang = 'ko'
                elif any('\u0900' <= c <= '\u097F' for c in token):  # Hindi/Devanagari
                    lang = 'hi'
                else:
                    # Default to English for Latin script
                    lang = 'en'
            
            language_ids.append(lang)
            
        return tokens, language_ids
    
    def process_examples(self, split: str = 'train', max_examples: Optional[int] = None):
        """
        Process examples from a dataset split and generate streaming labels.
        
        Args:
            split: Dataset split to process ('train', 'validation', 'test')
            max_examples: Maximum number of examples to process (None = all)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        print(f"\nProcessing {split} split...")
        data = self.dataset[split]
        
        # Limit number of examples if specified
        num_examples = len(data) if max_examples is None else min(max_examples, len(data))
        
        self.processed_data = []
        processed_count = 0
        
        for idx in range(num_examples):
            try:
                example = data[idx]
                
                # Extract the code-switched text from data_generation_result
                text_data = example.get('data_generation_result', '')
                
                # Skip if no text data
                if not text_data:
                    continue
                
                # Parse tokens and language IDs
                tokens, language_ids = self._parse_switchlingua_text(text_data)
                
                # Skip if data is invalid or too short
                if not tokens or len(tokens) < 2:
                    continue
                
                # Generate streaming labels with causal constraint
                streaming_labels = self.generate_streaming_labels(tokens, language_ids)
                
                # Skip if no labels generated
                if not streaming_labels:
                    continue
                
                # Store processed example
                self.processed_data.append({
                    'example_id': idx,
                    'tokens': tokens,
                    'language_ids': language_ids,
                    'streaming_labels': streaming_labels,
                    'num_switches': sum(1 for label in streaming_labels if label['switch_label'] == 1),
                    'sequence_length': len(tokens),
                    'first_language': example.get('first_language', 'unknown'),
                    'second_language': example.get('second_language', 'unknown'),
                    'cs_type': example.get('cs_type', 'unknown')
                })
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} examples (scanned {idx + 1}/{num_examples})...")
                    
            except Exception as e:
                # Skip problematic examples
                continue
        
        print(f"✓ Processing complete: {len(self.processed_data)} valid examples from {num_examples} total")
        
    def compute_statistics(self):
        """
        Compute comprehensive statistics on the processed data.
        """
        if not self.processed_data:
            raise ValueError("No processed data. Call process_examples() first.")
        
        print("\nComputing dataset statistics...")
        
        # Initialize counters
        total_positions = 0
        total_switches = 0
        duration_counts = Counter()
        language_counts = Counter()
        sequence_lengths = []
        switches_per_sequence = []
        
        # Collect statistics
        for example in self.processed_data:
            sequence_lengths.append(example['sequence_length'])
            switches_per_sequence.append(example['num_switches'])
            
            for label_info in example['streaming_labels']:
                total_positions += 1
                
                if label_info['switch_label'] == 1:
                    total_switches += 1
                    duration_counts[label_info['duration_label']] += 1
                
                language_counts[label_info['current_lang']] += 1
        
        # Calculate percentages
        switch_rate = (total_switches / total_positions * 100) if total_positions > 0 else 0
        
        duration_percentages = {
            'small': (duration_counts[0] / total_switches * 100) if total_switches > 0 else 0,
            'medium': (duration_counts[1] / total_switches * 100) if total_switches > 0 else 0,
            'large': (duration_counts[2] / total_switches * 100) if total_switches > 0 else 0
        }
        
        # Store statistics
        self.statistics = {
            'num_examples': len(self.processed_data),
            'total_positions': total_positions,
            'total_switches': total_switches,
            'switch_rate': switch_rate,
            'duration_distribution': {
                'small (1-2 tokens)': duration_counts[0],
                'medium (3-6 tokens)': duration_counts[1],
                'large (7+ tokens)': duration_counts[2]
            },
            'duration_percentages': duration_percentages,
            'language_distribution': dict(language_counts.most_common()),
            'sequence_lengths': {
                'mean': np.mean(sequence_lengths),
                'std': np.std(sequence_lengths),
                'min': np.min(sequence_lengths),
                'max': np.max(sequence_lengths)
            },
            'switches_per_sequence': {
                'mean': np.mean(switches_per_sequence),
                'std': np.std(switches_per_sequence),
                'min': np.min(switches_per_sequence),
                'max': np.max(switches_per_sequence)
            }
        }
        
        # Print statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"\nBasic Counts:")
        print(f"  Number of examples: {self.statistics['num_examples']}")
        print(f"  Total prediction positions: {self.statistics['total_positions']}")
        print(f"  Total code-switches: {self.statistics['total_switches']}")
        print(f"  Switch rate: {self.statistics['switch_rate']:.2f}%")
        
        print(f"\nDuration Distribution:")
        for duration, count in self.statistics['duration_distribution'].items():
            pct = duration_percentages[duration.split()[0]]
            print(f"  {duration}: {count} ({pct:.2f}%)")
        
        print(f"\nSequence Lengths:")
        print(f"  Mean: {self.statistics['sequence_lengths']['mean']:.2f}")
        print(f"  Std: {self.statistics['sequence_lengths']['std']:.2f}")
        print(f"  Range: [{self.statistics['sequence_lengths']['min']}, "
              f"{self.statistics['sequence_lengths']['max']}]")
        
        print(f"\nSwitches per Sequence:")
        print(f"  Mean: {self.statistics['switches_per_sequence']['mean']:.2f}")
        print(f"  Std: {self.statistics['switches_per_sequence']['std']:.2f}")
        
        print(f"\nTop 5 Languages:")
        for lang, count in list(self.statistics['language_distribution'].items())[:5]:
            pct = count / total_positions * 100
            print(f"  {lang}: {count} ({pct:.2f}%)")
        
        print("="*60)
        
        return self.statistics
    
    def visualize_statistics(self, save_dir: str = "./figures"):
        """
        Create visualizations of dataset statistics.
        
        Args:
            save_dir: Directory to save figure files
        """
        if not self.statistics:
            raise ValueError("No statistics computed. Call compute_statistics() first.")
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nGenerating visualizations...")
        
        # Figure 1: Duration Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Duration bar chart
        durations = ['Small\n(1-2)', 'Medium\n(3-6)', 'Large\n(7+)']
        counts = [
            self.statistics['duration_distribution']['small (1-2 tokens)'],
            self.statistics['duration_distribution']['medium (3-6 tokens)'],
            self.statistics['duration_distribution']['large (7+ tokens)']
        ]
        
        axes[0].bar(durations, counts, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0].set_xlabel('Duration Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Code-Switch Duration Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (dur, cnt) in enumerate(zip(durations, counts)):
            pct = cnt / sum(counts) * 100
            axes[0].text(i, cnt, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Switch vs No-Switch pie chart
        switch_count = self.statistics['total_switches']
        no_switch_count = self.statistics['total_positions'] - switch_count
        
        axes[1].pie([switch_count, no_switch_count], 
                   labels=['Code-Switch', 'No Switch'],
                   autopct='%1.1f%%',
                   colors=['#e74c3c', '#95a5a6'],
                   startangle=90)
        axes[1].set_title('Switch vs No-Switch Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/duration_and_switch_distribution.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: duration_and_switch_distribution.png")
        plt.close()
        
        # Figure 2: Language Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top 10 languages
        top_langs = list(self.statistics['language_distribution'].items())[:10]
        langs, counts = zip(*top_langs)
        
        ax.barh(range(len(langs)), counts, color='#3498db')
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels(langs)
        ax.set_xlabel('Token Count', fontsize=12)
        ax.set_ylabel('Language', fontsize=12)
        ax.set_title('Top 10 Language Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/language_distribution.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: language_distribution.png")
        plt.close()
        
        # Figure 3: Sequence Length Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lengths = [ex['sequence_length'] for ex in self.processed_data]
        
        ax.hist(lengths, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axvline(self.statistics['sequence_lengths']['mean'], 
                  color='red', linestyle='--', linewidth=2, 
                  label=f"Mean: {self.statistics['sequence_lengths']['mean']:.1f}")
        ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sequence_length_distribution.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: sequence_length_distribution.png")
        plt.close()
        
        print(f"\n✓ All visualizations saved to {save_dir}/")
    
    def save_processed_data(self, output_path: str = "./data/processed"):
        """
        Save processed data to disk for later use.
        
        Args:
            output_path: Directory to save processed data
        """
        import numpy as np
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\nSaving processed data to {output_path}...")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Save processed examples
        processed_data_serializable = convert_to_serializable(self.processed_data)
        with open(f"{output_path}/processed_data.json", 'w', encoding='utf-8') as f:
            json.dump(processed_data_serializable, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        statistics_serializable = convert_to_serializable(self.statistics)
        with open(f"{output_path}/statistics.json", 'w', encoding='utf-8') as f:
            json.dump(statistics_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved processed_data.json ({len(self.processed_data)} examples)")
        print(f"  ✓ Saved statistics.json")
    
    def get_streaming_batches(self, batch_size: int = 32) -> List[Dict]:
        """
        Generate batches for streaming prediction training.
        
        This simulates the streaming scenario where we process sequences
        position by position with causal constraints.
        
        Args:
            batch_size: Number of prediction positions per batch
            
        Returns:
            List of batches, each containing multiple streaming prediction instances
        """
        if not self.processed_data:
            raise ValueError("No processed data available")
        
        all_instances = []
        
        # Flatten all streaming labels from all examples
        for example in self.processed_data:
            for label_info in example['streaming_labels']:
                all_instances.append(label_info)
        
        # Create batches
        batches = []
        for i in range(0, len(all_instances), batch_size):
            batch = all_instances[i:i + batch_size]
            batches.append(batch)
        
        print(f"\nGenerated {len(batches)} streaming batches (batch_size={batch_size})")
        print(f"  Total prediction instances: {len(all_instances)}")
        
        return batches


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("SwitchLingua Data Processing Pipeline")
    print("="*60)
    
    # Initialize processor
    processor = SwitchLinguaProcessor(cache_dir="./data/cache")
    
    # Load dataset
    processor.load_dataset()

    # Process a subset for testing (remove max_examples to process all)
    processor.process_examples(split='train', max_examples=1000)
    
    # Compute statistics
    stats = processor.compute_statistics()
    
    # Create visualizations
    processor.visualize_statistics(save_dir="./figures")
    
    # Save processed data
    processor.save_processed_data(output_path="./data/processed")
    
    # Generate streaming batches
    batches = processor.get_streaming_batches(batch_size=32)
    print(f"\nExample batch structure:")
    print(f"  First batch has {len(batches[0])} instances")
    print(f"  Sample instance: {batches[0][0]}")
    
    print("\n" + "="*60)
    print("✓ Data processing complete!")
    print("="*60)