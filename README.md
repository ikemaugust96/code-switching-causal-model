# code-switching-prediction
Predictive Multitask Learning for Streaming Code-Switching
## ⚠️ Note on Data Files

The processed data files are **not included in this repository** due to GitHub's file size limits. 

To generate the data:
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Add it to `data_processing.py`
3. Run: `python3 main.py --max_examples 5000`


# Code-Switching Causal Prediction Project

## Overview

This project studies **causal code-switch prediction** and evaluates whether models can generalize across unseen language pairs.

We focus on:

* Switch prediction (binary)
* Duration prediction (3-class)
* Universality via leave-one-pair-out evaluation

---

## Setup

```bash
git clone <your_repo>
cd code-switching-causal-model

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Data Processing

```bash
python3 data_processing.py
```

This generates:

```
data/processed/processed_data.json
```

---

## Baseline Models

```bash
python3 main.py --skip_lr
```

Outputs:

```
results/baseline_results.json
results/pair_baseline_results.json
```

---

## Proposed Model (GRU)

```bash
python3 proposed_model.py \
  --epochs 5 \
  --batch_size 256 \
  --max_train_samples 800000 \
  --max_test_samples 200000
```

Outputs:

```
results/proposed_model_results.json
models/causal_multitask_gru.pt
```

---

## Universality Evaluation

```bash
python3 main.py \
  --run_universality \
  --skip_lr \
  --max_universality_pairs 8 \
  --universality_epochs 3
```

Outputs:

```
results/universality_results.json
```

---

## Visualization

```bash
python3 plot_results.py
```

Outputs:

```
figures/
  overall_comparison.png
  universality_summary.png
  per_pair.png
```

---

## Reproducibility

All experiments can be reproduced with the commands above.
Random seed is fixed to ensure consistent results.

---

## Key Result

The GRU model significantly outperforms N-gram baselines under unseen language pair evaluation, demonstrating partial universality.
