import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# A. UTILITIES
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


# ============================================================
# B. DATASET
# ============================================================

class StreamingCodeSwitchDataset(Dataset):
    """
    Position-level dataset for causal code-switch prediction.

    Each sample uses prefix [0:t] to predict:
    1) switch at t+1
    2) duration class at t+1
    """

    def __init__(
        self,
        data,
        token2id,
        lang2id,
        max_len=40,
        max_samples=None,
        sample_switch_ratio=1.0,
        sample_noswitch_ratio=0.25
    ):
        self.samples = []
        self.token2id = token2id
        self.lang2id = lang2id
        self.max_len = max_len

        for ex in data:
            tokens = ex["tokens"]
            langs = ex["language_ids"]
            labels = ex["streaming_labels"]

            for label in labels:
                pos = label["position"]
                switch_label = int(label["switch_label"])
                duration_label = int(label["duration_label"])

                # B1. Position-level sampling to control dataset size
                if switch_label == 1:
                    if random.random() > sample_switch_ratio:
                        continue
                else:
                    if random.random() > sample_noswitch_ratio:
                        continue

                prefix_tokens = tokens[:pos + 1]
                prefix_langs = langs[:pos + 1]

                if len(prefix_tokens) > max_len:
                    prefix_tokens = prefix_tokens[-max_len:]
                    prefix_langs = prefix_langs[-max_len:]

                token_ids = [token2id.get(tok, token2id["<UNK>"]) for tok in prefix_tokens]
                lang_ids = [lang2id.get(lang, lang2id["<UNK>"]) for lang in prefix_langs]

                # B2. Causal scalar features
                current_lang = langs[pos]

                stability = 0
                for i in range(pos, -1, -1):
                    if langs[i] == current_lang:
                        stability += 1
                    else:
                        break

                switches_so_far = 0
                for i in range(pos):
                    if langs[i] != langs[i + 1]:
                        switches_so_far += 1

                context_len = pos + 1
                switch_rate_so_far = safe_div(switches_so_far, max(1, pos))

                # Simple punctuation feature
                current_token = tokens[pos]
                is_punct = 1.0 if not any(ch.isalnum() for ch in current_token) else 0.0

                features = np.array([
                    min(context_len, max_len) / max_len,
                    min(stability, 10) / 10.0,
                    switch_rate_so_far,
                    is_punct
                ], dtype=np.float32)

                self.samples.append({
                    "token_ids": token_ids,
                    "lang_ids": lang_ids,
                    "features": features,
                    "switch_label": switch_label,
                    "duration_label": duration_label
                })

        if max_samples is not None and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_vocab(data, min_freq=2):
    token_counter = Counter()
    lang_counter = Counter()

    for ex in data:
        token_counter.update(ex["tokens"])
        lang_counter.update(ex["language_ids"])

    token2id = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for tok, freq in token_counter.items():
        if freq >= min_freq:
            token2id[tok] = len(token2id)

    lang2id = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for lang in sorted(lang_counter.keys()):
        lang2id[lang] = len(lang2id)

    return token2id, lang2id


def collate_fn(batch):
    max_seq_len = max(len(x["token_ids"]) for x in batch)

    token_batch = []
    lang_batch = []
    mask_batch = []
    feat_batch = []
    switch_batch = []
    duration_batch = []

    for item in batch:
        seq_len = len(item["token_ids"])
        pad_len = max_seq_len - seq_len

        token_batch.append(item["token_ids"] + [0] * pad_len)
        lang_batch.append(item["lang_ids"] + [0] * pad_len)
        mask_batch.append([1] * seq_len + [0] * pad_len)
        feat_batch.append(item["features"])
        switch_batch.append(item["switch_label"])
        duration_batch.append(item["duration_label"])

    return {
        "token_ids": torch.tensor(token_batch, dtype=torch.long),
        "lang_ids": torch.tensor(lang_batch, dtype=torch.long),
        "mask": torch.tensor(mask_batch, dtype=torch.float32),
        "features": torch.tensor(np.stack(feat_batch), dtype=torch.float32),
        "switch_labels": torch.tensor(switch_batch, dtype=torch.long),
        "duration_labels": torch.tensor(duration_batch, dtype=torch.long)
    }


# ============================================================
# C. MODEL
# ============================================================

class CausalMultitaskGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_langs,
        token_emb_dim=128,
        lang_emb_dim=24,
        feat_dim=4,
        hidden_dim=192,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(num_langs, lang_emb_dim, padding_idx=0)

        input_dim = token_emb_dim + lang_emb_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        combined_dim = hidden_dim + feat_dim

        self.switch_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, token_ids, lang_ids, mask, features):
        tok = self.token_emb(token_ids)
        lng = self.lang_emb(lang_ids)

        x = torch.cat([tok, lng], dim=-1)

        out, _ = self.gru(x)

        lengths = mask.sum(dim=1).long() - 1
        lengths = torch.clamp(lengths, min=0)

        batch_idx = torch.arange(out.size(0), device=out.device)
        last_hidden = out[batch_idx, lengths]

        combined = torch.cat([self.dropout(last_hidden), features], dim=-1)

        switch_logits = self.switch_head(combined)
        duration_logits = self.duration_head(combined)

        return switch_logits, duration_logits


# ============================================================
# D. TRAINING
# ============================================================

def compute_class_weights(dataset):
    switch_counter = Counter()
    duration_counter = Counter()

    for sample in dataset.samples:
        switch_counter[sample["switch_label"]] += 1
        if sample["duration_label"] != -1:
            duration_counter[sample["duration_label"]] += 1

    switch_weights = []
    total_switch = sum(switch_counter.values())
    for cls in [0, 1]:
        switch_weights.append(total_switch / max(1, switch_counter[cls]))

    duration_weights = []
    total_duration = sum(duration_counter.values())
    for cls in [0, 1, 2]:
        duration_weights.append(total_duration / max(1, duration_counter[cls]))

    return torch.tensor(switch_weights, dtype=torch.float32), torch.tensor(duration_weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, switch_loss_fn, duration_loss_fn, lambda_duration=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        mask = batch["mask"].to(device)
        features = batch["features"].to(device)
        switch_labels = batch["switch_labels"].to(device)
        duration_labels = batch["duration_labels"].to(device)

        optimizer.zero_grad()

        switch_logits, duration_logits = model(token_ids, lang_ids, mask, features)

        switch_loss = switch_loss_fn(switch_logits, switch_labels)

        valid = duration_labels != -1
        if valid.any():
            duration_loss = duration_loss_fn(duration_logits[valid], duration_labels[valid])
        else:
            duration_loss = torch.tensor(0.0, device=device)

        loss = switch_loss + lambda_duration * duration_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    switch_true = []
    switch_pred = []

    duration_true = []
    duration_pred = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        mask = batch["mask"].to(device)
        features = batch["features"].to(device)
        switch_labels = batch["switch_labels"].to(device)
        duration_labels = batch["duration_labels"].to(device)

        switch_logits, duration_logits = model(token_ids, lang_ids, mask, features)

        switch_hat = switch_logits.argmax(dim=1)
        duration_hat = duration_logits.argmax(dim=1)

        switch_true.extend(switch_labels.cpu().numpy().tolist())
        switch_pred.extend(switch_hat.cpu().numpy().tolist())

        valid = duration_labels != -1
        if valid.any():
            duration_true.extend(duration_labels[valid].cpu().numpy().tolist())
            duration_pred.extend(duration_hat[valid].cpu().numpy().tolist())

    results = {
        "switch_accuracy": accuracy_score(switch_true, switch_pred),
        "switch_f1": f1_score(switch_true, switch_pred, average="binary"),
        "duration_accuracy": accuracy_score(duration_true, duration_pred) if len(duration_true) > 0 else 0.0,
        "duration_f1_macro": f1_score(duration_true, duration_pred, average="macro") if len(duration_true) > 0 else 0.0,
    }

    return results


# ============================================================
# E. MAIN
# ============================================================

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open("./data/processed/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open("./data/processed/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded train examples: {len(train_data)}")
    print(f"Loaded test examples: {len(test_data)}")

    token2id, lang2id = build_vocab(train_data, min_freq=2)

    print(f"Token vocab size: {len(token2id)}")
    print(f"Language vocab size: {len(lang2id)}")

    # E1. Sample positions to make training feasible
    train_dataset = StreamingCodeSwitchDataset(
        train_data,
        token2id,
        lang2id,
        max_len=40,
        max_samples=800000,
        sample_switch_ratio=1.0,
        sample_noswitch_ratio=0.20
    )

    test_dataset = StreamingCodeSwitchDataset(
        test_data,
        token2id,
        lang2id,
        max_len=40,
        max_samples=200000,
        sample_switch_ratio=1.0,
        sample_noswitch_ratio=0.20
    )

    print(f"Train position samples: {len(train_dataset)}")
    print(f"Test position samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = CausalMultitaskGRU(
        vocab_size=len(token2id),
        num_langs=len(lang2id),
        token_emb_dim=128,
        lang_emb_dim=24,
        feat_dim=4,
        hidden_dim=192,
        num_layers=2,
        dropout=0.3
    ).to(device)

    switch_weights, duration_weights = compute_class_weights(train_dataset)
    switch_weights = switch_weights.to(device)
    duration_weights = duration_weights.to(device)

    switch_loss_fn = nn.CrossEntropyLoss(weight=switch_weights)
    duration_loss_fn = nn.CrossEntropyLoss(weight=duration_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_switch_f1 = -1.0
    best_results = None

    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            switch_loss_fn=switch_loss_fn,
            duration_loss_fn=duration_loss_fn,
            lambda_duration=1.0
        )

        results = evaluate(model, test_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Switch Accuracy: {results['switch_accuracy']:.4f}")
        print(f"Switch F1: {results['switch_f1']:.4f}")
        print(f"Duration Accuracy: {results['duration_accuracy']:.4f}")
        print(f"Duration Macro F1: {results['duration_f1_macro']:.4f}")

        if results["switch_f1"] > best_switch_f1:
            best_switch_f1 = results["switch_f1"]
            best_results = results
            torch.save(model.state_dict(), "./models/causal_multitask_gru.pt")
            print("✓ Saved best model")

    print("\nBest Results:")
    print(best_results)

    with open("./results/proposed_model_results.json", "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)

    print("✓ Saved proposed_model_results.json")


if __name__ == "__main__":
    main()
