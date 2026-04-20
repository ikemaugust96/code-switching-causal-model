import json
import os
import matplotlib.pyplot as plt

# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)


# ================================
# 图1：Baseline vs GRU（Random Split）
# ================================
def plot_overall():
    with open("results/baseline_results.json") as f:
        baseline = json.load(f)

    with open("results/proposed_model_results.json") as f:
        model = json.load(f)

    models = []
    switch_f1 = []

    for k, v in baseline.items():
        models.append(k)
        switch_f1.append(v["switch_f1"])

    models.append("GRU")
    switch_f1.append(model["best_results"]["switch_f1"])

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, switch_f1)

    plt.title("Switch F1 (Random Split)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 0.01, f"{y:.2f}", ha='center')

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/overall_comparison.png")
    plt.close()

    print("✓ Saved figures/overall_comparison.png")


# ================================
# 图2：Universality Summary ⭐最重要
# ================================
def plot_universality_summary():
    with open("results/universality_results.json") as f:
        data = json.load(f)

    summary = data["summary_by_model"]

    models = [x["model"] for x in summary]
    f1_scores = [x["avg_switch_f1"] for x in summary]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, f1_scores)

    plt.title("Universality (LOPO) - Average Switch F1")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 0.01, f"{y:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig("figures/universality_summary.png")
    plt.close()

    print("✓ Saved figures/universality_summary.png")


# ================================
# 图3：Per-pair 对比
# ================================
def plot_per_pair():
    with open("results/universality_results.json") as f:
        data = json.load(f)

    rows = data["rows"]

    pairs = list(set([r["held_out_pair"] for r in rows]))
    pairs.sort()

    models = list(set([r["model"] for r in rows]))

    pair_scores = {m: [] for m in models}

    for p in pairs:
        for m in models:
            found = False
            for r in rows:
                if r["held_out_pair"] == p and r["model"] == m:
                    pair_scores[m].append(r["switch_f1"])
                    found = True
                    break
            if not found:
                pair_scores[m].append(0)

    x = range(len(pairs))
    width = 0.2

    plt.figure(figsize=(10, 5))

    for i, m in enumerate(models):
        plt.bar(
            [xi + i * width for xi in x],
            pair_scores[m],
            width,
            label=m
        )

    plt.xticks([xi + width for xi in x], pairs, rotation=30)
    plt.ylabel("Switch F1")
    plt.title("Per-Pair Performance (LOPO)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/per_pair.png")
    plt.close()

    print("✓ Saved figures/per_pair.png")


# ================================
# 运行全部
# ================================
if __name__ == "__main__":
    plot_overall()
    plot_universality_summary()
    plot_per_pair()