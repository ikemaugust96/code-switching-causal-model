import matplotlib.pyplot as plt
import numpy as np

# =========================
# Switch F1 数据
# =========================
models = ["3-gram", "GRU"]

cantonese_switch = [0.5667, 0.9299]
arabic_switch = [0.4095, 0.7206]

x = np.arange(len(models))
width = 0.35

plt.figure()
plt.bar(x - width/2, cantonese_switch, width)
plt.bar(x + width/2, arabic_switch, width)

plt.xticks(x, models)
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("Universality: Switch F1 Comparison")

plt.legend(["Cantonese-English", "Arabic-English"])

# 🔥 新文件名（避免冲突）
plt.savefig("universality_switch_f1.png")
plt.close()


# =========================
# Duration Macro F1 数据
# =========================
cantonese_duration = [0.3172, 0.5250]
arabic_duration = [0.2614, 0.4448]

plt.figure()
plt.bar(x - width/2, cantonese_duration, width)
plt.bar(x + width/2, arabic_duration, width)

plt.xticks(x, models)
plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.title("Universality: Duration Macro F1 Comparison")

plt.legend(["Cantonese-English", "Arabic-English"])

# 🔥 新文件名（避免冲突）
plt.savefig("universality_duration_f1.png")
plt.close()

print("Saved:")
print(" - universality_switch_f1.png")
print(" - universality_duration_f1.png")
